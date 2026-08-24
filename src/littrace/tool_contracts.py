from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ToolContract(BaseModel):
    """Stable metadata for a callable LitTrace tool."""

    name: str
    version: str = "v1"
    category: str = "local"
    description: str
    input_schema: str | None = None
    output_schema: str | None = None
    side_effects: list[str] = Field(default_factory=list)
    requires_network: bool = False
    mutates_workspace: bool = False
    allow_in_react: bool = True
    idempotent: bool = True
    cache_policy: str = "none"
    provenance_outputs: list[str] = Field(default_factory=list)
    quality_requirements: list[str] = Field(default_factory=list)
    budget_cost: dict[str, float] = Field(default_factory=dict)

    @property
    def contract_id(self) -> str:
        return f"{self.name}:{self.version}"


class ToolResult(BaseModel, Generic[OutputT]):
    """Uniform result envelope for wrapped tools."""

    tool: str
    contract_id: str
    ok: bool
    output: OutputT | None = None
    output_ref: "ToolArtifactRef | None" = None
    output_bytes_estimate: int = 0
    output_redacted: bool = False
    error: str | None = None
    failure_class: str | None = None
    warnings: list[str] = Field(default_factory=list)
    started_at: str
    elapsed_ms: float
    output_summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ToolExecutionLedger:
    """Task-scoped mutable state for contract budgets and idempotent results."""

    remaining_budget: dict[str, float] = field(default_factory=dict)
    cached_results: dict[str, ToolResult[Any]] = field(default_factory=dict)


class ToolCallContext(BaseModel):
    caller: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    intent: str | None = None
    react_step: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolArtifactRef(BaseModel):
    artifact_id: str
    kind: str
    producer: str
    summary: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionPolicy(BaseModel):
    allow_network: bool = True
    allow_workspace_mutation: bool = True
    allow_side_effects: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    react_only: bool = False
    budget_limits: dict[str, float] = Field(default_factory=dict)

    def violation_for(self, contract: ToolContract) -> str | None:
        if self.allowed_tools and contract.name not in self.allowed_tools:
            return f"Tool '{contract.name}' is not in the allowed tool set."
        if self.react_only and not contract.allow_in_react:
            return f"Tool '{contract.name}' is not allowed in ReAct loops."
        if contract.requires_network and not self.allow_network:
            return f"Tool '{contract.name}' requires network access, but policy forbids it."
        if contract.mutates_workspace and not self.allow_workspace_mutation:
            return f"Tool '{contract.name}' mutates workspace, but policy forbids it."
        forbidden = [
            side_effect
            for side_effect in contract.side_effects
            if side_effect not in self.allow_side_effects
        ]
        if forbidden:
            return (
                f"Tool '{contract.name}' requires forbidden side effects: {', '.join(forbidden)}."
            )
        return None


class ToolCallRecord(BaseModel):
    contract: ToolContract
    context: ToolCallContext = Field(default_factory=ToolCallContext)
    result: ToolResult[Any]


@dataclass
class _PreparedToolCall:
    started_at: str
    started: float
    metadata: dict[str, Any]
    policy_violation: str | None


def _prepare_tool_call(
    contract: ToolContract,
    context: ToolCallContext | None,
    policy: ToolExecutionPolicy | None,
    metadata: dict[str, Any] | None,
) -> _PreparedToolCall:
    context_metadata = context.metadata if context else {}
    return _PreparedToolCall(
        started_at=datetime.now().isoformat(timespec="seconds"),
        started=perf_counter(),
        metadata={**context_metadata, **(metadata or {})},
        policy_violation=policy.violation_for(contract) if policy else None,
    )


def _failed_tool_result(
    contract: ToolContract,
    call: _PreparedToolCall,
    error: str,
) -> ToolResult[Any]:
    return ToolResult(
        tool=contract.name,
        contract_id=contract.contract_id,
        ok=False,
        error=error,
        failure_class=_failure_class(error),
        started_at=call.started_at,
        elapsed_ms=(perf_counter() - call.started) * 1000,
        metadata=call.metadata,
    )


def _failure_class(error: str) -> str:
    lowered = error.lower()
    if "policy" in lowered or "forbid" in lowered:
        return "policy_blocked"
    if "timeout" in lowered or "connect" in lowered or "transport" in lowered:
        return "transient"
    if "valueerror" in lowered or "validation" in lowered:
        return "invalid_input"
    return "source_unavailable"


def _successful_tool_result(
    contract: ToolContract,
    call: _PreparedToolCall,
    value: OutputT,
    output_mode: str,
    output_ref: ToolArtifactRef | None,
) -> ToolResult[OutputT]:
    redacted = output_mode == "summary"
    return ToolResult(
        tool=contract.name,
        contract_id=contract.contract_id,
        ok=True,
        output=None if redacted else value,
        output_ref=output_ref,
        output_bytes_estimate=_estimate_output_bytes(value),
        output_redacted=redacted,
        warnings=_extract_warnings(value),
        started_at=call.started_at,
        elapsed_ms=(perf_counter() - call.started) * 1000,
        output_summary=_summarize_output(value),
        metadata=call.metadata,
    )


def _execution_key(
    contract: ToolContract,
    payload: object,
    context: ToolCallContext | None,
    explicit_key: str | None,
) -> str | None:
    if explicit_key:
        return f"{contract.contract_id}:{explicit_key}"
    if context and isinstance(context.metadata.get("idempotency_key"), str):
        return f"{contract.contract_id}:{context.metadata['idempotency_key']}"
    return None


def _can_reuse_result(contract: ToolContract) -> bool:
    """Only cache read-only tools; workspace updates must always execute."""
    return contract.idempotent and not contract.mutates_workspace


def _budget_violation(
    contract: ToolContract,
    ledger: ToolExecutionLedger | None,
) -> str | None:
    if ledger is None:
        return None
    for name, cost in contract.budget_cost.items():
        if name in ledger.remaining_budget and ledger.remaining_budget[name] < cost:
            return f"Tool '{contract.name}' exceeds remaining {name} budget."
    return None


def _charge_budget(contract: ToolContract, ledger: ToolExecutionLedger | None) -> dict[str, float]:
    if ledger is None:
        return {}
    charged: dict[str, float] = {}
    for name, cost in contract.budget_cost.items():
        if name in ledger.remaining_budget:
            ledger.remaining_budget[name] -= cost
            charged[name] = cost
    return charged


def _result_with_execution_metadata(
    result: ToolResult[OutputT],
    *,
    contract: ToolContract,
    charged: dict[str, float] | None = None,
    reused: bool = False,
) -> ToolResult[OutputT]:
    return result.model_copy(
        update={
            "metadata": {
                **result.metadata,
                "cache_policy": contract.cache_policy,
                "idempotent": contract.idempotent,
                "idempotency_reused": reused,
                "budget_charged": charged or {},
            }
        }
    )


async def run_tool(
    contract: ToolContract,
    func: Callable[[InputT], OutputT] | Callable[[InputT], Awaitable[OutputT]],
    payload: InputT,
    *,
    context: ToolCallContext | None = None,
    policy: ToolExecutionPolicy | None = None,
    metadata: dict[str, Any] | None = None,
    output_mode: str = "inline",
    output_ref: ToolArtifactRef | None = None,
    ledger: ToolExecutionLedger | None = None,
    idempotency_key: str | None = None,
) -> ToolResult[OutputT]:
    call = _prepare_tool_call(contract, context, policy, metadata)
    if call.policy_violation:
        return _failed_tool_result(
            contract,
            call,
            f"ToolPolicyViolation: {call.policy_violation}",
        )
    if ledger is not None and not ledger.remaining_budget:
        ledger.remaining_budget.update(policy.budget_limits if policy else {})
    key = _execution_key(contract, payload, context, idempotency_key)
    if _can_reuse_result(contract) and key and ledger and key in ledger.cached_results:
        return _result_with_execution_metadata(
            ledger.cached_results[key].model_copy(deep=True),
            contract=contract,
            reused=True,
        )
    budget_violation = _budget_violation(contract, ledger)
    if budget_violation:
        return _failed_tool_result(contract, call, f"ToolPolicyViolation: {budget_violation}")
    try:
        value = func(payload)
        if hasattr(value, "__await__"):
            value = await value  # type: ignore[assignment]
        result = _successful_tool_result(contract, call, value, output_mode, output_ref)
        result = _result_with_execution_metadata(
            result,
            contract=contract,
            charged=_charge_budget(contract, ledger),
        )
        if _can_reuse_result(contract) and key and ledger:
            ledger.cached_results[key] = result.model_copy(deep=True)
        return result
    except Exception as exc:
        return _failed_tool_result(contract, call, f"{exc.__class__.__name__}: {exc}")


def run_sync_tool(
    contract: ToolContract,
    func: Callable[[InputT], OutputT],
    payload: InputT,
    *,
    context: ToolCallContext | None = None,
    policy: ToolExecutionPolicy | None = None,
    metadata: dict[str, Any] | None = None,
    output_mode: str = "inline",
    output_ref: ToolArtifactRef | None = None,
    ledger: ToolExecutionLedger | None = None,
    idempotency_key: str | None = None,
) -> ToolResult[OutputT]:
    call = _prepare_tool_call(contract, context, policy, metadata)
    if call.policy_violation:
        return _failed_tool_result(
            contract,
            call,
            f"ToolPolicyViolation: {call.policy_violation}",
        )
    if ledger is not None and not ledger.remaining_budget:
        ledger.remaining_budget.update(policy.budget_limits if policy else {})
    key = _execution_key(contract, payload, context, idempotency_key)
    if _can_reuse_result(contract) and key and ledger and key in ledger.cached_results:
        return _result_with_execution_metadata(
            ledger.cached_results[key].model_copy(deep=True),
            contract=contract,
            reused=True,
        )
    budget_violation = _budget_violation(contract, ledger)
    if budget_violation:
        return _failed_tool_result(contract, call, f"ToolPolicyViolation: {budget_violation}")
    try:
        value = func(payload)
        result = _successful_tool_result(contract, call, value, output_mode, output_ref)
        result = _result_with_execution_metadata(
            result,
            contract=contract,
            charged=_charge_budget(contract, ledger),
        )
        if _can_reuse_result(contract) and key and ledger:
            ledger.cached_results[key] = result.model_copy(deep=True)
        return result
    except Exception as exc:
        return _failed_tool_result(contract, call, f"{exc.__class__.__name__}: {exc}")


LITTRACE_TOOL_CONTRACTS: dict[str, ToolContract] = {
    "skill_creator": ToolContract(
        name="skill_creator",
        category="meta",
        description="Generate a SKILL.md + run.py skeleton for a new skill.",
        input_schema="skill_name + description + when_to_use",
        output_schema="dict[str, str]",
    ),
    "review_agent": ToolContract(
        name="review_agent",
        category="meta",
        description="Self-check the current workspace before submission.",
        input_schema="ChatSession + LiteratureWorkspace",
        output_schema="dict with score and findings",
    ),
    "build_research_plan": ToolContract(
        name="build_research_plan",
        category="planning",
        description="Build an evidence-first plan for a research objective.",
        input_schema="Research objective + LiteratureWorkspace",
        output_schema="ResearchPlan",
    ),
    "search_papers": ToolContract(
        name="search_papers",
        category="retrieval",
        description="Search literature sources and return ranked paper metadata.",
        input_schema="PaperSearchRequest",
        output_schema="PaperSearchResult",
        requires_network=True,
        cache_policy="source_ttl",
        provenance_outputs=["SourceRecord", "CanonicalWork", "ResolutionDecision"],
        quality_requirements=["identity_resolution"],
        budget_cost={"requests": 2.0},
    ),
    "resolve_workspace_full_text": ToolContract(
        name="resolve_workspace_full_text",
        category="access",
        description="Resolve full-text candidates for active workspace papers.",
        input_schema="LiteratureWorkspace",
        output_schema="LiteratureWorkspace",
        requires_network=True,
        mutates_workspace=True,
    ),
    "build_download_plan": ToolContract(
        name="build_download_plan",
        category="access",
        description="Plan compliant local PDF downloads for active workspace papers.",
        input_schema="LiteratureWorkspace",
        output_schema="DownloadPlan",
    ),
    "execute_downloads": ToolContract(
        name="execute_downloads",
        category="access",
        description="Execute compliant PDF downloads for selected active workspace papers.",
        input_schema="DownloadExecutionRequest + LiteratureWorkspace",
        output_schema="DownloadExecutionResult",
        side_effects=["network", "filesystem"],
        requires_network=True,
        idempotent=False,
    ),
    "parse_workspace_papers": ToolContract(
        name="parse_workspace_papers",
        category="document_parsing",
        description="Parse available local PDFs into structured document evidence.",
        input_schema="LiteratureWorkspace",
        output_schema="tuple[LiteratureWorkspace, parse_report]",
        mutates_workspace=True,
    ),
    "audit_citation_links": ToolContract(
        name="audit_citation_links",
        category="evidence",
        description="Check citation links for active papers.",
        input_schema="list[PaperMetadata]",
        output_schema="CitationAudit",
        requires_network=True,
    ),
    "extract_performance_cells": ToolContract(
        name="extract_performance_cells",
        category="extraction",
        description="Extract performance metrics from parsed paper evidence.",
        input_schema="LiteratureWorkspace",
        output_schema="tuple[LiteratureWorkspace, harness_report]",
        mutates_workspace=True,
        provenance_outputs=["EvidenceSpan"],
        quality_requirements=["extraction_verifier"],
    ),
    "build_comparison_matrices": ToolContract(
        name="build_comparison_matrices",
        category="synthesis",
        description="Build comparison matrices from extracted performance cells.",
        input_schema="LiteratureWorkspace",
        output_schema="ComparisonMatrixReport",
    ),
    "build_storyline_from_workspace": ToolContract(
        name="build_storyline_from_workspace",
        category="synthesis",
        description="Build evidence-linked storyline claims from workspace papers.",
        input_schema="LiteratureWorkspace",
        output_schema="list[StorylineClaim]",
    ),
    "build_research_document_report": ToolContract(
        name="build_research_document_report",
        category="synthesis",
        description="Compose an auditable research document from workspace evidence.",
        input_schema="LiteratureWorkspace",
        output_schema="ResearchDocumentReport",
        provenance_outputs=["Claim", "VerificationReport", "ReleaseSnapshot"],
        quality_requirements=["claim_verifier", "publication_gate"],
    ),
    "build_quality_metrics": ToolContract(
        name="build_quality_metrics",
        category="evaluation",
        description="Compute retrieval, parsing, storyline, and interaction quality metrics.",
        input_schema="LiteratureWorkspace",
        output_schema="dict[str, object]",
    ),
    "quality_report": ToolContract(
        name="quality_report",
        category="evaluation",
        description="Build a quality report for the current workspace.",
        input_schema="LiteratureWorkspace + LitTraceConfig",
        output_schema="QualityReport",
    ),
    "export_session_bundle": ToolContract(
        name="export_session_bundle",
        category="export",
        description="Export a session bundle of research artifacts and summaries.",
        input_schema="ChatSession + LiteratureWorkspace",
        output_schema="dict[str, str]",
        mutates_workspace=False,
        provenance_outputs=["ReleaseSnapshot"],
        quality_requirements=["publication_gate"],
    ),
}


def tool_contract(name: str) -> ToolContract:
    return LITTRACE_TOOL_CONTRACTS[name]


def list_tool_contracts() -> list[ToolContract]:
    return list(LITTRACE_TOOL_CONTRACTS.values())


def tool_contract_summary() -> dict[str, object]:
    contracts = list_tool_contracts()
    categories: dict[str, int] = {}
    for contract in contracts:
        categories[contract.category] = categories.get(contract.category, 0) + 1
    return {
        "schema": "littrace.tool_contract_summary.v1",
        "count": len(contracts),
        "categories": categories,
        "network_tools": [contract.name for contract in contracts if contract.requires_network],
        "idempotent_tools": [contract.name for contract in contracts if contract.idempotent],
        "cached_tools": [
            contract.name for contract in contracts if contract.cache_policy != "none"
        ],
        "workspace_mutation_tools": [
            contract.name for contract in contracts if contract.mutates_workspace
        ],
        "react_allowed_tools": [contract.name for contract in contracts if contract.allow_in_react],
    }


def _extract_warnings(value: object) -> list[str]:
    warnings: list[str] = []
    if isinstance(value, tuple):
        for item in value:
            warnings.extend(_extract_warnings(item))
        return warnings
    if isinstance(value, dict):
        raw = value.get("warnings")
        if isinstance(raw, list):
            warnings.extend(str(item) for item in raw)
    else:
        raw = getattr(value, "warnings", None)
        if isinstance(raw, list):
            warnings.extend(str(item) for item in raw)
    return warnings


def _summarize_output(value: object) -> dict[str, Any]:
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_summarize_output(item) for item in value],
        }
    if isinstance(value, list):
        return {"type": "list", "count": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(key) for key in value.keys())[:20]}
    summary: dict[str, Any] = {"type": value.__class__.__name__}
    for attr in ("papers", "items", "matrices", "sections", "records", "claims"):
        raw = getattr(value, attr, None)
        if isinstance(raw, list):
            summary[f"{attr}_count"] = len(raw)
    return summary


def _estimate_output_bytes(value: object) -> int:
    try:
        if hasattr(value, "model_dump_json"):
            return len(value.model_dump_json())
        return len(str(value).encode("utf-8"))
    except Exception:
        return 0


ToolResult.model_rebuild()
