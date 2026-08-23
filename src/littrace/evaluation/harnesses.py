"""Harness Engine — pluggable quality-gate system for LitTrace.

Architecture:
    HarnessConfig      — centralised thresholds (replaces hardcoded magic numbers)
    HarnessResult      — legacy flat result (kept for backward compatibility)
    HarnessReport      — structured per-check report with severity, remediation hints
    HarnessCheck       — base class for registered quality checks
    HarnessRegistry    — decorator-based registry for plug-in checks
    HarnessEngine      — orchestrates multiple checks with dependency ordering

Existing check_* functions (check_citations, check_performance_cells, etc.) are
preserved as thin wrappers that delegate to registered checks, so all current
call sites continue to work without modification.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from littrace.models import (
    CitationRecord,
    LinkStatus,
    PerformanceCell,
    StorylineClaim,
    StructuredArtifact,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class HarnessConfig(BaseModel):
    """Centralised thresholds for all harness checks.

    Defaults match the original hardcoded values so behaviour is unchanged.
    """

    # check_performance_cells
    performance_confidence_threshold: float = 0.65

    # check_structured_artifacts
    artifact_confidence_threshold: float = 0.6
    allowed_artifact_types: set[str] = Field(
        default_factory=lambda: {"table", "figure", "equation", "formula"}
    )

    # check_storyline_claims
    storyline_confidence_threshold: float = 0.7
    storyline_chain_min_evidence: int = 3
    storyline_multi_paper_types: set[str] = Field(
        default_factory=lambda: {"trend_by_year_and_method", "later_response"}
    )
    allowed_storyline_types: set[str] = Field(
        default_factory=lambda: {
            "prior_solution",
            "remaining_limitation",
            "later_response",
            "solution_limit_response_chain",
            "unresolved_gap",
            "trend_by_year_and_method",
        }
    )

    # Dimension 1: Retry health thresholds
    max_retry_rate: float = 0.5
    max_failure_rate: float = 0.2

    # Dimension 3: Schema validation
    schema_strict: bool = True
    schema_enabled: bool = True

    # Dimension 4: Cost budget
    budget_warning_threshold: float = 0.8

    @classmethod
    def from_littrace_config(cls, config: Any) -> HarnessConfig:
        """Build HarnessConfig from a LitTraceConfig instance.

        Reads harness, retry, cost_budget, and schema_validation configs.
        """
        harness_cfg = getattr(config, "harness", None)
        retry_cfg = getattr(config, "retry", None)
        cost_cfg = getattr(config, "cost_budget", None)
        schema_cfg = getattr(config, "schema_validation", None)

        kwargs: dict[str, Any] = {}
        if harness_cfg is not None:
            kwargs.update(
                performance_confidence_threshold=harness_cfg.performance_confidence,
                artifact_confidence_threshold=harness_cfg.artifact_confidence,
                storyline_confidence_threshold=harness_cfg.storyline_confidence,
                storyline_chain_min_evidence=harness_cfg.storyline_chain_min_evidence,
            )
        if retry_cfg is not None:
            kwargs["max_retry_rate"] = retry_cfg.max_retry_rate
            kwargs["max_failure_rate"] = retry_cfg.max_failure_rate
        if cost_cfg is not None:
            kwargs["budget_warning_threshold"] = cost_cfg.budget_warning_threshold
        # schema_validation used to be a nested SchemaValidationConfig; the
        # 0a85241 refactor flattened it into top-level booleans on
        # LitTraceConfig. Prefer the nested object when present, otherwise
        # fall back to the flat fields (which always exist on the config).
        if schema_cfg is not None:
            kwargs["schema_strict"] = schema_cfg.strict
            kwargs["schema_enabled"] = schema_cfg.enabled
        else:
            kwargs["schema_strict"] = getattr(config, "schema_validation_strict", True)
            kwargs["schema_enabled"] = getattr(config, "schema_validation_enabled", True)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class HarnessResult(BaseModel):
    """Legacy flat result — kept for backward compatibility."""

    passed: bool
    score: float
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Severity(str):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class HarnessFinding(BaseModel):
    """A single issue found by a harness check."""

    severity: str = Severity.ERROR
    message: str
    paper_id: str | None = None
    remediation_hint: str | None = None


class HarnessReport(BaseModel):
    """Structured per-check report."""

    check_name: str
    passed: bool
    score: float
    findings: list[HarnessFinding] = Field(default_factory=list)
    checked_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    item_count: int = 0

    @property
    def errors(self) -> list[str]:
        """Backward-compatible: error messages as flat strings."""
        return [f.message for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[str]:
        """Backward-compatible: warning messages as flat strings."""
        return [f.message for f in self.findings if f.severity == Severity.WARNING]

    def to_result(self) -> HarnessResult:
        """Convert to legacy HarnessResult."""
        return HarnessResult(
            passed=self.passed,
            score=self.score,
            errors=self.errors,
            warnings=self.warnings,
        )


# ---------------------------------------------------------------------------
# Check protocol & registry
# ---------------------------------------------------------------------------


@runtime_checkable
class CheckCallable(Protocol):
    """Protocol for registered check functions."""

    def __call__(self, items: list[Any], config: HarnessConfig | None = None) -> HarnessReport: ...


class HarnessCheck:
    """Base class for registered harness checks.

    Subclasses (or functions registered via @register_check) implement the
    actual checking logic. Each check has:
        - name: unique identifier
        - group: logical grouping (e.g. "tables", "storyline", "citations")
        - depends_on: names of checks that must run before this one
    """

    name: str = ""
    group: str = "default"
    depends_on: list[str] = Field(default_factory=list)

    def run(self, items: list[Any], config: HarnessConfig | None = None) -> HarnessReport:
        raise NotImplementedError


class _FunctionCheck(HarnessCheck):
    """Wraps a plain function as a HarnessCheck."""

    def __init__(
        self,
        func: CheckCallable,
        name: str,
        group: str = "default",
        depends_on: list[str] | None = None,
    ):
        self._func = func
        self.name = name
        self.group = group
        self.depends_on = depends_on or []

    def run(self, items: list[Any], config: HarnessConfig | None = None) -> HarnessReport:
        return self._func(items, config)


class HarnessRegistry:
    """Registry for harness checks — supports decorator and programmatic registration."""

    def __init__(self) -> None:
        self._checks: dict[str, HarnessCheck] = {}

    def register(
        self,
        name: str | None = None,
        group: str = "default",
        depends_on: list[str] | None = None,
    ) -> callable:
        """Decorator to register a check function."""

        def decorator(func: CheckCallable) -> CheckCallable:
            check_name = name or func.__name__
            self._checks[check_name] = _FunctionCheck(func, check_name, group, depends_on)
            return func

        return decorator

    def register_instance(self, check: HarnessCheck) -> None:
        """Programmatically register a check instance."""
        if not check.name:
            raise ValueError("Check must have a non-empty name")
        self._checks[check.name] = check

    def get(self, name: str) -> HarnessCheck | None:
        return self._checks.get(name)

    def list_checks(self) -> list[str]:
        return list(self._checks.keys())

    def checks_in_group(self, group: str) -> list[str]:
        return [name for name, check in self._checks.items() if check.group == group]

    def all_checks(self) -> dict[str, HarnessCheck]:
        return dict(self._checks)


# Global registry instance
registry = HarnessRegistry()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class HarnessEngine:
    """Orchestrates multiple harness checks with dependency ordering.

    Usage:
        engine = HarnessEngine(registry, config)
        report = engine.run("check_citations", citation_records)
        # Run all checks in a group:
        reports = engine.run_group("tables", [cells, artifacts])
        # Run with dependency resolution:
        reports = engine.run_with_deps("check_storyline_claims", items_map)
    """

    def __init__(
        self,
        reg: HarnessRegistry | None = None,
        config: HarnessConfig | None = None,
    ):
        self.registry = reg or registry
        self.config = config or HarnessConfig()

    def run(self, check_name: str, items: list[Any]) -> HarnessReport:
        """Run a single registered check by name."""
        check = self.registry.get(check_name)
        if check is None:
            raise KeyError(f"Harness check '{check_name}' is not registered")
        return check.run(items, self.config)

    def run_group(self, group: str, items_map: dict[str, list[Any]]) -> dict[str, HarnessReport]:
        """Run all checks in a group.

        Args:
            group: check group name
            items_map: maps check name to its input items
        """
        results: dict[str, HarnessReport] = {}
        for name in self.registry.checks_in_group(group):
            items = items_map.get(name, [])
            results[name] = self.run(name, items)
        return results

    def run_with_deps(
        self, check_name: str, items_map: dict[str, list[Any]]
    ) -> dict[str, HarnessReport]:
        """Run a check and all its dependencies in topological order.

        Args:
            check_name: target check to run
            items_map: maps check name to its input items
        """
        ordered = self._resolve_deps(check_name)
        results: dict[str, HarnessReport] = {}
        for name in ordered:
            items = items_map.get(name, [])
            results[name] = self.run(name, items)
        return results

    def _resolve_deps(self, check_name: str) -> list[str]:
        """Topological sort of a check and its dependencies."""
        visited: set[str] = set()
        ordered: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            check = self.registry.get(name)
            if check is not None:
                for dep in check.depends_on:
                    visit(dep)
            ordered.append(name)

        visit(check_name)
        return ordered

    def combine(self, reports: list[HarnessReport]) -> HarnessReport:
        """Combine multiple reports into one aggregate report."""
        if not reports:
            return HarnessReport(
                check_name="combined",
                passed=True,
                score=1.0,
                item_count=0,
            )
        total_score = sum(r.score for r in reports) / len(reports)
        all_passed = all(r.passed for r in reports)
        all_findings: list[HarnessFinding] = []
        for r in reports:
            all_findings.extend(r.findings)
        return HarnessReport(
            check_name="combined",
            passed=all_passed,
            score=total_score,
            findings=all_findings,
            item_count=sum(r.item_count for r in reports),
        )


# ---------------------------------------------------------------------------
# Built-in checks — registered via decorator
# ---------------------------------------------------------------------------


@registry.register(name="check_citations", group="citations")
def check_citations(
    records: list[CitationRecord],
    config: HarnessConfig | None = None,
) -> HarnessReport:
    """Verify citation records have verified links and non-empty text."""
    config = config or HarnessConfig()
    findings: list[HarnessFinding] = []

    for record in records:
        if record.link_status in {LinkStatus.FAILED, LinkStatus.UNCHECKED}:
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=f"{record.paper_id}: access link is not verified",
                    paper_id=record.paper_id,
                    remediation_hint="Re-run citation link audit or manually verify the URL.",
                )
            )
        if not record.citation_text.strip():
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=f"{record.paper_id}: missing citation text",
                    paper_id=record.paper_id,
                    remediation_hint="Fetch the paper metadata to populate citation text.",
                )
            )

    total = max(len(records), 1)
    score = (total - len([f for f in findings if f.severity == Severity.ERROR])) / total
    errors = [f for f in findings if f.severity == Severity.ERROR]
    return HarnessReport(
        check_name="check_citations",
        passed=not errors,
        score=score,
        findings=findings,
        item_count=len(records),
    )


@registry.register(name="check_performance_cells", group="tables")
def check_performance_cells(
    cells: list[PerformanceCell],
    config: HarnessConfig | None = None,
) -> HarnessReport:
    """Verify performance cells have traceable evidence and sufficient confidence."""
    config = config or HarnessConfig()
    threshold = config.performance_confidence_threshold
    findings: list[HarnessFinding] = []

    for cell in cells:
        evidence = cell.evidence
        if evidence.page is None and evidence.snippet is None:
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=f"{cell.paper_id}: metric {cell.metric} lacks traceable evidence",
                    paper_id=cell.paper_id,
                    remediation_hint="Re-parse the paper with OCR to capture page number or snippet.",
                )
            )
        if cell.higher_is_better is None:
            findings.append(
                HarnessFinding(
                    severity=Severity.WARNING,
                    message=f"{cell.paper_id}: metric direction missing for {cell.metric}",
                    paper_id=cell.paper_id,
                    remediation_hint="Add higher_is_better to the metric definition in METRIC_DIRECTIONS.",
                )
            )
        if evidence.confidence < threshold:
            findings.append(
                HarnessFinding(
                    severity=Severity.WARNING,
                    message=f"{cell.paper_id}: low extraction confidence for {cell.metric}",
                    paper_id=cell.paper_id,
                    remediation_hint=f"Confidence below {threshold}. Consider re-parsing with a different engine.",
                )
            )

    total = max(len(cells), 1)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    score = (total - len(errors)) / total
    return HarnessReport(
        check_name="check_performance_cells",
        passed=not errors,
        score=score,
        findings=findings,
        item_count=len(cells),
    )


@registry.register(name="check_structured_artifacts", group="tables")
def check_structured_artifacts(
    artifacts: list[StructuredArtifact],
    config: HarnessConfig | None = None,
) -> HarnessReport:
    """Verify structured artifacts have valid type, content, and evidence."""
    config = config or HarnessConfig()
    threshold = config.artifact_confidence_threshold
    allowed = config.allowed_artifact_types
    findings: list[HarnessFinding] = []

    for artifact in artifacts:
        if artifact.artifact_type not in allowed:
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=f"{artifact.paper_id}: unsupported artifact type {artifact.artifact_type}",
                    paper_id=artifact.paper_id,
                    remediation_hint=f"Use one of: {', '.join(sorted(allowed))}.",
                )
            )
        if not artifact.text.strip():
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=f"{artifact.paper_id}: empty {artifact.artifact_type} artifact",
                    paper_id=artifact.paper_id,
                    remediation_hint="Re-extract the artifact with a different parser.",
                )
            )
        if artifact.evidence.page is None and artifact.evidence.snippet is None:
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=f"{artifact.paper_id}: {artifact.artifact_type} lacks evidence",
                    paper_id=artifact.paper_id,
                    remediation_hint="Capture page number or snippet during extraction.",
                )
            )
        if artifact.confidence < threshold:
            findings.append(
                HarnessFinding(
                    severity=Severity.WARNING,
                    message=f"{artifact.paper_id}: low-confidence {artifact.artifact_type} artifact",
                    paper_id=artifact.paper_id,
                    remediation_hint=f"Confidence below {threshold}. Consider manual review.",
                )
            )

    total = max(len(artifacts), 1)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    score = (total - len(errors)) / total
    return HarnessReport(
        check_name="check_structured_artifacts",
        passed=not errors,
        score=score,
        findings=findings,
        item_count=len(artifacts),
    )


@registry.register(name="check_storyline_claims", group="storyline")
def check_storyline_claims(
    claims: list[StorylineClaim],
    config: HarnessConfig | None = None,
) -> HarnessReport:
    """Verify storyline claims are grounded and properly typed."""
    config = config or HarnessConfig()
    threshold = config.storyline_confidence_threshold
    allowed = config.allowed_storyline_types
    multi_paper_types = config.storyline_multi_paper_types
    chain_min = config.storyline_chain_min_evidence
    findings: list[HarnessFinding] = []

    for claim in claims:
        if claim.claim_type not in allowed:
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=f"Unsupported storyline claim type: {claim.claim_type}",
                    remediation_hint=f"Use one of: {', '.join(sorted(allowed))}.",
                )
            )
        unique_papers = {item.paper_id for item in claim.evidence}
        if len(unique_papers) < 1:
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=f"Ungrounded storyline claim: {claim.claim}",
                    remediation_hint="Add at least one evidence paper to support this claim.",
                )
            )
        if len(unique_papers) < 2 and claim.claim_type in multi_paper_types:
            findings.append(
                HarnessFinding(
                    severity=Severity.WARNING,
                    message=f"Claim should have at least two supporting papers: {claim.claim}",
                    remediation_hint="Find additional papers that support this trend or response.",
                )
            )
        if claim.claim_type == "solution_limit_response_chain" and len(claim.evidence) < chain_min:
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=f"Storyline chain lacks solution-limit-response evidence: {claim.claim}",
                    remediation_hint=f"Provide at least {chain_min} evidence items for chain claims.",
                )
            )
        if claim.confidence < threshold:
            findings.append(
                HarnessFinding(
                    severity=Severity.WARNING,
                    message=f"Low-confidence storyline claim: {claim.claim}",
                    remediation_hint=f"Confidence below {threshold}. Strengthen evidence or re-evaluate.",
                )
            )

    total = max(len(claims), 1)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    score = (total - len(errors)) / total
    return HarnessReport(
        check_name="check_storyline_claims",
        passed=not errors,
        score=score,
        findings=findings,
        item_count=len(claims),
    )


# ===========================================================================
# Dimension 1: Retry & Fallback Health Check
# ===========================================================================


@dataclass
class RetryHealthItem:
    """Input item for retry health check — describes one operation's retry stats."""

    operation: str
    total_calls: int
    total_retries: int
    failed_calls: int
    retry_rate: float = 0.0
    failure_rate: float = 0.0


@registry.register(name="check_retry_health", group="reliability")
def check_retry_health(
    items: list[RetryHealthItem],
    config: HarnessConfig | None = None,
) -> HarnessReport:
    """Assess retry health across operations.

    Flags operations with high retry rates or failure rates, indicating
    unreliable external dependencies or insufficient retry configuration.
    """
    config = config or HarnessConfig()
    # Read thresholds from config (attached dynamically by from_littrace_config)
    max_retry_rate = getattr(config, "max_retry_rate", 0.5)
    max_failure_rate = getattr(config, "max_failure_rate", 0.2)
    findings: list[HarnessFinding] = []

    for item in items:
        if item.failure_rate > max_failure_rate:
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=(
                        f"Operation '{item.operation}' has failure rate "
                        f"{item.failure_rate:.1%} (threshold {max_failure_rate:.1%})"
                    ),
                    remediation_hint=(
                        "Check if the external service is down, increase max_attempts, "
                        "or add a fallback endpoint."
                    ),
                )
            )
        if item.retry_rate > max_retry_rate:
            findings.append(
                HarnessFinding(
                    severity=Severity.WARNING,
                    message=(
                        f"Operation '{item.operation}' has retry rate "
                        f"{item.retry_rate:.1%} (threshold {max_retry_rate:.1%})"
                    ),
                    remediation_hint=(
                        "High retry rate suggests intermittent failures. "
                        "Consider increasing base_delay or switching backoff strategy."
                    ),
                )
            )
        if item.failed_calls > 0 and item.total_retries == 0:
            findings.append(
                HarnessFinding(
                    severity=Severity.WARNING,
                    message=(
                        f"Operation '{item.operation}' had {item.failed_calls} failures "
                        f"with 0 retries — retry may not be configured"
                    ),
                    remediation_hint=(
                        "Wrap the operation with @retry_async to enable automatic retries."
                    ),
                )
            )

    total = max(len(items), 1)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    score = (total - len(errors)) / total
    return HarnessReport(
        check_name="check_retry_health",
        passed=not errors,
        score=score,
        findings=findings,
        item_count=len(items),
    )


# ===========================================================================
# Dimension 2: Hallucination / Grounding Check
# ===========================================================================


@dataclass
class HallucinationCheckItem:
    """Input item for hallucination grounding check.

    Describes one LLM-generated text segment and its anchor statistics.
    """

    text: str
    checked_sentence_count: int
    unsupported_sentence_count: int
    unsupported_sentences: list[str]
    source: str = "chat"  # "chat", "storyline", "table_extraction", etc.


@registry.register(name="check_hallucination_grounding", group="grounding")
def check_hallucination_grounding(
    items: list[HallucinationCheckItem],
    config: HarnessConfig | None = None,
) -> HarnessReport:
    """Verify that LLM-generated text is grounded in evidence.

    Integrates citation_guard results into the harness system.
    Flags any LLM output containing sentences that lack citation anchors.
    """
    config = config or HarnessConfig()
    findings: list[HarnessFinding] = []

    for item in items:
        if item.unsupported_sentence_count > 0:
            severity = Severity.ERROR if item.unsupported_sentence_count > 2 else Severity.WARNING
            preview = item.unsupported_sentences[0][:120]
            findings.append(
                HarnessFinding(
                    severity=severity,
                    message=(
                        f"[{item.source}] {item.unsupported_sentence_count} unsupported sentences "
                        f"detected (out of {item.checked_sentence_count} checked). "
                        f'First: "{preview}..."'
                    ),
                    remediation_hint=(
                        "Remove unsupported sentences or add paper_id/DOI/title anchors. "
                        "Use remove_unsupported_sentences() to auto-strip them."
                    ),
                )
            )
        if item.checked_sentence_count == 0 and item.text.strip():
            findings.append(
                HarnessFinding(
                    severity=Severity.WARNING,
                    message=f"[{item.source}] Text was generated but no claim-bearing sentences were checked",
                    remediation_hint=(
                        "The CLAIM_HINTS word list may not cover this domain. "
                        "Consider extending it or lowering the checking threshold."
                    ),
                )
            )

    total = max(len(items), 1)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    score = (total - len(errors)) / total
    return HarnessReport(
        check_name="check_hallucination_grounding",
        passed=not errors,
        score=score,
        findings=findings,
        item_count=len(items),
    )


# ===========================================================================
# Dimension 3: Schema Compliance Check
# ===========================================================================


@dataclass
class SchemaCheckItem:
    """Input item for schema compliance check.

    Describes one LLM output and its validation result.
    """

    source: str  # "tables.py:extract_performance_cells", etc.
    total_items: int
    valid_items: int
    invalid_items: list[str]  # per-item error messages
    schema_name: str = ""  # e.g. "PerformanceCell", "ChatIntent"


@registry.register(name="check_schema_compliance", group="schema")
def check_schema_compliance(
    items: list[SchemaCheckItem],
    config: HarnessConfig | None = None,
) -> HarnessReport:
    """Verify that LLM outputs conform to expected Pydantic schemas.

    Catches schema violations that would otherwise be silently dropped
    by json.loads + manual field extraction.
    """
    config = config or HarnessConfig()
    strict = getattr(config, "schema_strict", True)
    findings: list[HarnessFinding] = []

    for item in items:
        invalid_count = len(item.invalid_items)
        if invalid_count == 0:
            continue
        for error_msg in item.invalid_items[:10]:  # cap findings per source
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR if strict else Severity.WARNING,
                    message=f"[{item.source}] Schema violation in {item.schema_name}: {error_msg}",
                    remediation_hint=(
                        "Fix the LLM prompt to produce schema-compliant JSON, "
                        "or add a post-processing step to normalize the output."
                    ),
                )
            )
        if invalid_count > 10:
            findings.append(
                HarnessFinding(
                    severity=Severity.WARNING,
                    message=f"[{item.source}] {invalid_count - 10} additional schema violations suppressed",
                    remediation_hint="Review the LLM output format and prompt engineering.",
                )
            )
        if item.valid_items == 0 and item.total_items > 0:
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=f"[{item.source}] All {item.total_items} items failed schema validation",
                    remediation_hint="The LLM output is completely malformed. Check if the model or prompt changed.",
                )
            )

    total = max(sum(max(item.total_items, 0) for item in items), 1)
    valid = sum(max(0, min(item.valid_items, item.total_items)) for item in items)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    score = valid / total
    return HarnessReport(
        check_name="check_schema_compliance",
        passed=not errors,
        score=score,
        findings=findings,
        item_count=len(items),
    )


# ===========================================================================
# Dimension 4: Cost Budget Check
# ===========================================================================


@dataclass
class CostCheckItem:
    """Input item for cost budget check.

    Describes current cost tracker state.
    """

    total_tokens: int
    total_cost_usd: float
    max_total_tokens: int  # 0 = unlimited
    budget_warning_threshold: float = 0.8
    by_model: dict[str, Any] = field(default_factory=dict)


@registry.register(name="check_cost_budget", group="cost")
def check_cost_budget(
    items: list[CostCheckItem],
    config: HarnessConfig | None = None,
) -> HarnessReport:
    """Verify that token usage stays within configured budget.

    Flags when cumulative token usage approaches or exceeds the budget.
    """
    config = config or HarnessConfig()
    findings: list[HarnessFinding] = []

    for item in items:
        if item.max_total_tokens > 0 and item.total_tokens >= item.max_total_tokens:
            findings.append(
                HarnessFinding(
                    severity=Severity.ERROR,
                    message=(
                        f"Token budget exhausted: {item.total_tokens} / {item.max_total_tokens} "
                        f"tokens used ({item.total_cost_usd:.4f} USD)"
                    ),
                    remediation_hint=(
                        "Increase max_total_tokens in config, or reduce the number of "
                        "LLM calls. Consider caching or batching prompts."
                    ),
                )
            )
        elif item.max_total_tokens > 0:
            usage_ratio = item.total_tokens / item.max_total_tokens
            if usage_ratio >= item.budget_warning_threshold:
                findings.append(
                    HarnessFinding(
                        severity=Severity.WARNING,
                        message=(
                            f"Token budget approaching limit: {usage_ratio:.1%} used "
                            f"({item.total_tokens} / {item.max_total_tokens} tokens, "
                            f"{item.total_cost_usd:.4f} USD)"
                        ),
                        remediation_hint="Monitor usage closely. Consider reducing LLM calls or increasing budget.",
                    )
                )
        # Per-model anomaly detection
        for model, stats in item.by_model.items():
            calls = stats.get("calls", 0)
            tokens = stats.get("total_tokens", 0)
            if calls > 0 and tokens > 0:
                avg_tokens = tokens / calls
                if avg_tokens > 100_000:
                    findings.append(
                        HarnessFinding(
                            severity=Severity.WARNING,
                            message=(
                                f"Model '{model}' has high avg tokens/call: "
                                f"{avg_tokens:.0f} ({calls} calls, {tokens} tokens)"
                            ),
                            remediation_hint=(
                                "High token usage per call may indicate overly long prompts. "
                                "Consider truncating context or using a more efficient prompt."
                            ),
                        )
                    )

    total = max(len(items), 1)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    score = (total - len(errors)) / total
    return HarnessReport(
        check_name="check_cost_budget",
        passed=not errors,
        score=score,
        findings=findings,
        item_count=len(items),
    )


# ---------------------------------------------------------------------------
# Dimension 5: Rate limiting health
# ---------------------------------------------------------------------------


class RateLimitHealthItem(BaseModel):
    """Snapshot of the rate limiter state for health checking."""

    max_concurrent: int = 0
    max_rpm: int = 0
    current_window: int = 0
    total_acquired: int = 0
    total_waited: int = 0
    # If True, rate limiting is configured but was never used
    configured_but_unused: bool = False


@registry.register(name="check_rate_limit", group="reliability")
def check_rate_limit(
    items: list[RateLimitHealthItem],
    config: HarnessConfig | None = None,
) -> HarnessReport:
    """Assess rate limiter health and configuration.

    Flags:
    - WARNING if rate limiting is not configured (max_concurrent=0 and max_rpm=0)
      but LLM calls are being made.
    - WARNING if rate limiting is configured but total_waited is high relative
      to total_acquired (indicates throughput bottleneck).
    - INFO if the limiter is configured but never used.
    """
    config = config or HarnessConfig()
    findings: list[HarnessFinding] = []

    for item in items:
        if item.max_concurrent == 0 and item.max_rpm == 0:
            findings.append(
                HarnessFinding(
                    severity=Severity.WARNING,
                    message=(
                        "Rate limiting is not configured. LLM API calls have no "
                        "concurrency or RPM protection."
                    ),
                    remediation_hint=(
                        "Set rate_limit.max_concurrent and/or rate_limit.max_requests_per_minute "
                        "in config.yaml to protect against API quota exhaustion."
                    ),
                )
            )
        elif item.configured_but_unused:
            findings.append(
                HarnessFinding(
                    severity=Severity.INFO,
                    message=("Rate limiter is configured but no LLM calls have been made yet."),
                )
            )
        else:
            # Check if wait ratio is high (bottleneck indicator)
            if item.total_acquired > 0:
                wait_ratio = item.total_waited / item.total_acquired
                if wait_ratio > 0.5:
                    findings.append(
                        HarnessFinding(
                            severity=Severity.WARNING,
                            message=(
                                f"Rate limiter wait ratio is high: {item.total_waited}/"
                                f"{item.total_acquired} ({wait_ratio:.1%}) requests had to wait. "
                                f"This indicates the RPM limit ({item.max_rpm}) may be too low "
                                f"for current workload."
                            ),
                            remediation_hint=(
                                "Consider increasing max_requests_per_minute or "
                                "max_concurrent in the rate_limit config."
                            ),
                        )
                    )

    total = max(len(items), 1)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    score = (total - len(errors)) / total
    return HarnessReport(
        check_name="check_rate_limit",
        passed=not errors,
        score=score,
        findings=findings,
        item_count=len(items),
    )
