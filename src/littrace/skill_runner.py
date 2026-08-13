from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from littrace.access_layer.download_planning import build_download_plan
from littrace.citations import audit_citation_links
from littrace.config import LitTraceConfig
from littrace.evidence.document_composer import build_research_document_report
from littrace.retrieval.full_text import resolve_workspace_full_text
from littrace.models import (
    DownloadExecutionRequest,
    DownloadExecutionResult,
    LiteratureWorkspace,
    PaperMetadata,
    PaperSearchRequest,
    PaperSearchResult,
)
from littrace.evidence.parsing import parse_workspace_papers
from littrace.evaluation.quality_report import build_quality_report
from littrace.research_planner import build_research_plan
from littrace.retrieval.search import LiveSearchClient, MockMaterialsSearchClient, SearchDiagnostics
from littrace.evidence.storyline import build_storyline_from_workspace
from littrace.evidence.tables import build_comparison_matrices, extract_performance_cells
from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
    ToolResult,
    run_sync_tool,
    run_tool,
    tool_contract,
)


@dataclass
class SearchSkillResult:
    result: PaperSearchResult
    diagnostics: SearchDiagnostics | None
    use_live: bool
    tool_result: ToolResult[PaperSearchResult]


async def _run_async_skill(
    name: str,
    func: Any,
    payload: Any,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolResult[Any]:
    return await run_tool(
        tool_contract(name),
        func,
        payload,
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
        metadata={"skill": name, **(metadata or {})},
    )


def _run_sync_skill(
    name: str,
    func: Any,
    payload: Any,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> ToolResult[Any]:
    return run_sync_tool(
        tool_contract(name),
        func,
        payload,
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
        metadata={"skill": name},
    )


def _output_or_raise(result: ToolResult[Any], skill_name: str) -> Any:
    if not result.ok or result.output is None:
        raise RuntimeError(result.error or f"{skill_name} failed")
    return result.output


def _active_papers(workspace: LiteratureWorkspace) -> list[PaperMetadata]:
    return [
        workspace.papers[paper_id]
        for paper_id in workspace.context.active_papers
        if paper_id in workspace.papers
    ]


async def search_papers_skill(
    request: PaperSearchRequest,
    config: LitTraceConfig,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> SearchSkillResult:
    use_live = config.api.enable_live_search if request.live is None else request.live
    client = LiveSearchClient(config, progress_callback=progress_callback) if use_live else MockMaterialsSearchClient()
    result = await _run_async_skill(
        "search_papers",
        client.fetch,
        request,
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
        metadata={"live": use_live},
    )
    diagnostics = client.diagnostics if use_live else None
    if not result.ok or result.output is None:
        if diagnostics:
            diagnostics.errors.append(result.error or "search_papers failed")
        return SearchSkillResult(
            result=PaperSearchResult(request=request, papers=[]),
            diagnostics=diagnostics,
            use_live=use_live,
            tool_result=result,
        )
    return SearchSkillResult(
        result=result.output,
        diagnostics=diagnostics,
        use_live=use_live,
        tool_result=result,
    )


async def parse_workspace_skill(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> tuple[LiteratureWorkspace, dict[str, object]]:
    result = await _run_async_skill(
        "parse_workspace_papers",
        lambda payload: parse_workspace_papers(payload["workspace"], payload["config"]),
        {"workspace": workspace, "config": config},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "parse_workspace_papers")


async def build_research_plan_skill(
    topic: str,
    workspace: LiteratureWorkspace,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    result = await _run_async_skill(
        "build_research_plan",
        lambda payload: build_research_plan(payload["topic"], payload["workspace"]),
        {"topic": topic, "workspace": workspace},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "build_research_plan")


async def extract_tables_skill(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> tuple[LiteratureWorkspace, object]:
    result = await _run_async_skill(
        "extract_performance_cells",
        lambda payload: extract_performance_cells(payload["workspace"], payload["config"]),
        {"workspace": workspace, "config": config},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "extract_performance_cells")


async def build_download_plan_skill(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    result = await _run_async_skill(
        "build_download_plan",
        lambda payload: build_download_plan(
            payload["config"], payload["papers"], set(payload["selected_for_download"])
        ),
        {
            "config": config,
            "papers": _active_papers(workspace),
            "selected_for_download": workspace.context.selected_for_download,
        },
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "build_download_plan")


async def build_research_report_skill(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    *,
    title: str | None = None,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    result = await _run_async_skill(
        "build_research_document_report",
        lambda payload: build_research_document_report(
            payload["workspace"], payload["config"], title=payload["title"]
        ),
        {"workspace": workspace, "config": config, "title": title},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "build_research_document_report")


async def export_session_bundle_skill(
    session,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> dict[str, str]:
    from littrace.export import export_session_bundle

    result = await _run_async_skill(
        "export_session_bundle",
        lambda payload: export_session_bundle(
            payload["session"], payload["workspace"], payload["config"]
        ),
        {"session": session, "workspace": workspace, "config": config},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "export_session_bundle")


async def audit_citation_links_skill(
    papers: list[PaperMetadata],
    config: LitTraceConfig,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    result = await _run_async_skill(
        "audit_citation_links",
        lambda payload: audit_citation_links(payload["papers"], payload["config"]),
        {"papers": papers, "config": config},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "audit_citation_links")


async def resolve_workspace_full_text_skill(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> LiteratureWorkspace:
    result = await _run_async_skill(
        "resolve_workspace_full_text",
        lambda payload: resolve_workspace_full_text(payload["workspace"], payload["config"]),
        {"workspace": workspace, "config": config},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "resolve_workspace_full_text")


def build_storyline_skill(
    workspace: LiteratureWorkspace,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    result = _run_sync_skill(
        "build_storyline_from_workspace",
        build_storyline_from_workspace,
        workspace,
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(
        result,
        "build_storyline_from_workspace",
    )


def build_quality_report_skill(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    return _output_or_raise(
        _run_sync_skill(
            "quality_report",
            lambda payload: build_quality_report(payload["config"], payload["workspace"]),
            {"config": config, "workspace": workspace},
            context=context,
            ledger=ledger,
            policy=policy,
            idempotency_key=idempotency_key,
        ),
        "quality_report",
    )


async def execute_downloads_skill(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    request: DownloadExecutionRequest,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> DownloadExecutionResult:
    from littrace.downloads import execute_downloads

    result = await _run_async_skill(
        "execute_downloads",
        lambda payload: execute_downloads(payload["config"], payload["papers"], payload["request"]),
        {"config": config, "papers": _active_papers(workspace), "request": request},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "execute_downloads")


def build_comparison_matrix_skill(
    workspace: LiteratureWorkspace,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    result = _run_sync_skill(
        "build_comparison_matrices",
        build_comparison_matrices,
        workspace,
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(
        result,
        "build_comparison_matrices",
    )
