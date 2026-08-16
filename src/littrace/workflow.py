from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TypedDict
from uuid import uuid4

from littrace.workflow_status import build_workflow_status
from littrace.autonomous_loop import run_review_loop
from littrace.context import add_ranked_candidate_papers, _merge_filters
from littrace.config import LitTraceConfig, load_config
from littrace.retrieval.full_text_context import build_full_text_context
from littrace.log import get_logger, timed
from littrace.models import (
    LiteratureWorkspace,
    PaperSearchRequest,
    ResearchRunResult,
    WorkflowTrace,
    WorkflowTraceStep,
)
from littrace.publisher_connectors import build_publisher_route_report, build_publisher_search_plan
from littrace.skill_runner import (
    SearchSkillResult,
    audit_citation_links_skill,
    build_comparison_matrix_skill,
    build_download_plan_skill,
    build_research_report_skill,
    build_storyline_skill,
    extract_tables_skill,
    parse_workspace_skill,
    search_papers_skill,
)
from littrace.retrieval.source_router import SourceRoute, route_sources
from littrace.retrieval.adapters import record_search_provenance
from littrace.evidence.storyline import verify_storyline_preview
from littrace.tool_contracts import ToolCallContext, ToolExecutionLedger, ToolExecutionPolicy

logger = get_logger("workflow")


class ResearchWorkflowState(TypedDict, total=False):
    request: PaperSearchRequest
    config: LitTraceConfig
    tool_context: ToolCallContext
    tool_ledger: ToolExecutionLedger
    tool_policy: ToolExecutionPolicy | None
    audit_citations_enabled: bool
    plan_downloads_enabled: bool
    route_publishers_enabled: bool
    parse_full_text_enabled: bool
    extract_tables_enabled: bool
    build_storyline_enabled: bool
    compose_document_enabled: bool
    autonomous_review_enabled: bool
    auto_replan_enabled: bool
    routes: list[SourceRoute]
    workspace: LiteratureWorkspace
    citation_audit: object
    download_plan: object
    publisher_routes: object
    parse_report: object
    table_harness: object
    comparison_matrix: object
    storyline: object
    storyline_harness: object
    document_report: object
    autonomous_loop_report: object
    workflow_status: object
    workflow_trace: WorkflowTrace


def _littrace_config_for_workflow() -> "LitTraceConfig | None":
    """Best-effort read of the global LitTraceConfig for workflow timeouts.

    Returns ``None`` if the config cannot be loaded (test environments,
    missing optional deps). Callers must tolerate ``None`` and fall back
    to a sane default.
    """
    try:
        from littrace.config import load_config
        return load_config()
    except Exception:
        return None


def _workflow_tool_context(state: ResearchWorkflowState, node: str) -> ToolCallContext:
    context = state["tool_context"]
    return context.model_copy(update={"metadata": {**context.metadata, "workflow_node": node}})


def _workflow_idempotency_key(request: PaperSearchRequest, node: str) -> str:
    payload = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return f"{node}:{payload}"


_OPTIONAL_NODES: tuple[tuple[str, str], ...] = (
    ("audit_citations_enabled", "audit_citations"),
    ("plan_downloads_enabled", "plan_downloads"),
    ("route_publishers_enabled", "route_publishers"),
    ("parse_full_text_enabled", "parse_full_text"),
    ("extract_tables_enabled", "extract_tables"),
    ("build_storyline_enabled", "build_storyline"),
    ("compose_document_enabled", "compose_document"),
    ("autonomous_review_enabled", "autonomous_review"),
)


def _next_after(state: ResearchWorkflowState, completed_node: str | None) -> tuple[str | None, str]:
    candidates = _OPTIONAL_NODES
    if completed_node:
        completed_index = next(
            (index for index, (_, node) in enumerate(candidates) if node == completed_node),
            len(candidates),
        )
        candidates = candidates[completed_index + 1 :]
    for flag, node in candidates:
        if state.get(flag, False):
            return node, f"{flag}=True，因此进入 {node}。"
    return None, "后续可选节点均未启用，流程结束。"


def _trace_step(
    state: ResearchWorkflowState,
    node: str,
    status: str,
    reason: str,
    *,
    inputs: dict[str, object] | None = None,
    outputs: dict[str, object] | None = None,
    next_node: str | None = None,
    next_reason: str | None = None,
) -> None:
    state.setdefault("workflow_trace", WorkflowTrace()).steps.append(
        WorkflowTraceStep(
            node=node,
            status=status,
            reason=reason,
            inputs=inputs or {},
            outputs=outputs or {},
            next_node=next_node,
            next_reason=next_reason,
        )
    )


async def _build_search_workspace(
    request: PaperSearchRequest,
    config: LitTraceConfig,
    *,
    routes=None,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> tuple[LiteratureWorkspace, SearchSkillResult]:
    routes = (
        routes if routes is not None else route_sources(request.discipline, request.wants_recent)
    )
    search = await search_papers_skill(
        request,
        config,
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    workspace = LiteratureWorkspace()
    _merge_filters(
        workspace.context.filters,
        {
            "discipline": request.discipline,
            "year_min": request.year_min,
            "source_routes": [route.name for route in routes],
            "publisher_search_plan": build_publisher_search_plan(request.topic).model_dump(),
            "search_mode": "live" if search.use_live else "mock",
            "search_completed_at": datetime.now(UTC).isoformat(),
            "search_diagnostics": (
                {
                    **search.diagnostics.__dict__,
                    "source_health": {
                        name: health.model_dump(mode="json")
                        for name, health in search.diagnostics.source_health.items()
                    },
                }
                if search.diagnostics
                else None
            ),
        },
    )
    workspace = add_ranked_candidate_papers(
        workspace,
        search.result.papers,
        request,
        active_limit=config.literature_context.active_context_limit,
    )
    if search.diagnostics:
        record_search_provenance(
            workspace,
            request,
            search.result.papers,
            search.diagnostics.source_health,
        )
    if search.use_live:
        context_result = await build_full_text_context(
            workspace,
            request,
            config,
            context=context,
            ledger=ledger,
            policy=policy,
        )
        workspace = context_result.workspace
        workspace.context.filters.full_text_context_warnings = context_result.warnings
    return workspace, search


async def run_search_preview(
    request: PaperSearchRequest,
    config: LitTraceConfig | None = None,
    *,
    tool_context: ToolCallContext | None = None,
    tool_ledger: ToolExecutionLedger | None = None,
    tool_policy: ToolExecutionPolicy | None = None,
) -> LiteratureWorkspace:
    context = tool_context or ToolCallContext(caller="workflow.preview", task_id=uuid4().hex)
    workspace, _ = await _build_search_workspace(
        request,
        config or load_config(),
        context=context,
        ledger=tool_ledger or ToolExecutionLedger(),
        policy=tool_policy,
        idempotency_key=_workflow_idempotency_key(request, "search_papers"),
    )
    return workspace


def build_littrace_graph():
    from langgraph.graph import END, StateGraph

    async def plan_sources(state: ResearchWorkflowState) -> ResearchWorkflowState:
        with timed("plan_sources"):
            request = state["request"]
            state["routes"] = route_sources(request.discipline, request.wants_recent)
            logger.info(
                "plan_sources",
                extra={
                    "discipline": request.discipline,
                    "routes": [r.name for r in state["routes"]],
                },
            )
            _trace_step(
                state,
                "plan_sources",
                "completed",
                "根据学科和时效偏好选择检索源。",
                inputs={"discipline": request.discipline, "wants_recent": request.wants_recent},
                outputs={"routes": [route.name for route in state["routes"]]},
                next_node="search_papers",
                next_reason="检索源规划完成后进入论文检索。",
            )
            return state

    async def search_papers(state: ResearchWorkflowState) -> ResearchWorkflowState:
        request = state["request"]
        config = state.get("config") or load_config()
        workspace, search = await _build_search_workspace(
            request,
            config,
            routes=state.get("routes", []),
            context=_workflow_tool_context(state, "search_papers"),
            ledger=state["tool_ledger"],
            policy=state.get("tool_policy"),
            idempotency_key=_workflow_idempotency_key(request, "search_papers"),
        )
        state["workspace"] = workspace
        next_node, next_reason = _next_after(state, None)
        _trace_step(
            state,
            "search_papers",
            "completed",
            "用户请求检索/调研文献，因此执行候选论文检索。",
            inputs={"topic": request.topic, "live": search.use_live, "year_min": request.year_min},
            outputs={
                "paper_count": len(state["workspace"].context.active_papers),
                "candidate_pool_count": getattr(
                    state["workspace"].context.filters, "candidate_pool_count", 0
                ),
                "valid_candidate_count": getattr(
                    state["workspace"].context.filters, "valid_candidate_count", 0
                ),
                "downloaded_full_text_count": getattr(
                    state["workspace"].context.filters, "downloaded_full_text_count", 0
                ),
                "parsed_full_text_count": getattr(
                    state["workspace"].context.filters, "parsed_full_text_count", 0
                ),
                "active_context_limit": getattr(
                    state["workspace"].context.filters, "active_context_limit", None
                ),
                "search_mode": "live" if search.use_live else "mock",
            },
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    async def audit_citations(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        workspace = state["workspace"]
        papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
        state["citation_audit"] = await audit_citation_links_skill(
            papers,
            config,
            context=_workflow_tool_context(state, "audit_citations"),
            ledger=state["tool_ledger"],
            policy=state.get("tool_policy"),
            idempotency_key="audit_citations",
        )
        next_node, next_reason = _next_after(state, "audit_citations")
        _trace_step(
            state,
            "audit_citations",
            "completed",
            "citation_audit_enabled=True，需要检查引用和访问链接。",
            inputs={"paper_count": len(papers)},
            outputs={
                "passed": state["citation_audit"].passed,
                "score": state["citation_audit"].score,
            },
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    async def plan_downloads(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        workspace = state["workspace"]
        selected_ids = set(workspace.context.selected_for_download)
        papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
        state["download_plan"] = await build_download_plan_skill(
            config,
            workspace,
            context=_workflow_tool_context(state, "plan_downloads"),
            ledger=state["tool_ledger"],
            policy=state.get("tool_policy"),
            idempotency_key="plan_downloads",
        )
        next_node, next_reason = _next_after(state, "plan_downloads")
        _trace_step(
            state,
            "plan_downloads",
            "completed",
            "plan_downloads_enabled=True，需要生成合规下载计划。",
            inputs={"paper_count": len(papers), "selected_count": len(selected_ids)},
            outputs={
                "downloadable_count": state["download_plan"].downloadable_count,
                "requires_login_count": state["download_plan"].requires_login_count,
            },
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    async def route_publishers(state: ResearchWorkflowState) -> ResearchWorkflowState:
        workspace = state["workspace"]
        papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
        state["publisher_routes"] = build_publisher_route_report(papers).model_dump()
        next_node, next_reason = _next_after(state, "route_publishers")
        _trace_step(
            state,
            "route_publishers",
            "completed",
            "route_publishers_enabled=True，需要识别出版商访问路线。",
            inputs={"paper_count": len(papers)},
            outputs={"route_count": len(state["publisher_routes"].get("routes", []))},
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    async def build_storyline(state: ResearchWorkflowState) -> ResearchWorkflowState:
        workspace = state["workspace"]
        claims = build_storyline_skill(
            workspace,
            context=_workflow_tool_context(state, "build_storyline"),
            ledger=state["tool_ledger"],
            policy=state.get("tool_policy"),
            idempotency_key="build_storyline",
        )
        state["storyline"] = claims
        state["storyline_harness"] = verify_storyline_preview(claims)
        next_node, next_reason = _next_after(state, "build_storyline")
        _trace_step(
            state,
            "build_storyline",
            "completed",
            "build_storyline_enabled=True，需要构建 solution-limit-response 脉络。",
            inputs={"paper_count": len(workspace.context.active_papers)},
            outputs={"claim_count": len(claims), "passed": state["storyline_harness"].passed},
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    async def compose_document(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        report = await build_research_report_skill(
            state["workspace"],
            config,
            context=_workflow_tool_context(state, "compose_document"),
            ledger=state["tool_ledger"],
            policy=state.get("tool_policy"),
            idempotency_key="compose_document",
        )
        state["workspace"].context.filters.document_report = report.model_dump(mode="json")
        state["document_report"] = report
        next_node, next_reason = _next_after(state, "compose_document")
        _trace_step(
            state,
            "compose_document",
            "completed" if report.release_ready else "blocked",
            "compose_document_enabled=True，需要生成学术化可审计报告。",
            inputs={"paper_count": len(state["workspace"].context.active_papers)},
            outputs={
                "section_count": len(report.sections),
                "evidence_count": report.evidence_count,
                "release_ready": report.release_ready,
                "release_blocker_count": len(report.release_blockers),
            },
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    async def autonomous_review(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        objective = state["request"].topic
        report = await run_review_loop(
            config,
            objective,
            state["workspace"],
            auto_replan=state.get("auto_replan_enabled", False),
        )
        state["workspace"].context.filters.autonomous_loop_report = report.model_dump(mode="json")
        state["autonomous_loop_report"] = report
        _trace_step(
            state,
            "autonomous_review",
            "completed",
            "autonomous_review_enabled=True，需要执行质量门和可选 Reviewer 审查。",
            inputs={"objective": objective, "auto_replan": state.get("auto_replan_enabled", False)},
            outputs={
                "round_count": len(report.rounds),
                "score": report.score,
                "passed": report.passed,
            },
            next_node=None,
            next_reason="复核循环完成，流程结束。",
        )
        return state

    async def parse_full_text(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        workspace, report = await parse_workspace_skill(
            state["workspace"],
            config,
            context=_workflow_tool_context(state, "parse_full_text"),
            ledger=state["tool_ledger"],
            policy=state.get("tool_policy"),
        )
        state["workspace"] = workspace
        state["parse_report"] = report
        next_node, next_reason = _next_after(state, "parse_full_text")
        _trace_step(
            state,
            "parse_full_text",
            "completed",
            "parse_full_text_enabled=True，需要解析本地 PDF；metadata/abstract fallback 已禁用。",
            inputs={"parse_strategy": config.parsing.parse_strategy},
            outputs={
                "parsed_count": report.get("parsed_count"),
                "failed_count": report.get("failed_count"),
                "missing_pdf_count": report.get("missing_pdf_count"),
            },
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    async def extract_tables(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        workspace, harness = await extract_tables_skill(
            state["workspace"],
            config,
            context=_workflow_tool_context(state, "extract_tables"),
            ledger=state["tool_ledger"],
            policy=state.get("tool_policy"),
        )
        state["workspace"] = workspace
        state["table_harness"] = harness.model_dump()
        state["comparison_matrix"] = build_comparison_matrix_skill(
            workspace,
            context=_workflow_tool_context(state, "extract_tables.comparison_matrix"),
            ledger=state["tool_ledger"],
            policy=state.get("tool_policy"),
            idempotency_key="extract_tables.comparison_matrix",
        )
        next_node, next_reason = _next_after(state, "extract_tables")
        _trace_step(
            state,
            "extract_tables",
            "completed",
            "extract_tables_enabled=True，需要抽取性能指标并保留 evidence。",
            inputs={"parsed_paper_count": len(workspace.parsed_papers)},
            outputs={
                "cell_count": len(workspace.performance_cells),
                "warning_count": len(harness.warnings),
            },
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    def _next_node(router):
        def select(state: ResearchWorkflowState) -> str:
            node, _reason = router(state)
            return node or END

        return select

    def _wrap_node(name: str, fn, *, timeout_seconds: float | None = None):
        """Wrap a LangGraph node with timed logging + per-node timeout.

        The timeout defaults to ``api.request_timeout_seconds * 4`` so a
        single stuck node cannot block the chat path indefinitely. Pass an
        explicit ``timeout_seconds`` to override for a specific node.
        """
        import functools

        effective_timeout = timeout_seconds
        if effective_timeout is None:
            try:
                effective_timeout = float(
                    getattr(_littrace_config_for_workflow(), "api.request_timeout_seconds", 60.0)
                ) * 4
            except Exception:
                effective_timeout = 240.0

        @functools.wraps(fn)
        async def wrapped(state, *args, **kwargs):
            with timed(f"node:{name}"):
                try:
                    result = await asyncio.wait_for(
                        fn(state, *args, **kwargs),
                        timeout=effective_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "workflow_node_timeout",
                        extra={
                            "node": name,
                            "timeout_seconds": effective_timeout,
                        },
                    )
                    raise
                ws = state.get("workspace")
                logger.info(
                    "node_completed",
                    extra={
                        "node": name,
                        "paper_count": len(ws.context.active_papers) if ws else 0,
                        "parsed_count": len(ws.parsed_papers) if ws else 0,
                    },
                )
                return result

        return wrapped

    graph = StateGraph(ResearchWorkflowState)
    graph.add_node("plan_sources", _wrap_node("plan_sources", plan_sources))
    graph.add_node("search_papers", _wrap_node("search_papers", search_papers))
    graph.add_node("audit_citations", _wrap_node("audit_citations", audit_citations))
    graph.add_node("plan_downloads", _wrap_node("plan_downloads", plan_downloads))
    graph.add_node("route_publishers", _wrap_node("route_publishers", route_publishers))
    graph.add_node("parse_full_text", _wrap_node("parse_full_text", parse_full_text))
    graph.add_node("extract_tables", _wrap_node("extract_tables", extract_tables))
    graph.add_node("build_storyline", _wrap_node("build_storyline", build_storyline))
    graph.add_node("compose_document", _wrap_node("compose_document", compose_document))
    graph.add_node("autonomous_review", _wrap_node("autonomous_review", autonomous_review))
    graph.set_entry_point("plan_sources")
    graph.add_edge("plan_sources", "search_papers")
    graph.add_conditional_edges("search_papers", _next_node(lambda state: _next_after(state, None)))
    graph.add_conditional_edges(
        "audit_citations", _next_node(lambda state: _next_after(state, "audit_citations"))
    )
    graph.add_conditional_edges(
        "plan_downloads", _next_node(lambda state: _next_after(state, "plan_downloads"))
    )
    graph.add_conditional_edges(
        "route_publishers", _next_node(lambda state: _next_after(state, "route_publishers"))
    )
    graph.add_conditional_edges(
        "parse_full_text", _next_node(lambda state: _next_after(state, "parse_full_text"))
    )
    graph.add_conditional_edges(
        "extract_tables", _next_node(lambda state: _next_after(state, "extract_tables"))
    )
    graph.add_conditional_edges(
        "build_storyline", _next_node(lambda state: _next_after(state, "build_storyline"))
    )
    graph.add_conditional_edges(
        "compose_document", _next_node(lambda state: _next_after(state, "compose_document"))
    )
    graph.add_edge("autonomous_review", END)
    return graph.compile()


async def run_research_graph(
    request: PaperSearchRequest,
    config: LitTraceConfig | None = None,
    audit_citations_enabled: bool = True,
    plan_downloads_enabled: bool = True,
    route_publishers_enabled: bool = True,
    parse_full_text_enabled: bool = False,
    extract_tables_enabled: bool = False,
    build_storyline_enabled: bool = False,
    compose_document_enabled: bool = False,
    autonomous_review_enabled: bool = False,
    auto_replan_enabled: bool = False,
    *,
    tool_context: ToolCallContext | None = None,
    tool_ledger: ToolExecutionLedger | None = None,
    tool_policy: ToolExecutionPolicy | None = None,
) -> ResearchRunResult:
    config = config or load_config()
    workflow_trace = WorkflowTrace()
    context = tool_context or ToolCallContext(caller="workflow", task_id=uuid4().hex)
    ledger = tool_ledger or ToolExecutionLedger()
    graph = build_littrace_graph()
    logger.info(
        "run_research_graph_start",
        extra={
            "topic": request.topic,
            "live": request.live,
            "runtime": "langgraph",
            "flags": {
                "audit": audit_citations_enabled,
                "downloads": plan_downloads_enabled,
                "publishers": route_publishers_enabled,
                "parse": parse_full_text_enabled,
                "tables": extract_tables_enabled,
                "storyline": build_storyline_enabled,
                "document": compose_document_enabled,
                "review": autonomous_review_enabled,
            },
        },
    )
    with timed("graph_ainvoke"):
        state = await graph.ainvoke(
            {
                "request": request,
                "config": config,
                "tool_context": context,
                "tool_ledger": ledger,
                "tool_policy": tool_policy,
                "audit_citations_enabled": audit_citations_enabled,
                "plan_downloads_enabled": plan_downloads_enabled,
                "route_publishers_enabled": route_publishers_enabled,
                "parse_full_text_enabled": parse_full_text_enabled,
                "extract_tables_enabled": extract_tables_enabled,
                "build_storyline_enabled": build_storyline_enabled,
                "compose_document_enabled": compose_document_enabled,
                "autonomous_review_enabled": autonomous_review_enabled,
                "auto_replan_enabled": auto_replan_enabled,
                "workflow_trace": workflow_trace,
            }
        )
    logger.info(
        "run_research_graph_done",
        extra={"path": "langgraph", "papers": len(state["workspace"].context.active_papers)},
    )
    return ResearchRunResult(
        workspace=state["workspace"],
        citation_audit=state.get("citation_audit"),
        download_plan=state.get("download_plan"),
        publisher_routes=state.get("publisher_routes"),
        workflow_status=build_workflow_status(state["workspace"]),
        parse_report=state.get("parse_report"),
        table_harness=state.get("table_harness"),
        comparison_matrix=state.get("comparison_matrix"),
        storyline=state.get("storyline"),
        document_report=state.get("document_report"),
        autonomous_loop_report=state.get("autonomous_loop_report"),
        workflow_trace=state.get("workflow_trace"),
    )
