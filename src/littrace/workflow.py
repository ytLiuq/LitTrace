from __future__ import annotations

from typing import TypedDict

from littrace.access import build_download_plan
from littrace.agent_interactions import build_agent_interaction_report
from littrace.autonomous_loop import run_autonomous_research_loop
from littrace.citations import audit_citation_links
from littrace.context import add_papers, add_ranked_candidate_papers
from littrace.config import LitTraceConfig, load_config
from littrace.document_composer import build_research_document_report
from littrace.models import (
    LiteratureWorkspace,
    PaperSearchResult,
    PaperSearchRequest,
    ResearchRunResult,
    WorkflowTrace,
    WorkflowTraceStep,
)
from littrace.parsing import parse_workspace_papers
from littrace.publisher_connectors import build_publisher_route_report, build_publisher_search_plan
from littrace.search import LiveSearchClient, MockMaterialsSearchClient
from littrace.source_router import SourceRoute, route_sources
from littrace.storyline import build_storyline_from_workspace, verify_storyline_preview
from littrace.tables import build_comparison_matrices, extract_performance_cells


class ResearchWorkflowState(TypedDict, total=False):
    request: PaperSearchRequest
    config: LitTraceConfig
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
    agent_interactions: object
    workflow_trace: WorkflowTrace


async def run_search_preview(
    request: PaperSearchRequest,
    config: LitTraceConfig | None = None,
) -> LiteratureWorkspace:
    config = config or load_config()
    routes = route_sources(request.discipline, request.wants_recent)
    use_live = config.api.enable_live_search if request.live is None else request.live
    if use_live:
        live_client = LiveSearchClient(config)
        try:
            result = await live_client.search(request)
            diagnostics = live_client.diagnostics
        except Exception as exc:
            result = PaperSearchResult(request=request, papers=[])
            diagnostics = live_client.diagnostics
            diagnostics.errors.append(f"live_search: {exc.__class__.__name__}: {exc}")
    else:
        result = await MockMaterialsSearchClient().search(request)
        diagnostics = None
    workspace = LiteratureWorkspace()
    workspace.context.filters = {
        "discipline": request.discipline,
        "year_min": request.year_min,
        "source_routes": [route.name for route in routes],
        "publisher_search_plan": build_publisher_search_plan(request.topic).model_dump(),
        "search_mode": "live" if use_live else "mock",
        "search_diagnostics": diagnostics.__dict__ if diagnostics else None,
    }
    return add_ranked_candidate_papers(
        workspace,
        result.papers,
        request,
        active_limit=config.literature_context.active_context_limit,
    )


def build_littrace_graph():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None

    async def plan_sources(state: ResearchWorkflowState) -> ResearchWorkflowState:
        request = state["request"]
        state["routes"] = route_sources(request.discipline, request.wants_recent)
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
        use_live = config.api.enable_live_search if request.live is None else request.live
        if use_live:
            live_client = LiveSearchClient(config)
            try:
                result = await live_client.search(request)
                diagnostics = live_client.diagnostics
            except Exception as exc:
                result = PaperSearchResult(request=request, papers=[])
                diagnostics = live_client.diagnostics
                diagnostics.errors.append(f"live_search: {exc.__class__.__name__}: {exc}")
        else:
            result = await MockMaterialsSearchClient().search(request)
            diagnostics = None
        workspace = LiteratureWorkspace()
        workspace.context.filters = {
            "discipline": request.discipline,
            "year_min": request.year_min,
            "source_routes": [route.name for route in state.get("routes", [])],
            "publisher_search_plan": build_publisher_search_plan(request.topic).model_dump(),
            "search_mode": "live" if use_live else "mock",
            "search_diagnostics": diagnostics.__dict__ if diagnostics else None,
        }
        state["workspace"] = add_ranked_candidate_papers(
            workspace,
            result.papers,
            request,
            active_limit=config.literature_context.active_context_limit,
        )
        next_node, next_reason = _next_after_search(state)
        _trace_step(
            state,
            "search_papers",
            "completed",
            "用户请求检索/调研文献，因此执行候选论文检索。",
            inputs={"topic": request.topic, "live": use_live, "year_min": request.year_min},
            outputs={
                "paper_count": len(state["workspace"].context.active_papers),
                "candidate_pool_count": state["workspace"].context.filters.get("candidate_pool_count", 0),
                "active_context_limit": state["workspace"].context.filters.get("active_context_limit"),
                "search_mode": "live" if use_live else "mock",
            },
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    async def audit_citations(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        workspace = state["workspace"]
        papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
        state["citation_audit"] = await audit_citation_links(papers, config)
        next_node, next_reason = _next_after_audit(state)
        _trace_step(
            state,
            "audit_citations",
            "completed",
            "citation_audit_enabled=True，需要检查引用和访问链接。",
            inputs={"paper_count": len(papers)},
            outputs={"passed": state["citation_audit"].passed, "score": state["citation_audit"].score},
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    async def plan_downloads(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        workspace = state["workspace"]
        selected_ids = set(workspace.context.selected_for_download)
        papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
        state["download_plan"] = build_download_plan(config, papers, selected_ids)
        next_node, next_reason = _next_after_download_plan(state)
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
        next_node, next_reason = _next_after_publisher_routes(state)
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
        claims = build_storyline_from_workspace(workspace)
        state["storyline"] = claims
        state["storyline_harness"] = verify_storyline_preview(claims)
        next_node, next_reason = _next_after_storyline(state)
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
        report = build_research_document_report(state["workspace"], config)
        state["workspace"].context.filters["document_report"] = report.model_dump(mode="json")
        state["document_report"] = report
        next_node, next_reason = _next_after_document(state)
        _trace_step(
            state,
            "compose_document",
            "completed",
            "compose_document_enabled=True，需要生成学术化可审计报告。",
            inputs={"paper_count": len(state["workspace"].context.active_papers)},
            outputs={"section_count": len(report.sections), "evidence_count": report.evidence_count},
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    async def autonomous_review(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        objective = state["request"].topic
        report = await run_autonomous_research_loop(
            config,
            objective,
            state["workspace"],
            auto_replan=state.get("auto_replan_enabled", False),
        )
        state["workspace"].context.filters["autonomous_loop_report"] = report.model_dump(mode="json")
        state["autonomous_loop_report"] = report
        _trace_step(
            state,
            "autonomous_review",
            "completed",
            "autonomous_review_enabled=True，需要多 Agent 复核、反驳和修订。",
            inputs={"objective": objective, "auto_replan": state.get("auto_replan_enabled", False)},
            outputs={"round_count": len(report.rounds), "score": report.score, "passed": report.passed},
            next_node=None,
            next_reason="复核循环完成，流程结束。",
        )
        return state

    async def parse_full_text(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        workspace, report = parse_workspace_papers(state["workspace"], config)
        state["workspace"] = workspace
        state["parse_report"] = report
        next_node, next_reason = _next_after_parse(state)
        _trace_step(
            state,
            "parse_full_text",
            "completed",
            "parse_full_text_enabled=True，需要解析本地 PDF 或元数据。",
            inputs={"parse_strategy": config.parsing.parse_strategy},
            outputs={
                "parsed_count": report.get("parsed_count"),
                "metadata_only_count": report.get("metadata_only_count"),
            },
            next_node=next_node,
            next_reason=next_reason,
        )
        return state

    async def extract_tables(state: ResearchWorkflowState) -> ResearchWorkflowState:
        workspace, harness = extract_performance_cells(state["workspace"])
        state["workspace"] = workspace
        state["table_harness"] = harness.model_dump()
        state["comparison_matrix"] = build_comparison_matrices(workspace)
        next_node, next_reason = _next_after_extract_tables(state)
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

    def after_search(state: ResearchWorkflowState) -> str:
        node, _reason = _next_after_search(state)
        return node or END

    def after_audit(state: ResearchWorkflowState) -> str:
        node, _reason = _next_after_audit(state)
        return node or END

    def after_download_plan(state: ResearchWorkflowState) -> str:
        node, _reason = _next_after_download_plan(state)
        return node or END

    def after_publisher_routes(state: ResearchWorkflowState) -> str:
        node, _reason = _next_after_publisher_routes(state)
        return node or END

    def after_parse(state: ResearchWorkflowState) -> str:
        node, _reason = _next_after_parse(state)
        return node or END

    def after_extract_tables(state: ResearchWorkflowState) -> str:
        node, _reason = _next_after_extract_tables(state)
        return node or END

    def after_storyline(state: ResearchWorkflowState) -> str:
        node, _reason = _next_after_storyline(state)
        return node or END

    def after_document(state: ResearchWorkflowState) -> str:
        node, _reason = _next_after_document(state)
        return node or END

    graph = StateGraph(ResearchWorkflowState)
    graph.add_node("plan_sources", plan_sources)
    graph.add_node("search_papers", search_papers)
    graph.add_node("audit_citations", audit_citations)
    graph.add_node("plan_downloads", plan_downloads)
    graph.add_node("route_publishers", route_publishers)
    graph.add_node("parse_full_text", parse_full_text)
    graph.add_node("extract_tables", extract_tables)
    graph.add_node("build_storyline", build_storyline)
    graph.add_node("compose_document", compose_document)
    graph.add_node("autonomous_review", autonomous_review)
    graph.set_entry_point("plan_sources")
    graph.add_edge("plan_sources", "search_papers")
    graph.add_conditional_edges("search_papers", after_search)
    graph.add_conditional_edges("audit_citations", after_audit)
    graph.add_conditional_edges("plan_downloads", after_download_plan)
    graph.add_conditional_edges("route_publishers", after_publisher_routes)
    graph.add_conditional_edges("parse_full_text", after_parse)
    graph.add_conditional_edges("extract_tables", after_extract_tables)
    graph.add_conditional_edges("build_storyline", after_storyline)
    graph.add_conditional_edges("compose_document", after_document)
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
) -> ResearchRunResult:
    config = config or load_config()
    workflow_trace = WorkflowTrace()
    graph = build_littrace_graph()
    if graph is None:
        workspace = await run_search_preview(request, config)
        _trace_direct(
            workflow_trace,
            "search_papers",
            "LangGraph 不可用，使用顺序 fallback 执行检索。",
            inputs={"topic": request.topic, "live": request.live, "year_min": request.year_min},
            outputs={
                "paper_count": len(workspace.context.active_papers),
                "candidate_pool_count": workspace.context.filters.get("candidate_pool_count", 0),
                "active_context_limit": workspace.context.filters.get("active_context_limit"),
            },
        )
        papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
        citation_audit = (
            await audit_citation_links(papers, config) if audit_citations_enabled else None
        )
        if citation_audit is not None:
            _trace_direct(
                workflow_trace,
                "audit_citations",
                "audit_citations_enabled=True，检查引用链接。",
                inputs={"paper_count": len(papers)},
                outputs={"passed": citation_audit.passed, "score": citation_audit.score},
            )
        else:
            _trace_direct(
                workflow_trace,
                "audit_citations",
                "audit_citations_enabled=False，跳过引用审计。",
                status="skipped",
            )
        download_plan = (
            build_download_plan(config, papers, set(workspace.context.selected_for_download))
            if plan_downloads_enabled
            else None
        )
        if download_plan is not None:
            _trace_direct(
                workflow_trace,
                "plan_downloads",
                "plan_downloads_enabled=True，生成下载计划。",
                outputs={
                    "downloadable_count": download_plan.downloadable_count,
                    "requires_login_count": download_plan.requires_login_count,
                },
            )
        publisher_routes = (
            build_publisher_route_report(papers).model_dump()
            if route_publishers_enabled
            else None
        )
        if publisher_routes is not None:
            _trace_direct(
                workflow_trace,
                "route_publishers",
                "route_publishers_enabled=True，识别出版商路线。",
                outputs={"route_count": len(publisher_routes.get("routes", []))},
            )
        parse_report = None
        if parse_full_text_enabled:
            workspace, parse_report = parse_workspace_papers(workspace, config)
            _trace_direct(
                workflow_trace,
                "parse_full_text",
                "parse_full_text_enabled=True，解析全文。",
                inputs={"parse_strategy": config.parsing.parse_strategy},
                outputs={
                    "parsed_count": parse_report.get("parsed_count"),
                    "metadata_only_count": parse_report.get("metadata_only_count"),
                },
            )
        table_harness = None
        comparison_matrix = None
        if extract_tables_enabled:
            workspace, harness = extract_performance_cells(workspace)
            table_harness = harness.model_dump()
            comparison_matrix = build_comparison_matrices(workspace)
            _trace_direct(
                workflow_trace,
                "extract_tables",
                "extract_tables_enabled=True，抽取性能指标。",
                outputs={"cell_count": len(workspace.performance_cells)},
            )
        storyline = (
            build_storyline_from_workspace(workspace) if build_storyline_enabled else None
        )
        if storyline is not None:
            _trace_direct(
                workflow_trace,
                "build_storyline",
                "build_storyline_enabled=True，构建发展脉络。",
                outputs={"claim_count": len(storyline)},
            )
        document_report = None
        if compose_document_enabled:
            document_report = build_research_document_report(workspace, config)
            workspace.context.filters["document_report"] = document_report.model_dump(mode="json")
            _trace_direct(
                workflow_trace,
                "compose_document",
                "compose_document_enabled=True，生成报告。",
                outputs={"section_count": len(document_report.sections)},
            )
        autonomous_loop_report = None
        if autonomous_review_enabled:
            autonomous_loop_report = await run_autonomous_research_loop(
                config,
                request.topic,
                workspace,
                auto_replan=auto_replan_enabled,
            )
            workspace.context.filters["autonomous_loop_report"] = autonomous_loop_report.model_dump(mode="json")
            _trace_direct(
                workflow_trace,
                "autonomous_review",
                "autonomous_review_enabled=True，执行多 Agent 复核。",
                outputs={"round_count": len(autonomous_loop_report.rounds)},
            )
        return ResearchRunResult(
            workspace=workspace,
            citation_audit=citation_audit,
            download_plan=download_plan,
            publisher_routes=publisher_routes,
            agent_interactions=build_agent_interaction_report(workspace),
            parse_report=parse_report,
            table_harness=table_harness,
            comparison_matrix=comparison_matrix,
            storyline=storyline,
            document_report=document_report,
            autonomous_loop_report=autonomous_loop_report,
            workflow_trace=workflow_trace,
        )

    state = await graph.ainvoke(
        {
            "request": request,
            "config": config,
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
    return ResearchRunResult(
        workspace=state["workspace"],
        citation_audit=state.get("citation_audit"),
        download_plan=state.get("download_plan"),
        publisher_routes=state.get("publisher_routes"),
        agent_interactions=build_agent_interaction_report(state["workspace"]),
        parse_report=state.get("parse_report"),
        table_harness=state.get("table_harness"),
        comparison_matrix=state.get("comparison_matrix"),
        storyline=state.get("storyline"),
        document_report=state.get("document_report"),
        autonomous_loop_report=state.get("autonomous_loop_report"),
        workflow_trace=state.get("workflow_trace"),
    )


def _trace_step(
    state: ResearchWorkflowState,
    node: str,
    status: str,
    reason: str,
    inputs: dict[str, object] | None = None,
    outputs: dict[str, object] | None = None,
    next_node: str | None = None,
    next_reason: str | None = None,
) -> None:
    trace = state.setdefault("workflow_trace", WorkflowTrace())
    trace.steps.append(
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


def _trace_direct(
    trace: WorkflowTrace,
    node: str,
    reason: str,
    status: str = "completed",
    inputs: dict[str, object] | None = None,
    outputs: dict[str, object] | None = None,
) -> None:
    trace.steps.append(
        WorkflowTraceStep(
            node=node,
            status=status,
            reason=reason,
            inputs=inputs or {},
            outputs=outputs or {},
        )
    )


def _first_enabled(state: ResearchWorkflowState, candidates: list[tuple[str, str]]) -> tuple[str | None, str]:
    for flag, node in candidates:
        if state.get(flag, False):
            return node, f"{flag}=True，因此进入 {node}。"
    return None, "后续可选节点均未启用，流程结束。"


def _next_after_search(state: ResearchWorkflowState) -> tuple[str | None, str]:
    return _first_enabled(
        state,
        [
            ("audit_citations_enabled", "audit_citations"),
            ("plan_downloads_enabled", "plan_downloads"),
            ("route_publishers_enabled", "route_publishers"),
            ("parse_full_text_enabled", "parse_full_text"),
            ("extract_tables_enabled", "extract_tables"),
            ("build_storyline_enabled", "build_storyline"),
            ("compose_document_enabled", "compose_document"),
            ("autonomous_review_enabled", "autonomous_review"),
        ],
    )


def _next_after_audit(state: ResearchWorkflowState) -> tuple[str | None, str]:
    return _first_enabled(
        state,
        [
            ("plan_downloads_enabled", "plan_downloads"),
            ("route_publishers_enabled", "route_publishers"),
            ("parse_full_text_enabled", "parse_full_text"),
            ("extract_tables_enabled", "extract_tables"),
            ("build_storyline_enabled", "build_storyline"),
            ("compose_document_enabled", "compose_document"),
            ("autonomous_review_enabled", "autonomous_review"),
        ],
    )


def _next_after_download_plan(state: ResearchWorkflowState) -> tuple[str | None, str]:
    return _first_enabled(
        state,
        [
            ("route_publishers_enabled", "route_publishers"),
            ("parse_full_text_enabled", "parse_full_text"),
            ("compose_document_enabled", "compose_document"),
            ("autonomous_review_enabled", "autonomous_review"),
        ],
    )


def _next_after_publisher_routes(state: ResearchWorkflowState) -> tuple[str | None, str]:
    return _first_enabled(
        state,
        [
            ("parse_full_text_enabled", "parse_full_text"),
            ("extract_tables_enabled", "extract_tables"),
            ("build_storyline_enabled", "build_storyline"),
            ("compose_document_enabled", "compose_document"),
            ("autonomous_review_enabled", "autonomous_review"),
        ],
    )


def _next_after_parse(state: ResearchWorkflowState) -> tuple[str | None, str]:
    return _first_enabled(
        state,
        [
            ("extract_tables_enabled", "extract_tables"),
            ("build_storyline_enabled", "build_storyline"),
            ("compose_document_enabled", "compose_document"),
            ("autonomous_review_enabled", "autonomous_review"),
        ],
    )


def _next_after_extract_tables(state: ResearchWorkflowState) -> tuple[str | None, str]:
    return _first_enabled(
        state,
        [
            ("build_storyline_enabled", "build_storyline"),
            ("compose_document_enabled", "compose_document"),
            ("autonomous_review_enabled", "autonomous_review"),
        ],
    )


def _next_after_storyline(state: ResearchWorkflowState) -> tuple[str | None, str]:
    return _first_enabled(
        state,
        [
            ("compose_document_enabled", "compose_document"),
            ("autonomous_review_enabled", "autonomous_review"),
        ],
    )


def _next_after_document(state: ResearchWorkflowState) -> tuple[str | None, str]:
    return _first_enabled(state, [("autonomous_review_enabled", "autonomous_review")])
