from __future__ import annotations

from typing import TypedDict

from littrace.access import build_download_plan
from littrace.agent_interactions import build_agent_interaction_report
from littrace.autonomous_loop import run_autonomous_research_loop
from littrace.citations import audit_citation_links
from littrace.context import add_papers
from littrace.config import LitTraceConfig, load_config
from littrace.document_composer import build_research_document_report
from littrace.models import LiteratureWorkspace, PaperSearchRequest, ResearchRunResult
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
            result = await MockMaterialsSearchClient().search(request)
            diagnostics = live_client.diagnostics
            diagnostics.used_fallback = True
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
    return add_papers(workspace, result.papers)


def build_littrace_graph():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None

    async def plan_sources(state: ResearchWorkflowState) -> ResearchWorkflowState:
        request = state["request"]
        state["routes"] = route_sources(request.discipline, request.wants_recent)
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
                result = await MockMaterialsSearchClient().search(request)
                diagnostics = live_client.diagnostics
                diagnostics.used_fallback = True
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
        state["workspace"] = add_papers(workspace, result.papers)
        return state

    async def audit_citations(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        workspace = state["workspace"]
        papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
        state["citation_audit"] = await audit_citation_links(papers, config)
        return state

    async def plan_downloads(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        workspace = state["workspace"]
        selected_ids = set(workspace.context.selected_for_download)
        papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
        state["download_plan"] = build_download_plan(config, papers, selected_ids)
        return state

    async def route_publishers(state: ResearchWorkflowState) -> ResearchWorkflowState:
        workspace = state["workspace"]
        papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
        state["publisher_routes"] = build_publisher_route_report(papers).model_dump()
        return state

    async def build_storyline(state: ResearchWorkflowState) -> ResearchWorkflowState:
        workspace = state["workspace"]
        claims = build_storyline_from_workspace(workspace)
        state["storyline"] = claims
        state["storyline_harness"] = verify_storyline_preview(claims)
        return state

    async def compose_document(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        report = build_research_document_report(state["workspace"], config)
        state["workspace"].context.filters["document_report"] = report.model_dump(mode="json")
        state["document_report"] = report
        return state

    async def autonomous_review(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        objective = state["request"].topic
        report = await run_autonomous_research_loop(config, objective, state["workspace"])
        state["workspace"].context.filters["autonomous_loop_report"] = report.model_dump(mode="json")
        state["autonomous_loop_report"] = report
        return state

    async def parse_full_text(state: ResearchWorkflowState) -> ResearchWorkflowState:
        config = state.get("config") or load_config()
        workspace, report = parse_workspace_papers(state["workspace"], config)
        state["workspace"] = workspace
        state["parse_report"] = report
        return state

    async def extract_tables(state: ResearchWorkflowState) -> ResearchWorkflowState:
        workspace, harness = extract_performance_cells(state["workspace"])
        state["workspace"] = workspace
        state["table_harness"] = harness.model_dump()
        state["comparison_matrix"] = build_comparison_matrices(workspace)
        return state

    def after_search(state: ResearchWorkflowState) -> str:
        if state.get("audit_citations_enabled", True):
            return "audit_citations"
        if state.get("plan_downloads_enabled", True):
            return "plan_downloads"
        if state.get("route_publishers_enabled", True):
            return "route_publishers"
        if state.get("parse_full_text_enabled", False):
            return "parse_full_text"
        if state.get("extract_tables_enabled", False):
            return "extract_tables"
        if state.get("build_storyline_enabled", False):
            return "build_storyline"
        if state.get("compose_document_enabled", False):
            return "compose_document"
        if state.get("autonomous_review_enabled", False):
            return "autonomous_review"
        return END

    def after_audit(state: ResearchWorkflowState) -> str:
        if state.get("plan_downloads_enabled", True):
            return "plan_downloads"
        if state.get("route_publishers_enabled", True):
            return "route_publishers"
        if state.get("parse_full_text_enabled", False):
            return "parse_full_text"
        if state.get("extract_tables_enabled", False):
            return "extract_tables"
        if state.get("build_storyline_enabled", False):
            return "build_storyline"
        if state.get("compose_document_enabled", False):
            return "compose_document"
        if state.get("autonomous_review_enabled", False):
            return "autonomous_review"
        return END

    def after_download_plan(state: ResearchWorkflowState) -> str:
        if state.get("route_publishers_enabled", True):
            return "route_publishers"
        if state.get("parse_full_text_enabled", False):
            return "parse_full_text"
        if state.get("compose_document_enabled", False):
            return "compose_document"
        if state.get("autonomous_review_enabled", False):
            return "autonomous_review"
        return END

    def after_publisher_routes(state: ResearchWorkflowState) -> str:
        if state.get("parse_full_text_enabled", False):
            return "parse_full_text"
        if state.get("extract_tables_enabled", False):
            return "extract_tables"
        if state.get("build_storyline_enabled", False):
            return "build_storyline"
        if state.get("compose_document_enabled", False):
            return "compose_document"
        if state.get("autonomous_review_enabled", False):
            return "autonomous_review"
        return END

    def after_parse(state: ResearchWorkflowState) -> str:
        if state.get("extract_tables_enabled", False):
            return "extract_tables"
        if state.get("build_storyline_enabled", False):
            return "build_storyline"
        if state.get("compose_document_enabled", False):
            return "compose_document"
        if state.get("autonomous_review_enabled", False):
            return "autonomous_review"
        return END

    def after_extract_tables(state: ResearchWorkflowState) -> str:
        if state.get("build_storyline_enabled", False):
            return "build_storyline"
        if state.get("compose_document_enabled", False):
            return "compose_document"
        if state.get("autonomous_review_enabled", False):
            return "autonomous_review"
        return END

    def after_storyline(state: ResearchWorkflowState) -> str:
        if state.get("compose_document_enabled", False):
            return "compose_document"
        if state.get("autonomous_review_enabled", False):
            return "autonomous_review"
        return END

    def after_document(state: ResearchWorkflowState) -> str:
        if state.get("autonomous_review_enabled", False):
            return "autonomous_review"
        return END

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
) -> ResearchRunResult:
    config = config or load_config()
    graph = build_littrace_graph()
    if graph is None:
        workspace = await run_search_preview(request, config)
        papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
        citation_audit = (
            await audit_citation_links(papers, config) if audit_citations_enabled else None
        )
        download_plan = (
            build_download_plan(config, papers, set(workspace.context.selected_for_download))
            if plan_downloads_enabled
            else None
        )
        publisher_routes = (
            build_publisher_route_report(papers).model_dump()
            if route_publishers_enabled
            else None
        )
        parse_report = None
        if parse_full_text_enabled:
            workspace, parse_report = parse_workspace_papers(workspace, config)
        table_harness = None
        comparison_matrix = None
        if extract_tables_enabled:
            workspace, harness = extract_performance_cells(workspace)
            table_harness = harness.model_dump()
            comparison_matrix = build_comparison_matrices(workspace)
        storyline = (
            build_storyline_from_workspace(workspace) if build_storyline_enabled else None
        )
        document_report = None
        if compose_document_enabled:
            document_report = build_research_document_report(workspace, config)
            workspace.context.filters["document_report"] = document_report.model_dump(mode="json")
        autonomous_loop_report = None
        if autonomous_review_enabled:
            autonomous_loop_report = await run_autonomous_research_loop(config, request.topic, workspace)
            workspace.context.filters["autonomous_loop_report"] = autonomous_loop_report.model_dump(mode="json")
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
    )
