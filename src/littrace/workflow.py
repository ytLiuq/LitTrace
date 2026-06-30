from __future__ import annotations

from typing import TypedDict

from littrace.access import build_download_plan
from littrace.citations import audit_citation_links
from littrace.context import add_papers
from littrace.config import LitTraceConfig, load_config
from littrace.models import LiteratureWorkspace, PaperSearchRequest, ResearchRunResult
from littrace.parsing import parse_workspace_papers
from littrace.search import LiveSearchClient, MockMaterialsSearchClient
from littrace.source_router import SourceRoute, route_sources
from littrace.storyline import build_storyline_from_workspace, verify_storyline_preview
from littrace.tables import build_comparison_matrices, extract_performance_cells


class ResearchWorkflowState(TypedDict, total=False):
    request: PaperSearchRequest
    config: LitTraceConfig
    audit_citations_enabled: bool
    plan_downloads_enabled: bool
    parse_full_text_enabled: bool
    extract_tables_enabled: bool
    build_storyline_enabled: bool
    routes: list[SourceRoute]
    workspace: LiteratureWorkspace
    citation_audit: object
    download_plan: object
    parse_report: object
    table_harness: object
    comparison_matrix: object
    storyline: object
    storyline_harness: object


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

    async def build_storyline(state: ResearchWorkflowState) -> ResearchWorkflowState:
        workspace = state["workspace"]
        claims = build_storyline_from_workspace(workspace)
        state["storyline"] = claims
        state["storyline_harness"] = verify_storyline_preview(claims)
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
        if state.get("parse_full_text_enabled", False):
            return "parse_full_text"
        if state.get("extract_tables_enabled", False):
            return "extract_tables"
        if state.get("build_storyline_enabled", False):
            return "build_storyline"
        return END

    def after_audit(state: ResearchWorkflowState) -> str:
        if state.get("plan_downloads_enabled", True):
            return "plan_downloads"
        if state.get("parse_full_text_enabled", False):
            return "parse_full_text"
        if state.get("extract_tables_enabled", False):
            return "extract_tables"
        if state.get("build_storyline_enabled", False):
            return "build_storyline"
        return END

    def after_download_plan(state: ResearchWorkflowState) -> str:
        if state.get("parse_full_text_enabled", False):
            return "parse_full_text"
        if state.get("extract_tables_enabled", False):
            return "extract_tables"
        if state.get("build_storyline_enabled", False):
            return "build_storyline"
        return END

    def after_parse(state: ResearchWorkflowState) -> str:
        if state.get("extract_tables_enabled", False):
            return "extract_tables"
        if state.get("build_storyline_enabled", False):
            return "build_storyline"
        return END

    def after_extract_tables(state: ResearchWorkflowState) -> str:
        if state.get("build_storyline_enabled", False):
            return "build_storyline"
        return END

    graph = StateGraph(ResearchWorkflowState)
    graph.add_node("plan_sources", plan_sources)
    graph.add_node("search_papers", search_papers)
    graph.add_node("audit_citations", audit_citations)
    graph.add_node("plan_downloads", plan_downloads)
    graph.add_node("parse_full_text", parse_full_text)
    graph.add_node("extract_tables", extract_tables)
    graph.add_node("build_storyline", build_storyline)
    graph.set_entry_point("plan_sources")
    graph.add_edge("plan_sources", "search_papers")
    graph.add_conditional_edges("search_papers", after_search)
    graph.add_conditional_edges("audit_citations", after_audit)
    graph.add_conditional_edges("plan_downloads", after_download_plan)
    graph.add_conditional_edges("parse_full_text", after_parse)
    graph.add_conditional_edges("extract_tables", after_extract_tables)
    graph.add_edge("build_storyline", END)
    return graph.compile()


async def run_research_graph(
    request: PaperSearchRequest,
    config: LitTraceConfig | None = None,
    audit_citations_enabled: bool = True,
    plan_downloads_enabled: bool = True,
    parse_full_text_enabled: bool = False,
    extract_tables_enabled: bool = False,
    build_storyline_enabled: bool = False,
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
        return ResearchRunResult(
            workspace=workspace,
            citation_audit=citation_audit,
            download_plan=download_plan,
            parse_report=parse_report,
            table_harness=table_harness,
            comparison_matrix=comparison_matrix,
            storyline=storyline,
        )

    state = await graph.ainvoke(
        {
            "request": request,
            "config": config,
            "audit_citations_enabled": audit_citations_enabled,
            "plan_downloads_enabled": plan_downloads_enabled,
            "parse_full_text_enabled": parse_full_text_enabled,
            "extract_tables_enabled": extract_tables_enabled,
            "build_storyline_enabled": build_storyline_enabled,
        }
    )
    return ResearchRunResult(
        workspace=state["workspace"],
        citation_audit=state.get("citation_audit"),
        download_plan=state.get("download_plan"),
        parse_report=state.get("parse_report"),
        table_harness=state.get("table_harness"),
        comparison_matrix=state.get("comparison_matrix"),
        storyline=state.get("storyline"),
    )
