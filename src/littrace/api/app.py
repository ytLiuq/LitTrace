from __future__ import annotations

from fastapi import FastAPI

from littrace.eval_api import (
    EvalMetricReport,
    parsing_metrics,
    retrieval_metrics,
    storyline_metrics,
)
from littrace.access import build_download_plan
from littrace.agents import AgentRoleSpec, AgentRuntimeStatus, agent_runtime_statuses, crew_role_specs
from littrace.attachments import (
    AttachmentResult,
    DownloadPresenceReport,
    attach_pdf_to_paper,
    check_download_presence,
)
from littrace.auto_resume import AutoResumeResult, auto_resume_downloaded_pdfs
from littrace.citations import audit_citation_links, citation_records_for_papers
from littrace.chat import handle_chat
from littrace.config import load_config
from littrace.context import apply_context_update
from littrace.downloads import execute_downloads
from littrace.export import export_session_bundle
from littrace.golden_eval import GoldenEvalReport, run_golden_eval
from littrace.login_flow import LoginLaunchResult, launch_login_for_paper
from littrace.models import (
    ChatRequest,
    ChatResponse,
    ComparisonMatrixReport,
    ContextUpdate,
    CitationAudit,
    CitationRecord,
    DownloadExecutionRequest,
    DownloadExecutionResult,
    DownloadPlan,
    LiteratureWorkspace,
    PaperSearchRequest,
    ResearchRunRequest,
    ResearchRunResult,
)
from littrace.parsing import parse_workspace_papers
from littrace.pdf_benchmark import PDFBenchmarkReport, benchmark_pdf_parsing
from littrace.publisher_connectors import (
    PublisherRouteReport,
    PublisherSearchPlanReport,
    build_publisher_search_plan,
    publisher_routes_for_workspace,
)
from littrace.publisher_retrieval import (
    BrowserRetrievalPlan,
    PublisherEnrichment,
    PublisherRetrievalResult,
    build_browser_retrieval_plan,
    fetch_publisher_search_results,
    parse_publisher_article_html,
)
from littrace.session import (
    append_message,
    load_or_create_session,
    load_workspace,
    save_workspace,
)
from littrace.tables import build_comparison_matrices, extract_performance_cells
from littrace.storyline import render_structured_storyline_report
from littrace.source_router import route_sources
from littrace.workflow import run_research_graph, run_search_preview

app = FastAPI(title="LitTrace API", version="0.1.0")

WORKSPACE = LiteratureWorkspace()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sources/materials-chemistry")
def materials_chemistry_sources(wants_recent: bool = True) -> list[dict[str, object]]:
    return [route.__dict__ for route in route_sources("materials chemistry", wants_recent)]


@app.get("/agents/crew", response_model=list[AgentRoleSpec])
def agents_crew() -> list[AgentRoleSpec]:
    return crew_role_specs()


@app.get("/agents/status", response_model=list[AgentRuntimeStatus])
def agents_status() -> list[AgentRuntimeStatus]:
    return agent_runtime_statuses()


@app.post("/search/preview", response_model=LiteratureWorkspace)
async def search_preview(request: PaperSearchRequest) -> LiteratureWorkspace:
    global WORKSPACE
    WORKSPACE = await run_search_preview(request, load_config())
    return WORKSPACE


@app.post("/workflow/research", response_model=ResearchRunResult)
async def workflow_research(request: ResearchRunRequest) -> ResearchRunResult:
    global WORKSPACE
    result = await run_research_graph(
        request.search,
        load_config(),
        audit_citations_enabled=request.audit_citations,
        plan_downloads_enabled=request.plan_downloads,
        route_publishers_enabled=request.route_publishers,
        parse_full_text_enabled=request.parse_full_text,
        extract_tables_enabled=request.extract_tables,
        build_storyline_enabled=request.build_storyline,
    )
    WORKSPACE = result.workspace
    return result


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    global WORKSPACE
    config = load_config()
    session = load_or_create_session(config, request.session_id)
    session_workspace = load_workspace(session)
    response, WORKSPACE = await handle_chat(request, session_workspace, config)
    response.session_id = session.session_id
    response.session_root = str(session.root)
    save_workspace(session, WORKSPACE)
    append_message(session, "user", request)
    append_message(session, "assistant", response)
    return response


@app.post("/sessions/{session_id}/export")
def export_session(session_id: str) -> dict[str, str]:
    config = load_config()
    session = load_or_create_session(config, session_id)
    workspace = load_workspace(session)
    return export_session_bundle(session, workspace)


@app.get("/context", response_model=LiteratureWorkspace)
def get_context() -> LiteratureWorkspace:
    return WORKSPACE


@app.patch("/context", response_model=LiteratureWorkspace)
def update_context(update: ContextUpdate) -> LiteratureWorkspace:
    global WORKSPACE
    WORKSPACE = apply_context_update(WORKSPACE, update)
    return WORKSPACE


@app.post("/downloads/plan", response_model=DownloadPlan)
def download_plan() -> DownloadPlan:
    config = load_config()
    selected_ids = set(WORKSPACE.context.selected_for_download)
    papers = [WORKSPACE.papers[paper_id] for paper_id in WORKSPACE.context.active_papers]
    return build_download_plan(config, papers, selected_ids)


@app.get("/publishers/routes", response_model=PublisherRouteReport)
def publisher_routes() -> PublisherRouteReport:
    return publisher_routes_for_workspace(WORKSPACE)


@app.get("/publishers/search-plan", response_model=PublisherSearchPlanReport)
def publisher_search_plan(topic: str) -> PublisherSearchPlanReport:
    return build_publisher_search_plan(topic)


@app.post("/publishers/retrieve", response_model=PublisherRetrievalResult)
async def publisher_retrieve(topic: str, family: str = "acs") -> PublisherRetrievalResult:
    plan_report = build_publisher_search_plan(topic, families=[family])
    if not plan_report.plans:
        raise KeyError(f"No publisher search plan for {family}")
    return await fetch_publisher_search_results(load_config(), plan_report.plans[0])


@app.get("/publishers/browser-plan", response_model=BrowserRetrievalPlan)
def publisher_browser_plan(topic: str, family: str = "acs") -> BrowserRetrievalPlan:
    plan_report = build_publisher_search_plan(topic, families=[family])
    if not plan_report.plans:
        raise KeyError(f"No publisher search plan for {family}")
    return build_browser_retrieval_plan(plan_report.plans[0])


@app.post("/publishers/enrich-html", response_model=PublisherEnrichment)
def publisher_enrich_html(html: str) -> PublisherEnrichment:
    return parse_publisher_article_html(html)


@app.post("/downloads/execute", response_model=DownloadExecutionResult)
async def downloads_execute(request: DownloadExecutionRequest) -> DownloadExecutionResult:
    config = load_config()
    papers = [WORKSPACE.papers[paper_id] for paper_id in WORKSPACE.context.active_papers]
    return await execute_downloads(config, papers, request)


@app.post("/downloads/login/{paper_id}", response_model=LoginLaunchResult)
def downloads_login(paper_id: str, dry_run: bool = False) -> LoginLaunchResult:
    config = load_config()
    paper = WORKSPACE.papers[paper_id]
    return launch_login_for_paper(config, paper, dry_run=dry_run)


@app.post("/downloads/check", response_model=DownloadPresenceReport)
def downloads_check() -> DownloadPresenceReport:
    return check_download_presence(load_config(), WORKSPACE)


@app.post("/downloads/resume", response_model=AutoResumeResult)
def downloads_resume(session_id: str | None = None) -> AutoResumeResult:
    global WORKSPACE
    config = load_config()
    session = load_or_create_session(config, session_id) if session_id else None
    WORKSPACE, result = auto_resume_downloaded_pdfs(config, WORKSPACE, session)
    if session:
        save_workspace(session, WORKSPACE)
    return result


@app.post("/papers/{paper_id}/attach-pdf", response_model=AttachmentResult)
def attach_pdf(paper_id: str, source_path: str) -> AttachmentResult:
    return attach_pdf_to_paper(load_config(), WORKSPACE, paper_id, source_path)


@app.post("/parse/context", response_model=LiteratureWorkspace)
def parse_context() -> LiteratureWorkspace:
    global WORKSPACE
    WORKSPACE, _ = parse_workspace_papers(WORKSPACE, load_config())
    return WORKSPACE


@app.post("/tables/extract", response_model=ResearchRunResult)
def tables_extract() -> ResearchRunResult:
    global WORKSPACE
    WORKSPACE, harness = extract_performance_cells(WORKSPACE)
    return ResearchRunResult(
        workspace=WORKSPACE,
        table_harness=harness.model_dump(),
        comparison_matrix=build_comparison_matrices(WORKSPACE),
    )


@app.get("/tables/matrix", response_model=ComparisonMatrixReport)
def tables_matrix() -> ComparisonMatrixReport:
    return build_comparison_matrices(WORKSPACE)


@app.get("/storyline/report")
def storyline_report() -> dict[str, str]:
    return {"markdown": render_structured_storyline_report(WORKSPACE)}


@app.get("/citations/context", response_model=list[CitationRecord])
def context_citations() -> list[CitationRecord]:
    papers = [WORKSPACE.papers[paper_id] for paper_id in WORKSPACE.context.active_papers]
    return citation_records_for_papers(papers)


@app.post("/citations/audit", response_model=CitationAudit)
async def audit_context_citations() -> CitationAudit:
    config = load_config()
    papers = [WORKSPACE.papers[paper_id] for paper_id in WORKSPACE.context.active_papers]
    return await audit_citation_links(papers, config)


@app.post("/eval/retrieval", response_model=EvalMetricReport)
def eval_retrieval(topic: str | None = None) -> EvalMetricReport:
    return EvalMetricReport(run_id="preview", topic=topic, metrics=retrieval_metrics())


@app.post("/eval/pdf-parsing", response_model=EvalMetricReport)
def eval_pdf_parsing(topic: str | None = None) -> EvalMetricReport:
    return EvalMetricReport(run_id="preview", topic=topic, metrics=parsing_metrics())


@app.get("/eval/pdf-benchmark", response_model=PDFBenchmarkReport)
def eval_pdf_benchmark() -> PDFBenchmarkReport:
    return benchmark_pdf_parsing(WORKSPACE, load_config())


@app.post("/eval/storyline", response_model=EvalMetricReport)
def eval_storyline(topic: str | None = None) -> EvalMetricReport:
    return EvalMetricReport(run_id="preview", topic=topic, metrics=storyline_metrics())


@app.post("/eval/end-to-end", response_model=EvalMetricReport)
def eval_end_to_end(topic: str | None = None) -> EvalMetricReport:
    metrics = {}
    metrics.update(retrieval_metrics())
    metrics.update(parsing_metrics())
    metrics.update(storyline_metrics())
    metrics["pdf_download_success_rate"] = 0.0
    metrics["citation_link_verified_rate"] = 0.0
    return EvalMetricReport(run_id="preview", topic=topic, metrics=metrics)


@app.get("/eval/golden", response_model=GoldenEvalReport)
def eval_golden() -> GoldenEvalReport:
    return run_golden_eval(load_config())
