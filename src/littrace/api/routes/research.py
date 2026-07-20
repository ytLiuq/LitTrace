from __future__ import annotations

from fastapi import APIRouter

from littrace.chat import handle_chat
from littrace.models import (
    ChatRequest,
    ChatResponse,
    LiteratureWorkspace,
    PaperSearchRequest,
    ResearchRunRequest,
    ResearchRunResult,
)
from littrace.session import (
    append_message,
    load_memory,
    load_or_create_session,
    load_workspace,
    save_workspace,
)
from littrace.skill_runner import export_session_bundle_skill
from littrace.workflow import run_research_graph, run_search_preview


class _AppProxy:
    def __getattr__(self, name: str):
        from littrace.api import app as api_app

        return getattr(api_app, name)


api_app = _AppProxy()
router = APIRouter()


@router.post("/search/preview", response_model=LiteratureWorkspace)
async def search_preview(request: PaperSearchRequest) -> LiteratureWorkspace:
    config = api_app.load_config()
    api_app._set_workspace(await run_search_preview(request, config))
    api_app.append_trace(
        config,
        "search_preview",
        {"topic": request.topic, "papers": len(api_app.WORKSPACE.papers)},
    )
    return api_app.WORKSPACE


@router.post("/workflow/research", response_model=ResearchRunResult)
async def workflow_research(request: ResearchRunRequest) -> ResearchRunResult:
    result = await run_research_graph(
        request.search,
        api_app.load_config(),
        audit_citations_enabled=request.audit_citations,
        plan_downloads_enabled=request.plan_downloads,
        route_publishers_enabled=request.route_publishers,
        parse_full_text_enabled=request.parse_full_text,
        extract_tables_enabled=request.extract_tables,
        build_storyline_enabled=request.build_storyline,
        compose_document_enabled=request.compose_document,
        autonomous_review_enabled=request.autonomous_review,
        auto_replan_enabled=request.auto_replan,
    )
    api_app._set_workspace(result.workspace)
    return result


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    config = api_app.load_config()
    session = load_or_create_session(config, request.session_id)
    session_workspace = load_workspace(session)
    session_memory = load_memory(session)
    response, session_workspace = await handle_chat(
        request,
        session_workspace,
        config,
        session_memory=session_memory,
    )
    api_app._set_workspace(session_workspace)
    response.session_id = session.session_id
    response.session_root = str(session.root)
    save_workspace(session, api_app.WORKSPACE)
    append_message(session, "user", request)
    append_message(session, "assistant", response)
    api_app.append_trace(config, "chat", {"action": response.action, "session_id": session.session_id})
    return response


@router.post("/sessions/{session_id}/export")
async def export_session(session_id: str) -> dict[str, str]:
    config = api_app.load_config()
    session = load_or_create_session(config, session_id)
    workspace = load_workspace(session)
    return await export_session_bundle_skill(session, workspace, config)
