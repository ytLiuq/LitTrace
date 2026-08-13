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
from littrace.research_background import (
    assess_research_background,
    mark_workspace_research_background_rejected,
    set_workspace_research_background,
    workspace_has_research_background,
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
    background_response = await _route_research_background_fast_gate(
        request,
        session_workspace,
        config,
    )
    if background_response is not None:
        api_app._set_workspace(session_workspace)
        background_response.session_id = session.session_id
        background_response.session_root = str(session.root)
        save_workspace(session, session_workspace, config=config)
        append_message(session, "user", request)
        append_message(session, "assistant", background_response)
        api_app.append_trace(
            config,
            "chat",
            {"action": background_response.action, "session_id": session.session_id},
        )
        return background_response
    request = request.model_copy(update={"session_id": session.session_id})
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
    try:
        save_workspace(session, api_app.WORKSPACE, config=config)
    except TypeError:
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


async def _route_research_background_fast_gate(
    request: ChatRequest,
    workspace: LiteratureWorkspace,
    config,
) -> ChatResponse | None:
    candidate = request.research_background
    should_check = bool(candidate)
    if not should_check and not workspace_has_research_background(workspace):
        should_check = _looks_like_obvious_non_topic(request.message)
        candidate = request.message if should_check else None
    if not should_check:
        return None

    assessment = await assess_research_background(candidate, config)
    if not assessment.accepted:
        if not workspace_has_research_background(workspace):
            mark_workspace_research_background_rejected(
                workspace,
                assessment.reason or "invalid",
            )
        return ChatResponse(
            reply=(
                "这个 session 需要先设置一个明确的研究背景，我暂时不会把当前内容作为长期科研主题。\n\n"
                + "\n".join(f"- {item}" for item in assessment.suggestions)
            ),
            action="research_background_required",
            workspace=workspace,
            warnings=[assessment.reason or "invalid_research_background"],
        )
    set_workspace_research_background(
        workspace,
        assessment.background or str(candidate or ""),
        topic=assessment.topic,
        retrieval_policy=assessment.retrieval_policy,
    )
    return ChatResponse(
        reply=(
            "已记录这个 session 的研究背景，并会把它作为长期记忆用于每日文献检索、PDF 下载和 RAG 更新。\n\n"
            f"研究主题：{workspace.context.filters.topic}\n\n"
            "接下来你可以告诉我具体要检索、下载、比较或分析什么。"
        ),
        action="research_background_set",
        workspace=workspace,
    )


def _looks_like_obvious_non_topic(message: str) -> bool:
    cleaned = " ".join(message.split()).lower()
    return cleaned in {"你好", "hello", "hi", "测试", "随便"} or len(cleaned) < 6
