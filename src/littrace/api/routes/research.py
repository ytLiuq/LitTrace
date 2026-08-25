from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from littrace.agent_runtime import handle_agent_chat
from littrace.api.auth import resolve_request_session
from littrace.api.backend import api_app
from littrace.codex_runtime.service import CodexAppServerChatService
from littrace.models import (
    ChatCancelRequest,
    ChatCancelResponse,
    ChatRequest,
    ChatResponse,
    ChatReviewRequest,
    ChatReviewResponse,
    ChatSteerRequest,
    ChatSteerResponse,
    LiteratureWorkspace,
    PaperSearchRequest,
    ResearchRunRequest,
    ResearchRunResult,
    WorkspaceSummary,
)
from littrace.research_background import (
    assess_research_background,
    mark_workspace_research_background_rejected,
    set_workspace_research_background,
    workspace_has_research_background,
)
from littrace.session import (
    append_message,
    load_or_create_session,
    load_workspace,
    save_workspace,
)
from littrace.skill_runner import export_session_bundle_skill
from littrace.workflow import run_research_graph, run_search_preview


def _workspace_summary(workspace: LiteratureWorkspace) -> WorkspaceSummary:
    """Project a full workspace down to a small, response-friendly summary.

    The full workspace carries every parsed paper, structured document,
    and evidence span — multi-MB for a real session. The API surface
    exposes only metadata + active-paper list, never full text.
    """
    return WorkspaceSummary.from_workspace(workspace)


router = APIRouter(tags=["research"])

_APP_SERVER_PERSISTED_ACTIONS = frozenset(
    {
        "codex_app_server_chat",
        "codex_app_server_committed_transport_failure",
        # Cancellation took the turn down before the route-level save
        # could run; the MCP gateway has already persisted whatever
        # mutations the turn committed, so the route must NOT re-save
        # the workspace and risk double-incrementing revision.
        "codex_app_server_interrupted",
        "codex_app_server_interrupted_failed",
    }
)


@router.post("/search/preview", response_model=WorkspaceSummary)
async def search_preview(
    request: PaperSearchRequest,
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> WorkspaceSummary:
    config = api_app.load_config()
    auth = resolve_request_session(
        config,
        header_session_id=x_littrace_session_id,
    )
    workspace = await run_search_preview(request, config)
    api_app._set_workspace(workspace)
    api_app.append_trace(
        config,
        "search_preview",
        {"topic": request.topic, "session_id": auth.session_id, "papers": len(workspace.papers)},
    )
    return _workspace_summary(workspace)


@router.post("/workflow/research", response_model=ResearchRunResult)
async def workflow_research(
    request: ResearchRunRequest,
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> ResearchRunResult:
    config = api_app.load_config()
    resolve_request_session(
        config,
        header_session_id=x_littrace_session_id,
    )
    result = await run_research_graph(
        request.search,
        config,
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
    result.workspace = result.workspace  # keep full workspace for the LLM path
    return result


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> ChatResponse:
    config = api_app.load_config()
    auth = resolve_request_session(
        config,
        route_session_id=request.session_id,
        header_session_id=x_littrace_session_id,
    )
    session = load_or_create_session(config, auth.session_id)
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
        # Project the full workspace down to a summary before returning —
        # the route response stays under 100 KB even for a 200-paper
        # session. The full workspace is still in the session_state row +
        # workspace_dir on disk.
        background_response.workspace = _workspace_summary(session_workspace)
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
    response, session_workspace = await handle_agent_chat(
        request,
        session_workspace,
        config,
        session=session,
    )
    api_app._set_workspace(session_workspace)
    response.session_id = session.session_id
    response.session_root = str(session.root)
    # Project the full workspace down to a summary before returning.
    response.workspace = _workspace_summary(session_workspace)
    # App Server domain commands commit their workspace change inside the MCP
    # Postgres transaction.  Saving again here would turn one logical command
    # into two revisions and bypass its idempotency boundary.  Read-only App
    # Server turns likewise have no workspace change to persist.  Legacy
    # responses still own an in-memory workspace and use the existing save.
    if _requires_route_workspace_save(response):
        try:
            save_workspace(session, api_app.WORKSPACE, config=config)
        except TypeError:
            save_workspace(session, api_app.WORKSPACE)
    append_message(session, "user", request)
    append_message(session, "assistant", response)
    api_app.append_trace(config, "chat", {"action": response.action, "session_id": session.session_id})
    return response


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> StreamingResponse:
    """Server-Sent Events variant of ``/chat``.

    The App Server pushes ``item/agentMessage/delta`` frames as the
    LLM streams. We forward each one as a ``delta`` SSE event and
    close the stream with a ``done`` event carrying the final
    ``ChatResponse`` shape. A failure mid-stream produces an
    ``error`` event with a stable code so the client can render a
    retry affordance.

    The legacy ``POST /chat`` endpoint remains untouched — clients
    that cannot consume SSE keep the buffered JSON response.
    """
    config = api_app.load_config()
    auth = resolve_request_session(
        config,
        route_session_id=request.session_id,
        header_session_id=x_littrace_session_id,
    )
    session = load_or_create_session(config, auth.session_id)
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
        background_response.workspace = _workspace_summary(session_workspace)
        save_workspace(session, session_workspace, config=config)
        append_message(session, "user", request)
        append_message(session, "assistant", background_response)
        api_app.append_trace(
            config,
            "chat_stream",
            {"action": background_response.action, "session_id": session.session_id},
        )
        async def _single_done() -> AsyncIterator[bytes]:
            yield _sse_event("done", background_response.model_dump(mode="json"))
        return _sse_response(_single_done())

    request = request.model_copy(update={"session_id": session.session_id})
    queue: asyncio.Queue = asyncio.Queue()

    def _enqueue_delta(delta: str) -> None:
        # The callback runs on the App Server's reader task; push
        # into an asyncio.Queue so the StreamingResponse generator
        # can yield without blocking the transport.
        queue.put_nowait(("delta", delta))

    service = CodexAppServerChatService(config)
    chat_task = asyncio.create_task(
        service.chat(
            request,
            session_workspace,
            session,
            on_delta=_enqueue_delta,
        )
    )

    async def _stream() -> AsyncIterator[bytes]:
        try:
            while True:
                if chat_task.done() and queue.empty():
                    break
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if kind == "delta":
                    yield _sse_event("delta", {"delta": payload})
        except asyncio.CancelledError:
            chat_task.cancel()
            raise
        try:
            response, latest_workspace = await chat_task
        except Exception as exc:
            # Surface the failure as a structured ``error`` event so a
            # client can render a retry affordance; close the response
            # cleanly so EventSource auto-reconnect does not loop.
            yield _sse_event("error", {
                "code": "codex_app_server_chat_failed",
                "message": f"{exc.__class__.__name__}: {exc}",
            })
            return
        api_app._set_workspace(latest_workspace)
        response.session_id = session.session_id
        response.session_root = str(session.root)
        response.workspace = _workspace_summary(latest_workspace)
        if _requires_route_workspace_save(response):
            try:
                save_workspace(session, api_app.WORKSPACE, config=config)
            except TypeError:
                save_workspace(session, api_app.WORKSPACE)
        append_message(session, "user", request)
        append_message(session, "assistant", response)
        api_app.append_trace(
            config,
            "chat_stream",
            {"action": response.action, "session_id": session.session_id},
        )
        yield _sse_event("done", response.model_dump(mode="json"))

    return _sse_response(_stream())


def _sse_response(generator: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse_event(event: str, data: dict[str, object]) -> bytes:
    # ``data:`` line carries the JSON payload, terminated by a
    # blank line. ``event:`` line is the SSE event name so a
    # EventSource client can attach an ``addEventListener``.
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _requires_route_workspace_save(response: ChatResponse) -> bool:
    """Return whether the route, rather than MCP, owns workspace persistence."""

    return response.action not in _APP_SERVER_PERSISTED_ACTIONS


@router.post("/chat/steer")
async def chat_steer(
    request: ChatSteerRequest,
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> "ChatSteerResponse":
    """Forward a mid-turn user input to the active App Server turn.

    Round 8 step 4: the desktop GUI uses this to redirect an
    in-flight turn without cancelling it (e.g. ``"actually
    focus on the failing tests first"``). The server returns
    ``active_turn_not_steerable`` for review / manual
    compaction turns, which surfaces as ``AppServerError`` and
    is rendered as a 409 by the route.
    """
    config = api_app.load_config()
    auth = resolve_request_session(
        config,
        route_session_id=request.session_id,
        header_session_id=x_littrace_session_id,
    )
    session = load_or_create_session(config, auth.session_id)
    service = CodexAppServerChatService(config)
    result = await service.steer(
        session,
        turn_id=request.turn_id,
        text=request.text,
        client_user_message_id=request.client_user_message_id,
    )
    api_app.append_trace(
        config,
        "chat_steer",
        {"turn_id": request.turn_id, "session_id": session.session_id},
    )
    return ChatSteerResponse(
        turn_id=result.turn_id,
        thread_id=result.thread_id,
        client_user_message_id=result.client_user_message_id,
        session_id=session.session_id,
    )


@router.post("/chat/review")
async def chat_review(
    request: ChatReviewRequest,
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> "ChatReviewResponse":
    """Kick off Codex's automated reviewer on the session's thread.

    Round 8 step 4: invokes ``review/start`` and returns the
    review turn id. The actual review text is delivered via
    the on-review-complete hook installed by the service
    layer; clients poll the session state or subscribe to a
    future notification stream for the verdict. The route
    is intentionally synchronous so a CLI caller can pipe
    the next prompt based on the review turn id.
    """
    config = api_app.load_config()
    auth = resolve_request_session(
        config,
        route_session_id=request.session_id,
        header_session_id=x_littrace_session_id,
    )
    session = load_or_create_session(config, auth.session_id)
    service = CodexAppServerChatService(config)

    completion: dict[str, object] = {}

    def _on_complete(item: dict[str, object]) -> None:
        completion["item"] = item

    result = await service.start_review(
        session,
        target=request.target,
        on_review_complete=_on_complete,
    )
    api_app.append_trace(
        config,
        "chat_review",
        {
            "turn_id": result.turn_id,
            "session_id": session.session_id,
            "status": result.status,
        },
    )
    return ChatReviewResponse(
        turn_id=result.turn_id,
        thread_id=result.thread_id,
        status=result.status,
        review_text=result.reply,
        exit_item=completion.get("item"),
        session_id=session.session_id,
    )


@router.post("/chat/{turn_id}/cancel")
async def chat_cancel(
    turn_id: str,
    request: ChatCancelRequest,
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> "ChatCancelResponse":
    """Cancel an in-flight turn and record the reason on the binding.

    Round 8 step 4: supersedes the implicit cancellation
    channel on ``/chat``. Callers pass a structured ``reason``
    (e.g. ``"user_pressed_esc"`` or ``"compaction_triggered"``)
    so an operator can later inspect the binding's
    ``last_error`` column to triage why a turn was abandoned.
    """
    config = api_app.load_config()
    auth = resolve_request_session(
        config,
        route_session_id=request.session_id,
        header_session_id=x_littrace_session_id,
    )
    session = load_or_create_session(config, auth.session_id)
    service = CodexAppServerChatService(config)
    acknowledged = await service.cancel_turn_with_reason(
        session,
        turn_id=turn_id,
        reason=request.reason,
    )
    api_app.append_trace(
        config,
        "chat_cancel",
        {
            "turn_id": turn_id,
            "session_id": session.session_id,
            "reason": request.reason,
            "acknowledged": acknowledged,
        },
    )
    return ChatCancelResponse(
        turn_id=turn_id,
        session_id=session.session_id,
        reason=request.reason,
        acknowledged=acknowledged,
    )


@router.post("/sessions/{session_id}/export")
async def export_session(
    session_id: str,
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> dict[str, str]:
    config = api_app.load_config()
    auth = resolve_request_session(
        config,
        route_session_id=session_id,
        header_session_id=x_littrace_session_id,
    )
    session = load_or_create_session(config, auth.session_id)
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
            workspace=WorkspaceSummary.from_workspace(workspace),
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
        workspace=WorkspaceSummary.from_workspace(workspace),
    )


def _looks_like_obvious_non_topic(message: str) -> bool:
    cleaned = " ".join(message.split()).lower()
    return cleaned in {"你好", "hello", "hi", "测试", "随便"} or len(cleaned) < 6
