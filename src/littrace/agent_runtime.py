"""Feature-flagged strangler facade for LitTrace interactive chat."""

from __future__ import annotations

import asyncio
import threading
from typing import Callable

from littrace.chat import handle_chat as handle_legacy_chat
from littrace.codex_runtime.errors import AppServerError
from littrace.codex_runtime.service import CodexAppServerChatService
from littrace.config import LitTraceConfig
from littrace.intent import parse_chat_intent
from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.runtime.memory import load_session_memory
from littrace.session import ChatSession, load_workspace

# Round 28 (Phase 6): the ``on_tool`` callback signature. The service
# emits tool-card events as the Codex App Server streams them; the
# GUI converts each event into a collapsible inline card. The
# callback runs on the asyncio thread; the GUI marshals back to the
# Tk thread via ``root.after``.
OnToolCallback = Callable[[str, str, float | None, str], None]

_LEGACY_DOMAIN_ACTIONS = {
    "storyline",
    "document",
    "autonomous_review",
    "show_context",
    "hide_context",
    "component_status",
}
_SELECTION_INTENT_ACTIONS = {
    "download",
    "select_downloads",
    "deselect_downloads",
}


async def handle_agent_chat(
    request: ChatRequest,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    *,
    session: ChatSession,
    cancellation: asyncio.Event | None = None,
    elicitation_handler=None,
    on_delta=None,
    on_phase=None,
    on_tool: OnToolCallback | None = None,
    stop_event: threading.Event | None = None,
) -> tuple[ChatResponse, LiteratureWorkspace]:
    """Route migrated capabilities to App Server and remaining mutations to legacy code.

    ``stop_event`` (Round 27) is a thread-safe ``threading.Event`` the
    GUI sets when the user clicks the Stop button during streaming.
    The service checks ``stop_event.is_set()`` between deltas and, if
    set, breaks out of the streaming loop early. The resulting
    ``ChatResponse`` carries ``truncated=True`` so the GUI can show a
    "已被用户中止" annotation. ``stop_event`` is independent of the
    existing ``asyncio.Event`` ``cancellation`` — the latter lives
    inside the asyncio loop, the former crosses the Tk/asyncio thread
    boundary.
    """

    if config.agent_runtime.mode != "codex_app_server":
        # Legacy path keeps the session_memory contract for
        # backward compatibility; the App Server path drops it
        # because the Codex-runtime stream is the canonical
        # context surface.
        session_memory = load_session_memory(session)
        return await handle_legacy_chat(
            request,
            workspace,
            config,
            session_memory=session_memory,
        )
    intent = parse_chat_intent(request.message)
    actions = set(intent.actions)
    starting_revision = workspace.context.filters.workspace_revision
    selection_only = (
        bool(actions & {"select_downloads", "deselect_downloads"})
        and actions <= _SELECTION_INTENT_ACTIONS
    )
    download_only = actions == {"download"}
    parse_only = actions == {"parse"}
    table_only = actions == {"table"}
    composite_search = "search" in actions and actions != {"search"}
    if (
        actions & _LEGACY_DOMAIN_ACTIONS
        or composite_search
        or ("download" in actions and not (selection_only or download_only))
        or ("parse" in actions and not parse_only)
        or ("table" in actions and not table_only)
    ):
        session_memory = load_session_memory(session)
        return await handle_legacy_chat(
            request,
            workspace,
            config,
            session_memory=session_memory,
        )
    try:
        return await CodexAppServerChatService(config).chat(
            request, workspace, session,
            cancellation=cancellation,
            elicitation_handler=elicitation_handler,
            on_delta=on_delta,
            on_phase=on_phase,
            on_tool=on_tool,
            stop_event=stop_event,
        )
    except AppServerError as exc:
        if not config.agent_runtime.fallback_to_legacy:
            raise
        # A mutating MCP tool may have committed before a later model/transport
        # failure. Reload Postgres truth before legacy fallback so stale input
        # cannot overwrite that committed command during the route-level save.
        workspace = load_workspace(session)
        if workspace.context.filters.workspace_revision > starting_revision:
            # A command may have committed before the App Server/model reply
            # failed. Never replay that user intent through legacy code: search
            # can perform network work and any mutation would bypass the MCP
            # idempotency key. Preserve Postgres truth and ask only for a reply
            # retry on the next turn.
            return (
                ChatResponse(
                    reply=(
                        "领域操作已经提交，但 Codex 回复链路随后中断。"
                        "已保留 Postgres 中的最新工作区，请重试查看当前结果。"
                    ),
                    action="codex_app_server_committed_transport_failure",
                    session_id=session.session_id,
                    warnings=[f"codex_app_server_post_commit_failure: {exc}"],
                ),
                workspace,
            )
        # The fallback path passes ``session_memory=None`` so
        # the legacy coordinator rebuilds memory from the
        # freshly-loaded workspace rather than from a stale
        # snapshot that pre-dates the App Server turn.
        response, updated_workspace = await handle_legacy_chat(
            request,
            workspace,
            config,
            session_memory=None,
        )
        response.warnings.append(f"codex_app_server_fallback: {exc}")
        return response, updated_workspace
