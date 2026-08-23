"""Feature-flagged strangler facade for LitTrace interactive chat."""

from __future__ import annotations

from littrace.chat import handle_chat as handle_legacy_chat
from littrace.codex_runtime.service import CodexAppServerChatService
from littrace.config import LitTraceConfig
from littrace.intent import parse_chat_intent
from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.session import ChatSession, load_workspace

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
    session_memory=None,
) -> tuple[ChatResponse, LiteratureWorkspace]:
    """Route migrated capabilities to App Server and remaining mutations to legacy code."""

    if config.agent_runtime.mode != "codex_app_server":
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
        return await handle_legacy_chat(
            request,
            workspace,
            config,
            session_memory=session_memory,
        )
    try:
        return await CodexAppServerChatService(config).chat(request, workspace, session)
    except Exception as exc:
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
        response, updated_workspace = await handle_legacy_chat(
            request,
            workspace,
            config,
            session_memory=session_memory,
        )
        response.warnings.append(f"codex_app_server_fallback: {exc}")
        return response, updated_workspace
