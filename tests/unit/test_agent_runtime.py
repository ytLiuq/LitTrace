from __future__ import annotations

import asyncio

import pytest

import littrace.agent_runtime as runtime
from littrace.api.routes.research import _requires_route_workspace_save
from littrace.config import AgentRuntimeMode, LitTraceConfig
from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.session import ChatSession


class _AppServerService:
    def __init__(self, _config) -> None:
        pass

    async def chat(
        self,
        _request,
        workspace,
        _session,
        *,
        cancellation=None,
        elicitation_handler=None,
        on_delta=None,
        on_phase=None,
        on_tool=None,
        stop_event=None,
    ):
        return ChatResponse(reply="app-server", action="codex"), workspace


class _FailingAppServerService:
    def __init__(self, _config) -> None:
        pass

    async def chat(
        self,
        _request,
        _workspace,
        _session,
        *,
        cancellation=None,
        elicitation_handler=None,
        on_delta=None,
        on_phase=None,
        on_tool=None,
        stop_event=None,
    ):
        from littrace.codex_runtime.errors import AppServerError, CodexErrorCode
        raise AppServerError(
            message="transport failed after tool commit",
            error_code=CodexErrorCode.OTHER,
        )


def _session(tmp_path, config: LitTraceConfig) -> ChatSession:
    return ChatSession.from_root(
        tmp_path / "session",
        "session-1",
        config=config,
    )


def test_selection_intent_moves_to_app_server(monkeypatch, tmp_path) -> None:
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    monkeypatch.setattr(runtime, "CodexAppServerChatService", _AppServerService)

    async def unexpected_legacy(*_args, **_kwargs):
        raise AssertionError("selection-only intent must not use the legacy coordinator")

    monkeypatch.setattr(runtime, "handle_legacy_chat", unexpected_legacy)

    response, _ = asyncio.run(
        runtime.handle_agent_chat(
            ChatRequest(message="选择第1篇下载"),
            LiteratureWorkspace(),
            config,
            session=_session(tmp_path, config),
        )
    )

    assert response.reply == "app-server"


def test_search_only_intent_moves_to_app_server(monkeypatch, tmp_path) -> None:
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    monkeypatch.setattr(runtime, "CodexAppServerChatService", _AppServerService)

    async def unexpected_legacy(*_args, **_kwargs):
        raise AssertionError("search-only intent must use the App Server domain command")

    monkeypatch.setattr(runtime, "handle_legacy_chat", unexpected_legacy)

    response, _ = asyncio.run(
        runtime.handle_agent_chat(
            ChatRequest(message="搜索 MXene 压力传感器论文"),
            LiteratureWorkspace(),
            config,
            session=_session(tmp_path, config),
        )
    )

    assert response.reply == "app-server"


def test_download_only_intent_moves_to_app_server(monkeypatch, tmp_path) -> None:
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    monkeypatch.setattr(runtime, "CodexAppServerChatService", _AppServerService)

    async def unexpected_legacy(*_args, **_kwargs):
        raise AssertionError("download-only intent must enqueue through App Server")

    monkeypatch.setattr(runtime, "handle_legacy_chat", unexpected_legacy)

    response, _ = asyncio.run(
        runtime.handle_agent_chat(
            ChatRequest(message="下载当前选中的论文"),
            LiteratureWorkspace(),
            config,
            session=_session(tmp_path, config),
        )
    )

    assert response.reply == "app-server"


def test_parse_only_intent_moves_to_app_server(monkeypatch, tmp_path) -> None:
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    monkeypatch.setattr(runtime, "CodexAppServerChatService", _AppServerService)

    async def unexpected_legacy(*_args, **_kwargs):
        raise AssertionError("parse-only intent must enqueue through App Server")

    monkeypatch.setattr(runtime, "handle_legacy_chat", unexpected_legacy)

    response, _ = asyncio.run(
        runtime.handle_agent_chat(
            ChatRequest(message="解析当前论文 PDF"),
            LiteratureWorkspace(),
            config,
            session=_session(tmp_path, config),
        )
    )

    assert response.reply == "app-server"


def test_table_only_intent_moves_to_app_server(monkeypatch, tmp_path) -> None:
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    monkeypatch.setattr(runtime, "CodexAppServerChatService", _AppServerService)

    async def unexpected_legacy(*_args, **_kwargs):
        raise AssertionError("table-only intent must enqueue through App Server")

    monkeypatch.setattr(runtime, "handle_legacy_chat", unexpected_legacy)

    response, _ = asyncio.run(
        runtime.handle_agent_chat(
            ChatRequest(message="生成当前论文的性能对比表"),
            LiteratureWorkspace(),
            config,
            session=_session(tmp_path, config),
        )
    )

    assert response.reply == "app-server"


def test_storyline_intent_moves_to_app_server(monkeypatch, tmp_path) -> None:
    """Round 20: storyline no longer falls through to legacy chat.

    The MCP command ``enqueue_storyline`` owns the mutation; the App
    Server must be the one to call it. If legacy still runs, the model
    cannot reason about job IDs or progress, so we fail loud.
    """
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    monkeypatch.setattr(runtime, "CodexAppServerChatService", _AppServerService)

    async def unexpected_legacy(*_args, **_kwargs):
        raise AssertionError("storyline intent must enqueue through App Server")

    monkeypatch.setattr(runtime, "handle_legacy_chat", unexpected_legacy)

    response, _ = asyncio.run(
        runtime.handle_agent_chat(
            ChatRequest(message="生成发展脉络"),
            LiteratureWorkspace(),
            config,
            session=_session(tmp_path, config),
        )
    )

    assert response.reply == "app-server"


def test_document_intent_moves_to_app_server(monkeypatch, tmp_path) -> None:
    """Round 20: document generation now flows through ``enqueue_document``."""
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    monkeypatch.setattr(runtime, "CodexAppServerChatService", _AppServerService)

    async def unexpected_legacy(*_args, **_kwargs):
        raise AssertionError("document intent must enqueue through App Server")

    monkeypatch.setattr(runtime, "handle_legacy_chat", unexpected_legacy)

    response, _ = asyncio.run(
        runtime.handle_agent_chat(
            ChatRequest(message="撰写一份研究报告"),
            LiteratureWorkspace(),
            config,
            session=_session(tmp_path, config),
        )
    )

    assert response.reply == "app-server"


def test_autonomous_review_intent_moves_to_app_server(monkeypatch, tmp_path) -> None:
    """Round 20: autonomous review now flows through ``enqueue_autonomous_review``."""
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    monkeypatch.setattr(runtime, "CodexAppServerChatService", _AppServerService)

    async def unexpected_legacy(*_args, **_kwargs):
        raise AssertionError("autonomous_review intent must enqueue through App Server")

    monkeypatch.setattr(runtime, "handle_legacy_chat", unexpected_legacy)

    response, _ = asyncio.run(
        runtime.handle_agent_chat(
            ChatRequest(message="运行自主审查"),
            LiteratureWorkspace(),
            config,
            session=_session(tmp_path, config),
        )
    )

    assert response.reply == "app-server"


def test_legacy_domain_actions_no_longer_lists_three_domain_jobs() -> None:
    """Round 20: lock the Phase 1 contract.

    ``storyline`` / ``document`` / ``autonomous_review`` must never
    re-enter ``_LEGACY_DOMAIN_ACTIONS``; they are durable Codex MCP
    commands and any future regression that puts them back into the
    legacy short-circuit would silently re-introduce the bug we are
    removing.
    """
    assert "storyline" not in runtime._LEGACY_DOMAIN_ACTIONS
    assert "document" not in runtime._LEGACY_DOMAIN_ACTIONS
    assert "autonomous_review" not in runtime._LEGACY_DOMAIN_ACTIONS


def test_legacy_domain_actions_set_is_empty() -> None:
    """Round 20 (Phase 6): the legacy-domain fast-path is gone.

    Once ``storyline`` / ``document`` / ``autonomous_review`` moved out,
    the only remaining entries (``show_context`` / ``hide_context`` /
    ``component_status``) are handled inside ``chat.py:_route_quick_action``
    when the legacy branch runs — they don't need a fast-path here.
    An empty set is the only way to ensure future regressions can't
    silently route domain work away from the App Server again.
    """
    assert runtime._LEGACY_DOMAIN_ACTIONS == set()


def test_repl_shell_bypasses_agent_runtime(monkeypatch) -> None:
    """Round 20 (Phase 6): the REPL escape hatch must remain intact.

    ``run_shell`` is the only surface that calls ``handle_chat`` directly,
    bypassing ``handle_agent_chat`` entirely. Lock this so a future
    refactor cannot accidentally funnel REPL input through the App Server
    facade (which would break the documented low-level escape-hatch
    behavior).
    """
    from littrace import cli

    # 1. ``cli`` must import ``handle_chat`` directly from ``chat`` (the
    #    legacy coordinator), not the strangler facade ``handle_agent_chat``.
    assert cli.handle_chat.__module__ == "littrace.chat"
    assert hasattr(cli, "handle_chat")
    assert "handle_agent_chat" not in cli.__dict__

    # 2. The REPL handler must not be re-routed at runtime. Walk the source
    #    of ``run_shell`` to confirm it calls ``handle_chat`` and never
    #    ``handle_agent_chat`` — this catches a refactor that wires REPL
    #    through the strangler facade.
    import inspect

    source = inspect.getsource(cli.run_shell)
    assert "handle_chat(" in source
    assert "handle_agent_chat" not in source


def test_post_commit_failure_never_replays_mutation_through_legacy(
    monkeypatch,
    tmp_path,
) -> None:
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    config.agent_runtime.fallback_to_legacy = True
    canonical = LiteratureWorkspace()
    canonical.context.selected_for_download = ["paper-1"]
    canonical.context.filters.workspace_revision = 4
    monkeypatch.setattr(runtime, "CodexAppServerChatService", _FailingAppServerService)
    monkeypatch.setattr(runtime, "load_workspace", lambda _session: canonical)

    async def legacy(*_args, **_kwargs):
        raise AssertionError("a committed App Server command must not be replayed")

    monkeypatch.setattr(runtime, "handle_legacy_chat", legacy)

    response, updated = asyncio.run(
        runtime.handle_agent_chat(
            ChatRequest(message="普通问题"),
            LiteratureWorkspace(),
            config,
            session=_session(tmp_path, config),
        )
    )

    assert updated is canonical
    assert response.action == "codex_app_server_committed_transport_failure"
    assert response.warnings == [
        "codex_app_server_post_commit_failure: transport failed after tool commit"
    ]


def test_pre_commit_app_server_failure_falls_back_to_legacy(monkeypatch, tmp_path) -> None:
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    config.agent_runtime.fallback_to_legacy = True
    canonical = LiteratureWorkspace()
    monkeypatch.setattr(runtime, "CodexAppServerChatService", _FailingAppServerService)
    monkeypatch.setattr(runtime, "load_workspace", lambda _session: canonical)

    async def legacy(_request, workspace, _config, *, session_memory=None):
        assert session_memory is None
        assert workspace is canonical
        return ChatResponse(reply="fallback", action="legacy"), workspace

    monkeypatch.setattr(runtime, "handle_legacy_chat", legacy)

    response, updated = asyncio.run(
        runtime.handle_agent_chat(
            ChatRequest(message="普通问题"),
            LiteratureWorkspace(),
            config,
            session=_session(tmp_path, config),
        )
    )

    assert updated is canonical
    assert response.reply == "fallback"
    assert response.warnings == ["codex_app_server_fallback: transport failed after tool commit"]


@pytest.mark.parametrize("message", ["搜索后选择第1篇下载", "解析并选择第1篇下载"])
def test_composite_mutations_stay_legacy(monkeypatch, tmp_path, message: str) -> None:
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    called = False

    async def legacy(_request, workspace, _config, *, session_memory=None):
        nonlocal called
        called = True
        return ChatResponse(reply="legacy", action="legacy"), workspace

    monkeypatch.setattr(runtime, "handle_legacy_chat", legacy)

    response, _ = asyncio.run(
        runtime.handle_agent_chat(
            ChatRequest(message=message),
            LiteratureWorkspace(),
            config,
            session=_session(tmp_path, config),
        )
    )

    assert called is True
    assert response.reply == "legacy"


def test_route_does_not_persist_app_server_workspace_twice() -> None:
    assert not _requires_route_workspace_save(
        ChatResponse(reply="ok", action="codex_app_server_chat")
    )
    assert not _requires_route_workspace_save(
        ChatResponse(
            reply="committed",
            action="codex_app_server_committed_transport_failure",
        )
    )
    assert _requires_route_workspace_save(ChatResponse(reply="legacy", action="search"))


def test_handle_agent_chat_raises_app_server_error_when_fallback_disabled(
    monkeypatch, tmp_path
) -> None:
    """Round 20: when ``fallback_to_legacy`` is off, ``AppServerError`` propagates.

    The TUI/Window strict-surface contract: a failed Codex turn must
    raise rather than silently invoke the legacy coordinator. The
    route layer is the only place that consults this flag.
    """
    from littrace.codex_runtime.errors import AppServerError

    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    config.agent_runtime.fallback_to_legacy = False

    class _RaisingAppServerService:
        def __init__(self, _config) -> None:
            pass

        async def chat(self, *args, **kwargs):
            raise AppServerError("Codex App Server is not running")

    monkeypatch.setattr(runtime, "CodexAppServerChatService", _RaisingAppServerService)

    async def unexpected_legacy(*_args, **_kwargs):
        raise AssertionError(
            "fallback_to_legacy=False must propagate AppServerError, not call legacy"
        )

    monkeypatch.setattr(runtime, "handle_legacy_chat", unexpected_legacy)

    with pytest.raises(AppServerError, match="Codex App Server is not running"):
        asyncio.run(
            runtime.handle_agent_chat(
                ChatRequest(message="检索 MXene 传感器论文"),
                LiteratureWorkspace(),
                config,
                session=_session(tmp_path, config),
            )
        )
