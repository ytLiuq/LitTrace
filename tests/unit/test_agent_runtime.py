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
