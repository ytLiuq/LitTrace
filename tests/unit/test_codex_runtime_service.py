from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest

from littrace.codex_runtime.client import AppServerError, AppServerTurnResult
from littrace.codex_runtime.service import CodexAppServerChatService
from littrace.config import LitTraceConfig
from littrace.models import ChatRequest, LiteratureWorkspace
from littrace.session import ChatSession
from littrace.state_db import SessionStateRecord


class _BindingStore:
    def __init__(self) -> None:
        self.binding = None
        self.state = None

    def get_agent_thread_binding(self, session_id: str):
        return self.binding if self.binding and self.binding.session_id == session_id else None

    def upsert_agent_thread_binding(self, binding):
        self.binding = binding
        return binding

    def get_session_state(self, session_id: str):
        return self.state if self.state and self.state.session_id == session_id else None


class _FakeClient:
    instances: ClassVar[list[_FakeClient]] = []
    on_turn: ClassVar[Callable[[], None] | None] = None

    def __init__(self, command, **kwargs) -> None:
        self.command = command
        self.kwargs = kwargs
        self.initialize_result = {"userAgent": "codex-test/1"}
        self.stderr_tail = ()
        self.started_with = None
        self.resumed = None
        type(self).instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def start_thread(self, params):
        self.started_with = params
        return {"id": "thread-1"}

    async def read_account(self, *, refresh_token=False):
        return {
            "account": {"type": "chatgpt"},
            "requiresOpenaiAuth": True,
        }

    async def resume_thread(self, thread_id, overrides):
        self.resumed = (thread_id, overrides)
        return {"id": thread_id}

    async def call_mcp_tool(self, thread_id, server, tool, arguments):
        return {"content": [{"type": "text", "text": "{}"}], "isError": False}

    async def run_turn(self, thread_id, text, *, timeout, cancellation=None):
        if type(self).on_turn is not None:
            type(self).on_turn()
        return AppServerTurnResult(
            thread_id=thread_id,
            turn_id="turn-1",
            status="completed",
            reply=f"answer: {text}",
            turn={"id": "turn-1", "status": "completed"},
        )


def test_service_persists_and_resumes_thread_binding(tmp_path: Path) -> None:
    _FakeClient.instances.clear()
    _FakeClient.on_turn = None
    config = LitTraceConfig()
    config.agent_runtime.scratch_root = tmp_path / "scratch"
    store = _BindingStore()
    session = ChatSession.from_root(tmp_path / "session", "session-1", config=config)
    workspace = LiteratureWorkspace()
    service = CodexAppServerChatService(
        config,
        state_store=store,
        client_factory=_FakeClient,
    )

    first, unchanged = asyncio.run(
        service.chat(ChatRequest(message="first"), workspace, session)
    )
    second, _ = asyncio.run(service.chat(ChatRequest(message="second"), workspace, session))

    assert first.reply == "answer: first"
    assert second.reply == "answer: second"
    assert unchanged is workspace
    assert store.binding.codex_thread_id == "thread-1"
    started = _FakeClient.instances[0].started_with
    assert started["sandbox"] == "read-only"
    assert started["approvalPolicy"] == "never"
    mcp = started["config"]["mcp_servers"]["littrace"]
    assert mcp["required"] is True
    assert "search_papers" in mcp["enabled_tools"]
    assert _FakeClient.instances[1].resumed[0] == "thread-1"
    assert _FakeClient.instances[0].kwargs["environment"]["CODEX_HOME"] == str(
        config.agent_runtime.codex_home.resolve()
    )


def test_service_namespaces_binding_when_codex_home_changes(tmp_path: Path) -> None:
    _FakeClient.instances.clear()
    _FakeClient.on_turn = None
    config = LitTraceConfig()
    config.agent_runtime.codex_home = tmp_path / "codex-a"
    store = _BindingStore()
    session = ChatSession.from_root(tmp_path / "session", "session-1", config=config)
    workspace = LiteratureWorkspace()

    first_service = CodexAppServerChatService(
        config,
        state_store=store,
        client_factory=_FakeClient,
    )
    asyncio.run(first_service.chat(ChatRequest(message="first"), workspace, session))
    first_kind = store.binding.runtime_kind

    config.agent_runtime.codex_home = tmp_path / "codex-b"
    second_service = CodexAppServerChatService(
        config,
        state_store=store,
        client_factory=_FakeClient,
    )
    asyncio.run(second_service.chat(ChatRequest(message="second"), workspace, session))

    assert store.binding.runtime_kind != first_kind
    assert _FakeClient.instances[1].started_with is not None
    assert _FakeClient.instances[1].resumed is None


class _UnauthenticatedClient(_FakeClient):
    async def read_account(self, *, refresh_token=False):
        return {"account": None, "requiresOpenaiAuth": True}


def test_isolated_home_requires_its_own_login(tmp_path: Path) -> None:
    _FakeClient.on_turn = None
    config = LitTraceConfig()
    config.agent_runtime.codex_home = tmp_path / "codex-home"
    session = ChatSession.from_root(tmp_path / "session", "session-1", config=config)
    service = CodexAppServerChatService(
        config,
        state_store=_BindingStore(),
        client_factory=_UnauthenticatedClient,
    )

    with pytest.raises(
        AppServerError,
        match="isolated LitTrace Codex home is not authenticated",
    ):
        asyncio.run(
            service.chat(ChatRequest(message="question"), LiteratureWorkspace(), session)
        )


def test_service_returns_canonical_workspace_after_mcp_mutation(tmp_path: Path) -> None:
    _FakeClient.instances.clear()
    config = LitTraceConfig()
    config.agent_runtime.codex_home = tmp_path / "codex-home"
    store = _BindingStore()
    workspace = LiteratureWorkspace()
    store.state = SessionStateRecord(
        session_id="session-1",
        workspace_json=workspace.model_dump(mode="json"),
    )
    session = ChatSession.from_root(tmp_path / "session", "session-1", config=config)

    def commit_mutation() -> None:
        mutated = LiteratureWorkspace.model_validate(store.state.workspace_json)
        mutated.context.selected_for_download = ["paper-1"]
        mutated.context.filters.workspace_revision = 1
        store.state = store.state.model_copy(
            update={
                "workspace_json": mutated.model_dump(mode="json"),
                "revision": 1,
            }
        )

    _FakeClient.on_turn = commit_mutation
    service = CodexAppServerChatService(
        config,
        state_store=store,
        client_factory=_FakeClient,
    )

    _, returned = asyncio.run(
        service.chat(ChatRequest(message="select"), workspace, session)
    )

    assert returned.context.selected_for_download == ["paper-1"]
    assert returned.context.filters.workspace_revision == 1
    assert store.binding.workspace_revision == 1
    _FakeClient.on_turn = None
