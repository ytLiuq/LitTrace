from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar
from unittest import mock

import pytest

from littrace.codex_runtime.client import AppServerError, AppServerTurnResult
from littrace.codex_runtime.errors import UnauthorizedError
from littrace.codex_runtime.service import CodexAppServerChatService
from littrace.config import LitTraceConfig, SandboxPolicy
from littrace.models import ChatRequest, LiteratureWorkspace
from littrace.session import ChatSession
from littrace.state_db import SessionStateRecord


class _BindingStore:
    def __init__(self) -> None:
        self.binding = None
        self.state = None
        self.compaction_due_calls: list[tuple[int, int, int]] = []
        self.enqueued_tasks: list[AsyncTaskRecord] = []
        self.bindings: dict[str, AgentThreadBindingRecord] = {}

    def session_write_lock(self, _session_id):
        # Round 4 P2 step 10: in-memory no-op.
        from contextlib import nullcontext
        return nullcontext()

    def get_agent_thread_binding(self, session_id: str):
        b = self.bindings.get(session_id)
        if b is not None and b.session_id == session_id:
            return b
        return None

    def upsert_agent_thread_binding(self, binding):
        self.bindings[binding.session_id] = binding
        self.binding = binding
        return binding

    def get_session_state(self, session_id: str):
        if self.state is not None and self.state.session_id == session_id:
            return self.state
        return None

    def upsert_session_state(self, state):
        # Round 4 P1 step 5: status machine (draft / idle / active /
        # systemError / archived) flips per turn, so the test fake
        # needs to persist the latest record. Tests that use a fake
        # client without seeding a session_state row pass ``None``;
        # treat that as a no-op so the chat path does not crash.
        if state is None:
            return None
        self.state = state.model_copy(deep=True)
        return state

    def compaction_due_sessions(
        self, *, threshold_turns, threshold_tokens, limit,
    ):
        from datetime import UTC, datetime, timedelta
        self.compaction_due_calls.append(
            (threshold_turns, threshold_tokens, limit)
        )
        out: list[tuple[str, str, int, int]] = []
        for sid, binding in self.bindings.items():
            # Mirror the production query: drive the decision off
            # the session_state status (which is what the Postgres
            # JOIN looks at), not the binding row, because
            # ``archived`` / ``systemError`` / ``draft`` only show
            # up on session_state — the binding always defaults to
            # ``active``.
            if self.state is not None and self.state.session_id == sid:
                if self.state.status not in ("active", "idle"):
                    continue
            if (
                binding.turn_count < threshold_turns
                and binding.last_total_tokens < threshold_tokens
            ):
                continue
            if (
                binding.turn_count < threshold_turns
                and binding.last_total_tokens < threshold_tokens
            ):
                continue
            last = binding.last_compacted_at
            if last and last >= (datetime.now(UTC) - timedelta(hours=1)).isoformat():
                continue
            out.append(
                (binding.session_id, binding.codex_thread_id,
                 binding.turn_count, binding.last_total_tokens)
            )
        out.sort(key=lambda r: -r[2])
        return out[:limit]

    def enqueue_async_task(self, task):
        self.enqueued_tasks.append(task)
        return task

    def list_async_tasks(self, *, kind=None, status=None, limit=20):
        rows = list(self.enqueued_tasks)
        if kind is not None:
            rows = [r for r in rows if r.kind == kind]
        if status is not None:
            rows = [r for r in rows if r.status == status]
        return rows[:limit]

    def update_async_task(self, task):
        # Replace in place by task_id; mirror the Postgres
        # ``update_async_task`` semantics.
        for i, existing in enumerate(self.enqueued_tasks):
            if existing.task_id == task.task_id:
                self.enqueued_tasks[i] = task
                break
        return task


class _FakeClient:
    instances: ClassVar[list[_FakeClient]] = []
    on_turn: ClassVar[Callable[[], None] | None] = None
    compact_calls: ClassVar[list[str]] = []

    def __init__(self, command, **kwargs) -> None:
        self.command = command
        self.kwargs = kwargs
        self.initialize_result = {"userAgent": "codex-test/1"}
        self.stderr_tail = ()
        self.started_with = None
        self.resumed = None
        # Mirror the real AppServerClient kwarg so rollout injection
        # round-trips through the same kwargs dict.
        self._rollout_recorder = kwargs.get("rollout_recorder")
        self._rollout_recorders: dict[str, object] = (
            {"__default__": self._rollout_recorder}
            if self._rollout_recorder is not None
            else {}
        )
        type(self).instances.append(self)

    def set_rollout_recorder(self, thread_id, recorder):  # type: ignore[no-untyped-def]
        # Round 4 P1 step 7 mirror.
        if recorder is None:
            self._rollout_recorders.pop(thread_id, None)
        else:
            self._rollout_recorders[thread_id] = recorder

    def set_elicitation_handler(self, handler):  # type: ignore[no-untyped-def]
        # Round 17 mirror. The production AppServerClient now exposes
        # an opt-in hook for ``mcpServer/elicitation/request``. The
        # fake records every set/clear so tests can assert the
        # service bound the right value around run_turn and cleared
        # it on exit.
        self._elicitation_handler = handler
        self._elicitation_history: list[object] = (
            getattr(self, "_elicitation_history", [])
        )
        self._elicitation_history.append(handler)

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

    async def compact_thread(self, thread_id, *, timeout=10.0):
        # Round 5 test stub. Real AppServerClient.compact_thread
        # fires ``thread/compact`` and returns a server reply; the
        # fake just records the call so a test can assert it ran
        # (or did not run) without involving the real RPC plumbing.
        type(self).compact_calls.append(thread_id)
        return {"ok": True, "thread_id": thread_id}

    async def run_turn(self, thread_id, text, *, timeout, cancellation=None, on_delta=None):
        if type(self).on_turn is not None:
            type(self).on_turn()
        # Mirror what AppServerClient.run_turn writes into the rollout
        # recorder so the service-layer tests exercise the same
        # append surface.
        recorder = getattr(self, "_rollout_recorder", None)
        if recorder is not None:
            recorder.append(type_="turn_start", turn_id="turn-1",
                             thread_id=thread_id, user_text=text)
            recorder.append(type_="event", method="item/agentMessage/delta",
                             turn_id="turn-1", params={"delta": text})
            recorder.append(type_="turn_complete", turn_id="turn-1",
                             status="completed", reply=f"answer: {text}",
                             turn={"id": "turn-1", "status": "completed"})
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
    assert started["approvalPolicy"] == "on-request"
    assert "writableRoots" not in started
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
        UnauthorizedError,
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


def test_service_passes_writable_roots_when_workspace_write(tmp_path) -> None:
    config = LitTraceConfig(
        agent_runtime=LitTraceConfig.model_fields["agent_runtime"].default_factory(),
    )
    config.agent_runtime.sandbox_policy = SandboxPolicy.WORKSPACE_WRITE
    config.agent_runtime.writable_roots = [tmp_path / "scratch"]
    session = ChatSession.from_root(tmp_path / "session", "session-1", config=config)
    service = CodexAppServerChatService(config)

    async def fake_run(self, *_args, **_kwargs):
        _FakeClient.instances[0].started_with = self.started_with
        return AppServerTurnResult(
            thread_id="thr-1", turn_id="turn-1",
            status="completed", reply="ok", turn={},
        )

    with mock.patch.object(
        _FakeClient, "start_thread",
        lambda self, params: asyncio.sleep(0, asyncio.Future())
        if False else _FakeClient.instances.__setitem__(
            len(_FakeClient.instances), self
        ) or asyncio.sleep(0)
    ):
        pass

    # drive directly via the service so we can assert on the captured
    # thread_overrides without a full chat round-trip.
    overrides = service._thread_overrides(session.root / "scratch")
    assert overrides["sandbox"] == "workspace-write"
    assert overrides["approvalPolicy"] == "on-request"
    assert overrides["writableRoots"] == [str(tmp_path / "scratch")]


def test_service_omits_writable_roots_for_danger_full_access(tmp_path) -> None:
    config = LitTraceConfig(
        agent_runtime=LitTraceConfig.model_fields["agent_runtime"].default_factory(),
    )
    config.agent_runtime.sandbox_policy = SandboxPolicy.DANGER_FULL_ACCESS
    session = ChatSession.from_root(tmp_path / "session", "session-1", config=config)
    service = CodexAppServerChatService(config)
    overrides = service._thread_overrides(session.root / "scratch")
    assert overrides["sandbox"] == "danger-full-access"
    assert overrides["approvalPolicy"] == "on-request"
    assert "writableRoots" not in overrides


def test_service_chat_returns_interrupted_action(monkeypatch, tmp_path) -> None:
    config = LitTraceConfig(
        agent_runtime=LitTraceConfig.model_fields["agent_runtime"].default_factory(),
    )
    session = ChatSession.from_root(tmp_path / "session", "session-1", config=config)
    store = _BindingStore()
    # Pre-seed the binding so service skips the upsert (and the FK to
    # session_state that would otherwise need a Postgres row).
    from littrace.state_db import AgentThreadBindingRecord
    store.binding = AgentThreadBindingRecord(
        session_id=session.session_id,
        codex_thread_id="thread-existing",
    )
    service = CodexAppServerChatService(
        config, state_store=store, client_factory=_FakeClient,
    )
    captured: list[dict[str, object]] = []

    async def fake_run(self, thread_id, text, *, timeout, cancellation=None, on_delta=None):
        captured.append({"cancellation": cancellation})
        return AppServerTurnResult(
            thread_id=thread_id, turn_id="turn-1",
            status="interrupted", reply="", turn={},
        )

    monkeypatch.setattr(_FakeClient, "run_turn", fake_run)
    cancellation = asyncio.Event()
    response, _ = asyncio.run(
        service.chat(
            ChatRequest(message="hi"), LiteratureWorkspace(), session,
            cancellation=cancellation,
        )
    )
    assert captured and captured[0]["cancellation"] is cancellation
    assert response.action == "codex_app_server_interrupted"


def test_service_chat_returns_interrupted_failed_action(monkeypatch, tmp_path) -> None:
    config = LitTraceConfig(
        agent_runtime=LitTraceConfig.model_fields["agent_runtime"].default_factory(),
    )
    session = ChatSession.from_root(tmp_path / "session", "session-1", config=config)
    store = _BindingStore()
    from littrace.state_db import AgentThreadBindingRecord
    store.binding = AgentThreadBindingRecord(
        session_id=session.session_id,
        codex_thread_id="thread-existing",
    )
    service = CodexAppServerChatService(
        config, state_store=store, client_factory=_FakeClient,
    )

    async def fake_run(self, thread_id, text, *, timeout, cancellation=None, on_delta=None):
        return AppServerTurnResult(
            thread_id=thread_id, turn_id="turn-1",
            status="failed", reply="", turn={},
        )

    monkeypatch.setattr(_FakeClient, "run_turn", fake_run)
    response, _ = asyncio.run(
        service.chat(
            ChatRequest(message="hi"), LiteratureWorkspace(), session,
        )
    )
    assert response.action == "codex_app_server_interrupted_failed"


def test_service_writes_rollout_file_when_enabled(monkeypatch, tmp_path) -> None:
    config = LitTraceConfig(
        agent_runtime=LitTraceConfig.model_fields["agent_runtime"].default_factory(),
    )
    config.agent_runtime.rollout_enabled = True
    session = ChatSession.from_root(tmp_path / "session", "session-1", config=config)
    store = _BindingStore()
    from littrace.state_db import AgentThreadBindingRecord
    store.binding = AgentThreadBindingRecord(
        session_id=session.session_id,
        codex_thread_id="thread-rollout",
    )
    service = CodexAppServerChatService(
        config, state_store=store, client_factory=_FakeClient,
    )

    async def fake_run(self, thread_id, text, *, timeout, cancellation=None, on_delta=None):
        # Mirror the real AppServerClient.run_turn surface that the
        # rollout recorder relies on so the service-level test
        # actually exercises the append path.
        recorder = getattr(self, "_rollout_recorder", None)
        if recorder is not None:
            recorder.append(
                type_="turn_start", turn_id="turn-1",
                thread_id=thread_id, user_text=text,
            )
            recorder.append(
                type_="turn_complete", turn_id="turn-1",
                status="completed", reply="ok",
            )
        return AppServerTurnResult(
            thread_id=thread_id, turn_id="turn-1",
            status="completed", reply="ok", turn={},
        )

    monkeypatch.setattr(_FakeClient, "run_turn", fake_run)
    asyncio.run(
        service.chat(
            ChatRequest(message="hi"), LiteratureWorkspace(), session,
        )
    )

    rollout_dir = session.root / "rollouts"
    files = list(rollout_dir.glob("*.jsonl"))
    assert files, f"expected a rollout JSONL file under {rollout_dir}"
    # File path itself encodes session_id.
    assert session.session_id in files[0].name
    contents = files[0].read_text(encoding="utf-8")
    assert '"type": "turn_start"' in contents
    assert '"type": "turn_complete"' in contents


def test_service_skips_rollout_when_disabled(monkeypatch, tmp_path) -> None:
    config = LitTraceConfig(
        agent_runtime=LitTraceConfig.model_fields["agent_runtime"].default_factory(),
    )
    # rollout_enabled stays False (the default).
    session = ChatSession.from_root(tmp_path / "session", "session-1", config=config)
    store = _BindingStore()
    from littrace.state_db import AgentThreadBindingRecord
    store.binding = AgentThreadBindingRecord(
        session_id=session.session_id,
        codex_thread_id="thread-no-rollout",
    )
    service = CodexAppServerChatService(
        config, state_store=store, client_factory=_FakeClient,
    )

    async def fake_run(self, thread_id, text, *, timeout, cancellation=None, on_delta=None):
        return AppServerTurnResult(
            thread_id=thread_id, turn_id="turn-1",
            status="completed", reply="ok", turn={},
        )

    monkeypatch.setattr(_FakeClient, "run_turn", fake_run)
    asyncio.run(
        service.chat(
            ChatRequest(message="hi"), LiteratureWorkspace(), session,
        )
    )

    rollout_dir = session.root / "rollouts"
    assert not rollout_dir.exists(), (
        "rollout directory must not exist when disabled"
    )


def test_looks_like_refusal_matches_upstream_guard_patterns() -> None:
    """Round 16: the upstream codex App Server exec-mode guard
    refuses with English and Chinese phrasings; both must trigger
    the legacy fallback."""
    from littrace.codex_runtime.service import _looks_like_refusal

    upstream_guard = (
        "MCP tool call requires approval, but approval policy is never",
        "LitTrace 工具调用需要批准，但当前审批策略为 never",
        "Tool call requires approval before proceeding",
    )
    for reply in upstream_guard:
        assert _looks_like_refusal(reply), f"upstream guard missed: {reply!r}"


def test_looks_like_refusal_matches_model_confusion_patterns() -> None:
    """Round 16: the ChatGPT.app bundled codex reaches the MCP
    server fine (the tool returns success:true) but the model
    then misreads the empty-but-valid response as a refusal and
    declines to call any further tool. Its confused phrasing is
    stable enough that substring matching covers it."""
    from littrace.codex_runtime.service import _looks_like_refusal

    model_confusion = (
        "LitTrace 的 `get_workspace_context` 工具调用被系统拒绝",
        "工作区上下文读取刚被工具层拒绝",
        "LitTrace MCP 工具调用被运行环境拦截",
        "工具调用连续被拒绝",
        "The tool call was rejected by the active session",
        "Tool call was denied because the workspace is empty",
    )
    for reply in model_confusion:
        assert _looks_like_refusal(reply), f"model confusion missed: {reply!r}"


def test_looks_like_refusal_ignores_normal_replies() -> None:
    """Round 16: normal assistant replies must not be flagged as
    refusals even if they happen to contain trigger substrings
    like "rejected" in unrelated contexts."""
    from littrace.codex_runtime.service import _looks_like_refusal

    benign = (
        "Here are the top 3 papers for MXene pressure sensor 2024.",
        "I'll search the workspace now.",
        "The previous results were rejected by the user, so I'm trying again.",
        "",
    )
    for reply in benign:
        assert not _looks_like_refusal(reply), f"false positive: {reply!r}"


def test_service_binds_and_clears_elicitation_handler(monkeypatch, tmp_path) -> None:
    """Round 17: ``chat()`` must install the caller's elicitation
    handler on the AppServerClient around ``run_turn`` and clear it
    on exit so a subsequent chat on the same shared client does
    NOT inherit a stale UI hook."""
    config = LitTraceConfig(
        agent_runtime=LitTraceConfig.model_fields["agent_runtime"].default_factory(),
    )
    session = ChatSession.from_root(tmp_path / "session", "session-1", config=config)
    store = _BindingStore()
    from littrace.state_db import AgentThreadBindingRecord
    store.binding = AgentThreadBindingRecord(
        session_id=session.session_id,
        codex_thread_id="thread-elic",
    )
    service = CodexAppServerChatService(
        config, state_store=store, client_factory=_FakeClient,
    )

    async def noop(params):
        return {"action": "decline", "content": None, "_meta": None}

    asyncio.run(
        service.chat(
            ChatRequest(message="hi"),
            LiteratureWorkspace(),
            session,
            elicitation_handler=noop,
        )
    )
    # The fake client records every (set_elicitation_handler)
    # call. The contract is: the caller's handler is installed
    # BEFORE run_turn and cleared AFTER — even on the happy path.
    client = _FakeClient.instances[-1]
    assert noop in client._elicitation_history, (
        "service.chat did not bind the caller's elicitation handler"
    )
    assert client._elicitation_history[-1] is None, (
        "service.chat did not clear the elicitation handler on exit"
    )
    # And the caller's handler was bound exactly once (set + clear = 2).
    handler_sets = [h for h in client._elicitation_history if h is not None]
    assert handler_sets == [noop], (
        f"handler was set/cleared extra times: {client._elicitation_history!r}"
    )
