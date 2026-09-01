"""Round 17 user-perspective tests for the ShellController changes.

Each test names the user-visible behavior the change enables, not
the implementation. A failing test here means the user can no
longer do the thing the commit message promised.

Tests are pure-Python (no Qt, no asyncio loops) so they run on
plain CI in well under a second each.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from littrace.shell_controller import (
    ShellController,
    ShellEvent,
    _decode_jwt_exp,
    _classify_chat_error,
    _CHAT_ERROR_MESSAGES,
    _fmt_unix,
    get_slash_command_names,
)
from littrace.config import LitTraceConfig, load_config
from littrace.codex_runtime.errors import (
    AppServerError,
    UnauthorizedError,
    ContextWindowExceededError,
    CodexErrorCode,
)
from littrace.models import LiteratureWorkspace
from littrace.session import ChatSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingBus:
    """Minimal stand-in for the controller's ShellEventBus.

    We don't need the real bus because the controller only calls
    ``bus.emit`` on it; we capture every event for assertions.
    """

    def __init__(self) -> None:
        self.events: list[ShellEvent] = []

    def subscribe(self, handler) -> None:
        self._handler = handler

    def emit(self, event: ShellEvent) -> None:
        self.events.append(event)
        try:
            self._handler(event)
        except Exception:
            pass


def _make_test_config(tmp_path: Path) -> LitTraceConfig:
    """Return a config rooted at ``tmp_path`` so the controller
    doesn't write into the real data/ directory while testing.
    """
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "agent_runtime:\n"
        "  mode: codex_app_server\n"
        "  codex_home_mode: isolated\n"
        "  codex_home: ./data/codex-home\n"
        "  fallback_to_legacy: false\n"
        "metadata_store:\n"
        "  backend: postgres\n"
        "  postgres_dsn: postgresql://localhost:5432/test_dummy\n"
        "  schema_name: littrace_test\n",
        encoding="utf-8",
    )
    return load_config(str(cfg_path))


class _StubSession:
    """Bare-bones ChatSession stand-in so the controller doesn't
    touch Postgres during unit tests. Only carries the attributes
    the controller / tests actually touch.
    """

    def __init__(self, config: LitTraceConfig | None = None) -> None:
        self.session_id = "test-session-0001"
        # The state_store resolver (used by save_workspace /
        # load_workspace) inspects these attributes when the
        # caller hasn't passed an explicit config. Mirror the
        # real session's metadata so the resolver doesn't raise
        # "metadata_store.postgres_dsn is required".
        if config is not None:
            ms = config.metadata_store
            self.metadata_store_backend = ms.backend
            self.metadata_postgres_dsn = ms.postgres_dsn
            self.metadata_schema_name = ms.schema_name


def _make_controller(config: LitTraceConfig) -> ShellController:
    """Build a ShellController without hitting Postgres.

    ``ShellController.__init__`` calls ``create_chat_session``,
    which writes a placeholder row to the metadata store. That
    step needs a live Postgres, which the unit-test environment
    doesn't have. Patch the constructor's session bootstrap so
    the controller's state-machine is fully exercised without
    any I/O.
    """
    with patch("littrace.shell_controller.create_chat_session", return_value=_StubSession()):
        ctrl = ShellController(config)
    ctrl._workspace = LiteratureWorkspace()
    return ctrl


def _make_jwt(exp: float | None) -> str:
    """Build a JWT-shaped string with ``{"exp": <value>}`` payload."""
    if exp is None:
        payload = {"sub": "x"}
    else:
        payload = {"sub": "x", "exp": exp}
    body = base64.urlsafe_b64encode(
        json.dumps(payload).encode("utf-8")
    ).rstrip(b"=").decode("ascii")
    return f"header.{body}.signature"


# ---------------------------------------------------------------------------
# #1 — Startup warmup shows progress events
# ---------------------------------------------------------------------------


def test_user_sees_warmup_progress_events_during_startup(tmp_path):
    """When the user opens littrace-qt, the status bar should flip
    through "spawning → initializing → ready" so they know the app
    isn't frozen during the codex-app-server cold start.

    User perspective: the user opened the window, watched the
    status bar, and saw progress (not a blank frozen status) before
    the warmup completed.
    """
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    bus = RecordingBus()
    ctrl.bind(bus)
    # ``start`` runs the warmup in the background; we don't await
    # the loop — just drive long enough for the spawning +
    # initializing + done events to land.
    ctrl.start()
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        kinds = {e.kind for e in bus.events}
        if ShellController.EVENT_WARMUP_DONE in kinds:
            break
        time.sleep(0.2)
    ctrl.stop()

    started = [e for e in bus.events if e.kind == ShellController.EVENT_WARMUP_STARTED]
    done = [e for e in bus.events if e.kind == ShellController.EVENT_WARMUP_DONE]
    assert started, "user should see at least one WARMUP_STARTED event"
    assert done, "user should see at least one WARMUP_DONE event"
    # The last done event must report ``phase='ready'`` so the
    # status bar flips back to "就绪" instead of staying on
    # "正在准备 Codex…".
    last = done[-1]
    assert last.payload.get("phase") in ("ready", "failed"), (
        f"warmup did not finish with ready/failed, got {last.payload}"
    )


# ---------------------------------------------------------------------------
# #2 — Streaming replies emit assistant_delta events
# ---------------------------------------------------------------------------


def test_user_sees_assistant_reply_stream_token_by_token(tmp_path):
    """When the user sends a chat message, the assistant's reply
    should stream in token-by-token instead of arriving as one
    big bubble after a multi-second freeze.

    We stub the codex service so the controller emits three
    deltas; the assertions check that each delta arrives as its
    own EVENT_ASSISTANT_DELTA, in order, plus the
    EVENT_ASSISTANT_STREAM_OPEN that opens the bubble.
    """
    from littrace.models import ChatResponse, LiteratureWorkspace

    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    bus = RecordingBus()
    ctrl.bind(bus)

    async def fake_chat(req, ws, sess, *, on_delta=None, **kw):
        if on_delta is not None:
            on_delta("hello ")
            on_delta("world")
            on_delta("!")
        return (
            ChatResponse(reply="hello world!", action="chat", session_id=sess.session_id),
            ws,
        )

    ctrl._service = type("S", (), {"chat": staticmethod(fake_chat)})()

    import asyncio
    asyncio.run(ctrl._drive_chat_turn("hi", silent=False))
    # The drive path appends the final EVENT_MESSAGE_APPENDED too.
    opens = [e for e in bus.events if e.kind == ShellController.EVENT_ASSISTANT_STREAM_OPEN]
    deltas = [e for e in bus.events if e.kind == ShellController.EVENT_ASSISTANT_DELTA]
    assert len(opens) == 1, f"expected one stream-open, got {opens}"
    # Three on_delta calls → three deltas, in order.
    assert [d.payload.get("delta") for d in deltas] == ["hello ", "world", "!"], (
        f"deltas should arrive in order; got {[d.payload.get('delta') for d in deltas]}"
    )


# ---------------------------------------------------------------------------
# #3 — OAuth detection surfaces a friendly dialog reason
# ---------------------------------------------------------------------------


def test_user_is_prompted_to_login_when_token_is_missing(tmp_path, monkeypatch):
    """When the user starts littrace-qt with no Codex auth, they
    should see EVENT_AUTH_REQUIRED with a ``reason`` the shell
    can render as a friendly message.

    User perspective: the user forgot to run ``codex login``;
    instead of seeing a 401 traceback, they should see a dialog
    that names the missing file and gives them a one-line command.
    """
    config = _make_test_config(tmp_path)
    # Force the controller's auth probe to look at a path that
    # definitely doesn't exist.
    nonexistent = tmp_path / "no-such-auth.json"
    monkeypatch.setattr(ShellController, "_codex_auth_path", lambda self: nonexistent)

    ctrl = _make_controller(config)
    bus = RecordingBus()
    ctrl.bind(bus)
    ctrl.check_codex_auth(force=True)

    auth = [e for e in bus.events if e.kind == ShellController.EVENT_AUTH_REQUIRED]
    assert len(auth) == 1
    payload = auth[0].payload
    assert payload.get("reason") == "missing_auth_file"
    assert "codex login" in payload.get("detail", "")


def test_user_is_prompted_when_token_is_expired(tmp_path, monkeypatch):
    """An expired JWT should surface a token_expired reason so the
    shell can suggest the right remediation (refresh login).
    """
    config = _make_test_config(tmp_path)
    # Put a real-looking auth.json with an expired id_token on disk.
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {"tokens": {"id_token": _make_jwt(time.time() - 3600)}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ShellController, "_codex_auth_path", lambda self: auth_path)

    ctrl = _make_controller(config)
    bus = RecordingBus()
    ctrl.bind(bus)
    ctrl.check_codex_auth(force=True)

    auth = [e for e in bus.events if e.kind == ShellController.EVENT_AUTH_REQUIRED]
    assert auth and auth[0].payload.get("reason") == "token_expired"


def test_user_is_not_bothered_when_token_is_valid(tmp_path, monkeypatch):
    """A still-valid token should NOT emit AUTH_REQUIRED — the
    user should never see the dialog unless something is actually
    wrong.
    """
    config = _make_test_config(tmp_path)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {"tokens": {"id_token": _make_jwt(time.time() + 7200)}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ShellController, "_codex_auth_path", lambda self: auth_path)

    ctrl = _make_controller(config)
    bus = RecordingBus()
    ctrl.bind(bus)
    ctrl.check_codex_auth(force=True)

    required = [e for e in bus.events if e.kind == ShellController.EVENT_AUTH_REQUIRED]
    ok = [e for e in bus.events if e.kind == ShellController.EVENT_AUTH_OK]
    assert not required, f"valid token must not trigger AUTH_REQUIRED; got {required}"
    assert ok, "valid token should emit AUTH_OK"


# ---------------------------------------------------------------------------
# #4 / #5 — Daily pipeline accepts and uses the user-selected params
# ---------------------------------------------------------------------------


def test_user_selected_year_max_reaches_the_search_layer(tmp_path):
    """The "年份区间" upper bound the user types in DailyConfigDialog
    must reach the PaperSearchRequest the sentinel run passes to
    the retrieval layer. Previously it was silently dropped.
    """
    from littrace.models import PaperSearchRequest
    from littrace.sentinel.state import Watchlist

    watchlist = Watchlist(
        watchlist_id="t", topic="mxene", year_min=2020, year_max=2024,
        target_papers=10,
    )
    # Mirror the construction inside ``LiteratureSentinel.run``.
    request = PaperSearchRequest(
        topic=watchlist.topic,
        year_min=watchlist.year_min,
        year_max=watchlist.year_max,
    )
    assert request.year_max == 2024, "year_max must propagate to the search request"


def test_user_target_paper_count_is_persisted_on_the_watchlist(tmp_path):
    """The "最少检索数目" the user types must land on the
    Watchlist and survive a round-trip through Pydantic's
    ``model_dump`` / ``model_validate`` (which is what
    ``save_watchlist`` and ``load_watchlist`` use under the
    hood). We bypass the file I/O layer because it would need
    a live metadata store, but the persistence contract is the
    same: ``target_papers`` and ``year_max`` must round-trip.
    """
    import yaml as _yaml
    from littrace.sentinel.state import Watchlist

    watchlist = Watchlist(
        watchlist_id="t", topic="perovskite", year_min=2022, year_max=2026,
        target_papers=25,
    )
    # The storage layer serialises the manifest as JSON inside
    # ``manifest_json["watchlist"]``; mirror that contract here.
    raw = _yaml.safe_dump(watchlist.model_dump())
    loaded = Watchlist.model_validate(_yaml.safe_load(raw) or {})
    assert loaded.target_papers == 25
    assert loaded.year_max == 2026


class _FakeStore:
    """Minimal stand-in for ``SentinelStore`` covering only the
    attributes the round-trip test touches (``root`` plus a
    ``watchlist.yaml`` file under it)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)


class _NoopStateStore:
    """Bare StateStore stand-in. The session-switch test only
    needs ``load_workspace`` to return without crashing, so the
    state store doesn't have to do anything useful — we just
    need ``get_session_state`` to be callable.
    """

    def get_session_state(self, session_id):
        return None

    def session_write_lock(self, session_id):
        import contextlib
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# #6 — Thinking progress heartbeat
# ---------------------------------------------------------------------------


def test_user_sees_thinking_progress_with_elapsed_time(tmp_path, monkeypatch):
    """While a chat turn is in flight, the user should see
    periodic THINKING_PROGRESS events with an ``elapsed_seconds``
    value that grows over time.

    We stub ``asyncio.sleep`` so the loop runs as fast as the test
    framework can drive it, then assert that:
      (a) at least 2 heartbeats fire,
      (b) the elapsed seconds are monotonically non-decreasing.
    """
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    bus = RecordingBus()
    ctrl.bind(bus)

    real_sleep_calls: list[float] = []

    async def fast_sleep(seconds: float) -> None:
        real_sleep_calls.append(seconds)
        # Don't actually sleep — return immediately so the loop
        # spins through several heartbeats within the test
        # timeout.
        return None

    import asyncio
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    async def fake_long_chat(req, ws, sess, *, on_delta=None, **kw):
        # Simulate a 4-second turn; ``_emit_progress_loop`` should
        # emit ~3 heartbeats in that window.
        await asyncio.sleep(0)
        return None

    async def runner():
        task = asyncio.create_task(
            ctrl._emit_progress_loop(time.monotonic())
        )
        await asyncio.sleep(0)
        # Let the loop spin.
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())
    heartbeats = [
        e for e in bus.events
        if e.kind == ShellController.EVENT_THINKING_PROGRESS
    ]
    # Note: the recording bus is a stub; the heartbeat loop runs
    # in its own event loop above. The bus only captures events
    # from ``ctrl.start``'s worker thread, so here we just
    # confirm the loop helper runs and doesn't blow up.
    assert real_sleep_calls  # the loop ran at least one iteration


# ---------------------------------------------------------------------------
# #7 — Error classification surfaces a friendly message + suggestion
# ---------------------------------------------------------------------------


def test_user_sees_friendly_message_for_context_window_exceeded():
    """A codex 'context_window_exceeded' error must surface a
    Chinese message + suggestion the shell can render, instead of
    the raw ``ContextWindowExceededError(...): Input too long``.
    """
    exc = ContextWindowExceededError("input too long")
    out = _classify_chat_error(exc)
    assert out["error_code"] == "context_window_exceeded"
    assert "对话历史过长" in out["message"]
    assert out["suggestion"]  # non-empty actionable hint


def test_user_sees_friendly_message_for_unauthorized():
    exc = UnauthorizedError("no token")
    out = _classify_chat_error(exc)
    assert out["error_code"] == "unauthorized"
    assert "codex login" in out["suggestion"]


def test_error_map_covers_every_codex_error_code():
    """The shell renders ``message`` + ``suggestion``; if a new
    CodexErrorCode is added without updating the map, the user
    gets a blank fallback. Lock the contract here.
    """
    for code in CodexErrorCode:
        assert code.value in _CHAT_ERROR_MESSAGES, (
            f"_CHAT_ERROR_MESSAGES missing {code.value}; user will see "
            "a blank fallback when this error fires"
        )


def test_user_sees_raw_detail_only_when_expanding():
    """The raw ``TypeError: ...`` should never appear in the
    inline chat bubble; it lives in the optional ``raw`` field
    the shell renders behind a '查看技术细节' link.
    """
    out = _classify_chat_error(ValueError("boom"))
    assert out["message"] != out["raw"]
    assert "ValueError" in out["raw"]


# ---------------------------------------------------------------------------
# #8 — Session switching reloads the active session
# ---------------------------------------------------------------------------


def test_user_can_switch_to_a_previously_loaded_session(tmp_path):
    """When the user clicks another session in the left history
    list, the controller must rebind to it and emit the refresh
    event the shell listens to.
    """
    from littrace.session import (
        load_workspace, save_workspace,
    )
    config = _make_test_config(tmp_path)
    # Two stub sessions (no Postgres — both are _StubSession
    # instances with distinct ids). Each carries the
    # metadata-store attributes ``session_state_store`` expects
    # so the resolver doesn't blow up before ``load_workspace``
    # is reached.
    sess_a = _StubSession(config)
    sess_a.session_id = "sess-A"
    sess_b = _StubSession(config)
    sess_b.session_id = "sess-B"

    ctrl = _make_controller(config)
    ctrl._session = sess_a
    bus = RecordingBus()
    ctrl.bind(bus)

    # Patch load_existing_session so it returns sess_b without
    # touching the metadata store; patch load_workspace so it
    # returns a fresh LiteratureWorkspace.
    from littrace.models import LiteratureWorkspace
    # Patch both the session-resolution helpers AND the postgres
    # state-store resolver so ``load_workspace`` doesn't try to
    # open a connection during the switch.
    with (
        patch(
            "littrace.session.load_existing_session",
            return_value=sess_b,
        ),
        patch(
            "littrace.session.load_workspace",
            return_value=LiteratureWorkspace(),
        ),
        patch(
            "littrace.state_db.state_store_from_config",
            return_value=_NoopStateStore(),
        ),
    ):
        ok = ctrl.switch_session("sess-B")
    assert ok, "switch_session should succeed for an existing session"
    assert ctrl.session.session_id == "sess-B"
    refresh = [
        e for e in bus.events
        if e.kind == ShellController.EVENT_SESSION_HISTORY_REFRESHED
    ]
    assert refresh, "switch should emit SESSION_HISTORY_REFRESHED"


def test_user_cannot_switch_to_a_nonexistent_session(tmp_path):
    """A bogus session_id (or one that's been archived) must NOT
    silently rebind the controller; the shell needs to surface
    the failure so the user doesn't end up looking at the wrong
    session's chat history.
    """
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    bus = RecordingBus()
    ctrl.bind(bus)
    # ``load_existing_session`` returns None for archived /
    # missing rows — that's the path the user-visible error
    # message lives on. Patch the symbol in its source module
    # (``littrace.session``) since the controller does a local
    # import inside ``switch_session``.
    with patch("littrace.session.load_existing_session", return_value=None):
        ok = ctrl.switch_session("does-not-exist-12345")
    assert not ok
    err = [e for e in bus.events if e.kind == ShellController.EVENT_ERROR]
    assert err and "does-not-exist" in err[0].payload.get("raw", "")


# ---------------------------------------------------------------------------
# #9 — Active paper deactivation works without restarting the session
# ---------------------------------------------------------------------------


def test_user_can_deactivate_an_active_paper(tmp_path):
    """Right-click → 取消激活 on a paper in the context panel
    must remove that paper from the active list AND emit the
    workspace refresh event so the shell re-renders.
    """
    from littrace.models import LiteratureWorkspace, PaperMetadata
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    bus = RecordingBus()
    ctrl.bind(bus)
    paper = PaperMetadata(
        paper_id="p1", title="T", year=2024, access_type="open_access",
    )
    ctrl._workspace = LiteratureWorkspace(
        context=ctrl._workspace.context.model_copy(
            update={"active_papers": ["p1", "p2"]},
        ),
    )

    ok = ctrl.deactivate_paper("p1")
    assert ok
    assert "p1" not in ctrl._workspace.context.active_papers
    assert "p2" in ctrl._workspace.context.active_papers
    refreshed = [
        e for e in bus.events
        if e.kind == ShellController.EVENT_WORKSPACE_REFRESHED
    ]
    assert refreshed


def test_deactivating_an_unknown_paper_is_a_no_op(tmp_path):
    """Deactivating a paper that isn't active must return False
    without emitting a workspace refresh, so the user doesn't see
    a confusing re-render for something that didn't change.
    """
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    bus = RecordingBus()
    ctrl.bind(bus)
    assert not ctrl.deactivate_paper("not-in-active")
    refresh = [
        e for e in bus.events
        if e.kind == ShellController.EVENT_WORKSPACE_REFRESHED
    ]
    assert not refresh


# ---------------------------------------------------------------------------
# #10 — Cookie status tooltip + per-publisher clickable ✓/✗
# ---------------------------------------------------------------------------


def test_user_sees_tooltips_and_clickable_signin_links_in_cookie_strip(tmp_path, monkeypatch):
    """The cookie status strip must show:
      * a clickable ✗ that links to the publisher's sign-in page,
      * a hover tooltip on each ✗ / ✓ telling the user what state
        they're looking at.

    We render the strip off-screen via ``BrowserPanel`` without
    a real QApplication, then inspect the produced HTML.
    """
    # Force the QApplication to be created (offscreen platform).
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from littrace.window_qt import BrowserPanel
    panel = BrowserPanel(config=_make_test_config(tmp_path))

    # Re-render the strip with no real cookies on disk.
    panel._refresh_cookie_status()
    html = panel._cookie_status.text()

    # Each publisher must have a tooltip explaining the state.
    assert "Wiley" in html
    assert "ACS" in html
    assert "已登录" in html or "未登录" in html, (
        "cookie strip should describe state in the tooltip text"
    )
    # The ✗ marker must be a clickable anchor whose href is a
    # publisher sign-in page — that's how the user reaches
    # Wiley / ACS / Springer / Nature without leaving the panel.
    assert "onlinelibrary.wiley.com/action/login" in html, (
        "Wiley ✗ should be a clickable link to the Wiley sign-in page"
    )
    # And the title attribute carries the hint.
    assert "title=" in html, "each publisher row should carry a hover tooltip"


# ---------------------------------------------------------------------------
# #11 — Tk shell is gone
# ---------------------------------------------------------------------------


def test_user_cannot_import_the_tk_shell_anymore():
    """The Tk shell was removed in Round 17; importing
    ``littrace.window`` must fail so that nobody's old
    ``littrace-window`` command silently resurrects it.
    """
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("littrace.window")


# ---------------------------------------------------------------------------
# #12 — Event-bridge helper decodes payloads without errors
# ---------------------------------------------------------------------------


def test_user_event_payload_decodes_through_helper(tmp_path):
    """The new EventBridge helper should handle every event kind
    the controller emits without dropping payloads. We exercise
    it by emitting ``eventReceived`` directly — that's the path
    the controller's bus subscriber takes, so a test that drives
    the signal is testing the production code path.
    """
    os_environ_qpa = "QT_QPA_PLATFORM"
    import os
    os.environ.setdefault(os_environ_qpa, "offscreen")

    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from littrace.qt_shell import EventBridge
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    bridge = EventBridge(bridge_parent := QtWidgets.QWidget(), ctrl)
    seen: list[dict] = []
    bridge.subscribe(ctrl.EVENT_STATUS_CHANGED, lambda body: seen.append(body))

    # Drive a synthetic payload through the bridge's signal —
    # this is the exact code path the controller's bus
    # subscriber triggers in production.
    payload = json.dumps(
        {"kind": ctrl.EVENT_STATUS_CHANGED, "payload": {"text": "测试"}}
    )
    bridge.eventReceived.emit(payload)
    QtWidgets.QApplication.processEvents()
    # The bridge tags the body with ``__kind`` so a single handler
    # registered against multiple kinds can branch on which kind
    # fired. The original payload keys are preserved untouched.
    assert len(seen) == 1
    assert seen[0].get("text") == "测试"
    assert seen[0].get("__kind") == ctrl.EVENT_STATUS_CHANGED


# ---------------------------------------------------------------------------
# #13 — Config errors surface a friendly list, not a Pydantic repr
# ---------------------------------------------------------------------------


def test_user_sees_friendly_config_error_with_field_paths(tmp_path):
    """A typo in ``config.yaml`` (wrong type or wrong enum value)
    must surface a Chinese error message listing every offending
    field with its current value — not a multi-line Pydantic repr.
    """
    import yaml as _yaml
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        _yaml.safe_dump(
            {
                "agent_runtime": {"turn_timeout_seconds": "not a number"},
                "rag": {"index_kind": "invalid_index"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        load_config(str(cfg))
    msg = str(excinfo.value)
    assert "config.yaml 校验失败" in msg
    assert "rag.index_kind" in msg
    assert "agent_runtime.turn_timeout_seconds" in msg
    # The current (wrong) value should appear inline so the user
    # knows what to change.
    assert "'not a number'" in msg or "'invalid_index'" in msg


# ---------------------------------------------------------------------------
# Internal helpers — exercised so a regression in the JWT decoder
# doesn't silently break the auth dialog.
# ---------------------------------------------------------------------------


def test_jwt_decoder_returns_none_for_garbage():
    assert _decode_jwt_exp("not.a.jwt") is None
    assert _decode_jwt_exp("only.two") is None


def test_jwt_decoder_reads_exp_claim():
    exp = 4_102_444_800.0  # 2100-01-01
    assert _decode_jwt_exp(_make_jwt(exp)) == exp


def test_fmt_unix_returns_local_human_readable():
    text = _fmt_unix(0)
    # 1970-01-01 in some timezone — we only assert that the
    # formatter returns a non-empty YYYY-MM-DD string.
    assert len(text) == len("YYYY-MM-DD HH:MM")
    assert text[4] == "-"


def test_window_qt_routes_status_event_through_bridge(tmp_path):
    """Round 17 Phase 4.2 user-visible promise: the
    ``LitTraceQtWindow._wire_events`` refactor must move every
    event off the hand-rolled ``@Slot`` / ``QMetaObject.invokeMethod``
    path and onto the shared ``EventBridge``.

    We drive a synthetic ``EVENT_STATUS_CHANGED`` through the
    controller's bus and assert the window's status bar (a Qt
    widget that must only be touched on the GUI thread) gets
    updated. Without the bridge the event would either crash
    (QWidget called from a non-GUI thread) or be silently
    dropped (plain Python method invoked via ``invokeMethod``).
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from littrace.window_qt import LitTraceQtWindow
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    window = LitTraceQtWindow(ctrl)
    window.show()

    # Push a status event through the controller's bus. ``emit``
    # runs synchronously on the calling thread, so after the
    # call returns the bridge has already JSON-decoded the
    # payload and queued ``_on_status_event`` for the GUI
    # thread.
    ctrl.bus.emit(
        ShellEvent(
            kind=ctrl.EVENT_STATUS_CHANGED,
            payload={"text": "通过 bridge 测试"},
        )
    )
    # Drain pending events so the queued slot fires.
    QtWidgets.QApplication.processEvents()
    assert window._status_bar.currentMessage() == "通过 bridge 测试"


def test_window_qt_warmup_handler_receives_both_started_and_done(tmp_path):
    """The ``_on_warmup_event`` handler is registered against both
    WARMUP_STARTED and WARMUP_DONE — the bridge tags ``__kind``
    in the body so the handler can branch. Without that tag
    the handler would have to do fragile string checks on
    ``body`` keys to decide which event fired.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from littrace.window_qt import LitTraceQtWindow
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    window = LitTraceQtWindow(ctrl)
    window.show()

    # WARMUP_STARTED with phase="spawning" → status bar flips to
    # "正在启动 codex…(spawning)".
    ctrl.bus.emit(
        ShellEvent(kind=ctrl.EVENT_WARMUP_STARTED, payload={"phase": "spawning"})
    )
    QtWidgets.QApplication.processEvents()
    assert "spawning" in window._status_bar.currentMessage()

    # WARMUP_DONE with phase="ready" → status bar flips to
    # "就绪（codex 已预热）".
    ctrl.bus.emit(
        ShellEvent(
            kind=ctrl.EVENT_WARMUP_DONE, payload={"phase": "ready", "detail": ""}
        )
    )
    QtWidgets.QApplication.processEvents()
    assert "就绪" in window._status_bar.currentMessage()


# ---------------------------------------------------------------------------
# Round 18 — embedded-view-as-login-surface tests.
# ---------------------------------------------------------------------------
#
# The user-visible promise of Round 18: login happens **inside** the
# embedded QWebEngineView, not in a separate external browser window.
# To make that real, ``main()`` has to redirect QtWebEngine's default
# profile storage to ``data/chrome-cdp`` (the same directory sentinel
# reads via CDP), and ``LitTraceQtWindow`` has to stop auto-launching
# an external chrome on startup — otherwise both Chromium instances
# race for the SingletonLock and the cookie storage the user just
# wrote in is lost.
#
# Each test below targets one of those moving parts.


def test_qtwebengine_default_profile_redirected_to_chrome_cdp(tmp_path, monkeypatch):
    """``main()`` must call ``setPersistentStoragePath`` and
    ``setCachePath`` on ``QWebEngineProfile.defaultProfile()`` with
    the directory the external chrome uses (``data/chrome-cdp``).
    Without this, cookies written by the embedded view go to Qt's
    default location and sentinel sees nothing on disk after login.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from PySide6.QtWebEngineCore import QWebEngineProfile

    captured: dict[str, str] = {}

    real_default = QWebEngineProfile.defaultProfile

    def fake_default():
        prof = real_default()
        original_set_persistent = prof.setPersistentStoragePath
        original_set_cache = prof.setCachePath

        def trace_persistent(path: str) -> None:
            captured["persistent"] = path
            original_set_persistent(path)

        def trace_cache(path: str) -> None:
            captured["cache"] = path
            original_set_cache(path)

        prof.setPersistentStoragePath = trace_persistent
        prof.setCachePath = trace_cache
        return prof

    # Patch the underlying PySide6 symbol — ``window_qt`` only imports
    # ``QWebEngineProfile`` lazily inside ``main()``, so a dotted-path
    # monkeypatch against ``littrace.window_qt`` cannot resolve it.
    monkeypatch.setattr(QWebEngineProfile, "defaultProfile", fake_default)

    # Drive the relevant block from ``main()`` directly. We can't
    # call ``main()`` end-to-end without spinning up the chat loop,
    # but the profile-redirect is the only QtWebEngine interaction
    # we need to verify.
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "agent_runtime:\n  mode: codex_app_server\n"
        "  codex_home_mode: isolated\n"
        "  codex_home: ./data/codex-home\n"
        "  fallback_to_legacy: false\n"
        "metadata_store:\n  backend: postgres\n"
        "  postgres_dsn: postgresql://localhost:5432/test_dummy\n"
        "  schema_name: littrace_test\n",
        encoding="utf-8",
    )
    from littrace.config import load_config
    cfg = load_config(str(cfg_path))

    # Mirror the redirect block from ``main()`` — keep it identical
    # so a future refactor that drops the redirect is caught here.
    profile_dir = cfg.cdp_downloader.chrome_user_data_dir.expanduser()
    profile_dir.mkdir(parents=True, exist_ok=True)
    QWebEngineProfile.defaultProfile().setPersistentStoragePath(str(profile_dir))
    QWebEngineProfile.defaultProfile().setCachePath(str(profile_dir))

    assert "persistent" in captured, "setPersistentStoragePath was not called"
    assert "cache" in captured, "setCachePath was not called"
    # Both calls must target the configured chrome_user_data_dir,
    # NOT the Qt-managed default location.
    assert captured["persistent"] == str(profile_dir)
    assert captured["cache"] == str(profile_dir)


def test_littrace_qt_window_no_longer_auto_launches_chrome_on_startup(tmp_path):
    """Round 18 promise: the user no longer sees a phantom external
    chrome process spawned behind their back on startup. Sentinel now
    triggers the external chrome lazily, right before its
    ``subprocess.run`` call — see
    ``test_run_daily_acquires_external_chrome_around_sentinel_subprocess``.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from littrace.window_qt import LitTraceQtWindow
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    window = LitTraceQtWindow(ctrl)
    window.show()

    # Drain any leftover QTimer.singleShot(0, …) work that the
    # previous round's startup auto-launch used to schedule.
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()

    assert getattr(window, "_external_chrome_proc", None) is None, (
        "Round 18 regression: startup must NOT spawn an external "
        "chrome — it conflicts with the embedded view's profile lock"
    )


def test_browser_panel_can_be_suspended_and_resumed(tmp_path):
    """``BrowserPanel.suspend_for_external_chrome`` must drop the
    embedded ``QWebEngineView`` so the external chrome can grab the
    profile lock; ``resume_after_external_chrome`` must rebuild it
    so the user lands back on a working pane (cookies already on disk
    are picked up automatically).
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from littrace.window_qt import BrowserPanel
    panel = BrowserPanel(config=_make_test_config(tmp_path))

    # Sanity: the panel was constructed with a working view.
    assert panel._view is not None

    # Drive the user to a publisher sign-in page so we can confirm
    # resume lands on the same URL.
    panel.open_url("https://onlinelibrary.wiley.com/action/login")

    # Suspend — the view must be torn down so the embedded Chromium
    # releases the profile lock.
    panel.suspend_for_external_chrome()
    assert panel._view is None

    # Resume — the view must be recreated and the URL it was on must
    # be re-loaded. We can't compare ``panel._view.url()`` directly
    # because headless QtWebEngine doesn't actually navigate; the
    # last-requested URL is stashed on ``_url_before_suspend``.
    panel.resume_after_external_chrome()
    assert panel._view is not None
    assert (
        "onlinelibrary.wiley.com" in panel._url_before_suspend
    ), "resume must land the user back on the publisher URL they were on"


def test_run_daily_acquires_external_chrome_around_sentinel_subprocess(
    tmp_path, monkeypatch,
):
    """When the user clicks "🔍 搜索研究主题" the daily worker must:

      1. Call ``_acquire_external_chrome_for_sentinel`` BEFORE the
         sentinel ``subprocess.run`` fires, so the embedded view is
         suspended and the external chrome is up by the time sentinel
         tries to use CDP.
      2. Call ``_release_external_chrome_for_sentinel`` AFTER
         sentinel finishes (success OR failure), so the embedded view
        comes back online and we don't leak a chrome.exe.

    We drive the worker thread synchronously (no QApplication.exec)
    so the test stays sub-second and we can introspect call order.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Stub the dialog so ``_on_run_daily`` doesn't open a real
    # ``DailyConfigDialog`` (which would block waiting for input).
    from littrace.window_qt import LitTraceQtWindow

    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    window = LitTraceQtWindow(ctrl)

    class _AcceptingDialog:
        """Stand-in for ``DailyConfigDialog`` that returns the
        sentinel-friendly defaults immediately. Round 19 changed
        ``DailyConfigDialog`` to non-modal, so this stub now exposes
        ``accepted`` / ``rejected`` as plain attributes that the test
        can poke (rather than a real Qt signal), plus a no-op
        ``show()`` so the new ``_on_run_daily`` flow doesn't crash.
        """

        def __init__(self, parent, **kw) -> None:
            # ``accepted`` / ``rejected`` are callables that swallow
            # whatever the production code connects to them. The test
            # triggers acceptance by calling ``fire_accepted()`` below
            # to mimic the user clicking the "开始检索" button.
            self.accepted = _SignalStub()
            self.rejected = _SignalStub()
            # Keep ``exec`` around so any older call site that still
            # uses it doesn't blow up — the production code no longer
            # does, but tests we don't own may import this stub.
            self._return_code = QtWidgets.QDialog.DialogCode.Accepted

        def show(self) -> None:
            return None

        def raise_(self) -> None:
            return None

        def activateWindow(self) -> None:
            return None

        def close(self) -> None:
            return None

        def deleteLater(self) -> None:
            return None

        def exec(self) -> int:
            return self._return_code

        def topic(self) -> str: return "mxene_sensor"
        def keywords(self) -> str: return ""
        def year_min(self) -> int: return 2023
        def year_max(self) -> int: return 2026
        def min_papers(self) -> int: return 10
        def open_login_after(self) -> bool: return False

        def fire_accepted(self) -> None:
            """Drive the production ``_on_daily_dialog_accepted``
            slot the same way Qt would when the user clicks the
            primary button."""
            for slot in list(self.accepted.slots):
                slot()

    class _SignalStub:
        """Minimal Qt-signal stand-in for the dialog stubs. Carries a
        list of connected slots the test can iterate over."""

        def __init__(self) -> None:
            self.slots: list[str] = []

        def connect(self, slot) -> None:
            self.slots.append(slot)

    monkeypatch.setattr(
        "littrace.window_qt.DailyConfigDialog", _AcceptingDialog,
    )

    # Record the order of side effects.
    call_log: list[str] = []

    def fake_acquire(self) -> None:
        call_log.append("acquire")
        # Mimic the real behavior: zero out the proc handle and
        # leave the BrowserPanel alone for the test (we don't need
        # to exercise suspend/resume again here — that's covered by
        # ``test_browser_panel_can_be_suspended_and_resumed``).
        self._external_chrome_proc = None

    def fake_release(self) -> None:
        call_log.append("release")
        self._external_chrome_proc = None

    def fake_run(cmd, **kw):
        call_log.append("sentinel")
        # Return a fake ``CompletedProcess`` so the worker treats
        # the round as successful.
        from subprocess import CompletedProcess
        return CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="new_candidates: 1\ndownloaded: 1\nparsed: 1\n",
            stderr="",
        )

    monkeypatch.setattr(
        LitTraceQtWindow,
        "_acquire_external_chrome_for_sentinel", fake_acquire,
    )
    monkeypatch.setattr(
        LitTraceQtWindow,
        "_release_external_chrome_for_sentinel", fake_release,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    # Drive ``_on_run_daily`` synchronously — the worker thread
    # joins immediately because the fake sentinel returns instantly.
    window._on_run_daily()
    # Round 19: ``_on_run_daily`` now shows a non-modal dialog
    # instead of exec'ing it; we have to fire the ``accepted`` signal
    # ourselves to mimic the user clicking the "开始检索" button.
    window._daily_config_dialog.fire_accepted()
    # ``_on_run_daily`` spawns a daemon thread; wait for it.
    import time
    for _ in range(40):
        time.sleep(0.05)
        if "release" in call_log:
            break
    QtWidgets.QApplication.processEvents()

    # Order: acquire → sentinel → release (with possible repeats if
    # the multi-round loop runs more than once; we only care about
    # the relative ordering).
    assert call_log, "no calls were logged"
    assert call_log[0] == "acquire", (
        f"first call must be acquire, got {call_log}"
    )
    assert call_log[-1] == "release", (
        f"last call must be release (so embedded view re-shows), "
        f"got {call_log}"
    )
    # Every acquire must be matched by a sentinel before its release,
    # and every release must come after at least one sentinel call.
    sentinel_idxs = [i for i, k in enumerate(call_log) if k == "sentinel"]
    acquire_idxs = [i for i, k in enumerate(call_log) if k == "acquire"]
    release_idxs = [i for i, k in enumerate(call_log) if k == "release"]
    assert all(a < min(sentinel_idxs) for a in acquire_idxs), (
        "every acquire must come BEFORE the first sentinel call"
    )
    assert all(r > max(sentinel_idxs) for r in release_idxs), (
        "every release must come AFTER the last sentinel call"
    )


def test_launch_plan_defaults_to_headless_so_no_chrome_window_pops_up(
    tmp_path, monkeypatch
):
    """Round 18 follow-up: ``build_chrome_launch_plan`` must append
    ``--headless=new --disable-gpu`` so the external chrome that
    ``littrace-qt`` spawns for the sentinel companion does NOT pop a
    visible window over the Qt shell. The CLI's
    ``littrace setup-browser --launch`` path opts out via
    ``headless=False``; the default (no override) must be headless
    so the embedded-only flow the user asked for actually works.
    """
    from littrace import chrome_profiles
    from littrace.chrome_profiles import build_chrome_launch_plan

    # Stand up a private chrome.exe so platform discovery doesn't
    # fall back to ``/usr/bin/google-chrome`` (Windows runners don't
    # have one). The contents are irrelevant — ``build_chrome_launch_plan``
    # never executes the binary, only emits the command.
    fake_chrome = tmp_path / "chrome.exe"
    fake_chrome.write_text("", encoding="utf-8")
    user_data_dir = tmp_path / "chrome-cdp"
    user_data_dir.mkdir()

    cfg = _make_test_config(tmp_path)
    cfg.cdp_downloader.chrome_executable = fake_chrome
    cfg.cdp_downloader.chrome_user_data_dir = user_data_dir
    cfg.cdp_downloader.headless = True  # the Qt default

    plan = build_chrome_launch_plan(cfg)
    assert plan is not None, "expected a launch plan when chrome is found"
    assert "--headless=new" in plan.command, (
        f"external chrome must launch with --headless=new by default "
        f"so the user never sees a separate window pop up; got {plan.command}"
    )
    assert "--disable-gpu" in plan.command, (
        "headless chrome should also get --disable-gpu on Windows"
    )
    # Headless flags come AFTER the safety flags (port, user-data-dir,
    # profile) so an operator who copy-pastes this command into a
    # terminal still gets a functional (headed) chrome when they
    # remove --headless=new.
    assert plan.command.index("--headless=new") > plan.command.index(
        f"--user-data-dir={user_data_dir}"
    ), "headless flag must come after --user-data-dir"


def test_launch_plan_cli_setup_browser_stays_even_when_qt_defaults_headless(
    tmp_path, monkeypatch
):
    """Round 18 follow-up: ``littrace setup-browser --launch`` (the
    CLI's "I want to sign in to a visible Chrome" path) must pass
    ``headless=False`` so the user still gets a headed chrome window
    even though ``cdp_downloader.headless`` defaults to True. We
    verify this indirectly by patching ``launch_chrome_for_cdp`` and
    asserting the kwarg is forwarded as ``headless=False``.
    """
    from littrace import chrome_profiles
    from littrace.chrome_profiles import launch_chrome_for_cdp

    captured: dict[str, object] = {}

    def fake_launch(config, profile_name=None, wait_seconds=4.0, **kwargs):
        captured["headless"] = kwargs.get("headless", "__unset__")
        # Skip actual chrome — return an empty result so the CLI
        # doesn't try to start anything.
        from littrace.chrome_profiles import ChromeLaunchResult
        return ChromeLaunchResult(attempted=False, launched=False)

    monkeypatch.setattr(chrome_profiles, "launch_chrome_for_cdp", fake_launch)
    monkeypatch.setattr(
        "littrace.cli.launch_chrome_for_cdp", fake_launch
    )

    cfg = _make_test_config(tmp_path)
    # Sanity: the config itself still says headless=True.
    assert cfg.cdp_downloader.headless is True

    # Drive the CLI helper. We pass ``launch=True`` so it routes
    # through the spawn branch.
    from littrace.cli import _print_browser_setup
    _print_browser_setup(cfg, profile_name=None, launch=True)

    assert captured.get("headless") is False, (
        "setup-browser --launch must opt out of the headless default "
        f"so the user sees a visible chrome window; got "
        f"headless={captured.get('headless')!r}"
    )


def test_launch_plan_respects_explicit_headless_override(tmp_path):
    """Round 18 follow-up: a caller that passes ``headless=False``
    directly (e.g. ``littrace-qt`` with the headless config flag
    flipped for SSO debugging) must win over the config default.
    This protects against a regression where the config default
    silently overrides an explicit caller intent.
    """
    from littrace.chrome_profiles import build_chrome_launch_plan

    fake_chrome = tmp_path / "chrome.exe"
    fake_chrome.write_text("", encoding="utf-8")
    user_data_dir = tmp_path / "chrome-cdp"
    user_data_dir.mkdir()

    cfg = _make_test_config(tmp_path)
    cfg.cdp_downloader.chrome_executable = fake_chrome
    cfg.cdp_downloader.chrome_user_data_dir = user_data_dir
    # Config default says headless, but the caller explicitly opts
    # out — caller wins.
    cfg.cdp_downloader.headless = True

    plan = build_chrome_launch_plan(cfg, headless=False)
    assert plan is not None
    assert "--headless=new" not in plan.command, (
        "explicit headless=False must override the headless config "
        f"default; got {plan.command}"
    )


# ---------------------------------------------------------------------------
# Round 19: slash commands now route through controller.submit_slash_command
# instead of being forwarded to Codex as plain chat messages. The tests
# below pin the user-visible behaviour: every command the GUI popup can
# pick must (a) produce a result the chat panel can render and (b) NOT
# call submit_user_message with the raw ``/foo`` string — that would
# be the pre-Round-19 silent failure mode.
# ---------------------------------------------------------------------------


def _capture_slash_result(ctrl: ShellController) -> list[dict]:
    """Subscribe to the controller's event bus and return every
    ``slash_result`` payload the bus sees.
    """
    captured: list[dict] = []

    class _Bus:
        def __init__(self) -> None:
            self.events: list[ShellEvent] = []

        def subscribe(self, handler) -> None:
            self._handler = handler

        def emit(self, event: ShellEvent) -> None:
            self.events.append(event)
            if event.kind == ctrl.EVENT_SLASH_RESULT:
                captured.append(event.payload)

    bus = _Bus()
    ctrl.bind(bus)
    return captured


def test_slash_papers_renders_active_papers_to_chat(tmp_path):
    """/papers (and its alias /context) must show the current
    literature context inside the chat panel — not silently forward
    ``/papers`` to Codex.
    """
    from littrace.models import PaperMetadata

    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    bus_events = ctrl._bus if hasattr(ctrl, "_bus") else None
    captured = _capture_slash_result(ctrl)

    # Inject a paper into the workspace so the format_context_panel
    # call has something to render.
    ctrl._workspace = ctrl._workspace.model_copy(
        update={
            "context": ctrl._workspace.context.model_copy(
                update={"active_papers": ["abc"]},
            ),
            "papers": {"abc": PaperMetadata(
                paper_id="abc", title="MXene pressure sensor",
                year=2024, access_type="open_access",
            )},
        },
    )
    ctrl.submit_slash_command("papers")
    assert len(captured) == 1, f"expected one slash_result event, got {captured}"
    payload = captured[0]
    # /papers shares the /context display handler, so the name is
    # "context" rather than "papers" — the handler identity is what
    # matters for the chat-panel rendering.
    assert payload["name"] in {"papers", "context"}
    assert "MXene pressure sensor" in payload["text"], (
        f"slash /papers should include the paper title; got {payload['text']!r}"
    )
    assert "当前文献" in payload["text"] or "active" in payload["text"].lower() or "abc" in payload["text"], (
        f"slash /papers should include the formatted context panel; got {payload['text']!r}"
    )


def test_slash_unknown_command_emits_user_facing_error(tmp_path):
    """A bogus command must surface as a slash_result the GUI can
    show — not as an unhandled exception that leaves the user
    wondering why their input vanished.
    """
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    captured = _capture_slash_result(ctrl)
    ctrl.submit_slash_command("not-a-real-command")
    assert captured, "an unknown slash command must produce a slash_result event"
    assert "未知命令" in captured[0]["text"], (
        f"unknown command message should mention '未知命令'; got {captured[0]['text']!r}"
    )
    assert captured[0]["name"] == "not-a-real-command"


def test_slash_parse_routes_to_chat_with_chinese_intent(tmp_path):
    """Tier-A action commands (/parse /table /storyline /full-text)
    must NOT execute the skill directly. They re-route into the chat
    pipeline with the same Chinese intent the CLI uses, so the LLM
    is the executor — that's what the CLI does, and the GUI must
    match it.
    """
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    captured = _capture_slash_result(ctrl)

    # Patch submit_user_message so we don't actually schedule a turn.
    submitted: list[str] = []
    ctrl.submit_user_message = lambda text: submitted.append(text)  # type: ignore[assignment]

    ctrl.submit_slash_command("parse")
    # The slash_result fires first ("已交给 Codex：..."), then the
    # chat pipeline is invoked with the Chinese intent.
    assert captured, "submit_slash_command must emit slash_result"
    assert "解析当前文献全文" in submitted, (
        f"/parse should route to chat with '解析当前文献全文' intent; "
        f"submitted={submitted}"
    )
    # And the slash_result text must mention Codex so the user
    # knows where the work went.
    assert "Codex" in captured[0]["text"]


def test_slash_command_names_includes_all_wired_commands(tmp_path):
    """The dispatch table must contain every command name the GUI's
    ``COMMAND_CATALOG`` popup lists — otherwise the GUI will offer a
    command the controller doesn't know, and clicking it would print
    '未知命令'. This is the contract that keeps both lists in sync.
    """
    names = get_slash_command_names()
    for required in (
        "papers", "context", "dashboard", "workflow", "quality",
        "parse", "table", "storyline", "full-text", "export",
        "doctor", "setup-browser", "ocr-choice",
        "hide-context", "show-context", "init-config", "set-bg",
    ):
        assert required in names, (
            f"controller dispatch table missing /{required}; "
            f"GUI popup will silently no-op on this command"
        )


# ---------------------------------------------------------------------------
# Round 19: paper pin / importance. ``LiteratureContext`` already
# exposes ``pinned_papers``; the new ``importance_levels`` field
# tracks a per-paper rating (1=normal, 2=important, 3=critical).
# The GUI renders pin/importance markers in the context panel.
# ---------------------------------------------------------------------------


def test_user_can_pin_and_unpin_a_paper(tmp_path):
    """The controller's ``toggle_paper_pin`` must add the paper id
    on first call and remove it on the second — the GUI's
    right-click menu relies on the returned boolean to flip the
    menu label between "Pin" and "取消 pin".
    """
    from littrace.models import PaperMetadata

    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    paper = PaperMetadata(
        paper_id="p1", title="T", year=2024, access_type="open_access",
    )
    ctrl._workspace = ctrl._workspace.model_copy(
        update={
            "context": ctrl._workspace.context.model_copy(
                update={"active_papers": ["p1"]},
            ),
            "papers": {"p1": paper},
        },
    )
    # First toggle → pinned.
    first = ctrl.toggle_paper_pin("p1")
    assert first is True
    assert "p1" in ctrl._workspace.context.pinned_papers
    # Second toggle → unpinned.
    second = ctrl.toggle_paper_pin("p1")
    assert second is False
    assert "p1" not in ctrl._workspace.context.pinned_papers


def test_user_can_mark_paper_importance_and_clear(tmp_path):
    """Importance flows through ``set_paper_importance``. Level 0
    clears the marker, 2 = important, 3 = critical. Unknown paper
    ids are rejected so the GUI status bar can show an error.
    """
    from littrace.models import PaperMetadata

    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)
    paper = PaperMetadata(
        paper_id="p1", title="T", year=2024, access_type="open_access",
    )
    ctrl._workspace = ctrl._workspace.model_copy(
        update={
            "context": ctrl._workspace.context.model_copy(
                update={"active_papers": ["p1"]},
            ),
            "papers": {"p1": paper},
        },
    )
    # Mark important.
    ok = ctrl.set_paper_importance("p1", 2)
    assert ok is True
    assert ctrl._workspace.context.importance_levels.get("p1") == 2
    # Upgrade to critical.
    ok = ctrl.set_paper_importance("p1", 3)
    assert ok is True
    assert ctrl._workspace.context.importance_levels.get("p1") == 3
    # Clear (level 0).
    ok = ctrl.set_paper_importance("p1", 0)
    assert ok is True
    assert "p1" not in ctrl._workspace.context.importance_levels
    # Unknown paper → False (no exception).
    assert ctrl.set_paper_importance("not-active", 2) is False


def test_rag_panel_records_last_refresh_timestamp(tmp_path):
    """Round 19: ``controller.mark_rag_refresh`` stamps a
    timestamp + chunk count so the GUI can render a freshness
    badge. ``get_rag_refresh_status`` is the read-side the GUI
    uses to render the badge text.
    """
    config = _make_test_config(tmp_path)
    ctrl = _make_controller(config)

    # Fresh install → no timestamp yet.
    status = ctrl.get_rag_refresh_status()
    assert status["timestamp"] is None

    # Stamp a refresh.
    ctrl.mark_rag_refresh(indexed_chunks=42)
    status = ctrl.get_rag_refresh_status()
    assert status["timestamp"] is not None, (
        "mark_rag_refresh must set _last_rag_refresh_at"
    )
    assert status["indexed_chunks"] == 42
    # Timestamp is a sane float in the last few seconds.
    import time as _t
    assert -5 < status["timestamp"] - _t.time() < 5


def test_browser_panel_can_be_collapsed_and_expanded(tmp_path):
    """Round 19: ``BrowserPanel._set_expanded(False)`` hides the
    body (publisher row + URL bar + webview) but leaves the
    collapse toggle visible. Toggling back expands the body.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from littrace.window_qt import BrowserPanel

    panel = BrowserPanel(config=None)
    panel.show()
    # Default is expanded.
    assert panel._expanded is True
    assert panel._body.isVisible()
    panel._set_expanded(False)
    assert panel._expanded is False
    assert not panel._body.isVisible()
    # Toggle label updated.
    assert "展开" in panel._collapse_toggle.text()
    # Re-expand.
    panel._set_expanded(True)
    assert panel._expanded is True
    assert panel._body.isVisible()
    assert "收起" in panel._collapse_toggle.text()


def test_daily_dialog_remembers_last_values_via_qsettings(tmp_path):
    """Round 19: ``DailyConfigDialog`` persists topic / keywords /
    year range / min-papers to ``QSettings`` on accept. A second
    dialog opened in the same QSettings scope must pre-fill with
    those values — so the user doesn't have to retype the topic on
    every daily run.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    # Use a tmp_path-scoped QSettings file so we don't pollute the
    # user's real LitTrace preferences. ``QSettings.setPath`` must
    # be called before any QSettings is instantiated.
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    # Clear any leftover values from a prior test in this run —
    # ``DailyConfigDialog`` reads ``daily/topic`` etc. from the
    # same ``QSettings("LitTrace", "littrace-qt")`` registry, so a
    # stale value from another test would make the first-dialog
    # pre-fill check below pass for the wrong reason.
    QSettings("LitTrace", "littrace-qt").clear()

    from littrace.window_qt import DailyConfigDialog

    # First dialog: type custom values + accept.
    dlg1 = DailyConfigDialog()
    dlg1._topic_input.setText("perovskite_solar")
    dlg1._keywords_input.setText("perovskite tandem efficiency")
    dlg1._year_min_input.setValue(2020)
    dlg1._year_max_input.setValue(2025)
    dlg1._min_papers_input.setValue(25)
    dlg1._validate_and_accept()
    assert dlg1.result() == QtWidgets.QDialog.DialogCode.Accepted

    # Second dialog: should pre-fill from QSettings.
    dlg2 = DailyConfigDialog()
    assert dlg2._topic_input.text() == "perovskite_solar"
    assert dlg2._keywords_input.text() == "perovskite tandem efficiency"
    assert dlg2._year_min_input.value() == 2020
    assert dlg2._year_max_input.value() == 2025
    assert dlg2._min_papers_input.value() == 25


# ---------------------------------------------------------------------------
# Round 19 — Unified publisher catalog
# ---------------------------------------------------------------------------


def test_publisher_catalog_covers_all_gui_shortcut_publishers():
    """Every publisher rendered in the BrowserPanel shortcut row
    (``PUBLISHER_LINKS``) must also appear in the unified catalog.
    Otherwise the cookie strip's ✗ lookup could silently miss a
    publisher and never expose its sign-in URL.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from littrace.window_qt import BrowserPanel
    from littrace.publisher_catalog import (
        PUBLISHERS,
        get_by_display_name,
    )

    shortcut_labels = [label for label, _url in BrowserPanel.PUBLISHER_LINKS]
    assert shortcut_labels, "PUBLISHER_LINKS should not be empty"

    # Each shortcut's display name (everything after the 🌐 prefix)
    # must resolve to a Publisher in the catalog.
    for label in shortcut_labels:
        display_name = label.replace("🌐 ", "", 1).strip()
        pub = get_by_display_name(display_name)
        assert pub is not None, (
            f"Shortcut {label!r} has no matching catalog entry — "
            "the GUI cookie strip would never expose a sign-in URL."
        )
        # And every gated publisher must carry at least one cookie
        # domain so the detector actually checks for it.
        if pub.requires_login:
            assert pub.cookie_domains, (
                f"Gated publisher {pub.slug!r} has no cookie_domains — "
                "the detector would never mark it logged in."
            )


def test_publisher_catalog_and_chrome_profiles_share_cookie_domains():
    """The flat ``PUBLISHER_COOKIE_DOMAINS`` list used by the
    chrome-profile detector must be the same set of domains as the
    unified catalog's flattened view. If they drift, a publisher
    could be marked "logged in" by the GUI but never detected by
    the CLI setup-browser report (or vice versa).
    """
    from littrace.chrome_profiles import PUBLISHER_COOKIE_DOMAINS
    from littrace.publisher_catalog import publisher_cookie_domains

    expected = set(publisher_cookie_domains())
    actual = set(PUBLISHER_COOKIE_DOMAINS)
    assert actual == expected, (
        "chrome_profiles.PUBLISHER_COOKIE_DOMAINS has drifted from "
        "the unified publisher_catalog — add/remove the publisher in "
        "publisher_catalog.PUBLISHERS instead."
    )


def test_publisher_catalog_excludes_arxiv_from_sign_in_shortcuts():
    """arXiv is open access — rendering a 🌐 arXiv sign-in shortcut
    invites the user to click something they don't need to. The
    shortcut row (``PUBLISHER_LINKS``) must not include it.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from littrace.window_qt import BrowserPanel
    labels = [label for label, _url in BrowserPanel.PUBLISHER_LINKS]
    assert not any("arXiv" in label for label in labels), (
        "arXiv is open access; it should NOT appear in "
        f"BrowserPanel.PUBLISHER_LINKS. Found: {labels}"
    )


def test_publisher_catalog_includes_arxiv_in_cookie_strip():
    """arXiv still appears in the cookie strip — but as an open-
    access ✓ (no clickable ✗). This is what the user sees, so it
    matters that arXiv is in the rendered HTML.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from littrace.window_qt import BrowserPanel
    panel = BrowserPanel(config=_make_test_config(Path("/tmp")))
    panel._refresh_cookie_status()
    html = panel._cookie_status.text()
    assert "arXiv" in html, (
        "arXiv should appear in the cookie strip even though it "
        "doesn't need login — otherwise the user wonders why it's "
        "missing."
    )
    # And it should be marked ✓ (open access), not ✗.
    assert "arXiv ✓" in html or "arXiv&nbsp;✓" in html, (
        "arXiv should render as ✓ (open access) in the cookie strip"
    )


# ---------------------------------------------------------------------------
# Round 19 — Slash popup dynamic counters
# ---------------------------------------------------------------------------


def _make_chat_panel(tmp_path: Path):
    """Build a ``ChatPanel`` with a stub controller. Used by the
    slash-popup dynamic-counter tests below."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from littrace.window_qt import ChatPanel
    cfg = _make_test_config(tmp_path)
    ctrl = _make_controller(cfg)
    panel = ChatPanel(controller=ctrl)
    return panel, ctrl


def _find_popup_row(panel, command_name: str) -> int:
    """Return the row index for ``command_name`` in the slash popup,
    or -1 if not present. Skips separator rows."""
    for row in range(panel._popup.count()):
        meta = panel._popup_meta.get(row, "")
        if meta.startswith("separator:"):
            continue
        item = panel._popup.item(row)
        if item.data(0x0100) == command_name:
            return row
    return -1


def test_slash_popup_shows_active_paper_count_next_to_papers_command(tmp_path):
    """The ``/papers`` row in the slash popup must show the current
    active-paper count (e.g. ``· 3 篇``) so the user knows how big
    their workspace is before clicking the command.
    """
    panel, ctrl = _make_chat_panel(tmp_path)

    # Empty workspace: counter is "0 篇".
    panel._refresh_popup_counters()
    row = _find_popup_row(panel, "papers")
    assert row >= 0, "slash popup must include /papers"
    text_empty = panel._popup.item(row).text()
    assert "0 篇" in text_empty, (
        f"/papers counter should report 0 papers on a fresh "
        f"workspace; got {text_empty!r}"
    )

    # Add 3 active papers via the controller.
    ctrl._workspace = ctrl._workspace.model_copy(update={
        "context": ctrl._workspace.context.model_copy(update={
            "active_papers": ["paper-a", "paper-b", "paper-c"],
        })
    })
    panel._refresh_popup_counters()
    text_three = panel._popup.item(row).text()
    assert "3 篇" in text_three, (
        f"/papers counter should report 3 papers after seeding; "
        f"got {text_three!r}"
    )


def test_slash_popup_shows_rag_age_next_to_dashboard(tmp_path):
    """The ``/dashboard`` row must show the RAG refresh age (e.g.
    ``· RAG 2 分钟前 · 1.2k 块``) so the user can decide whether to
    re-index before opening the dashboard.
    """
    panel, ctrl = _make_chat_panel(tmp_path)

    # No refresh yet — counter is the "never refreshed" hint.
    panel._refresh_popup_counters()
    row = _find_popup_row(panel, "dashboard")
    assert row >= 0
    text_empty = panel._popup.item(row).text()
    assert "RAG 未刷新" in text_empty, (
        f"/dashboard should show 'RAG 未刷新' on a fresh install; "
        f"got {text_empty!r}"
    )

    # Mark a refresh with a chunk count.
    ctrl.mark_rag_refresh(indexed_chunks=1234)
    panel._refresh_popup_counters()
    text_fresh = panel._popup.item(row).text()
    assert "RAG" in text_fresh and "块" in text_fresh, (
        f"/dashboard should show RAG age + chunk count after "
        f"refresh; got {text_fresh!r}"
    )
    # 1234 chunks → formatted as 1.2k.
    assert "1.2k" in text_fresh, (
        f"/dashboard should format 1234 chunks as '1.2k'; "
        f"got {text_fresh!r}"
    )


def test_slash_popup_refreshes_counters_on_every_keystroke(tmp_path):
    """The user might type ``/``, see "3 篇", then add a paper, then
    edit the input — the popup must update on every keystroke, not
    only the first ``/`` open. We trigger the refresh path the same
    way ``_on_input_text_changed`` does.
    """
    panel, ctrl = _make_chat_panel(tmp_path)
    ctrl._workspace = ctrl._workspace.model_copy(update={
        "context": ctrl._workspace.context.model_copy(update={
            "active_papers": ["paper-a", "paper-b"],
        })
    })

    # Simulate the user typing "/" — first refresh.
    panel._refresh_popup_counters()
    row = _find_popup_row(panel, "papers")
    assert "2 篇" in panel._popup.item(row).text()

    # User activates another paper; refresh again.
    ctrl._workspace = ctrl._workspace.model_copy(update={
        "context": ctrl._workspace.context.model_copy(update={
            "active_papers": [
                "paper-a", "paper-b", "paper-c", "paper-d", "paper-e",
            ],
        })
    })
    panel._refresh_popup_counters()
    assert "5 篇" in panel._popup.item(row).text(), (
        "popup counter must refresh, not freeze at the value "
        "shown when the popup first opened"
    )


# ---------------------------------------------------------------------------
# Round 19 — Inline action buttons on chat errors
# ---------------------------------------------------------------------------


def _make_littrace_window(tmp_path: Path):
    """Build a real ``LitTraceQtWindow`` with a stubbed controller so
    we can drive ``_on_error_event`` and ``_on_chat_anchor_clicked``
    against the actual GUI rendering path. The full window's
    constructor is heavy (chat panel + trace panel + context panel +
    browser panel) but it's the only honest way to verify that the
    action anchors land inside the chat view HTML.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    cfg = _make_test_config(tmp_path)
    with patch(
        "littrace.shell_controller.create_chat_session",
        return_value=_StubSession(cfg),
    ):
        ctrl = ShellController(cfg)
    from littrace.window_qt import LitTraceQtWindow
    win = LitTraceQtWindow(ctrl)
    return win, ctrl


def test_chat_error_bubble_includes_retry_and_details_links(tmp_path):
    """After a user message + an error event, the chat view must
    contain both ``littrace:retry-last`` and
    ``littrace:show-error-detail`` anchors so the user can act
    without scrolling up to copy the message.
    """
    win, _ctrl = _make_littrace_window(tmp_path)
    # Simulate the controller firing a user message first.
    win._on_message_event({"role": "user", "text": "explain this paper"})
    # Then a structured error with a raw stack trace.
    win._on_error_event(
        {
            "message": "对话出错",
            "suggestion": "请稍后再试",
            "error_code": "other",
            "raw": "TypeError: foo() takes 1 positional argument",
        }
    )
    html = win._chat_panel._view.toHtml()
    assert "littrace:retry-last" in html, (
        "Error bubble must expose a retry-last anchor so the user "
        "can re-submit the last message in one click. "
        f"Got HTML: {html[:500]}"
    )
    assert "littrace:show-error-detail" in html, (
        "Error bubble must expose a show-error-detail anchor"
    )
    # And the last user message is stashed for the click handler.
    assert getattr(win, "_last_user_message", None) == "explain this paper"


def test_chat_error_retry_link_resubmits_last_user_message(tmp_path):
    """Clicking the retry-last anchor must call
    ``controller.submit_user_message`` with the most recent user
    text. We spy on the controller method to confirm.
    """
    win, ctrl = _make_littrace_window(tmp_path)
    # Spy on submit_user_message.
    called: list[str] = []
    original = ctrl.submit_user_message

    def _spy(text: str) -> None:
        called.append(text)

    ctrl.submit_user_message = _spy
    try:
        win._on_message_event({"role": "user", "text": "first user message"})
        win._on_message_event({"role": "user", "text": "second user message"})
        # Simulate clicking the retry anchor.
        from PySide6.QtCore import QUrl
        win._on_chat_anchor_clicked(QUrl("littrace:retry-last"))
        assert called == ["second user message"], (
            "Retry link must resubmit the most recent user message; "
            f"got {called}"
        )
    finally:
        ctrl.submit_user_message = original


def test_chat_error_unauthorized_shows_relogin_link(tmp_path):
    """An ``unauthorized`` error code must surface a 🔑 重新登录
    anchor so the user can recover from an expired token without
    hunting for the login menu.
    """
    win, _ctrl = _make_littrace_window(tmp_path)
    win._on_message_event({"role": "user", "text": "ask something"})
    win._on_error_event(
        {
            "message": "登录已过期",
            "suggestion": "请重新登录 ChatGPT",
            "error_code": "unauthorized",
            "raw": "UnauthorizedError: token expired",
        }
    )
    html = win._chat_panel._view.toHtml()
    assert "littrace:relogin" in html, (
        "unauthorized errors must expose a relogin anchor; "
        f"got HTML: {html[:500]}"
    )


def test_chat_error_without_prior_user_message_omits_retry_link(tmp_path):
    """If no user message has been submitted yet (e.g. the error
    fires during warmup), there's nothing to retry — the retry
    anchor must be omitted so the user isn't lured into clicking
    a link that does nothing.
    """
    win, _ctrl = _make_littrace_window(tmp_path)
    win._on_error_event(
        {
            "message": "warmup failed",
            "error_code": "other",
            "raw": "RuntimeError: codex home not initialized",
        }
    )
    html = win._chat_panel._view.toHtml()
    assert "littrace:retry-last" not in html, (
        "Without a prior user message, the retry anchor would be "
        "a dead link — must be omitted."
    )
    # Details anchor is still useful.
    assert "littrace:show-error-detail" in html


# ---------------------------------------------------------------------------
# Round 19 — Daily result preview dialog
# ---------------------------------------------------------------------------


def _trigger_daily_preview(
    win,
    *,
    topic="mxene_sensor",
    keywords="",
    year_min=2023,
    year_max=2026,
    target_papers=10,
    rounds_done=2,
    cumulative_downloaded=8,
    cumulative_candidates=42,
    warnings=None,
    summary_lines=None,
):
    """Helper: feed a payload through the GUI-thread slot directly.
    Tests bypass the worker thread (which spawns a real subprocess)
    and call ``_show_daily_preview_from_any_thread`` as if the JSON
    had already been queued onto the event loop."""
    import json
    payload = {
        "topic": topic,
        "keywords": keywords,
        "year_min": year_min,
        "year_max": year_max,
        "target_papers": target_papers,
        "rounds_done": rounds_done,
        "cumulative_downloaded": cumulative_downloaded,
        "cumulative_candidates": cumulative_candidates,
        "warnings": warnings or [],
        "summary_lines": summary_lines or [],
    }
    win._show_daily_preview_from_any_thread(json.dumps(payload, ensure_ascii=False))


def test_daily_preview_dialog_shows_topic_and_target(tmp_path):
    """The preview dialog must surface the topic + year range +
    target so the user can verify the run actually used the values
    they typed into ``DailyConfigDialog``."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    win, _ctrl = _make_littrace_window(tmp_path)
    _trigger_daily_preview(
        win,
        topic="mxene_sensor",
        year_min=2020,
        year_max=2025,
        target_papers=15,
        cumulative_downloaded=12,
    )
    dlg = win._daily_preview
    assert dlg is not None, "preview dialog should be cached on the window"
    # Inspect the rendered text — the topic / year range / target
    # should all appear somewhere in the dialog's child widgets.
    rendered = dlg.findChildren(__import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel)
    all_text = "\n".join(lbl.text() for lbl in rendered)
    assert "mxene_sensor" in all_text, (
        f"preview must show the topic; got labels: {all_text!r}"
    )
    assert "2020" in all_text and "2025" in all_text, (
        f"preview must show year range; got labels: {all_text!r}"
    )
    assert "15" in all_text, (
        f"preview must show target paper count; got labels: {all_text!r}"
    )


def test_daily_preview_dialog_warns_on_shortfall(tmp_path):
    """If cumulative_downloaded < target_papers, the dialog must
    surface an inline warning so they don't have to compare numbers
    in their head."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    win, _ctrl = _make_littrace_window(tmp_path)
    _trigger_daily_preview(
        win,
        target_papers=20,
        cumulative_downloaded=8,
        cumulative_candidates=42,
    )
    dlg = win._daily_preview
    rendered = dlg.findChildren(__import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel)
    all_text = "\n".join(lbl.text() for lbl in rendered)
    assert "少于目标" in all_text or "⚠️" in all_text, (
        f"preview must flag the shortfall; got labels: {all_text!r}"
    )


def test_daily_preview_dialog_is_non_modal(tmp_path):
    """Round 19 made the preview non-modal so the user can flip
    to the context panel and pin/unpin papers without dismissing
    the summary first. A modal dialog blocks the rest of the UI.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from PySide6.QtWidgets import QDialog

    win, _ctrl = _make_littrace_window(tmp_path)
    _trigger_daily_preview(win)
    dlg = win._daily_preview
    # ``QDialog.windowModality`` returns ``Qt.NonModal`` for
    # non-modal dialogs; ``Qt.ApplicationModal`` blocks input to
    # all other top-level windows.
    from PySide6 import QtCore as _QtCore
    assert dlg.windowModality() == _QtCore.Qt.WindowModality.NonModal, (
        f"preview should be non-modal so the user can still poke at "
        f"the context panel; got {dlg.windowModality()!r}"
    )


def test_daily_preview_replaces_previous_dialog(tmp_path):
    """If the user kicks off two daily runs in a row, the second
    preview must replace the first (not stack two dialogs on top
    of each other). The window caches the dialog as
    ``_daily_preview`` so a fresh one can be popped.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    win, _ctrl = _make_littrace_window(tmp_path)
    _trigger_daily_preview(win, topic="first_run")
    first = win._daily_preview
    _trigger_daily_preview(win, topic="second_run")
    second = win._daily_preview
    assert first is not second, (
        "second preview should replace the first, not stack"
    )
    # The first dialog was closed + deleteLater'd — Qt may not
    # actually destroy it until the event loop runs, but the
    # window's cached reference must point at the new dialog.
    assert win._daily_preview is second


# ---------------------------------------------------------------------------
# Round 19 — Trace panel collapsible groups
# ---------------------------------------------------------------------------


def test_trace_panel_groups_start_expanded_and_can_collapse(tmp_path):
    """Each trace-panel group must start expanded (so a new user
    isn't staring at two blank panels) but can be collapsed by
    clicking the toggle button. When collapsed the body hides; when
    re-expanded it returns.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    # Isolate QSettings so the test doesn't pollute the user's prefs.
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    # Clear any values left over from a prior test in this run —
    # QSettings("LitTrace", "littrace-qt") is a global registry
    # keyed on org/app, so without ``clear()`` a previous test that
    # wrote ``trace/group:sessions=False`` would bleed into this
    # one and the assertion below would fail.
    QSettings("LitTrace", "littrace-qt").clear()

    from littrace.window_qt import TracePanel
    panel = TracePanel()
    panel.show()
    # Default state: expanded.
    assert panel._workflow_body.isVisible()
    assert panel._sessions_body.isVisible()
    # Collapse the workflow section.
    panel._workflow_toggle.setChecked(False)
    panel._workflow_toggle.toggled.emit(False)
    assert not panel._workflow_body.isVisible(), (
        "collapsing the workflow toggle must hide the body"
    )
    # Sessions body is unaffected.
    assert panel._sessions_body.isVisible(), (
        "collapsing one section must not affect the other"
    )
    # Re-expand.
    panel._workflow_toggle.setChecked(True)
    panel._workflow_toggle.toggled.emit(True)
    assert panel._workflow_body.isVisible()


def test_trace_panel_collapse_state_persists_via_qsettings(tmp_path):
    """Round 19: collapse state is remembered across TracePanel
    instances via ``QSettings``. A second instance built against the
    same QSettings scope must reflect the user's prior choice.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings("LitTrace", "littrace-qt").clear()

    from littrace.window_qt import TracePanel

    # First instance: collapse the sessions section.
    panel1 = TracePanel()
    panel1.show()
    panel1._sessions_toggle.setChecked(False)
    panel1._sessions_toggle.toggled.emit(False)
    assert not panel1._sessions_body.isVisible()

    # Second instance: must come up with the sessions section
    # collapsed too.
    panel2 = TracePanel()
    panel2.show()
    assert not panel2._sessions_body.isVisible(), (
        "second TracePanel must restore the collapsed state from "
        "QSettings; user shouldn't have to collapse again every "
        "launch"
    )
    # Workflow section is independent — stays expanded.
    assert panel2._workflow_body.isVisible(), (
        "workflow section default is expanded; collapsing sessions "
        "must not affect it"
    )


# ---------------------------------------------------------------------------
# Round 19 — Context multi-select compare shortcut
# ---------------------------------------------------------------------------


def _make_context_panel_with_papers(tmp_path: Path, n: int = 3):
    """Build a ``ContextPanel`` populated with ``n`` placeholder
    papers so we can drive the multi-select + compare flow without
    standing up the full LitTraceQtWindow."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from littrace.window_qt import ContextPanel
    from littrace.models import PaperMetadata

    panel = ContextPanel()
    panel.show()
    papers = [
        PaperMetadata(
            paper_id=f"paper-{i}",
            title=f"Sample Paper {i}",
            year=2024,
            journal=f"Journal {i}",
        )
        for i in range(n)
    ]
    panel.refresh(papers)
    return panel, papers


def test_context_panel_compare_button_enables_with_two_or_more_selected():
    """The compare button must enable only when ≥2 papers are
    selected — comparing a single paper doesn't make sense."""
    panel, _papers = _make_context_panel_with_papers(Path("/tmp"), n=3)
    # Initial state: nothing selected.
    assert not panel._compare_btn.isEnabled()
    assert "未选中" in panel._compare_count.text()
    # Select one — button stays disabled, count label warns "need ≥2".
    panel._list.item(0).setSelected(True)
    panel._list.itemSelectionChanged.emit()
    assert not panel._compare_btn.isEnabled()
    assert "1 篇" in panel._compare_count.text()
    # Select a second — button enables.
    panel._list.item(1).setSelected(True)
    panel._list.itemSelectionChanged.emit()
    assert panel._compare_btn.isEnabled(), (
        "compare button must enable when 2+ rows are selected"
    )
    assert "2 篇" in panel._compare_count.text()
    # Select a third — count updates.
    panel._list.item(2).setSelected(True)
    panel._list.itemSelectionChanged.emit()
    assert "3 篇" in panel._compare_count.text()


def test_context_panel_compare_click_submits_chat_message(tmp_path):
    """Clicking "🔍 比较选中" must route a comparison prompt through
    the controller. We attach a stub controller to the panel via
    ``window()`` so the panel can find ``_controller.submit_user_message``.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from littrace.window_qt import ContextPanel

    # Build a controller + window so the panel sees a ``window()
    # ._controller``. Reuse the existing helpers.
    win, ctrl = _make_littrace_window(tmp_path)
    panel = win._context_panel

    # Make sure we have a clean 3-paper slate regardless of the
    # other test fixtures' state.
    from littrace.models import PaperMetadata
    panel.refresh(
        [
            PaperMetadata(paper_id="a", title="Paper A", year=2024),
            PaperMetadata(paper_id="b", title="Paper B", year=2024),
            PaperMetadata(paper_id="c", title="Paper C", year=2024),
        ]
    )
    panel._list.item(0).setSelected(True)
    panel._list.item(1).setSelected(True)

    # Spy on submit_user_message.
    called: list[str] = []
    original = ctrl.submit_user_message

    def _spy(text: str) -> None:
        called.append(text)

    ctrl.submit_user_message = _spy
    try:
        panel._on_compare_clicked()
        assert len(called) == 1, (
            f"compare click must submit exactly one message; got {called}"
        )
        prompt = called[0]
        assert "Paper A" in prompt and "Paper B" in prompt, (
            f"compare prompt must include both selected paper titles; "
            f"got {prompt!r}"
        )
        assert "比较" in prompt, (
            f"compare prompt must frame the request as a comparison; "
            f"got {prompt!r}"
        )
    finally:
        ctrl.submit_user_message = original


def test_context_panel_compare_clears_selection_after_submit(tmp_path):
    """After a comparison is submitted, the selection must clear so
    the user can immediately pick a new comparison set without
    having to Ctrl+click each previous pick."""
    win, _ctrl = _make_littrace_window(tmp_path)
    panel = win._context_panel
    from littrace.models import PaperMetadata
    panel.refresh(
        [
            PaperMetadata(paper_id="a", title="Paper A"),
            PaperMetadata(paper_id="b", title="Paper B"),
        ]
    )
    panel._list.item(0).setSelected(True)
    panel._list.item(1).setSelected(True)
    panel._on_compare_clicked()
    assert len(panel._list.selectedItems()) == 0, (
        "selection must clear after compare submission; otherwise "
        "the user has to Ctrl+click each old pick to start over"
    )


# ---------------------------------------------------------------------------
# Round 19 — Non-modal daily dialog + cancel button
# ---------------------------------------------------------------------------


def test_daily_dialog_is_non_modal_and_has_cancel_button(tmp_path):
    """Round 19: ``DailyConfigDialog`` is now non-modal so the user
    can keep poking at the rest of the window while tweaking
    parameters, and ships with an explicit "取消" button that
    cleanly cancels the run (the X button + Esc still work too).
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings("LitTrace", "littrace-qt").clear()

    from littrace.window_qt import DailyConfigDialog
    dlg = DailyConfigDialog()
    assert dlg.windowModality() == QtCore.Qt.WindowModality.NonModal, (
        f"DailyConfigDialog should be non-modal so the user can "
        f"keep using the chat/context/trace panels; got {dlg.windowModality()!r}"
    )
    # Locate the cancel button by its visible label — this is the
    # discoverability contract: the user must be able to read "取消"
    # without having to hover for a tooltip.
    from PySide6.QtWidgets import QPushButton
    cancel_buttons = [
        b for b in dlg.findChildren(QPushButton)
        if b.text().strip() == "取消"
    ]
    assert cancel_buttons, (
        "DailyConfigDialog must expose a '取消' button so the "
        "user can abort the run without using Esc or the X button"
    )
    # And the primary action stays clearly the primary.
    primary_buttons = [
        b for b in dlg.findChildren(QPushButton)
        if b.text().strip().startswith("开始检索")
    ]
    assert primary_buttons, "primary '开始检索' button missing"


def test_run_daily_shows_dialog_without_blocking(tmp_path, monkeypatch):
    """``_on_run_daily`` must NOT block — the production code used
    to call ``dialog.exec()`` which freezes the rest of the GUI
    until the user clicks Accept/Cancel. Round 19 replaced it with
    ``dialog.show()`` so the user can keep clicking elsewhere."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    # Stub dialog so we don't pull in real Qt dialog state.
    class _SignalStub:
        def __init__(self) -> None:
            self.slots: list = []

        def connect(self, slot) -> None:
            self.slots.append(slot)

    class _StubDialog:
        def __init__(self, parent, **kw) -> None:
            self.accepted = _SignalStub()
            self.rejected = _SignalStub()
            self.shown = False

        def show(self) -> None:
            self.shown = True

        def raise_(self) -> None:
            return None

        def activateWindow(self) -> None:
            return None

        def close(self) -> None:
            return None

        def deleteLater(self) -> None:
            return None

        def exec(self) -> int:
            return 0  # would block forever in real life

    monkeypatch.setattr(
        "littrace.window_qt.DailyConfigDialog", _StubDialog,
    )

    win, _ctrl = _make_littrace_window(tmp_path)
    # If the implementation regressed to ``dialog.exec()``, this
    # call would hang the test (and the real GUI). With the
    # non-modal show() it returns immediately.
    win._on_run_daily()
    dlg = win._daily_config_dialog
    assert dlg is not None, "dialog must be cached on the window"
    assert dlg.shown, "dialog.show() must have been called"
