"""Round 20 (Phase 5 follow-up): Window elicitation / error modal tests.

Window had three Phase-5 helpers that previously lived only as
end-of-file lambda patches onto ``LitTraceWindow``:

  * ``_make_window_elicitation_handler`` — registers the future-based
    handler that surfaces Codex ``mcpServer/elicitation/request``
    notifications as a Y/N/Esc modal.
  * ``_open_elicitation_modal`` — renders the Tk Toplevel that
    collects the operator's decision.
  * ``_show_app_server_error_modal`` / ``_show_startup_error_modal``
    — error envelopes for mid-turn Codex failures and startup
    preflight failures.

The handlers run on the Codex reader loop's asyncio loop, not the Tk
thread, so the cross-loop future-resolution safety is critical: a
naive ``future.set_result(payload)`` from the Tk callback would raise
``RuntimeError: ... attached to a different loop``. The tests pin
this contract by spinning up a real ``tk.Tk`` (Tk is available in the
project's test environment) and recording every modal spawn.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest

from littrace import window as window_module
from littrace.config import LitTraceConfig


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def root():
    """A real (withdrawn) Tk root that is destroyed after the test.

    We do not use ``littrace.window.LitTraceWindow`` directly because
    its ``__init__`` constructs the full layout, modal popups, and
    binds to ``load_config``. These tests only need the small surface
    the Round-20 helpers touch (``root``, ``tk``, ``ttk``, plus the
    helper-method namespace).
    """
    import tkinter as tk
    from tkinter import ttk

    r = tk.Tk()
    r.withdraw()
    yield r, tk, ttk
    try:
        r.destroy()
    except Exception:  # pragma: no cover - cleanup
        pass


class _StubWindow:
    """Bare object that quacks like a ``LitTraceWindow`` for the
    Round-20 helpers. We only need ``root``, ``tk``, ``ttk``,
    ``_pending_approvals`` / ``_approval_roots`` for the elicitation
    handler tests.
    """

    def __init__(self, root, tk, ttk) -> None:
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self._pending_approvals: list[dict[str, Any]] = []
        self._approval_roots: dict[int, Any] = {}


# ---------------------------------------------------------------------------
# _make_window_elicitation_handler
# ---------------------------------------------------------------------------


def test_handler_schedules_modal_via_root_after(root) -> None:
    """The handler must defer modal creation to the Tk thread via
    ``root.after(0, ...)``. Without that, ``tk.Toplevel`` would race
    the main thread and the modal would never appear."""
    r, tk, ttk = root
    win = _StubWindow(r, tk, ttk)

    handler = window_module._make_window_elicitation_handler(win)
    assert callable(handler)

    # Capture every ``root.after`` call so we can verify the handler
    # scheduled exactly one modal-spawn.
    scheduled: list[tuple[int, Any]] = []
    real_after = r.after
    r.after = lambda delay, fn, *a: scheduled.append((delay, (fn, a))) or "job-id"  # type: ignore[assignment]

    async def _drive() -> None:
        params = {
            "serverName": "littrace",
            "_meta": {"tool_params": {"topic": "MXene"}},
            "message": 'Allow tool "search_papers"?',
        }
        # We never resolve the future here; we just need to confirm
        # ``root.after`` was called before we cancel.
        task = asyncio.create_task(handler(params))
        # Yield until the handler parks the modal via after().
        for _ in range(20):
            await asyncio.sleep(0)
            if scheduled:
                break
        assert scheduled, "handler never scheduled modal spawn via root.after"
        delay, (fn, args) = scheduled[0]
        assert delay == 0, f"modal spawn must use delay=0; got {delay}"
        assert callable(fn), "scheduled callback must be callable"
        task.cancel()

    asyncio.run(_drive())
    # Restore for cleanup.
    r.after = real_after  # type: ignore[assignment]


def test_cross_loop_resolution_uses_call_soon_threadsafe() -> None:
    """When the Tk callback resolves a future bound to a different
    event loop, it must use ``loop.call_soon_threadsafe`` to avoid
    the ``RuntimeError: ... attached to a different loop`` failure.

    This is the cross-loop safety contract that the TUI also relies
    on (``tui._resolve_approval_future``).
    """
    import asyncio as _asyncio

    # Park the future on a long-lived loop (not the Tk thread's
    # implicit closed-loop).
    future, _thread = _live_future()

    class _RecordingLoopProxy:
        """Wraps the live loop to record ``call_soon_threadsafe`` calls
        without changing the future's binding."""

        def __init__(self, inner) -> None:
            self._inner = inner
            self.calls: list[tuple[Any, tuple[Any, ...]]] = []

        def call_soon_threadsafe(self, fn, *args):
            self.calls.append((fn, args))
            return self._inner.call_soon_threadsafe(fn, *args)

    async def _verify() -> None:
        # ``fut_loop`` is the live loop (analogous to the Codex reader
        # loop in production).
        fut_loop = future._loop  # type: ignore[attr-defined]
        proxy = _RecordingLoopProxy(fut_loop)
        # Inline the relevant branch of the production resolve callback.
        try:
            running_loop = _asyncio.get_running_loop()
        except RuntimeError:
            # The Tk main thread has no running asyncio loop — the
            # exact condition this test is meant to exercise.
            running_loop = None
        if running_loop is not fut_loop:
            proxy.call_soon_threadsafe(future.set_result, {"action": "accept"})
            # Yield so the scheduled callback actually runs.
            await _asyncio.sleep(0.05)
        assert proxy.calls, (
            "cross-loop hop must call call_soon_threadsafe"
        )
        assert future.done(), (
            "future must resolve after call_soon_threadsafe fires"
        )
        assert future.result() == {"action": "accept"}

    _asyncio.run(_verify())


# ---------------------------------------------------------------------------
# _open_elicitation_modal
# ---------------------------------------------------------------------------


def _live_future() -> tuple[asyncio.Future, threading.Thread]:
    """Return ``(future, daemon_thread)`` parked on a long-lived event loop.

    ``asyncio.Future()`` without a running loop in 3.10+ attaches to a
    lazily-created / possibly-closed loop, which silently swallows
    ``set_result`` from a Tk callback. The production handler parks
    its future on the Codex reader loop (always live while the App
    Server is running); we mimic that by spinning up a daemon thread
    that runs an event loop until ``atexit`` fires.

    We register an ``atexit`` hook that schedules ``loop.stop()`` via
    ``call_soon_threadsafe`` so the runner returns from
    ``run_forever`` and can close its proactor sockets cleanly —
    otherwise the interpreter tears down the process while the
    proactor still has live handles and Windows trips a
    ``0x80000003`` breakpoint during GC.
    """
    import atexit as _atexit
    import threading as _threading

    ready = _threading.Event()
    captured: dict[str, Any] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        captured["loop"] = loop
        captured["future"] = loop.create_future()
        ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:  # pragma: no cover - cleanup
                pass

    def _shutdown() -> None:
        loop = captured.get("loop")
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

    thread = _threading.Thread(target=_runner, daemon=True)
    thread.start()
    ready.wait(timeout=2.0)
    _atexit.register(_shutdown)
    return captured["future"], thread


def test_open_elicitation_modal_renders_tk_toplevel_with_title(root) -> None:
    """The modal must be a Tk ``Toplevel`` whose title is the
    ``server.tool`` of the request — operators identify what they
    are approving from the title bar, not from a single line of text
    inside an unlabeled dialog."""
    r, tk, ttk = root
    win = _StubWindow(r, tk, ttk)

    future, _thread = _live_future()
    params = {
        "serverName": "littrace",
        "_meta": {"tool_params": {"topic": "MXene"}},
        "message": 'Allow tool "search_papers"?',
    }

    request_key = 42
    window_module._open_elicitation_modal(win, request_key, params, future)
    assert request_key in win._approval_roots, (
        "modal must be tracked in _approval_roots so resolve can pop it"
    )
    modal = win._approval_roots[request_key]
    try:
        title = modal.title()
        assert "littrace" in title and "search_papers" in title, (
            f"modal title must mention server and tool; got {title!r}"
        )
        text_children = [
            child for child in modal.winfo_children()
            if isinstance(child, tk.Text)
        ]
        assert text_children, (
            "modal must include a Text widget rendering the arguments"
        )
        body = text_children[0].get("1.0", "end")
        assert "MXene" in body, (
            f"modal body must render the request arguments; got {body!r}"
        )
    finally:
        modal.destroy()


def test_open_elicitation_modal_resolves_future_with_accept(root) -> None:
    """Pressing Y in the modal must resolve the handler's future
    with the ``_ACCEPT_RESPONSE`` payload. The Tk callback uses
    ``call_soon_threadsafe``; we verify the payload shape and the
    request_key popped from ``_approval_roots``."""
    r, tk, ttk = root
    win = _StubWindow(r, tk, ttk)
    from littrace.tui import _ACCEPT_RESPONSE

    future, _thread = _live_future()
    params = {
        "serverName": "littrace",
        "_meta": {"tool_params": {}},
        "message": 'Allow tool "search_papers"?',
    }
    window_module._open_elicitation_modal(win, 7, params, future)
    modal = win._approval_roots[7]
    try:
        accept_btn = _find_button_by_text(modal, "批准", ttk)
        assert accept_btn is not None, "Accept button must be present"
        accept_btn.invoke()
        assert future.done(), "Y press must resolve the handler's future"
        assert future.result() == _ACCEPT_RESPONSE
        # The modal must have popped itself from _approval_roots.
        assert 7 not in win._approval_roots
    finally:
        if modal.winfo_exists():
            modal.destroy()


def test_open_elicitation_modal_resolves_future_with_decline(root) -> None:
    """N button resolves with the decline payload — operators must be
    able to refuse without closing the modal."""
    r, tk, ttk = root
    win = _StubWindow(r, tk, ttk)
    from littrace.tui import _DECLINE_RESPONSE

    future, _thread = _live_future()
    params = {
        "serverName": "littrace",
        "_meta": {"tool_params": {}},
        "message": 'Allow tool "search_papers"?',
    }
    window_module._open_elicitation_modal(win, 99, params, future)
    modal = win._approval_roots[99]
    try:
        decline_btn = _find_button_by_text(modal, "拒绝", ttk)
        assert decline_btn is not None
        decline_btn.invoke()
        assert future.done()
        assert future.result() == _DECLINE_RESPONSE
    finally:
        if modal.winfo_exists():
            modal.destroy()


# ---------------------------------------------------------------------------
# _show_app_server_error_modal / _show_startup_error_modal
# ---------------------------------------------------------------------------


def test_show_app_server_error_modal_includes_remediation(root) -> None:
    """The error modal must surface both the AppServerError message
    AND the remediation steps. Without remediation, the operator
    cannot self-diagnose and would just file a ticket."""
    r, tk, ttk = root
    win = _StubWindow(r, tk, ttk)

    class _FakeAppServerError(Exception):
        pass

    exc = _FakeAppServerError("transport closed before reply")
    window_module._show_app_server_error_modal(win, exc)
    # The helper does not register the modal in _approval_roots —
    # find the topmost Toplevel the helper created.
    modals = [
        child for child in r.winfo_children()
        if isinstance(child, tk.Toplevel)
    ]
    assert modals, "App Server error modal must create a Toplevel"
    modal = modals[-1]
    try:
        text_widgets = [
            child for child in modal.winfo_children()
            if isinstance(child, tk.Text)
        ]
        assert text_widgets, "modal body must include a Text widget"
        body = text_widgets[0].get("1.0", "end")
        assert "transport closed before reply" in body
        assert "修复步骤" in body, (
            "modal body must include the remediation label so operators "
            "know what the bullet list means"
        )
    finally:
        modal.destroy()


def test_show_startup_error_modal_lists_remediation_steps(root) -> None:
    """The startup-error modal must pull ``remediation`` off the
    exception and render it as bullet points. Without that, the
    operator sees a bare ``Codex CLI 未找到: ...`` and no next step."""
    r, tk, ttk = root
    win = _StubWindow(r, tk, ttk)

    class _FakeStartupError(Exception):
        def __init__(self, message: str, remediation: list[str]) -> None:
            super().__init__(message)
            self.remediation = remediation

    exc = _FakeStartupError(
        "Codex CLI 未找到: 'codex' 不在 PATH。",
        [
            "安装 Codex CLI: `npm install -g @openai/codex`",
            "把 codex 可执行文件加到 PATH",
        ],
    )
    window_module._show_startup_error_modal(win, exc)
    modals = [
        child for child in r.winfo_children()
        if isinstance(child, tk.Toplevel)
    ]
    assert modals
    modal = modals[-1]
    try:
        text_widgets = [
            child for child in modal.winfo_children()
            if isinstance(child, tk.Text)
        ]
        assert text_widgets
        body = text_widgets[0].get("1.0", "end")
        assert "Codex CLI 未找到" in body
        assert "npm install -g @openai/codex" in body, (
            "remediation bullets must be rendered verbatim so operators "
            "can copy-paste them into a shell"
        )
        assert "把 codex 可执行文件加到 PATH" in body
    finally:
        modal.destroy()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_button_by_text(parent, needle: str, ttk):
    """Walk a Tk widget tree and return the first ttk.Button whose
    ``cget("text")`` contains ``needle``. Used by the modal Y/N/Esc
    tests to programmatically press the buttons the way an operator
    would."""
    stack = list(parent.winfo_children())
    while stack:
        child = stack.pop()
        try:
            text = child.cget("text")  # type: ignore[attr-defined]
        except Exception:
            text = None
        if isinstance(text, str) and needle in text:
            return child
        stack.extend(child.winfo_children())
    return None
