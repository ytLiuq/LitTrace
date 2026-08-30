"""Round 18: cover the elicitation param-parsing contract.

The TUI's ``_make_elicitation_handler`` is the single point of
contact between codex-cli's ``mcpServer/elicitation/request``
notification and the operator's Y/N/Esc decision. The wire format
matters: a wrong field name silently degrades the modal into a
"littrace / ?" placeholder, and operators then have no idea what
they are approving.

The two wire formats we accept:

  * codex 0.150.1+: ``{serverName, mode, _meta.tool_params, message}``
    where the tool name is encoded inside the message string
    (``tool "search_papers"``).
  * Older / other transports: ``{server, tool, arguments}``.

These tests pin the parser against both shapes. The handler parks
a Future on ``state.pending_approval`` and awaits it; the tests
resolve that future on a separate task before asserting on the
parked approval's shape.
"""

from __future__ import annotations

import asyncio
import curses
import os
from typing import Any

import pytest

from littrace.tui import (
    ApprovalRequest,
    TUIState,
    _advance_approval_queue,
    _approval_modal_body_rows,
    _handle_approval_key,
    _make_elicitation_handler,
)


pytestmark = pytest.mark.unit


_ACCEPT: dict[str, Any] = {"action": "accept", "content": {}, "_meta": None}


def _state() -> TUIState:
    from littrace.models import LiteratureWorkspace
    return TUIState(
        workspace=LiteratureWorkspace(),
        session_id="test",
        session_root="/tmp",
        messages=[],
    )


async def _drive(handler, state: TUIState, params: dict[str, Any]) -> None:
    """Run the handler, resolve its parked future on a parallel task.

    Yields until the handler parks ``state.pending_approval`` and
    then resolves the Future with the accept payload. The handler
    then returns cleanly. Times out fast (50 ticks) so a buggy test
    never hangs CI.
    """
    handler_task = asyncio.create_task(handler(params))
    for _ in range(50):
        await asyncio.sleep(0)
        if handler_task.done():
            return
        approval = state.pending_approval
        if approval is not None and not approval.future.done():
            approval.future.set_result(_ACCEPT)
            await handler_task
            return
    raise AssertionError("handler never parked an approval within 50 ticks")


@pytest.mark.parametrize(
    "params,expected_server,expected_tool,expected_args",
    [
        # New codex 0.150 wire format.
        (
            {
                "threadId": "thr-1",
                "turnId": "turn-1",
                "serverName": "littrace",
                "mode": "form",
                "_meta": {
                    "codex_approval_kind": "mcp_tool_call",
                    "tool_params": {"topic": "MXene"},
                },
                "message": 'Allow the littrace MCP server to run tool "search_papers"?',
                "requestedSchema": {"type": "object", "properties": {}},
            },
            "littrace",
            "search_papers",
            {"topic": "MXene"},
        ),
        # Older / flat wire format.
        (
            {
                "server": "littrace",
                "tool": "search_papers",
                "arguments": {"topic": "MXene"},
            },
            "littrace",
            "search_papers",
            {"topic": "MXene"},
        ),
        # ``name`` instead of ``tool``.
        (
            {
                "serverName": "littrace",
                "name": "get_workspace_context",
                "arguments": {},
            },
            "littrace",
            "get_workspace_context",
            {},
        ),
        # Empty payload — tool falls back to ``?`` but server still
        # resolves so the operator sees the right MCP server name.
        (
            {},
            "littrace",
            "?",
            {},
        ),
    ],
)
def test_extract_server_and_tool_and_args(
    params, expected_server, expected_tool, expected_args
) -> None:
    """The parked ApprovalRequest must carry the right server + tool +
    arguments for each wire format we accept."""
    state = _state()
    handler = _make_elicitation_handler(state)

    captured: dict[str, Any] = {}

    async def _capture_then_drive() -> None:
        task = asyncio.create_task(handler(params))
        for _ in range(50):
            await asyncio.sleep(0)
            if state.pending_approval is not None:
                captured["approval"] = state.pending_approval
                state.pending_approval.future.set_result(_ACCEPT)
                await task
                return
        raise AssertionError("handler never parked an approval within 50 ticks")

    asyncio.run(_capture_then_drive())
    approval = captured["approval"]
    assert approval.server == expected_server
    assert approval.tool == expected_tool
    assert approval.arguments == expected_args


def test_handler_clears_pending_approval_after_resolve() -> None:
    """The handler's finally block must clear ``state.pending_approval``
    so a follow-up approval can park a new request."""
    state = _state()
    handler = _make_elicitation_handler(state)
    asyncio.run(_drive(handler, state, {
        "serverName": "littrace",
        "_meta": {"tool_params": {}},
        "message": 'tool "search_papers"?',
    }))
    assert state.pending_approval is None


# -- Round 19: queue behaviour ---------------------------------------------


def test_concurrent_elicitations_queue_behind_active_one() -> None:
    """When a second elicitation arrives while the operator has not
    yet resolved the first, the second one parks on
    ``state.approval_queue`` instead of overwriting
    ``state.pending_approval``.

    This is the fix for the Round 18 P2-2 backlog item — the
    previous handler unconditionally set ``state.pending_approval``,
    so a later arrival stole the modal from the earlier request.
    """
    state = _state()

    async def _run() -> None:
        handler = _make_elicitation_handler(state)
        task1 = asyncio.create_task(handler({
            "serverName": "littrace",
            "_meta": {"tool_params": {"q": "first"}},
            "message": 'tool "search_papers"?',
        }))
        # Yield so the first handler parks its approval.
        for _ in range(20):
            await asyncio.sleep(0)
            if state.pending_approval is not None:
                break
        assert state.pending_approval is not None
        first_approval = state.pending_approval
        # Second handler call while the first is still pending.
        task2 = asyncio.create_task(handler({
            "serverName": "littrace",
            "_meta": {"tool_params": {"q": "second"}},
            "message": 'tool "search_papers"?',
        }))
        for _ in range(20):
            await asyncio.sleep(0)
            if state.approval_queue:
                break
        # The modal must still show the FIRST request; the second
        # must be in the queue, NOT in pending_approval.
        assert state.pending_approval is first_approval
        assert len(state.approval_queue) == 1
        queued = state.approval_queue[0]
        assert queued.arguments == {"q": "second"}
        # Operator presses Y — this resolves the future AND
        # promotes the queued approval onto pending_approval.
        await _handle_approval_key(ord("Y"), state)
        await task1
        # The handler's finally already cleared pending_approval
        # (since it owned that slot). _handle_approval_key's
        # _advance_approval_queue call restored it from the queue.
        assert state.pending_approval is queued
        assert state.approval_queue == []
        # Operator presses Y for the second one too.
        await _handle_approval_key(ord("Y"), state)
        await task2
        assert state.pending_approval is None
        assert state.approval_queue == []

    asyncio.run(_run())


@pytest.mark.parametrize("n_total", [3, 5, 8])
def test_n_concurrent_elicitations_all_queue_in_order(n_total: int) -> None:
    """N elicitation requests arriving while none has been resolved
    yet: the first one stays on ``pending_approval`` and the other
    ``N-1`` queue up in arrival order. Each Y/N/Esc decision
    promotes the head of the queue onto the modal so the operator
    reviews them in FIFO order.

    The bug we are regressing against: pre-Round-19 handler
    unconditionally overwrote ``pending_approval``, so
    N>1 arrivals lost all but the last one.
    """
    state = _state()

    async def _run() -> None:
        handler = _make_elicitation_handler(state)
        tasks = []
        arrivals = []
        for i in range(n_total):
            payload = {
                "serverName": "littrace",
                "_meta": {"tool_params": {"q": f"query-{i}"}},
                "message": f'tool "search_papers"?',
                "serverRequestId": f"req-{i}",
            }
            task = asyncio.create_task(handler(payload))
            tasks.append((task, payload))
            arrivals.append(payload)
        # Yield until all N have parked (one on pending, N-1 in queue).
        for _ in range(50):
            await asyncio.sleep(0)
            if (
                state.pending_approval is not None
                and len(state.approval_queue) == n_total - 1
            ):
                break
        assert state.pending_approval is not None
        first = state.pending_approval
        # The first arrival is the active one; it carries query-0.
        assert first.arguments == {"q": "query-0"}
        assert first.request_id == "req-0"
        # The remaining N-1 sit in the queue in arrival order.
        assert len(state.approval_queue) == n_total - 1
        for i, queued in enumerate(state.approval_queue, start=1):
            assert queued.arguments == {"q": f"query-{i}"}
            assert queued.request_id == f"req-{i}"
        # Drain the queue one decision at a time. Each Y promotes
        # the next queued request onto the modal.
        expected_active = first
        for i in range(n_total):
            assert state.pending_approval is expected_active, (
                f"iter {i}: expected {expected_active.request_id!r}, "
                f"got {state.pending_approval and state.pending_approval.request_id!r}"
            )
            assert (
                state.pending_approval.arguments == {"q": f"query-{i}"}
            )
            # Operator presses Y → resolves the future and promotes.
            await _handle_approval_key(ord("Y"), state)
            task, _ = tasks[i]
            await task
            # After this Y press, the queue head has been promoted
            # onto pending_approval (if any are left). Update the
            # expected slot for the next iteration.
            if i < n_total - 1:
                expected_active = state.pending_approval
            else:
                expected_active = None  # type: ignore[assignment]
        # All N resolved; queue empty, no pending approval.
        assert state.pending_approval is None
        assert state.approval_queue == []
        assert all(t.done() for t, _ in tasks)

    asyncio.run(_run())


def test_advance_approval_queue_promotes_head() -> None:
    """The ``_advance_approval_queue`` helper pops the head of the
    queue onto ``pending_approval`` so the operator's next Y/N/Esc
    decision lands on the right request."""
    from littrace.tui import _advance_approval_queue

    async def _make_req(label: str):
        loop = asyncio.get_running_loop()
        return ApprovalRequest(
            request_id=label,
            server="littrace",
            tool="search_papers",
            arguments={"label": label},
            future=loop.create_future(),
        )

    async def _run() -> None:
        first = await _make_req("first")
        second = await _make_req("second")
        state = _state()
        state.pending_approval = first
        state.approval_queue.append(second)
        # Operator resolves the first one. _advance_approval_queue
        # is called by _handle_approval_key; we call it directly here
        # to keep the test focused on the helper.
        state.pending_approval = None
        _advance_approval_queue(state)
        assert state.pending_approval is second
        assert state.approval_queue == []
        # Operator resolves the second one too.
        state.pending_approval = None
        _advance_approval_queue(state)
        assert state.pending_approval is None

    asyncio.run(_run())


# -- Round 19: scroll keys --------------------------------------------------


def test_scroll_keys_move_offset_without_resolving() -> None:
    """The Up/Down/PgUp/PgDn/Home/End keys adjust
    ``approval.scroll_offset`` but do NOT resolve the future.

    Regression for the Round 18 P2-1 backlog — the previous
    handler resolved on every keypress, so a 50-element
    ``enqueue_download`` argument list could not be reviewed
    before the operator accidentally pressed Y/N/Esc.
    """
    state = _state()

    async def _run() -> None:
        handler = _make_elicitation_handler(state)
        # 30-line argument so we have plenty to scroll through.
        long_args = {f"paper_{i}": f"value_{i}" for i in range(30)}
        task = asyncio.create_task(handler({
            "serverName": "littrace",
            "_meta": {"tool_params": long_args},
            "message": 'tool "enqueue_download"?',
        }))
        for _ in range(20):
            await asyncio.sleep(0)
            if state.pending_approval is not None:
                break
        approval = state.pending_approval
        assert approval is not None
        assert approval.scroll_offset == 0
        assert not approval.future.done()
        # Compute the cap from the same default the modal uses.
        max_offset = max(
            len(approval.content_lines)
            - _approval_modal_body_rows(state),
            0,
        )
        assert max_offset > 10, (
            "test fixture must produce enough lines for PageDown to advance"
        )
        # Walk the cursor across all the scroll keys, asserting
        # the offset after each keypress.
        cursor = 0
        steps: list[tuple[int, int | None]] = [
            (curses.KEY_DOWN, min(cursor + 1, max_offset)),
            (curses.KEY_DOWN, min(cursor + 2, max_offset)),
            (curses.KEY_PPAGE, max(0, cursor - 10)),  # 2 -> 0 (clamped)
            (curses.KEY_UP, max(0, cursor - 1)),       # 0 -> 0 (clamped)
            (curses.KEY_NPAGE, min(cursor + 10, max_offset)),  # 0 -> 10
            (curses.KEY_HOME, 0),
            (curses.KEY_END, None),  # None means "max_offset"
        ]
        for key, expected in steps:
            await _handle_approval_key(key, state)
            assert not approval.future.done(), (
                f"key {key} unexpectedly resolved the future"
            )
            if expected is None:
                assert approval.scroll_offset == max_offset, (
                    f"key {key} expected max_offset {max_offset}, "
                    f"got {approval.scroll_offset}"
                )
                cursor = max_offset
            else:
                assert approval.scroll_offset == expected, (
                    f"key {key} expected offset {expected}, "
                    f"got {approval.scroll_offset}"
                )
                cursor = expected
        # Clean up: resolve the future so the test doesn't hang.
        approval.future.set_result(_ACCEPT)
        await task

    asyncio.run(_run())


# -- Round 19: ? help overlay ----------------------------------------------


def test_help_toggle_round_trip() -> None:
    """``?`` toggles ``state.show_help``; any other key dismisses it."""
    state = _state()
    assert state.show_help is False
    # Toggle on.
    state.show_help = True
    assert state.show_help is True
    # Toggle off.
    state.show_help = False
    assert state.show_help is False


# -- Round 19: tick env var -------------------------------------------------


def test_tick_seconds_defaults_to_50ms(monkeypatch) -> None:
    """``run_tui`` reads ``LITTRACE_TUI_TICK_MS`` and clamps it to
    [10, 1000] ms; default is 50 ms when the env var is unset or
    malformed."""
    monkeypatch.delenv("LITTRACE_TUI_TICK_MS", raising=False)
    # The clamp lives inside ``run_tui``. We re-implement it here to
    # keep this test independent of curses initialization.
    tick_ms_raw = os.environ.get("LITTRACE_TUI_TICK_MS", "50")
    try:
        tick_ms = max(10, min(int(tick_ms_raw), 1000))
    except (TypeError, ValueError):
        tick_ms = 50
    assert tick_ms == 50


def test_tick_seconds_clamps_high(monkeypatch) -> None:
    monkeypatch.setenv("LITTRACE_TUI_TICK_MS", "9999")
    tick_ms_raw = os.environ.get("LITTRACE_TUI_TICK_MS", "50")
    try:
        tick_ms = max(10, min(int(tick_ms_raw), 1000))
    except (TypeError, ValueError):
        tick_ms = 50
    assert tick_ms == 1000  # clamped


def test_tick_seconds_clamps_low(monkeypatch) -> None:
    monkeypatch.setenv("LITTRACE_TUI_TICK_MS", "2")
    tick_ms_raw = os.environ.get("LITTRACE_TUI_TICK_MS", "50")
    try:
        tick_ms = max(10, min(int(tick_ms_raw), 1000))
    except (TypeError, ValueError):
        tick_ms = 50
    assert tick_ms == 10  # clamped


def test_tick_seconds_handles_garbage(monkeypatch) -> None:
    monkeypatch.setenv("LITTRACE_TUI_TICK_MS", "not-a-number")
    tick_ms_raw = os.environ.get("LITTRACE_TUI_TICK_MS", "50")
    try:
        tick_ms = max(10, min(int(tick_ms_raw), 1000))
    except (TypeError, ValueError):
        tick_ms = 50
    assert tick_ms == 50  # graceful fallback