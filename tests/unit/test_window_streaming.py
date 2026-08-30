"""Unit tests for the Round 23 streaming UX.

The GUI now wires ``on_delta`` (per-token text) and ``on_phase``
``co (coarse state transitions) through ``handle_agent_chat`` into a
streaming bubble + status bar. These tests pin the contract without
spinning up a real Codex process:

  * ``_emit_phase`` swallows exceptions so a broken UI hook cannot
    poison the transport.
  * ``CodexAppServerChatService.chat`` accepts ``on_delta`` and
    ``on_phase`` kwargs and forwards them to the turn loop.
  * The GUI's streaming helpers create a single bubble and append
    text without leaking the placeholder.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Phase helper
# ---------------------------------------------------------------------------


def test_emit_phase_with_no_callback_is_noop() -> None:
    """``_emit_phase`` must be safe to call when no callback is set.

    Most code paths fire phases unconditionally; a ``None`` callback
    (e.g. from a CLI caller that doesn't care about progress) must
    not raise.
    """
    from littrace.codex_runtime.service import _emit_phase

    # Just verify it does not raise.
    _emit_phase(None, "authenticated")
    _emit_phase(None, "model_thinking")


def test_emit_phase_swallows_callback_exceptions() -> None:
    """A broken UI hook must NEVER poison the turn loop.

    Round 23 ships the phase hook into the hot path of
    ``CodexAppServerChatService._chat_with_client``. If a GUI callback
    raises (e.g. a Tk race during shutdown) the turn must still
    complete. We assert the exception is swallowed.
    """
    from littrace.codex_runtime.service import _emit_phase

    def boom(_phase: str) -> None:
        raise RuntimeError("GUI is gone")

    # Should NOT raise.
    _emit_phase(boom, "turn_completed")


# ---------------------------------------------------------------------------
# Service-level forwarding (chat signature)
# ---------------------------------------------------------------------------


def test_service_chat_signature_accepts_streaming_kwargs() -> None:
    """``CodexAppServerChatService.chat`` must accept ``on_delta`` and
    ``on_phase`` keyword args. A regression here would break every
    GUI/TUI caller that wants live updates.
    """
    import inspect

    from littrace.codex_runtime.service import CodexAppServerChatService

    sig = inspect.signature(CodexAppServerChatService.chat)
    assert "on_delta" in sig.parameters, "chat() must accept on_delta"
    assert "on_phase" in sig.parameters, "chat() must accept on_phase"
    # Both should default to None so legacy callers keep working.
    assert sig.parameters["on_delta"].default is None
    assert sig.parameters["on_phase"].default is None


def test_handle_agent_chat_signature_accepts_streaming_kwargs() -> None:
    """``handle_agent_chat`` must forward ``on_delta`` / ``on_phase``
    through to the service.
    """
    import inspect

    from littrace.agent_runtime import handle_agent_chat

    sig = inspect.signature(handle_agent_chat)
    assert "on_delta" in sig.parameters
    assert "on_phase" in sig.parameters
    assert sig.parameters["on_delta"].default is None
    assert sig.parameters["on_phase"].default is None


def test_handle_agent_chat_forwards_callbacks_to_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: when ``handle_agent_chat`` runs the App Server path,
    it MUST pass its ``on_delta`` / ``on_phase`` arguments through to
    ``CodexAppServerChatService.chat``. Otherwise the GUI would never
    receive a single delta.
    """
    from littrace.agent_runtime import handle_agent_chat
    from littrace.config import AgentRuntimeMode
    from littrace.intent import ChatIntent

    captured: dict[str, object] = {}

    class _RecordingService:
        def __init__(self, _config) -> None:
            pass

        async def chat(self, _request, workspace, _session, **_kwargs):
            # Capture everything we care about so the test can assert
            # the wiring without dictating argument order.
            captured.update(_kwargs)
            from littrace.models import ChatResponse

            return ChatResponse(reply="ok", action="ok"), workspace

    monkeypatch.setattr(
        "littrace.agent_runtime.CodexAppServerChatService",
        _RecordingService,
    )
    # Skip the legacy fallback so we hit the App Server branch
    # regardless of the agent_runtime.mode default.
    monkeypatch.setattr(
        "littrace.agent_runtime.parse_chat_intent",
        lambda _msg: ChatIntent(actions=["search"], topic="t"),
    )

    from littrace.config import LitTraceConfig

    cfg = LitTraceConfig()
    cfg.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER

    from littrace.models import ChatRequest, LiteratureWorkspace
    from littrace.session import ChatSession

    sentinel_delta = lambda _d: None
    sentinel_phase = lambda _p: None
    workspace = LiteratureWorkspace()
    session = MagicMock(spec=ChatSession)

    import asyncio

    asyncio.run(
        handle_agent_chat(
            ChatRequest(message="hi"),
            workspace,
            cfg,
            session=session,
            on_delta=sentinel_delta,
            on_phase=sentinel_phase,
        )
    )
    # The two callbacks must have been forwarded.
    assert captured.get("on_delta") is sentinel_delta
    assert captured.get("on_phase") is sentinel_phase


# ---------------------------------------------------------------------------
# GUI helpers — streaming bubble creation / finalisation
# ---------------------------------------------------------------------------


def _make_window_stub():
    import tkinter as tk
    from tkinter import ttk as _ttk

    from littrace.window import DESIGN, LitTraceWindow, _font

    root = tk.Tk()
    root.withdraw()
    win = LitTraceWindow.__new__(LitTraceWindow)
    win.root = root
    win.tk = tk
    win.ttk = _ttk
    win.style = _ttk.Style(root)
    win.chat_text = tk.Text(root, height=4, width=40)
    win.chat_text.configure(state="normal")
    win.status_var = tk.StringVar(value="")
    win.input_entry = tk.Text(root, height=2, width=40)
    win.input_entry.configure(state="normal")
    # Round 26: animation / debounce state. Initialised explicitly so
    # the new methods (which call ``getattr(self, "_input_md_after_id",
    # None)`` internally) behave the same on a stub as on a real
    # ``__init__``-built window.
    win._input_md_after_id = None
    win._status_anim_after_id = None
    win._status_dot_count = 0
    win._status_base_label = ""
    # Round 27: per-bubble action row state + Stop button anchoring.
    # Initialised to the same defaults ``__init__`` writes so the
    # stub behaves like a fully-built window.
    win._bubble_records = []
    # Round 28 (Phase 6): inline tool-card record list, parallel to
    # ``_bubble_records``. The session switcher drains both lists
    # before replaying history.
    win._tool_card_records = []
    win._last_user_message = None
    win._streaming_stop_event = None
    # Round 27 legacy stop-button attrs — kept for back-compat with
    # older tests that still reference them.
    win._stop_button_window = None
    win._stop_button = None
    win._stop_button_frame = None
    win._stop_button_line = None
    # Round 28 (Phase 4): the inline ``Pill.TButton`` stop pill.
    win._stop_pill = None
    win._stop_pill_index = None
    win._streaming_chars = 0
    # Register markdown tags so tests can probe tag ranges — both on
    # the input box (Round 26) and on the chat pane (Round 27).
    win._configure_input_tags()
    win._configure_chat_markdown_tags()
    # Round 27: also configure the chat-pane bubble tags (bubble_system
    # + status_pulse) and the ttk styles that the embedded action
    # Frames depend on. Real ``__init__`` wires this inside
    # ``_build_layout``; the stub skips that to stay headless.
    win.chat_text.tag_configure(
        "bubble_system",
        foreground=DESIGN["primary"],
        background=DESIGN["accent_teal_subtle"],
        font=_font("fine"),
        justify=tk.LEFT,
        lmargin1=14,
        lmargin2=14,
        rmargin=170,
        spacing1=8,
        spacing3=8,
    )
    win.chat_text.tag_configure(
        "status_pulse",
        foreground=DESIGN["accent_coral"],
        font=_font("body_strong"),
    )
    win.style.configure(
        "Secondary.TButton",
        background=DESIGN["pearl"],
        foreground=DESIGN["ink"],
        padding=(15, 8),
        font=_font("button_utility"),
    )
    win.style.configure(
        "Link.TButton",
        background=DESIGN["canvas"],
        foreground=DESIGN["primary"],
        borderwidth=0,
        relief=tk.FLAT,
        padding=(6, 2),
        font=_font("fine"),
    )
    return win, root


def test_begin_streaming_bubble_inserts_placeholder() -> None:
    """``_begin_streaming_bubble`` must leave a non-empty region that
    ``_append_streaming_delta`` can fill. The cursor must be parked
    at ``_streaming_mark``.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._begin_streaming_bubble()
        # The mark must point at a valid Tk index.
        assert win._streaming_mark is not None
        assert win._streaming_mark != ""
        # The bubble content must be at least the trailing padding
        # (two newlines) plus the cursor.
        end_index = win.chat_text.index(tk.END)
        assert end_index != "1.0"
    finally:
        root.destroy()


def test_append_streaming_delta_appends_text_into_bubble() -> None:
    """``_append_streaming_delta`` must insert text near the streaming
    mark and bump the char counter.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._begin_streaming_bubble()
        win._append_streaming_delta("Hello ")
        win._append_streaming_delta("world!")
        # Finalise to read the bubble body back.
        win._finalize_streaming_bubble("Hello world!")
        body = win.chat_text.get("1.0", tk.END)
        assert "Hello world!" in body
        # The streaming mark must be cleared after finalisation so a
        # subsequent turn starts fresh.
        assert not hasattr(win, "_streaming_mark")
    finally:
        root.destroy()


def test_finalize_streaming_bubble_with_none_shows_placeholder() -> None:
    """If the final reply is empty (e.g. early_response with no deltas)
    we still show ``（无回复）`` so the user doesn't see a stranded
    bubble that never resolved.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._begin_streaming_bubble()
        win._finalize_streaming_bubble(None)
        body = win.chat_text.get("1.0", tk.END)
        assert "（无回复）" in body
    finally:
        root.destroy()


def test_phase_labels_cover_all_emitted_ids() -> None:
    """The GUI must know how to render every phase id the service
    fires. If a new id is added to ``_emit_phase`` calls without a
    matching label, the status bar silently keeps its old text — this
    test catches that gap.
    """
    from littrace.window import LitTraceWindow

    declared_ids = {
        "authenticated",
        "thread_started",
        "thread_resumed",
        "mcp_ready",
        "model_thinking",
        "turn_completed",
    }
    # Every id the service emits must have a label entry. The union
    # of declared ids and labels must agree (modulo labels we might
    # add for purely-cosmetic reasons — none today).
    missing = declared_ids - set(LitTraceWindow._PHASE_LABELS)
    assert not missing, f"Missing GUI phase labels for: {missing}"
    extra = set(LitTraceWindow._PHASE_LABELS) - declared_ids
    assert not extra, f"GUI phase labels with no emitter: {extra}"


# ---------------------------------------------------------------------------
# Round 25: Codex status bubble lives inside the chat area
# ---------------------------------------------------------------------------


def test_set_status_in_chat_inserts_bubble_on_first_call() -> None:
    """First call to ``_set_status_in_chat`` must insert a bubble
    containing the supplied label.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._set_status_in_chat("Codex 已认证")
        body = win.chat_text.get("1.0", tk.END)
        assert "Codex 已认证" in body
        # The bubble range must be recorded so subsequent calls can
        # replace it instead of stacking duplicate bubbles.
        assert hasattr(win, "_status_bubble_start")
        assert hasattr(win, "_status_bubble_end")
    finally:
        root.destroy()


def test_set_status_in_chat_replaces_previous_bubble() -> None:
    """Calling ``_set_status_in_chat`` twice must REPLACE the first
    bubble — never append a second one. Round 25 explicit goal: only
    one Codex status bubble visible at any time.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._set_status_in_chat("Codex 已认证")
        win._set_status_in_chat("Codex thread 已建立")
        win._set_status_in_chat("Codex 思考中…")
        body = win.chat_text.get("1.0", tk.END)
        # Only the LATEST label should remain; the earlier labels were
        # overwritten in place, not appended.
        assert "Codex 思考中…" in body
        assert "Codex 已认证" not in body
        assert "Codex thread 已建立" not in body
        # Count occurrences of "Codex " — must be exactly one.
        assert body.count("Codex ") == 1
    finally:
        root.destroy()


def test_clear_status_in_chat_removes_bubble() -> None:
    """``_clear_status_in_chat`` must drop the bubble and reset the
    start/end marks so the next turn starts fresh.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._set_status_in_chat("Codex 已认证")
        win._clear_status_in_chat()
        body = win.chat_text.get("1.0", tk.END)
        assert "Codex 已认证" not in body
        assert not hasattr(win, "_status_bubble_start")
        assert not hasattr(win, "_status_bubble_end")
        # Calling clear when no bubble exists must be a no-op.
        win._clear_status_in_chat()
    finally:
        root.destroy()


def test_send_clears_status_bubble_for_new_turn() -> None:
    """``_send`` must drop the previous turn's Codex status bubble
    before kicking off the next turn so the chat doesn't show stale
    state.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    # Stub the heavy bits ``_send`` touches so we can isolate the
    # clear-status behaviour without spinning up a real Codex run.
    win._append_message = lambda *_args, **_kwargs: None
    win._show_execution_path = lambda *_args, **_kwargs: None
    win.input_entry.delete("1.0", tk.END)
    win.input_entry.insert("1.0", "hi")
    threading = __import__("threading")
    started = []

    class _NoopThread:
        def __init__(self, target=None, args=(), daemon=False):
            started.append(target)

        def start(self):
            pass

    win.threading = threading
    win.threading.Thread = _NoopThread  # type: ignore[attr-defined]

    try:
        # Simulate a finished previous turn with a stale status bubble.
        win._set_status_in_chat("Codex 回复完成 · search")
        assert "Codex 回复完成" in win.chat_text.get("1.0", tk.END)
        # Sending a new message must clear it.
        win._send()
        body = win.chat_text.get("1.0", tk.END)
        assert "Codex 回复完成" not in body
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# Round 26: CJK font fallback, live markdown rendering, status animation
# ---------------------------------------------------------------------------


def test_font_recipes_include_cjk_fallback() -> None:
    """Round 26: the ``sans`` family chain must lead with a CJK font
    so pinyin IME composition text and committed Chinese characters
    render at the same metrics. The new recipes (``mono``, ``h1``,
    ``h2``, ``h3``, ``body_italic``) must all resolve to a non-None
    Font object.
    """
    import tkinter as tk

    from littrace.window import _font

    # _font() builds ``tkfont.Font`` instances internally — that needs
    # a default Tk root. Create it before invoking _font().
    root = tk.Tk()
    root.withdraw()
    try:
        for token in ("body", "mono", "h1", "h2", "h3", "body_italic"):
            font_obj = _font(token)
            assert font_obj is not None, f"_font({token!r}) returned None"
        body_font = _font("body")
        # Tk resolves the family chain to the first available face;
        # ``actual('family')`` returns what the platform is using.
        actual = body_font.actual("family")
        assert isinstance(actual, str)
    finally:
        root.destroy()


def test_input_entry_has_markdown_tags_configured() -> None:
    """Round 26: every ``md_*`` tag must be registered after
    ``_configure_input_tags()`` so the renderer can apply them.
    """
    import tkinter as tk

    from littrace.window import _INPUT_MD_TAGS

    win, root = _make_window_stub()
    try:
        names = set(win.input_entry.tag_names())
        for tag in _INPUT_MD_TAGS:
            assert tag in names, f"Missing markdown tag {tag!r}"
    finally:
        root.destroy()


def test_render_input_markdown_bold_range() -> None:
    """Round 26: ``**world**`` in the buffer must produce a single
    ``md_bold`` tag covering the inner ``world`` range.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win.input_entry.delete("1.0", tk.END)
        win.input_entry.insert("1.0", "hi **world** ok")
        win._render_input_markdown()
        ranges = win.input_entry.tag_ranges("md_bold")
        # Tk stores tag ranges as alternating (start, end) indices;
        # we only care about the first pair for a single bold span.
        assert len(ranges) >= 2, f"expected at least one bold range, got {ranges}"
        start, end = str(ranges[0]), str(ranges[1])
        # Tk indices for ``hi **world** ok`` (1-indexed lines):
        #   ``hi ``     ends at column 3 (exclusive) -> the first ``*`` sits at "1.3"
        #   ``world``   starts at column 5 -> "1.5", ends at column 10 -> "1.10"
        # (the regex strips the surrounding ``**`` markers; group(1) is
        # the inner text only).
        assert start == "1.5", f"bold start was {start}, expected 1.5"
        assert end == "1.10", f"bold end was {end}, expected 1.10"
    finally:
        root.destroy()


def test_render_input_markdown_inline_code() -> None:
    """Round 26: `` `pip install` `` must produce an ``md_code`` tag
    covering just ``pip install`` (group 1 of the inline-code regex).
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win.input_entry.delete("1.0", tk.END)
        win.input_entry.insert("1.0", "use `pip install littrace` here")
        win._render_input_markdown()
        ranges = win.input_entry.tag_ranges("md_code")
        assert len(ranges) >= 2, f"expected at least one code range, got {ranges}"
        start, end = str(ranges[0]), str(ranges[1])
        # ``use ``     is 4 chars (columns 0..3), then opening backtick at col 4.
        # ``pip install littrace`` is 20 chars (columns 5..24), so group(1)
        # spans columns 5..25 (end exclusive).
        assert start == "1.5", f"code start was {start}, expected 1.5"
        assert end == "1.25", f"code end was {end}, expected 1.25"
    finally:
        root.destroy()


def test_render_input_markdown_headings() -> None:
    """Round 26: ``#``, ``##``, ``###`` prefixes must each produce
    the corresponding ``md_h1`` / ``md_h2`` / ``md_h3`` tag range.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win.input_entry.delete("1.0", tk.END)
        win.input_entry.insert("1.0", "# A\n## B\n### C")
        win._render_input_markdown()
        h1 = win.input_entry.tag_ranges("md_h1")
        h2 = win.input_entry.tag_ranges("md_h2")
        h3 = win.input_entry.tag_ranges("md_h3")
        assert len(h1) >= 2, f"md_h1 ranges: {h1}"
        assert len(h2) >= 2, f"md_h2 ranges: {h2}"
        assert len(h3) >= 2, f"md_h3 ranges: {h3}"
    finally:
        root.destroy()


def test_render_input_markdown_idempotent() -> None:
    """Round 26: calling the renderer twice on the same buffer must
    produce identical tag ranges — the strip pass at the top of every
    call guarantees this.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win.input_entry.delete("1.0", tk.END)
        win.input_entry.insert("1.0", "# Title\n**bold** and `code`")
        win._render_input_markdown()
        h1_first = list(win.input_entry.tag_ranges("md_h1"))
        bold_first = list(win.input_entry.tag_ranges("md_bold"))
        code_first = list(win.input_entry.tag_ranges("md_code"))
        # Second render must not double the tag ranges.
        win._render_input_markdown()
        assert list(win.input_entry.tag_ranges("md_h1")) == h1_first
        assert list(win.input_entry.tag_ranges("md_bold")) == bold_first
        assert list(win.input_entry.tag_ranges("md_code")) == code_first
    finally:
        root.destroy()


def test_render_input_markdown_preserves_text() -> None:
    """Round 26: the renderer only manipulates tags, never text. The
    buffer must be byte-identical before and after a render — the
    model still receives the raw ``**bold**`` markdown verbatim.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win.input_entry.delete("1.0", tk.END)
        sample = "# Title\nhello **world** and `code`\n- list item\n"
        win.input_entry.insert("1.0", sample)
        before = win.input_entry.get("1.0", tk.END)
        win._render_input_markdown()
        after = win.input_entry.get("1.0", tk.END)
        assert before == after, (
            f"render mutated text:\nbefore={before!r}\nafter={after!r}"
        )
    finally:
        root.destroy()


def test_status_animation_starts_on_thinking_label() -> None:
    """Round 26: ``_set_status_in_chat`` with an animatable label
    must arm the trailing-dot cycle (set base + schedule a tick).
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._set_status_in_chat("Codex 思考中")
        assert win._status_base_label == "Codex 思考中"
        assert win._status_anim_after_id is not None, (
            "expected animation tick to be scheduled"
        )
    finally:
        root.destroy()


def test_status_animation_does_not_start_on_completed_label() -> None:
    """Round 26: a non-animatable label (e.g. ``Codex 已认证``) must
    NOT start the cycle. The previous tick (if any) is cancelled.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        # Pre-arm a fake tick so we can prove the call clears it.
        win._status_anim_after_id = "fake-id"
        win._set_status_in_chat("Codex 已认证")
        assert win._status_anim_after_id is None, (
            "non-animatable label should not schedule a tick"
        )
        assert win._status_base_label == ""
    finally:
        root.destroy()


def test_status_animation_stops_on_clear() -> None:
    """Round 26: ``_clear_status_in_chat`` must cancel any pending
    tick so the timer never fires against a deleted bubble range.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._set_status_in_chat("Codex 思考中")
        assert win._status_anim_after_id is not None
        win._clear_status_in_chat()
        assert win._status_anim_after_id is None
        assert win._status_base_label == ""
    finally:
        root.destroy()


def test_status_animation_step_increments_dots() -> None:
    """Round 26: ``_animate_status_bubble_step`` bumps the dot count
    and the bubble body must reflect the new suffix.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._set_status_in_chat("Codex 思考中")
        win._cancel_status_animation()  # stop the auto-schedule
        # Manually invoke two ticks (dot 0 -> 1 -> 2).
        win._animate_status_bubble_step()
        win._cancel_status_animation()
        win._animate_status_bubble_step()
        win._cancel_status_animation()
        assert win._status_dot_count == 2, (
            f"dot count was {win._status_dot_count}, expected 2"
        )
        body = win.chat_text.get("1.0", tk.END)
        assert "思考中.." in body, f"bubble missing dot suffix:\n{body!r}"
    finally:
        root.destroy()


def test_status_animation_re_entry_does_not_double_schedule() -> None:
    """Round 26: a fresh ``_set_status_in_chat`` of an animatable
    label must cancel any in-flight tick (via ``_start_status_animation``
    -> ``_cancel_status_animation``) before scheduling a new one.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._status_anim_after_id = "fake-stale-id"
        win._set_status_in_chat("Codex 思考中")
        # The fake id must have been cancelled; a fresh id scheduled.
        assert win._status_anim_after_id is not None
        assert win._status_anim_after_id != "fake-stale-id"
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# Round 27 — Tier A UX overhaul
#   Fix 1: assistant markdown rendering (render-on-finalize)
#   Fix 2: status bubble visual + 200ms animation tick
#   Fix 4: streaming Stop button (threading.Event bridge)
#   Fix 5: per-bubble Copy / 重新生成 / 编辑 actions
#   Fix 6: adaptive input height (1..8 lines)
# ---------------------------------------------------------------------------


def test_chat_markdown_tags_configured() -> None:
    """Fix 1: every ``md_*`` tag must be registered on ``chat_text``
    after ``_configure_chat_markdown_tags()`` runs.
    """
    import tkinter as tk

    from littrace.window import _CHAT_MD_TAGS

    win, root = _make_window_stub()
    try:
        names = set(win.chat_text.tag_names())
        for tag in _CHAT_MD_TAGS:
            assert tag in names, f"Missing chat markdown tag {tag!r}"
    finally:
        root.destroy()


def test_render_chat_markdown_bold_range() -> None:
    """Fix 1: assistant reply ``hi **world** ok`` must produce a
    single ``md_bold`` tag covering just ``world``.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win.chat_text.delete("1.0", tk.END)
        win.chat_text.insert("1.0", "hi **world** ok")
        pre = win.chat_text.index("1.0")
        win._render_chat_markdown(pre, "hi **world** ok")
        ranges = win.chat_text.tag_ranges("md_bold")
        assert len(ranges) >= 2, f"expected at least one bold range, got {ranges}"
        start, end = str(ranges[0]), str(ranges[1])
        # ``hi `` ends at col 3 (exclusive); ``world`` is cols 5..10.
        assert start == "1.5", f"bold start was {start}, expected 1.5"
        assert end == "1.10", f"bold end was {end}, expected 1.10"
    finally:
        root.destroy()


def test_render_chat_markdown_inline_code() -> None:
    """Fix 1: backtick code spans render with the ``md_code`` tag
    over just the inner token.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win.chat_text.delete("1.0", tk.END)
        win.chat_text.insert("1.0", "use `pip install littrace` here")
        pre = win.chat_text.index("1.0")
        win._render_chat_markdown(pre, "use `pip install littrace` here")
        ranges = win.chat_text.tag_ranges("md_code")
        assert len(ranges) >= 2, f"expected code range, got {ranges}"
        start, end = str(ranges[0]), str(ranges[1])
        # ``use `` is 4 chars; ``pip install littrace`` is 20 chars.
        assert start == "1.5", f"code start was {start}, expected 1.5"
        assert end == "1.25", f"code end was {end}, expected 1.25"
    finally:
        root.destroy()


def test_render_chat_markdown_headings() -> None:
    """Fix 1: ``#``, ``##``, ``###`` prefixes render as
    ``md_h1`` / ``md_h2`` / ``md_h3``.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win.chat_text.delete("1.0", tk.END)
        win.chat_text.insert("1.0", "# A\n## B\n### C")
        pre = win.chat_text.index("1.0")
        win._render_chat_markdown(pre, "# A\n## B\n### C")
        h1 = win.chat_text.tag_ranges("md_h1")
        h2 = win.chat_text.tag_ranges("md_h2")
        h3 = win.chat_text.tag_ranges("md_h3")
        assert len(h1) >= 2, f"md_h1 ranges: {h1}"
        assert len(h2) >= 2, f"md_h2 ranges: {h2}"
        assert len(h3) >= 2, f"md_h3 ranges: {h3}"
    finally:
        root.destroy()


def test_render_chat_markdown_preserves_text() -> None:
    """Fix 1: render must not mutate buffer text — the model
    output stays verbatim even after markdown styling lands.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win.chat_text.delete("1.0", tk.END)
        sample = "# Title\nhello **world** and `code`\n- list item\n"
        win.chat_text.insert("1.0", sample)
        pre = win.chat_text.index("1.0")
        before = win.chat_text.get("1.0", tk.END)
        win._render_chat_markdown(pre, sample.rstrip("\n"))
        after = win.chat_text.get("1.0", tk.END)
        assert before == after, (
            f"render mutated text:\nbefore={before!r}\nafter={after!r}"
        )
    finally:
        root.destroy()


def test_render_chat_markdown_idempotent() -> None:
    """Fix 1: second render on the same region must not double
    the tag ranges — the strip pass at the top of every call
    guarantees a stable render.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win.chat_text.delete("1.0", tk.END)
        win.chat_text.insert("1.0", "# Title\n**bold** and `code`")
        pre = win.chat_text.index("1.0")
        win._render_chat_markdown(pre, "# Title\n**bold** and `code`")
        h1_first = list(win.chat_text.tag_ranges("md_h1"))
        bold_first = list(win.chat_text.tag_ranges("md_bold"))
        code_first = list(win.chat_text.tag_ranges("md_code"))
        win._render_chat_markdown(pre, "# Title\n**bold** and `code`")
        assert list(win.chat_text.tag_ranges("md_h1")) == h1_first
        assert list(win.chat_text.tag_ranges("md_bold")) == bold_first
        assert list(win.chat_text.tag_ranges("md_code")) == code_first
    finally:
        root.destroy()


def test_status_bubble_visual_upgrade() -> None:
    """Fix 2: ``bubble_system`` tag must use the new teal-on-canvas
    palette so it reads as an active pill, not a footer.

    We assert the foreground colour was switched from the muted ink
    to the primary token, and the background to the teal subtle.
    """
    import tkinter as tk

    from littrace.window import DESIGN

    win, root = _make_window_stub()
    try:
        win._set_status_in_chat("Codex 思考中")
        fg = win.chat_text.tag_cget("bubble_system", "foreground")
        bg = win.chat_text.tag_cget("bubble_system", "background")
        assert fg == DESIGN["primary"], (
            f"bubble_system foreground was {fg}, expected primary {DESIGN['primary']}"
        )
        assert bg == DESIGN["accent_teal_subtle"], (
            f"bubble_system background was {bg}, expected teal {DESIGN['accent_teal_subtle']}"
        )
    finally:
        root.destroy()


def test_status_animation_interval_is_200ms() -> None:
    """Fix 3: the trailing-dot tick interval must be 200 ms — double
    the speed of the previous 400 ms cycle.
    """
    from littrace.window import _STATUS_ANIM_INTERVAL_MS

    assert _STATUS_ANIM_INTERVAL_MS == 200


def test_stop_event_is_thread_event() -> None:
    """Fix 4: ``_streaming_stop_event`` is a ``threading.Event`` so
    the Tk click handler can flip it from the GUI thread without
    contending with the asyncio reader thread.
    """
    import threading

    import tkinter as tk

    win, root = _make_window_stub()
    try:
        # ``_handle_message_thread`` is what creates the Event; on a
        # stub we exercise the same one-line assignment manually.
        win._streaming_stop_event = threading.Event()
        assert isinstance(win._streaming_stop_event, threading.Event)
        assert not win._streaming_stop_event.is_set()
    finally:
        root.destroy()


def test_request_stop_sets_event() -> None:
    """Fix 4: ``_request_stop_streaming`` must flip the
    ``threading.Event`` flag the asyncio loop polls between deltas.
    """
    import threading
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        # We need ``_begin_streaming_bubble`` to have run so the Stop
        # button exists; this also drops a real ``threading.Event``
        # into ``_streaming_stop_event`` — exactly the same way
        # ``_handle_message_thread`` does on a real turn.
        win._streaming_stop_event = threading.Event()
        win._begin_streaming_bubble()
        assert not win._streaming_stop_event.is_set()
        win._request_stop_streaming()
        assert win._streaming_stop_event.is_set()
    finally:
        root.destroy()


def test_stop_button_destroyed_on_finalize() -> None:
    """Fix 4: the embedded Stop button must be torn down when
    ``_finalize_streaming_bubble`` runs — otherwise it lingers
    next to the next turn's bubble.

    Round 28: the Stop button is now the inline ``Pill.TButton`` —
    the test pivots to the new ``_stop_pill`` / ``_stop_pill_index``
    attributes. The legacy ``_stop_button_frame`` / ``_stop_button_line``
    attributes are no longer populated by ``_begin_streaming_bubble``.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._begin_streaming_bubble()
        # The Pill is a direct child of chat_text; we track the
        # Button widget + the line index of the embedded window slot.
        assert win._stop_pill is not None
        assert win._stop_pill_index is not None
        # Simulate the canonical end-of-turn path: append a delta,
        # then finalise.
        win._append_streaming_delta("partial reply")
        win._finalize_streaming_bubble("full reply")
        # After finalize the Pill widget and line index are cleared.
        assert win._stop_pill is None
        assert win._stop_pill_index is None
    finally:
        root.destroy()


def test_bubble_actions_attached_after_append() -> None:
    """Fix 5: ``_append_message`` for a user/assistant bubble must
    append a record to ``_bubble_records`` so the action Frame is
    tracked alongside the bubble body.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._bubble_records = []
        win._append_message("assistant", "hello")
        assert len(win._bubble_records) == 1, (
            f"expected 1 bubble record, got {len(win._bubble_records)}"
        )
        record = win._bubble_records[0]
        assert record["role"] == "assistant"
        assert record["text"] == "hello"
        # The record must hold both the Frame widget and the line
        # index of its embedded window slot — these are what the
        # session switcher needs to clean up.
        assert record["frame"] is not None
        assert record["line_index"] is not None
    finally:
        root.destroy()


def test_regenerate_calls_send_with_last_message() -> None:
    """Fix 5: 重新生成 must replay ``_last_user_message`` via
    ``_send_with_text`` so the same Codex call is re-driven end to
    end (clear input, append user bubble, spawn worker thread).
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        win._last_user_message = "find me transformer papers"
        # Stub the worker-spawning bits so ``_send_with_text`` does
        # not block on a real Codex call.
        win._append_message = lambda *_a, **_k: None
        win._show_execution_path = lambda *_a, **_k: None
        win._clear_status_in_chat = lambda: None
        # ``_send_with_text`` calls ``input_entry.delete`` which we
        # want to be a no-op so the stub state stays clean.
        win.input_entry.delete("1.0", tk.END)
        sent = []
        win._send_with_text = lambda text: sent.append(text)  # type: ignore[method-assign]
        # No active streaming -> the guard passes.
        win._regenerate_last()
        assert sent == ["find me transformer papers"]
    finally:
        root.destroy()


def test_input_height_adapts_to_lines() -> None:
    """Fix 6: ``_resize_input_to_content`` must grow ``input_entry``
    to fit content up to 8 lines, then reset to 1 when empty.
    """
    import tkinter as tk

    win, root = _make_window_stub()
    try:
        # 1 line: stays at 1.
        win.input_entry.delete("1.0", tk.END)
        win.input_entry.insert("1.0", "single line")
        win._resize_input_to_content()
        assert int(win.input_entry.cget("height")) == 1, (
            f"height for 1 line was {win.input_entry.cget('height')}"
        )
        # 5 newlines -> 5 visual lines.
        win.input_entry.delete("1.0", tk.END)
        win.input_entry.insert("1.0", "line1\nline2\nline3\nline4\nline5")
        win._resize_input_to_content()
        assert int(win.input_entry.cget("height")) == 5, (
            f"height for 5 lines was {win.input_entry.cget('height')}"
        )
        # Empty -> 1.
        win.input_entry.delete("1.0", tk.END)
        win._resize_input_to_content()
        assert int(win.input_entry.cget("height")) == 1, (
            f"height for empty was {win.input_entry.cget('height')}"
        )
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# Round 28 — Codex Light visual overhaul. The following 15 tests lock the
# design-token palette, font recipes, flat-message avatar render, inline
# Stop pill, and tool-card subsystem. They are deliberately small and
# synchronous so the probe script can mirror them headlessly.
# ---------------------------------------------------------------------------


def test_design_tokens_include_codex_palette() -> None:
    """Round 28: ``DESIGN`` exposes the Codex Light palette keys.

    Anchored values: warm-white canvas ``#faf9f5``, dark ink ``#1f1f1f``,
    hairline ``#e8e6e0``, teal primary ``#0f7a7b``. Plus the 6 Round 28
    keys for avatars / pill / tool card.
    """
    from littrace.window import DESIGN

    required = {
        "primary": "#0f7a7b",
        "ink": "#1f1f1f",
        "canvas": "#faf9f5",
        "hairline": "#e8e6e0",
        "avatar_bg": "#1f1f1f",
        "user_avatar_bg": "#e6efee",
        "pill_bg": "#0f7a7b",
        "pill_fg": "#ffffff",
        "tool_card_bg": "#f7f6f1",
        "tool_card_border": "#e8e6e0",
    }
    for key, expected in required.items():
        assert key in DESIGN, f"missing DESIGN key: {key}"
        assert DESIGN[key] == expected, (
            f"DESIGN[{key!r}]={DESIGN[key]!r}, expected {expected!r}"
        )


def test_font_recipe_avatar_resolves() -> None:
    """Round 28: the new ``avatar`` / ``tool_name`` / ``tool_meta`` /
    ``timestamp`` recipes resolve to ``tk.Font`` instances.
    """
    from littrace.window import _font, _reset_font_cache

    # ``_font`` needs a Tk root; borrow the stub window.
    win, root = _make_window_stub()
    try:
        # Pre-existing tests may have cached fonts against a destroyed
        # root — reset so this test sees a fresh, realized Font.
        _reset_font_cache()
        for recipe in ("avatar", "tool_name", "tool_meta", "timestamp"):
            font = _font(recipe)
            assert font is not None, f"_font({recipe!r}) returned None"
    finally:
        root.destroy()


def test_font_recipe_body_size_is_15() -> None:
    """Round 28: body recipe is 15pt (Codex desktop reading size)."""
    from littrace.window import _font, _reset_font_cache

    win, root = _make_window_stub()
    try:
        _reset_font_cache()
        body = _font("body")
        # tk may report size as int or str depending on platform.
        assert body.cget("size") in (15, "15"), (
            f"body size was {body.cget('size')!r}"
        )
    finally:
        root.destroy()


def test_append_message_renders_avatar_for_assistant() -> None:
    """Round 28: assistant messages get a flat-message avatar header."""
    win, root = _make_window_stub()
    try:
        win._bubble_records = []
        win._append_message("assistant", "hello")
        text = win.chat_text.get("1.0", __import__("tkinter").END)
        assert "LitTrace" in text, f"role label missing: {text!r}"
        # Bubble record tracks the action Frame.
        assert len(win._bubble_records) == 1
    finally:
        root.destroy()


def test_append_message_renders_avatar_for_user() -> None:
    """Round 28: user messages get a flat-message avatar header."""
    win, root = _make_window_stub()
    try:
        win._bubble_records = []
        win._append_message("user", "hi")
        text = win.chat_text.get("1.0", __import__("tkinter").END)
        assert "你" in text, f"role label missing: {text!r}"
        assert len(win._bubble_records) == 1
    finally:
        root.destroy()


def test_status_indicator_replaces_in_place() -> None:
    """Round 28: status indicator replaces the previous line in place.

    Codex-style: the inline avatar + label sits at the end of the
    chat timeline. Two consecutive calls must NOT grow the timeline
    by a full line — the new label rewrites the existing one.
    """
    win, root = _make_window_stub()
    try:
        win._set_status_in_chat("Codex 思考中")
        lines_after_first = int(
            win.chat_text.index("end-1c").split(".")[0]
        )
        win._set_status_in_chat("Codex 正在搜索…")
        lines_after_second = int(
            win.chat_text.index("end-1c").split(".")[0]
        )
        # The second call replaces in place; line count must not grow.
        assert lines_after_second <= lines_after_first + 1, (
            f"status indicator grew by {lines_after_second - lines_after_first} lines"
        )
    finally:
        root.destroy()


def test_status_indicator_clears_on_clear() -> None:
    """Round 28: ``_clear_status_in_chat`` removes the trailing indicator."""
    win, root = _make_window_stub()
    try:
        win._set_status_in_chat("Codex 思考中")
        before = int(win.chat_text.index("end-1c").split(".")[0])
        win._clear_status_in_chat()
        after = int(win.chat_text.index("end-1c").split(".")[0])
        assert after <= before, (
            f"status indicator not cleared: {before} -> {after}"
        )
    finally:
        root.destroy()


def test_relative_timestamp_formatter() -> None:
    """Round 28: ``_format_relative_time`` walks the boundary ladder."""
    from datetime import datetime, timedelta, timezone

    win, root = _make_window_stub()
    try:
        now = datetime.now(timezone.utc)
        cases = [
            (now, "Just now"),
            (now - timedelta(minutes=3), "3m ago"),
            (now - timedelta(hours=2), "2h ago"),
        ]
        for stamp, expected in cases:
            got = win._format_relative_time(stamp.isoformat())
            assert got == expected, (
                f"{stamp.isoformat()}: expected {expected!r}, got {got!r}"
            )
    finally:
        root.destroy()


def test_stop_pill_inline_anchor() -> None:
    """Round 28: ``_begin_streaming_bubble`` parks a ``Pill.TButton``
    at ``_stop_pill_index`` so deltas land AFTER the pill."""
    win, root = _make_window_stub()
    try:
        win._streaming_stop_event = __import__("threading").Event()
        win._begin_streaming_bubble()
        assert win._stop_pill is not None
        assert win._stop_pill_index is not None
        # The streaming start mark is on a NEW line below the pill —
        # deltas append to it, not to the pill.
        start = win._streaming_start_mark
        pill = win._stop_pill_index
        assert start != pill
    finally:
        root.destroy()


def test_destroy_stop_pill_drops_window_slot() -> None:
    """Round 28: ``_destroy_stop_button_window`` (which is what
    ``_finalize_streaming_bubble`` calls) tears down the Pill widget
    AND deletes the line that held its Tk window slot."""
    win, root = _make_window_stub()
    try:
        win._streaming_stop_event = __import__("threading").Event()
        win._begin_streaming_bubble()
        assert win._stop_pill is not None
        pill_line = win._stop_pill_index
        win._destroy_stop_button_window()
        assert win._stop_pill is None
        assert win._stop_pill_index is None
    finally:
        root.destroy()


def test_card_helper_returns_ttk_frame() -> None:
    """Round 28: ``_render_tool_card`` returns a tracked ToolCard Frame
    on the chat text buffer."""
    win, root = _make_window_stub()
    try:
        before = len(win._tool_card_records)
        win._render_tool_card(
            tool_name="search_papers",
            status="success",
            duration=0.1,
            body='{"papers": []}',
        )
        assert len(win._tool_card_records) == before + 1
        rec = win._tool_card_records[-1]
        assert rec["frame"] is not None
        assert rec["name"] == "search_papers"
        assert rec["status"] == "success"
    finally:
        root.destroy()


def test_render_tool_card_failed_status_expands_body() -> None:
    """Round 28: failed tool cards auto-expand the body so the user
    sees the error context without an extra click."""
    win, root = _make_window_stub()
    try:
        win._tool_card_records.clear()
        win._render_tool_card(
            tool_name="enqueue_download",
            status="failed",
            duration=1.5,
            body='{"error": "queue full"}',
        )
        rec = win._tool_card_records[-1]
        # Headless: ``winfo_manager()`` reports "pack" if the body is
        # attached to the geometry manager — without a visible parent
        # ``winfo_ismapped`` returns False even when packed.
        assert rec["body_text"].winfo_manager() == "pack", (
            f"failed body should be packed, manager={rec['body_text'].winfo_manager()!r}"
        )
    finally:
        root.destroy()


def test_toggle_tool_card_body_collapses_and_expands() -> None:
    """Round 28: success card body is collapsed by default; toggle
    via the chevron packs/unpacks it."""
    win, root = _make_window_stub()
    try:
        win._tool_card_records.clear()
        win._render_tool_card(
            tool_name="search_papers",
            status="success",
            duration=0.2,
            body='{"papers": ["a", "b"]}',
        )
        rec = win._tool_card_records[-1]
        # Collapsed by default — manager is "" (unmapped).
        assert rec["body_text"].winfo_manager() == "", (
            f"success body should be collapsed: {rec['body_text'].winfo_manager()!r}"
        )
    finally:
        root.destroy()


def test_agent_runtime_forwards_on_tool_kwarg() -> None:
    """Round 28 (Phase 6): ``handle_agent_chat`` forwards ``on_tool``
    to the underlying service so tool events emitted by the Codex App
    Server land on the GUI's ``on_tool`` callback."""
    import asyncio
    import tempfile
    from pathlib import Path

    import littrace.agent_runtime as runtime_mod
    from littrace.config import AgentRuntimeMode, LitTraceConfig
    from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
    from littrace.session import ChatSession

    captured: dict = {}

    class _CapturingService:
        def __init__(self, _config) -> None:
            pass

        async def chat(self, *_args, **kwargs):
            captured.update(kwargs)
            return ChatResponse(reply="ok", action="codex"), LiteratureWorkspace()

    cfg = LitTraceConfig()
    cfg.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    runtime_mod.CodexAppServerChatService = _CapturingService

    def _on_tool(name, status, duration, body):
        pass

    with tempfile.TemporaryDirectory() as tmp:
        session = ChatSession.from_root(
            Path(tmp) / "session", "session-1", config=cfg,
        )
        asyncio.run(
            runtime_mod.handle_agent_chat(
                ChatRequest(message="hi"),
                LiteratureWorkspace(),
                cfg,
                session=session,
                on_tool=_on_tool,
            )
        )

    assert "on_tool" in captured, f"on_tool not forwarded; kwargs={list(captured)}"
    assert captured["on_tool"] is _on_tool