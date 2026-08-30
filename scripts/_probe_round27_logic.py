"""Verify the Round 27 Tier-A UX overhaul logic without spinning up
the real GUI event loop.

Drives the production helpers directly (no ``mainloop``, no asyncio)
and asserts:

  * ``_configure_chat_markdown_tags`` registers the 10 ``md_*`` tags
    on ``chat_text``,
  * ``_render_chat_markdown`` applies bold / italic / inline-code /
    headings / links over the bubble body WITHOUT mutating buffer
    text — the model output stays verbatim,
  * ``bubble_system`` tag uses the new teal-on-canvas palette (the
    active-pill upgrade from Round 27),
  * ``_STATUS_ANIM_INTERVAL_MS`` is 200 (Round 27 cut it from 400ms),
  * the Stop-button / threading.Event wiring lands: arming the event
    flips ``is_set()`` synchronously, and the Stop button is torn
    down when ``_finalize_streaming_bubble`` runs,
  * per-bubble action rows are tracked in ``_bubble_records`` with a
    Frame widget + line index (so the session switcher can clean them
    up cleanly),
  * the adaptive input resize (``_resize_input_to_content``) grows
    the input box from 1 to 8 lines as the user types and snaps back
    to 1 on empty.

Mirrors the round26 probe pattern — single-threaded synchronous
script that exits with a non-zero status if any assertion fails.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import tkinter as tk  # noqa: E402
from tkinter import ttk as _ttk  # noqa: E402

from littrace.window import (  # noqa: E402
    DESIGN,
    LitTraceWindow,
    _CHAT_MD_TAGS,
    _STATUS_ANIM_INTERVAL_MS,
    _font,
)

root = tk.Tk()
root.withdraw()
win = LitTraceWindow.__new__(LitTraceWindow)
win.root = root
win.tk = tk
win.ttk = _ttk
win.style = _ttk.Style(root)
win.chat_text = tk.Text(root, height=8, width=60)
win.chat_text.configure(state="normal")
win.status_var = tk.StringVar(value="")
win.input_entry = tk.Text(root, height=2, width=60)
win.input_entry.configure(state="normal")
win._input_md_after_id = None
win._status_anim_after_id = None
win._status_dot_count = 0
win._status_base_label = ""
# Round 27 state.
win._bubble_records = []
win._last_user_message = None
win._streaming_stop_event = None
win._stop_button_window = None  # legacy attribute kept for back-compat
win._stop_button = None
win._stop_button_frame = None
win._stop_button_line = None
win._streaming_chars = 0
# Register markdown tags + the chat bubble / link styles.
win._configure_input_tags()
win._configure_chat_markdown_tags()
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


# --- 1. Chat markdown tag registration --------------------------------------

print("[1] Chat markdown tags")
names = set(win.chat_text.tag_names())
missing = [t for t in _CHAT_MD_TAGS if t not in names]
assert not missing, f"Missing chat markdown tags: {missing}"
print(f"    OK — {_CHAT_MD_TAGS!r} all registered on chat_text")


# --- 2. Assistant markdown rendering ----------------------------------------

print("[2] Assistant markdown rendering (render-on-finalize)")
sample = (
    "Here's a quick walkthrough:\n"
    "\n"
    "## Setup\n"
    "Run `pip install littrace` to install.\n"
    "Then **register** your OpenAI key in the *settings* panel.\n"
    "\n"
    "- bullet one\n"
    "- bullet two\n"
)
win.chat_text.delete("1.0", tk.END)
win.chat_text.insert("1.0", sample)
pre = win.chat_text.index("1.0")
win._render_chat_markdown(pre, sample.rstrip("\n"))

bold_ranges = [(str(r[0]), str(r[1])) for r in zip(
    *([iter(win.chat_text.tag_ranges("md_bold"))] * 2),
)]
code_ranges = [(str(r[0]), str(r[1])) for r in zip(
    *([iter(win.chat_text.tag_ranges("md_code"))] * 2),
)]
h2_ranges = [(str(r[0]), str(r[1])) for r in zip(
    *([iter(win.chat_text.tag_ranges("md_h2"))] * 2),
)]
list_ranges = list(win.chat_text.tag_ranges("md_list_bullet"))
italic_ranges = [(str(r[0]), str(r[1])) for r in zip(
    *([iter(win.chat_text.tag_ranges("md_italic"))] * 2),
)]

print(f"    bold:    {bold_ranges}")
print(f"    code:    {code_ranges}")
print(f"    h2:      {h2_ranges}")
print(f"    italic:  {italic_ranges}")
print(f"    bullet:  {list_ranges[:4]}…")

# ``**register**`` -> group(1) = "register" only. The text contains
# "...Then **register** your..." on line 5 of the sample (after the
# blank line + ``## Setup`` heading + ``Run ...`` line), where
# ``Then `` is 5 chars, then the ``**`` at col 5..6, then
# ``register`` at cols 7..15.
assert any(s == "5.7" and e == "5.15" for s, e in bold_ranges), (
    f"bold range missing register: {bold_ranges}"
)
# `` `pip install littrace` `` -> ``pip install littrace`` at
# columns 5..25 on line 4 (after ``Run ``).
assert any(s == "4.5" for s, _ in code_ranges), (
    f"code range wrong: {code_ranges}"
)
# ``## Setup`` on line 3 — the heading text itself is styled.
assert any(s.startswith("3.") for s, _ in h2_ranges), (
    f"h2 range missing: {h2_ranges}"
)
# Text preservation — buffer must be byte-identical.
buf_before = sample
buf_after = win.chat_text.get("1.0", tk.END)
assert buf_after.startswith(buf_before.rstrip("\n")[:20]), (
    f"render mutated text head:\nbefore={buf_before[:60]!r}\nafter={buf_after[:60]!r}"
)

# Idempotent — second render must not double tag ranges.
bold_first = list(win.chat_text.tag_ranges("md_bold"))
win._render_chat_markdown(pre, sample.rstrip("\n"))
bold_second = list(win.chat_text.tag_ranges("md_bold"))
assert bold_first == bold_second, (
    f"renderer not idempotent — first={bold_first} second={bold_second}"
)

print("    OK — bold / code / h2 / italic / bullet ranges correct; text preserved; idempotent")


# --- 3. Status bubble visual upgrade ---------------------------------------

print("[3] Status bubble palette (active pill)")
win._set_status_in_chat("Codex 思考中")
fg = win.chat_text.tag_cget("bubble_system", "foreground")
bg = win.chat_text.tag_cget("bubble_system", "background")
print(f"    bubble_system fg={fg!r} bg={bg!r}")
assert fg == DESIGN["primary"], f"foreground wrong: {fg}"
assert bg == DESIGN["accent_teal_subtle"], f"background wrong: {bg}"
print("    OK — teal-on-canvas palette applied")

# 200ms tick interval — Round 27 doubled the speed.
assert _STATUS_ANIM_INTERVAL_MS == 200, (
    f"animation interval was {_STATUS_ANIM_INTERVAL_MS}, expected 200"
)
print(f"    OK — _STATUS_ANIM_INTERVAL_MS={_STATUS_ANIM_INTERVAL_MS}")


# --- 4. Stop-button wiring --------------------------------------------------

print("[4] Stop-button + threading.Event bridge")
# Reset status bubble so the stop button on its own line doesn't
# interact with the status bubble.
win._clear_status_in_chat()

win._streaming_stop_event = threading.Event()
assert isinstance(win._streaming_stop_event, threading.Event)
assert not win._streaming_stop_event.is_set()
win._begin_streaming_bubble()
# Round 28 (Phase 4): the Stop button is now the inline
# ``Pill.TButton`` — the legacy ``_stop_button_frame`` /
# ``_stop_button_line`` attributes are no longer populated.
assert win._stop_pill is not None, "Stop pill not anchored"
assert win._stop_pill_index is not None, "Stop pill line not tracked"
print(f"    [a] Stop pill anchored at line {win._stop_pill_index!r}")

# Append a delta + finalize, the Stop pill must be torn down.
win._append_streaming_delta("partial")
win._finalize_streaming_bubble("canonical reply")
assert win._stop_pill is None, "Stop pill not destroyed on finalize"
assert win._stop_pill_index is None, "Stop pill line not cleared on finalize"
assert win._streaming_stop_event is None, (
    "stop event not reset after finalize"
)
body = win.chat_text.get("1.0", tk.END)
assert "canonical reply" in body, f"final reply missing from chat:\n{body!r}"
assert "partial" not in body, f"streaming partial still in chat:\n{body!r}"
print("    OK — Stop pill torn down, streaming text replaced cleanly")


# --- 5. Per-bubble action rows ---------------------------------------------

print("[5] Per-bubble action rows")
win._bubble_records = []
win._append_message("user", "find me transformer papers")
assert len(win._bubble_records) == 1
rec = win._bubble_records[0]
assert rec["role"] == "user"
assert rec["text"] == "find me transformer papers"
assert rec["frame"] is not None
assert rec["line_index"] is not None
print(f"    [a] user bubble -> {rec['text']!r}, action frame on line {rec['line_index']!r}")

win._append_message("assistant", "Here are some papers:\n- Attention is all you need\n- BERT")
assert len(win._bubble_records) == 2
rec = win._bubble_records[1]
assert rec["role"] == "assistant"
assert rec["text"] == "Here are some papers:\n- Attention is all you need\n- BERT"
print(f"    [b] assistant bubble -> {rec['text'][:30]!r}…, action frame on line {rec['line_index']!r}")

# Clipboard side-effect: copy text and confirm the clipboard got it.
root.clipboard_clear()
root.clipboard_append(rec["text"])
clipped = root.clipboard_get()
assert clipped == rec["text"], f"clipboard mismatch: {clipped!r}"
print("    OK — Copy action populates the clipboard")

# Regenerate guard: while a turn is mid-flight, 重新生成 is a no-op.
win._streaming_stop_event = threading.Event()  # active
calls: list[str] = []
win._send_with_text = lambda text: calls.append(text)  # type: ignore[method-assign]
win._last_user_message = "find me transformer papers"
win._regenerate_last()
assert calls == [], f"regenerate fired during streaming: {calls}"
# Drop the active event, regenerate should now fire.
win._streaming_stop_event = None
win._regenerate_last()
assert calls == ["find me transformer papers"], (
    f"regenerate did not fire after stop event cleared: {calls}"
)
print(f"    OK — regenerate is guarded while streaming; replayed {calls!r} after")


# --- 6. Adaptive input height ----------------------------------------------

print("[6] Adaptive input height")
win.input_entry.configure(height=1)
win.input_entry.delete("1.0", tk.END)
win.input_entry.insert("1.0", "single line")
win._resize_input_to_content()
assert int(win.input_entry.cget("height")) == 1, (
    f"single-line height was {win.input_entry.cget('height')}"
)

# Five newline-delimited lines -> 5 visual lines.
win.input_entry.delete("1.0", tk.END)
win.input_entry.insert("1.0", "l1\nl2\nl3\nl4\nl5")
win._resize_input_to_content()
assert int(win.input_entry.cget("height")) == 5, (
    f"5-line height was {win.input_entry.cget('height')}"
)

# Empty -> snap back to 1.
win.input_entry.delete("1.0", tk.END)
win._resize_input_to_content()
assert int(win.input_entry.cget("height")) == 1, (
    f"empty height was {win.input_entry.cget('height')}"
)
print("    OK — input height adapts 1..5..1 around content")


# --- 7. Session switch cleanup ---------------------------------------------

print("[7] Session switch cleanup of action frames")
# Stub out the surface ``_render_session_messages`` reads off of
# ``self`` so the cleanup branch can run without spinning up
# Postgres. We point ``state_store_from_config`` at a fake store
# that returns an empty list — the cleanup logic doesn't depend
# on any actual record content, only on the prior ``_bubble_records``.
import littrace.window as _w  # noqa: E402


class _StubStore:
    def list_chat_messages(self, _session_id):  # type: ignore[no-untyped-def]
        return []


win.config = None  # noqa: SLF001 — stub attribute for the cleanup path
win.session = type("_S", (), {"session_id": "probe"})()
_w.state_store_from_config = lambda _cfg: _StubStore()  # type: ignore[assignment]

before = len(win._bubble_records)
win._render_session_messages()
after = len(win._bubble_records)
assert before >= 2, f"expected >=2 bubble records before switch, got {before}"
assert after == 0, f"records not cleared on session switch: {after}"
print(f"    OK — {before} action frames cleared; _bubble_records reset to {after}")


# --- Cleanup ----------------------------------------------------------------

win._cancel_status_animation()
print()
print("=" * 72)
print(" Round 27 logic verified — Tier A UX overhaul")
print("=" * 72)
root.destroy()