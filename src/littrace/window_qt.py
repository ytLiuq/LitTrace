"""Qt-based shell for LitTrace.

This is the ``littrace-qt`` entry point. It mirrors the layout and
feature set of the Tk shell (``littrace.window``) while delegating all
business logic to ``ShellController``. ``QWebEngineView`` is a real
``QWidget``, so the embedded Chromium pane lives in the same splitter
as the chat / context / trace panels rather than opening a separate
top-level window like pywebview would.

The Tk shell (``littrace-window``) and this Qt shell can coexist on the
same ``config.yaml``; both go through the same controller, session
backend, and codex MCP tools.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("QT_LOGGING_RULES", "qt.webengine.*=false")

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWebEngineWidgets import QWebEngineView

from littrace.config import load_config
from littrace.models import PaperMetadata
from littrace.session import list_chat_sessions
from littrace.shell_controller import ShellController, ShellEvent


# Animated "thinking" dots used in the chat-panel status strip. Cycles
# through 1..3 dots so the user gets a clear heartbeat while a Codex
# turn is in flight — the previous static "处理中…" looked frozen.
_THINKING_FRAMES = ("·  ·  ·", "·· ··", "·····")


# ---------------------------------------------------------------------------
# Design tokens — kept in sync with the Tk shell's DESIGN dict.
# ---------------------------------------------------------------------------

DESIGN = {
    "primary": "#3a8a8c",
    "primary_hover": "#4ea3a5",
    "ink": "#0b0c0e",
    "ink_muted": "#5c6068",
    "ink_subtle": "#a4a7ad",
    "canvas": "#fbfbfc",
    "parchment": "#f5f6f6",
    "pearl": "#ffffff",
    "surface_1": "#ffffff",
    "surface_2": "#f7f8f8",
    "hairline": "#d8d9dc",
    "black": "#0b0c0e",
    "dark_tile": "#0f1011",
    "on_dark": "#f7f8f8",
    "accent_coral": "#cc785c",
    "accent_teal_subtle": "#e6efee",
}


logger = logging.getLogger("littrace.window_qt")


def _summarise_sentinel_output(stdout: str) -> str:
    """Pick the user-facing one-liner out of a ``littrace sentinel run``
    stdout stream.

    Sentinel prints ``run_id:`` / ``new_candidates:`` / ``downloaded:`` /
    ``parsed:`` / ``access_tasks:`` lines near the top of its run summary
    block, followed by a closing ``warnings:`` blob whose individual
    items ("Dataset is missing for all rows; comparison may be unfair.")
    are useful to the parse pipeline but pure noise on a status badge.
    Extract just the run_id + the four counter lines.
    """
    keys = ("run_id:", "new_candidates:", "downloaded:", "parsed:", "access_tasks:")
    picked: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        for key in keys:
            if line.startswith(key):
                picked.append(line)
                break
    if not picked:
        # Fall back to a short tail when the run did not match the
        # expected format (older codex output, broken upgrade, …).
        tail_lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
        return " · ".join(tail_lines[-2:])
    return " · ".join(picked)


def _summary_value(summary: str, key: str) -> int | None:
    """Pull the integer out of a sentinel counter line that
    ``_summarise_sentinel_output`` already picked out of stdout, e.g.
    ``downloaded: 1`` → 1. Returns ``None`` if the line isn't in the
    summary or the value can't be parsed.
    """
    for line in summary.split("·"):
        line = line.strip()
        if line.startswith(key):
            tail = line[len(key):].strip()
            token = tail.split()[0] if tail else ""
            try:
                return int(token)
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# Message body rendering
# ---------------------------------------------------------------------------
#
# Codex replies come back as plain text that happens to use Markdown
# conventions: bold with **...**, inline code with `...`, code fences
# with ```...```, headings with ``#``/``##``, list items with ``- ``,
# and block quotes with ``> ``. ``QTextBrowser.append`` accepts an HTML
# fragment, so we convert Markdown to a small whitelist of HTML tags
# here and inject the result. We escape the input first, then run
# Markdown rules against the escaped text — that way any literal
# ``<``/``>`` in the reply never opens a tag the browser interprets.
# No external Markdown dependency; the rule set is intentionally
# minimal (whatever Codex 0.149 emits today, not CommonMark).


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_message_html(text: str) -> str:
    """Render a Codex reply as the small HTML subset Qt's text browser
    understands. ``text`` is treated as Markdown; the rules above cover
    the cases Codex 0.149 actually emits (bold / code / fenced code /
    headings / lists / block quotes).

    Implementation note: escape-then-rewrite would re-escape the ``<b>``
    / ``<code>`` / ``<pre>`` we just inserted when a later rule (e.g.
    the list rule) sees those tags inside its match group. So we do
    three passes:

      1. Run the markdown rules on the raw text. Each rule's callback
         emits the *final* HTML for its match (escaping the captured
         text group itself), so the result is already valid HTML.
      2. ``_xml_escape`` the whole string once. That escapes any
         literal ``<``/``>``/``&`` in the reply that was outside the
         markdown patterns, but leaves the already-emitted HTML tags
         (we swapped them for sentinels just before escaping so they
         pass through untouched).
      3. Swap the sentinels back to their final ``<b>``/``<code>``/etc.
    """
    sentinel_map = {
        "@@LT_B@@": "<b>",
        "@@RT_B@@": "</b>",
        "@@LT_I@@": "<i>",
        "@@RT_I@@": "</i>",
        "@@LT_CODE@@": (
            "<code style='background:" + DESIGN["surface_2"]
            + ";padding:1px 4px;border-radius:3px;"
            + "font-family:Menlo,Consolas,monospace;font-size:12px;'>"
        ),
        "@@RT_CODE@@": "</code>",
        "@@LT_PRE@@": (
            "<pre style='background:" + DESIGN["surface_2"]
            + ";padding:6px 8px;border-radius:4px;"
            + "white-space:pre-wrap;font-family:Menlo,Consolas,monospace;"
            + "font-size:12px;'>"
        ),
        "@@RT_PRE@@": "</pre>",
        "@@LT_H3@@": "<h3 style='margin:6px 0;'>",
        "@@RT_H3@@": "</h3>",
        "@@LT_H4@@": "<h4>",
        "@@RT_H4@@": "</h4>",
        "@@LT_H5@@": "<h5>",
        "@@RT_H5@@": "</h5>",
        "@@LT_H6@@": "<h6>",
        "@@RT_H6@@": "</h6>",
        "@@LT_BQ@@": (
            "<blockquote style='margin:4px 0;padding:2px 8px;"
            + "border-left:3px solid " + DESIGN["hairline"] + ";"
            + "color:" + DESIGN["ink_muted"] + ";'>"
        ),
        "@@RT_BQ@@": "</blockquote>",
        "@@LI_START@@": (
            "<div style='margin-left:12px;'>"
        ),
        "@@LI_END@@": "</div>",
    }

    def fence(m):
        return "@@LT_PRE@@" + _xml_escape(m.group(1).strip()) + "@@RT_PRE@@"

    def code(m):
        return "@@LT_CODE@@" + _xml_escape(m.group(1)) + "@@RT_CODE@@"

    def bold(m):
        return "@@LT_B@@" + _xml_escape(m.group(1)) + "@@RT_B@@"

    def italic(m):
        return "@@LT_I@@" + _xml_escape(m.group(1)) + "@@RT_I@@"

    def heading(level: str):
        return lambda m: f"@@LT_H{level}@@" + _xml_escape(m.group(1)) + f"@@RT_H{level}@@"

    def bq(m):
        return "@@LT_BQ@@" + _xml_escape(m.group(1)) + "@@RT_BQ@@"

    def li_dash(m):
        return "@@LI_START@@• " + _xml_escape(m.group(1)) + "@@LI_END@@"

    def li_num(m):
        return "@@LI_START@@" + m.group(0) + "@@LI_END@@"

    body = text
    body = re.sub(r"```([\s\S]*?)```", fence, body)
    body = re.sub(r"`([^`\n]+)`", code, body)
    body = re.sub(r"\*\*([^*\n]+)\*\*", bold, body)
    body = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\w)", italic, body)
    body = re.sub(r"^###### (.+)$", heading("6"), body, flags=re.MULTILINE)
    body = re.sub(r"^##### (.+)$", heading("5"), body, flags=re.MULTILINE)
    body = re.sub(r"^#### (.+)$", heading("5"), body, flags=re.MULTILINE)
    body = re.sub(r"^### (.+)$", heading("4"), body, flags=re.MULTILINE)
    body = re.sub(r"^## (.+)$", heading("4"), body, flags=re.MULTILINE)
    body = re.sub(r"^# (.+)$", heading("3"), body, flags=re.MULTILINE)
    body = re.sub(r"^> (.+)$", bq, body, flags=re.MULTILINE)
    body = re.sub(r"^(?:[-*] )(.+)$", li_dash, body, flags=re.MULTILINE)
    body = re.sub(r"^\d+\. (.+)$", li_num, body, flags=re.MULTILINE)
    # Escape anything left over so a literal ``<`` or ``&`` in the
    # user's reply can't open a tag.
    body = _xml_escape(body)
    # Re-emit the previously-substituted HTML tags. Order doesn't
    # matter because the sentinels are unique placeholder strings.
    for sentinel, html in sentinel_map.items():
        body = body.replace(sentinel, html)
    body = body.replace("\n", "<br>")
    return body


# Codex 0.149 alpha frequently leaks sentences of internal narration
# into its replies. Some are prepended ("I'll keep it to two concise
# comparison bullets.", "我会按 Codex/OpenAI 产品说明来回答；这里只
# 需要确认当前助手身份，不会改动 LitTrace 会话。"), but a common new
# pattern is the *trailing* sentence the model uses to "tidy up":
# "用户不关注的信息不用说", "I'll keep the rest concise.", "The rest
# is internal.". Strip both at the start of the reply and immediately
# before the first block-level tag so the user's eyes land on the
# substantive answer, not a line of model bookkeeping.

_NARRATION_PATTERNS = [
    # Each pattern matches ONE leading sentence up to its first ``。``
    # so we never bleed past the narration into the real answer.
    r"^我[^。]*?(?:确认|只需|只.{0,8}需要|会.{0,8}改动|会.{0,8}修改|会.{0,8}影响|不.{0,4}会)[^。]*?[。.]\s*",
    r"^I['’]?ll[^.!?]*?(?:keep|just|only)[^.!?]*?[.!?]\s*",
    r"^Sure[,!.][^.!?]*?[.!?]\s*",
    r"^Here'?s[^.!?]*?[.!?]\s*",
    r"^当然[,，.][^。]*?[。.]\s*",
]
# Trailing sentences: Codex often tacks a directive-style line at
# the end ("用户不关注的信息不用说", "I'll keep the rest concise.",
# "The rest is internal."). These are stripped before the closing
# block boundary so the chat scrollback ends on the substantive
# content.
_TRAILING_NARRATION_PATTERNS = [
    r"[\s\S]*?(?:用户不关注.{0,30}|用户不需要.{0,30}|不.{0,4}直接展示给用户|不.{0,4}展示给用户|不.{0,8}说给用户|不.{0,8}用.{0,4}看)[。.]?\s*$",
    r"[\s\S]*?(?:I['’]?ll keep the rest concise\.?|Keep the rest concise\.?|I['’]?ll be concise\.?|I'll skip the rest\.?)\s*$",
    r"[\s\S]*?(?:I['’]?ll skip the internal details\.?|Skipping internal details\.?)\s*$",
    r"[\s\S]*?(?:The rest is internal\.?|Internal notes removed\.?)\s*$",
]


def _strip_leading_narration(html: str) -> str:
    """Drop leading sentences that match common Codex internal-
    narration templates. Operates on the already-rendered HTML
    fragment by stripping a leading text-run before any block tag.
    """
    for pattern in _NARRATION_PATTERNS:
        m = re.match(pattern, html, re.DOTALL)
        if m:
            return html[m.end():].lstrip()
    return html


def _strip_trailing_narration(html: str) -> str:
    """Drop a trailing internal-narration sentence from the end of the
    reply, just before the closing block boundary. The patterns match
    Codex's common "tidy up" lines ("用户不关注的信息不用说",
    "I'll keep the rest concise.", etc.) — these leak the model's
    own bookkeeping into the visible chat.
    """
    for pattern in _TRAILING_NARRATION_PATTERNS:
        m = re.search(pattern, html)
        if m:
            return html[: m.start()].rstrip()
    return html


# Late import kept out of the top of the file on purpose; ``re`` is
# already imported above near the standard-library block.


def _littrace_cmd(*args: str) -> tuple[list[str], dict[str, str], str]:
    """Resolve the ``littrace`` console-script entry point and the
    working-directory / environment needed to invoke it.

    Hard-coding ``["littrace", ...]`` broke when the shell was launched
    via PyInstaller/py2app or any environment where the ``$PATH`` did
    not include the venv bin directory: ``subprocess.run`` raised
    ``FileNotFoundError`` and the user clicked "运行今日管线" only to
    see the status bar silently stay on "运行中…" forever. A second,
    subtler bug surfaced once that was fixed: ``python -m littrace.cli``
    resolves ``config.yaml`` relative to the *child's* cwd (not the
    LitTrace repository root), so the fallback path raised another
    ``FileNotFoundError`` from ``load_config``. Anchor both paths to
    the project root so the subprocess always sees the right
    ``config.yaml`` regardless of how ``littrace-qt`` was launched.
    """
    project_root = Path(__file__).resolve().parent.parent.parent  # src/littrace/window_qt.py -> repo root
    # Prefer the installed console script. If that fails (e.g. frozen
    # distribution where the entry wasn't generated) fall back to
    # ``python -m littrace.cli`` but still anchor cwd + config.
    exe = shutil.which("littrace")
    if exe:
        cmd = [exe, *args]
    else:
        cmd = [sys.executable, "-m", "littrace.cli", *args]
    env = {
        "LITTRACE_CONFIG": str(project_root / "config.yaml"),
        "PYTHONPATH": str(project_root / "src"),
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    return cmd, env, str(project_root)


# ---------------------------------------------------------------------------
# Design tokens — kept in sync with the Tk shell's DESIGN dict.
# ---------------------------------------------------------------------------

WINDOW_QSS = f"""
QMainWindow, QWidget#root {{
    background: {DESIGN["parchment"]};
    color: {DESIGN["ink"]};
    font-family: -apple-system, "Helvetica Neue", "PingFang SC", sans-serif;
    font-size: 13px;
}}

QFrame#nav, QFrame#nav * {{
    background: {DESIGN["black"]};
    color: {DESIGN["on_dark"]};
}}

QLabel#brand {{
    color: {DESIGN["ink"]};
    font-size: 22px;
    font-weight: 600;
}}

QLabel#tagline {{
    color: {DESIGN["ink_muted"]};
    font-size: 12px;
}}

QFrame#subnav, QPushButton#subnav_btn {{
    background: {DESIGN["parchment"]};
}}
QPushButton#subnav_btn {{
    border: 1px solid {DESIGN["hairline"]};
    color: {DESIGN["ink"]};
    border-radius: 4px;
    padding: 6px 14px;
}}
QPushButton#subnav_btn:hover {{
    background: {DESIGN["surface_1"]};
}}
QPushButton#subnav_btn_primary {{
    background: {DESIGN["primary"]};
    color: white;
    border: none;
    padding: 6px 14px;
    border-radius: 4px;
    font-weight: 600;
}}
QPushButton#subnav_btn_primary:hover {{
    background: {DESIGN["primary_hover"]};
}}

QFrame#tile, QFrame#chat_tile, QFrame#trace_tile, QFrame#context_tile, QFrame#browser_tile {{
    background: {DESIGN["surface_1"]};
    border: 1px solid {DESIGN["hairline"]};
    border-radius: 6px;
}}

QLabel#pane_title {{
    color: {DESIGN["ink"]};
    font-weight: 600;
    font-size: 14px;
    padding: 2px 4px;
}}

QTextBrowser#trace_view, QTextBrowser#chat_view {{
    background: {DESIGN["surface_1"]};
    border: none;
    color: {DESIGN["ink"]};
}}

QLineEdit#input {{
    background: {DESIGN["surface_2"]};
    border: 1px solid {DESIGN["hairline"]};
    border-radius: 6px;
    padding: 8px 10px;
    color: {DESIGN["ink"]};
    selection-background-color: {DESIGN["accent_teal_subtle"]};
}}

QPushButton#send {{
    background: {DESIGN["primary"]};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
}}
QPushButton#send:hover {{
    background: {DESIGN["primary_hover"]};
}}

QListWidget#context, QListWidget#sessions {{
    background: {DESIGN["surface_1"]};
    border: 1px solid {DESIGN["hairline"]};
    border-radius: 6px;
    color: {DESIGN["ink"]};
}}

QListWidget#command_popup {{
    background: {DESIGN["surface_1"]};
    border: 1px solid {DESIGN["hairline"]};
    border-radius: 4px;
    color: {DESIGN["ink"]};
}}
QListWidget#command_popup::item:selected {{
    background: {DESIGN["accent_teal_subtle"]};
    color: {DESIGN["ink"]};
}}

QLabel#status {{
    color: {DESIGN["ink_subtle"]};
}}

QStatusBar {{
    background: {DESIGN["parchment"]};
    color: {DESIGN["ink_muted"]};
}}

QLabel#thinking_strip {{
    color: {DESIGN["ink_muted"]};
    background: {DESIGN["surface_2"]};
    border: 1px solid {DESIGN["hairline"]};
    border-radius: 4px;
    padding: 4px 10px;
    font-style: italic;
}}
QLabel#thinking_strip[active="true"] {{
    color: {DESIGN["primary"]};
}}

QSplitter::handle {{
    background: {DESIGN["parchment"]};
}}
QSplitter::handle:horizontal {{
    width: 8px;
}}
"""


# ---------------------------------------------------------------------------
# Slash command catalog — mirrored from window.COMMAND_CATALOG so the
# autocomplete popup behaves identically to the Tk shell.
# ---------------------------------------------------------------------------

COMMAND_CATALOG: list[tuple[str, str, bool]] = [
    ("context", "显示 / 隐藏当前文献上下文", True),
    ("papers", "列出当前上下文文献", True),
    ("parse", "按当前解析模式处理 PDF", False),
    ("parse --ocr", "强制使用 OCR 解析", False),
    ("parse --text", "强制使用文本层解析", False),
    ("table", "抽取性能指标并生成对比表", True),
    ("storyline", "梳理论文回应关系", True),
    ("storyline-report", "导出 storyline 报告", True),
    ("storyline-review", "Reviewer 审阅 storyline", True),
    ("dashboard", "打开 RAG / Daily 仪表盘", True),
    ("quality", "运行质量门", True),
    ("agents", "列出可用 agents", False),
    ("workflow", "显示当前 workflow trace", True),
    ("quality-audits", "运行质量审计", True),
    ("plan", "显示当前执行计划", False),
    ("init-config", "运行 config wizard", False),
    ("login", "打开授权登录弹窗", True),
    ("attach", "手动附加本地 PDF", False),
    ("attach-si", "附加 SI / 补充材料", False),
    ("full-text", "构建 full-text context", True),
    ("backfill-dois", "回填 DOI", False),
    ("publisher-retrieve", "按 publisher 抓取", False),
    ("check-downloads", "检查当前下载计划", True),
    ("resume-downloads", "恢复下载（等待用户授权）", True),
    ("benchmark", "运行评测基准", False),
    ("golden-eval", "运行 golden set 评估", False),
    ("export", "导出当前 session", False),
    ("quit", "关闭窗口", False),
    ("全部下载", "选择当前上下文中全部待下载文献", True),
    ("选择第 N 篇下载", "选择第 N 篇进入下载计划", True),
    ("取消选择第 N 篇", "从下载计划中移除第 N 篇", True),
]


# ---------------------------------------------------------------------------
# Panel widgets
# ---------------------------------------------------------------------------


class TracePanel(QtWidgets.QFrame):
    """Left column — workflow trace + session history."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trace_tile")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("执行 Trace")
        title.setObjectName("pane_title")
        layout.addWidget(title)

        self._view = QtWidgets.QTextBrowser()
        self._view.setObjectName("trace_view")
        self._view.setOpenExternalLinks(False)
        layout.addWidget(self._view, stretch=3)

        # Session history sub-panel
        history_title = QtWidgets.QLabel("历史 Session")
        history_title.setObjectName("pane_title")
        layout.addWidget(history_title)

        self._sessions = QtWidgets.QListWidget()
        self._sessions.setObjectName("sessions")
        self._sessions.setMaximumHeight(180)
        layout.addWidget(self._sessions, stretch=2)

    def render_workflow_trace(self, trace_steps: Iterable[str]) -> None:
        body = "<br>".join(f"• {step}" for step in trace_steps) or "(等待任务…)"
        self._view.setHtml(f"<div>{body}</div>")

    def render_execution_path(self, steps: Iterable[str]) -> None:
        body = "<br>".join(f"→ {step}" for step in steps) or ""
        self._view.append(f'<div style="margin-top:6px;color:#5c6068;">{body}</div>')

    def set_sessions(self, sessions: list[Any], current_session_id: str) -> None:
        self._sessions.clear()
        for session in sessions:
            label = f"{session.session_id}  ({session.created_at[:19]})"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, session.session_id)
            if session.session_id == current_session_id:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._sessions.addItem(item)


class ChatPanel(QtWidgets.QFrame):
    """Middle column — chat scrollback + slash autocomplete + input."""

    def __init__(self, controller: ShellController, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chat_tile")
        self._controller = controller

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("对话")
        title.setObjectName("pane_title")
        layout.addWidget(title)

        self._view = QtWidgets.QTextBrowser()
        self._view.setObjectName("chat_view")
        self._view.setOpenExternalLinks(True)
        self._view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._on_chat_context_menu)
        layout.addWidget(self._view, stretch=1)

        # Thinking strip — sits between the chat scrollback and the input
        # box, mirroring ChatGPT Codex App's "thinking / researching" row.
        # Animated dots + the current pipeline label so the user can see
        # the controller is alive while a turn is in flight.
        self._thinking = QtWidgets.QLabel("")
        self._thinking.setObjectName("thinking_strip")
        self._thinking.setMinimumHeight(22)
        self._thinking.setStyleSheet(
            f"color:{DESIGN['ink_muted']};padding:2px 6px;font-style:italic;"
        )
        self._thinking.hide()
        layout.addWidget(self._thinking)
        self._thinking_frame = 0
        self._thinking_timer = QtCore.QTimer(self)
        self._thinking_timer.setInterval(450)
        self._thinking_timer.timeout.connect(self._tick_thinking)
        self._thinking_label = ""

        # ChatGPT-Codex-style input bar: a single rounded container with
        # a left attachment slot, a multi-line text editor in the
        # middle, and a send button on the right. The container's
        # background lightens on focus, mirroring the Codex App.
        input_row = QtWidgets.QFrame()
        input_row.setObjectName("input_bar")
        input_row.setStyleSheet(
            "QFrame#input_bar {"
            f"  background: {DESIGN['surface_2']};"
            f"  border: 1px solid {DESIGN['hairline']};"
            "  border-radius: 22px;"
            "}"
            "QFrame#input_bar:focus-within {"
            f"  border: 1px solid {DESIGN['ink']};"
            "}"
        )
        input_layout = QtWidgets.QHBoxLayout(input_row)
        input_layout.setContentsMargins(8, 4, 8, 4)
        input_layout.setSpacing(6)

        # Left slot — placeholder for an "attach file" button (future
        # work). For now it's a thin visual anchor.
        # No left-side button — the previous "+" attachment slot was
        # unused and made the bar look busy. The input starts directly
        # at the rounded container's left edge.

        # Multi-line text input (was a single-line QLineEdit; Codex App
        # wraps on Enter when Shift isn't held and inserts a literal
        # newline when it is, so the user can paste multi-line
        # context).
        self._input = QtWidgets.QTextEdit()
        self._input.setObjectName("input")
        self._input.setPlaceholderText("Message LitTrace…  (Enter 发送 / Shift+Enter 换行)")
        self._input.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._input.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._input.setAcceptRichText(False)
        self._input.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._input.setFixedHeight(40)
        self._input.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        # Install an event filter so the existing slash-popup keypress
        # logic (Up / Down / Enter / Esc) keeps working against the new
        # widget.
        self._input.installEventFilter(self)
        # Re-route textChanged to the existing handler.
        self._input.textChanged.connect(self._on_input_text_changed)
        input_layout.addWidget(self._input, stretch=1)

        # Send button — round, primary-coloured, only enabled when the
        # text editor holds non-whitespace content.
        self._send = QtWidgets.QPushButton("➤")
        self._send.setObjectName("send")
        self._send.setFixedSize(32, 32)
        self._send.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._send.setStyleSheet(
            "QPushButton#send {"
            f"  background: {DESIGN['primary']};"
            "  color: white;"
            "  border: none;"
            "  border-radius: 16px;"
            "  font-size: 14px;"
            "  font-weight: bold;"
            "}"
            "QPushButton#send:hover {"
            f"  background: {DESIGN['primary_hover']};"
            "}"
            "QPushButton#send:disabled {"
            "  background: #d8d9dc;"
            "  color: #a4a7ad;"
            "}"
        )
        self._send.clicked.connect(self._on_send)
        self._send.setEnabled(False)
        input_layout.addWidget(self._send)

        layout.addWidget(input_row)

        # Slash-command autocomplete popup, lazily shown beneath the
        # input box when the user types a leading "/". Pre-create every
        # catalog entry so the first keystroke never pays a 500 ms
        # ``addItem x 30 + show()`` cost (measured on this machine; see
        # commit message for trace). Filtering is done by toggling
        # ``setHidden`` on the existing items, which is O(N) Qt flag work
        # and stays single-digit-ms even for the full catalog.
        self._popup = QtWidgets.QListWidget()
        self._popup.setObjectName("command_popup")
        self._popup.setWindowFlags(QtCore.Qt.WindowType.Popup)
        self._popup.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._popup.itemActivated.connect(self._commit_popup_item)
        self._popup.itemClicked.connect(self._commit_popup_item)
        self._popup.setFixedWidth(420)
        self._popup.setMinimumHeight(220)
        self._popup.setUniformItemSizes(True)
        for name, desc, _ctx in COMMAND_CATALOG:
            item = QtWidgets.QListWidgetItem(f"/{name}    — {desc}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            self._popup.addItem(item)
        self._popup.hide()

        self._popup_index = 0

    # ---- Event filter for ↑/↓/Esc -------------------------------------

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if source is self._input and event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()
            modifiers = event.modifiers()
            if self._popup.isVisible():
                if key == QtCore.Qt.Key.Key_Down:
                    self._navigate_popup(+1)
                    return True
                if key == QtCore.Qt.Key.Key_Up:
                    self._navigate_popup(-1)
                    return True
                if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    self._commit_popup_at(self._popup.currentRow())
                    return True
                if key == QtCore.Qt.Key.Key_Escape:
                    self._popup.hide()
                    return True
            else:
                # Send-on-Enter when the slash popup isn't up — matches
                # the ChatGPT Codex App behaviour. Shift+Enter still
                # inserts a literal newline.
                if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    if not (modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier):
                        self._on_send()
                        return True
        return super().eventFilter(source, event)

    # ---- Thinking strip --------------------------------------------------

    def _tick_thinking(self) -> None:
        self._thinking_frame = (self._thinking_frame + 1) % len(_THINKING_FRAMES)
        dots = _THINKING_FRAMES[self._thinking_frame]
        self._thinking.setText(f"{self._thinking_label}  {dots}")

    def set_thinking(self, active: bool, label: str = "") -> None:
        if active:
            self._thinking_label = label or "思考中"
            self._thinking.show()
            self._tick_thinking()
            self._thinking_timer.start()
        else:
            self._thinking_timer.stop()
            self._thinking.hide()
            self._thinking_label = ""

    # ---- Slash popup logic ----------------------------------------------

    def _on_input_text_changed(self) -> None:
        text = self._input.toPlainText()
        # Keep the send button in sync with whether the user has
        # anything to send. ``CursorPosition`` updates fire this slot
        # as well, so re-read the current value each time.
        self._send.setEnabled(bool(text.strip()))
        if not text.startswith("/"):
            self._popup.hide()
            return
        query = text[1:].lower()
        # Toggle ``setHidden`` on the pre-populated items instead of
        # ``clear()`` + ``addItem`` per keystroke. ``clear/addItem`` paid
        # ~500 ms on the very first keystroke because Qt had to allocate
        # every item, lay it out, and then position+show the popup window;
        # toggling the hidden flag is flag-bit work on existing items.
        visible_rows: list[int] = []
        for row in range(self._popup.count()):
            item = self._popup.item(row)
            name = item.data(QtCore.Qt.ItemDataRole.UserRole) or ""
            match = not query or name.lower().startswith(query)
            item.setHidden(not match)
            if match:
                visible_rows.append(row)
        if not visible_rows:
            self._popup.hide()
            return
        # Preserve current selection if it still matches; otherwise jump
        # to the first visible row so ↑/↓ stays predictable.
        current = self._popup.currentRow()
        if current < 0 or self._popup.item(current).isHidden():
            self._popup.setCurrentRow(visible_rows[0])
        self._popup_index = self._popup.currentRow()
        # Position popup beneath the input line edit (cached while
        # visible so repeated keystrokes don't re-query the screen).
        if not self._popup.isVisible():
            anchor = self._input.mapToGlobal(QtCore.QPoint(0, self._input.height()))
            self._popup.move(anchor)
            self._popup.show()

    def _navigate_popup(self, delta: int) -> None:
        count = self._popup.count()
        if count == 0:
            return
        new_row = (self._popup.currentRow() + delta) % count
        self._popup.setCurrentRow(new_row)
        self._popup_index = new_row

    def _commit_popup_at(self, row: int) -> None:
        if row < 0 or row >= self._popup.count():
            return
        self._commit_popup_item(self._popup.item(row))

    def _commit_popup_item(self, item: QtWidgets.QListWidgetItem) -> None:
        name = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not name:
            return
        self._input.setText(f"/{name} ")
        self._popup.hide()

    # ---- Send ---------------------------------------------------------

    def _on_send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self._popup.hide()
        self._send.setEnabled(False)
        if text == "/quit":
            QtWidgets.QApplication.instance().quit()
            return
        self._controller.submit_user_message(text)

    # ---- Event handlers wired by LitTraceQtWindow ----------------------

    def append_message(self, role: str, text: str, **extras: Any) -> None:
        # Render a Codex reply as a left/right chat bubble. ChatGPT
        # Codex App shows user on the right, assistant on the left, with
        # no per-message name label (the visual side is the speaker
        # marker). Markdown is rendered by ``_render_message_html``;
        # any leading Codex internal-narration sentence is stripped.
        body = _strip_leading_narration(_render_message_html(text))
        body = _strip_trailing_narration(body)
        if role == "user":
            bubble_color = DESIGN["primary"]
            text_color = "#ffffff"
            align = "right"
            wrap = "12px 12px 4px 12px"  # tail bottom-right
            max_width = "78%"
        elif role == "system":
            bubble_color = DESIGN["surface_2"]
            text_color = DESIGN["ink_muted"]
            align = "left"
            wrap = "8px"
            max_width = "78%"
        else:  # assistant
            bubble_color = "#ffffff"
            text_color = DESIGN["ink"]
            align = "left"
            wrap = "12px 12px 12px 4px"  # tail bottom-left
            max_width = "78%"
        from PySide6.QtGui import QTextBlockFormat

        cursor = self._view.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)

        # Set the block format on a new block (so the previous bubble's
        # alignment doesn't bleed in) then ``insertHtml`` the bubble.
        # ``<p align="...">`` in HTML was silently coerced to right by
        # QTextBrowser's HTML parser, so we have to set the alignment
        # via the block format instead.
        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            if align == "right"
            else QtCore.Qt.AlignmentFlag.AlignLeft
        )
        block_fmt.setTopMargin(8)
        block_fmt.setBottomMargin(8)
        cursor.insertBlock(block_fmt)
        cursor.insertHtml(
            f'<span style="display:inline-block;max-width:{max_width};'
            f"background:{bubble_color};color:{text_color};"
            f"border:1px solid {DESIGN['hairline']};"
            f"border-radius:{wrap};padding:8px 12px;"
            f"line-height:1.4;"
            f'">'
            f"{body}"
            f"</span>"
        )
        self._view.setTextCursor(cursor)
        self._view.ensureCursorVisible()

    def _on_chat_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        copy_action = menu.addAction("复制")
        copy_action.setEnabled(self._view.textCursor().hasSelection())
        copy_action.triggered.connect(self._view.copy)
        select_all = menu.addAction("全选")
        select_all.triggered.connect(self._view.selectAll)
        clear_action = menu.addAction("清屏")
        clear_action.triggered.connect(self._view.clear)
        menu.exec(self._view.mapToGlobal(pos))

    # ---- Right-click copy/paste menu -----------------------------------

    def _on_chat_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        copy_action = menu.addAction("复制")
        copy_action.setEnabled(self._view.textCursor().hasSelection())
        copy_action.triggered.connect(self._view.copy)
        select_all = menu.addAction("全选")
        select_all.triggered.connect(self._view.selectAll)
        clear_action = menu.addAction("清屏")
        clear_action.triggered.connect(self._view.clear)
        menu.exec(self._view.mapToGlobal(pos))


class ContextPanel(QtWidgets.QFrame):
    """Right column upper — active literature context."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("context_tile")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("文献上下文")
        title.setObjectName("pane_title")
        layout.addWidget(title)

        self._list = QtWidgets.QListWidget()
        self._list.setObjectName("context")
        layout.addWidget(self._list, stretch=1)

    def refresh(self, papers: list[PaperMetadata]) -> None:
        self._list.clear()
        if not papers:
            placeholder = QtWidgets.QListWidgetItem("暂无激活文献 — 在聊天中提需求即可加入")
            placeholder.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self._list.addItem(placeholder)
            return
        for index, paper in enumerate(papers, start=1):
            year = paper.year or "n.d."
            source = paper.journal or paper.publisher or "unknown source"
            item = QtWidgets.QListWidgetItem(
                f"{index}. {paper.title}  ({year}, {source})"
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, paper)
            self._list.addItem(item)


class RAGPanel(QtWidgets.QFrame):
    """Right column middle — single Daily run entry point.

    The original shell exposed three buttons ("刷新当前 session", "全量刷新",
    "Full daily") whose semantics were unclear without reading the parser
    code. Collapse them into one explicit action: **"运行今日管线"** runs
    the daily retrieval + parse + RAG refresh against all sessions, and
    the panel shows the result of the most recent run.
    """

    def __init__(
        self,
        on_run_daily,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("tile")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("今日管线")
        title.setObjectName("pane_title")
        layout.addWidget(title)

        helper = QtWidgets.QLabel(
            "检索最新文献 → 下载开放访问 PDF → 用 docling 解析 → 写入 RAG 索引。"
        )
        helper.setObjectName("status")
        helper.setWordWrap(True)
        layout.addWidget(helper)

        run_btn = QtWidgets.QPushButton("🔍 检索并补全文献")
        run_btn.setObjectName("subnav_btn_primary")
        run_btn.clicked.connect(on_run_daily)
        layout.addWidget(run_btn)

        self._status = QtWidgets.QLabel("未运行")
        self._status.setObjectName("status")
        self._status.setWordWrap(True)
        layout.addWidget(self._status, stretch=1)

    def set_status(self, text: str) -> None:
        self._status.setText(text)


class DailyConfigDialog(QtWidgets.QDialog):
    """Collect the parameters that drive ``littrace sentinel run`` from
    the user before the daily pipeline kicks off. Four fields:

      * 研究主题 (used as sentinel watchlist id, required)
      * 开始 / 结束年份 (year range for retrieval)
      * 最少检索数目 (target count; warns if the run comes up short)

    When the user wants publisher-only papers (not just open-access
    ones), they tick the **「打开 publisher 登录」** button after the
    dialog closes — that fires ``chromium --remote-debugging-port=19222``
    on LitTrace's private profile (``./data/chrome-cdp``) and the user
    signs in to each publisher in turn. The next ``sentinel run`` then
    has access to the gated PDFs.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        default_topic: str = "mxene_sensor",
        default_year_min: int = 2023,
        default_year_max: int = 2026,
        default_min_papers: int = 10,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("检索并补全文献 · 配置")
        self.setModal(True)
        self.resize(540, 380)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("检索并补全文献")
        title.setStyleSheet(
            f"font-size:18px;font-weight:600;color:{DESIGN['ink']};"
        )
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "告诉 sentinel 要研究什么主题、什么年份区间、最少要几篇。"
            "只检索开放访问论文；要看 publisher 内的论文，先点下面的「打开 publisher 登录」。"
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"font-size:12px;color:{DESIGN['ink_muted']};"
        )
        layout.addWidget(subtitle)

        form = QtWidgets.QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        # 主题（watchlist id，必填）
        self._topic_input = QtWidgets.QLineEdit(default_topic)
        self._topic_input.setPlaceholderText("e.g. mxene_sensor, perovskite_solar, mof_co2")
        self._topic_input.selectAll()
        form.addRow("研究主题 *", self._topic_input)

        # 关键词
        self._keywords_input = QtWidgets.QLineEdit()
        self._keywords_input.setPlaceholderText("可选：更精确的检索词（多个用空格分隔）")
        form.addRow("关键词", self._keywords_input)

        # 年份范围
        year_row = QtWidgets.QHBoxLayout()
        year_row.setSpacing(6)
        self._year_min_input = QtWidgets.QSpinBox()
        self._year_min_input.setRange(1990, 2030)
        self._year_min_input.setValue(default_year_min)
        year_row.addWidget(self._year_min_input)
        year_row.addWidget(QtWidgets.QLabel("至"))
        self._year_max_input = QtWidgets.QSpinBox()
        self._year_max_input.setRange(1990, 2030)
        self._year_max_input.setValue(default_year_max)
        year_row.addWidget(self._year_max_input)
        year_row.addStretch(1)
        form.addRow("年份区间", year_row)

        # 最少检索数
        self._min_papers_input = QtWidgets.QSpinBox()
        self._min_papers_input.setRange(1, 200)
        self._min_papers_input.setValue(default_min_papers)
        self._min_papers_input.setSuffix(" 篇")
        form.addRow("最少检索数目", self._min_papers_input)

        layout.addLayout(form)
        layout.addStretch(1)

        # 错误提示
        self._error_label = QtWidgets.QLabel("")
        self._error_label.setStyleSheet(
            f"color:{DESIGN['accent_coral']};font-size:12px;"
        )
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)

        # 按钮
        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(8)
        login_btn = QtWidgets.QPushButton("🌐 打开 publisher 登录")
        login_btn.setObjectName("subnav_btn")
        login_btn.setToolTip(
            "启动 LitTrace 自己的 Chrome（独立 profile，避开你日常 Chrome 的 user-data-dir 锁），"
            "在打开的浏览器里登录各 publisher。登录态会保留，下次 sentinel run 可访问 gated PDF。"
        )
        login_btn.clicked.connect(self._open_publisher_login)
        button_row.addWidget(login_btn)
        button_row.addStretch(1)
        cancel_btn = QtWidgets.QPushButton("取消")
        cancel_btn.setObjectName("subnav_btn")
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(cancel_btn)
        run_btn = QtWidgets.QPushButton("开始检索")
        run_btn.setObjectName("subnav_btn_primary")
        run_btn.setDefault(True)
        run_btn.clicked.connect(self._validate_and_accept)
        button_row.addWidget(run_btn)
        layout.addLayout(button_row)

        self._topic_input.returnPressed.connect(self._validate_and_accept)

    def _open_publisher_login(self) -> None:
        # Persist the dialog's settings before opening the browser so
        # when the user returns and clicks "开始检索" they keep what
        # they typed. We do that by accepting the dialog and storing
        # the chosen values back through the parent (``_on_run_daily``
        # reads them). The parent then launches the browser.
        self._validate_and_accept()
        if self.result() == QtWidgets.QDialog.DialogCode.Accepted:
            # Mark "open publisher login" intent on the parent so it
            # knows to launch the browser after the dialog returns.
            self._open_login_after = True

    def _validate_and_accept(self) -> None:
        topic = self._topic_input.text().strip()
        if not topic:
            self._error_label.setText("研究主题是必填的（用于 sentinel watchlist id）")
            return
        if not topic.replace("_", "").replace("-", "").isalnum():
            self._error_label.setText(
                "主题只能是字母数字 + _ + -（用作 watchlist id 文件名）"
            )
            return
        if self._year_min_input.value() > self._year_max_input.value():
            self._error_label.setText("开始年份不能晚于结束年份")
            return
        self.accept()

    def open_login_after(self) -> bool:
        return getattr(self, "_open_login_after", False)

    # Accessors
    def topic(self) -> str:
        return self._topic_input.text().strip()

    def keywords(self) -> str:
        return self._keywords_input.text().strip()

    def year_min(self) -> int:
        return self._year_min_input.value()

    def year_max(self) -> int:
        return self._year_max_input.value()

    def min_papers(self) -> int:
        return self._min_papers_input.value()


class BrowserPanel(QtWidgets.QFrame):
    """Right column lower — QWebEngineView pane for publisher pages.

    The panel exposes two shortcuts that come up often when the user
    wants LitTrace to be able to pull gated PDFs:

      * A row of one-click publisher login buttons (Wiley / ACS /
        Springer / Nature / arXiv) right above the URL bar. Each opens
        that publisher's sign-in page in the embedded Chromium, so the
        user can sign in once and the session cookies live in
        ``./data/chrome-cdp`` for the next ``sentinel run`` to use.
      * A free-form URL line for ad-hoc navigation.

    The previous round removed the back/forward/reload arrows because
    the URL bar already accepts any input; this round keeps that
    decision and only adds the publisher row.
    """

    HOME_URL = "about:blank"

    # Sign-in URLs for the publishers LitTrace is wired to handle. These
    # land the user on whichever page surfaces a "Sign in" link near
    # the top-right after the federated SSO redirects land.
    PUBLISHER_LINKS: list[tuple[str, str]] = [
        ("🌐 Wiley", "https://onlinelibrary.wiley.com/action/login"),
        ("🌐 ACS", "https://pubs.acs.org/action/showLogin"),
        ("🌐 Springer", "https://link.springer.com/signup-login"),
        ("🌐 Nature", "https://idp.nature.com/authorize?response_type=cookie"),
        ("🌐 arXiv", "https://arxiv.org/login"),
    ]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("browser_tile")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Publisher shortcut row — small text buttons so the row stays
        # compact even with five entries.
        publisher_row = QtWidgets.QHBoxLayout()
        publisher_row.setContentsMargins(8, 8, 8, 0)
        publisher_row.setSpacing(4)
        for label, url in self.PUBLISHER_LINKS:
            btn = QtWidgets.QPushButton(label)
            btn.setObjectName("publisher_btn")
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(url)
            btn.clicked.connect(lambda _checked=False, u=url: self.open_url(u))
            publisher_row.addWidget(btn)
        publisher_row.addStretch(1)
        layout.addLayout(publisher_row)

        # Free-form URL bar for ad-hoc navigation.
        url_row = QtWidgets.QHBoxLayout()
        url_row.setContentsMargins(8, 4, 8, 4)
        url_row.addWidget(QtWidgets.QLabel("URL"))
        self._url = QtWidgets.QLineEdit()
        self._url.setPlaceholderText("粘贴 publisher 链接（Enter 打开）…")
        self._url.returnPressed.connect(self._on_url_entered)
        url_row.addWidget(self._url, stretch=1)
        layout.addLayout(url_row)

        self._view = QWebEngineView()
        self._view.setUrl(QtCore.QUrl(self.HOME_URL))
        self._view.urlChanged.connect(self._on_url_changed)
        layout.addWidget(self._view, stretch=1)

    def _on_url_entered(self) -> None:
        self.open_url(self._url.text())

    def _on_url_changed(self, url: QtCore.QUrl) -> None:
        # Don't echo about:blank into the URL bar — leaves a stale-looking
        # field on startup.
        text = url.toString()
        if text == "about:blank":
            self._url.clear()
        else:
            self._url.setText(text)

    def open_url(self, url: str) -> None:
        """Load ``url`` in the embedded Chromium. Normalises bare hosts
        (``example.com``) to ``https://example.com``) and tolerates
        missing scheme prefixes; safe to call from any caller.
        """
        text = (url or "").strip()
        if not text:
            return
        if not text.startswith(("http://", "https://")):
            text = "https://" + text
        self._view.setUrl(QtCore.QUrl(text))
        text = url.toString()
        if text == "about:blank":
            self._url.clear()
        else:
            self._url.setText(text)

    def open_url(self, url: str) -> None:
        self._view.setUrl(QtCore.QUrl(url))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class LitTraceQtWindow(QtWidgets.QMainWindow):
    def __init__(self, controller: ShellController) -> None:
        super().__init__()
        self._controller = controller

        self.setWindowTitle("LitTrace")
        self.resize(1280, 820)
        self.setMinimumSize(960, 600)
        self.setStyleSheet(WINDOW_QSS)

        self._build_layout()
        self._wire_events()
        self._refresh_initial_state()

    # ---- Cross-thread bridge for "运行今日管线" status ----------------

    @QtCore.Slot("QString")
    def _set_rag_status_from_any_thread(self, text: str) -> None:
        # Slot target for ``QMetaObject.invokeMethod``. ``QLabel.setText``
        # is a plain Python method, not a registered Qt slot, so calling
        # ``invokeMethod(label, "setText", ...)`` returns ``True`` but
        # never actually fires. ``@Slot("QString")`` registers this
        # method in Qt's meta-object system so cross-thread
        # ``invokeMethod(..., QueuedConnection, ...)`` posts the call onto
        # the GUI event loop and the actual ``setText`` runs there.
        # Use ``QString`` (not the bare ``str``) — PySide6's invokeMethod
        # matches a Qt string Q_ARG against ``@Slot("QString")`` but
        # silently drops the call against a bare ``@Slot(str)``, so the
        # status label never updates.
        self._rag_panel._status.setText(text)

    def _post_status(self, text: str) -> None:
        # Posted from a daemon worker thread (``_on_run_daily``). The
        # ``@Slot(str)`` decorator on ``_set_rag_status_from_any_thread``
        # lets ``QMetaObject.invokeMethod`` route this onto the GUI
        # event loop. Earlier attempts that used
        # ``QTimer.singleShot(0, ...)`` from the worker thread were
        # silently dropped because the timer was bound to a thread
        # with no Qt event loop.
        QtCore.QMetaObject.invokeMethod(
            self,
            "_set_rag_status_from_any_thread",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, text),
        )

    # ---- UI construction -------------------------------------------------

    def _build_layout(self) -> None:
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_nav())
        outer.addWidget(self._build_brand_strip())
        outer.addWidget(self._build_subnav())

        # Main three-column splitter: trace | chat | right column
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_trace_panel())
        splitter.addWidget(self._build_chat_panel())
        splitter.addWidget(self._build_right_column())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 4)
        splitter.setSizes([280, 520, 520])
        outer.addWidget(splitter, stretch=1)

        self._status_bar = QtWidgets.QStatusBar()
        self._status_bar.setObjectName("status")
        self.setStatusBar(self._status_bar)

    def _build_nav(self) -> QtWidgets.QWidget:
        nav = QtWidgets.QFrame()
        nav.setObjectName("nav")
        nav.setFixedHeight(48)
        layout = QtWidgets.QHBoxLayout(nav)
        layout.setContentsMargins(20, 0, 20, 0)
        title = QtWidgets.QLabel("LitTrace")
        title.setStyleSheet(f"color:{DESIGN['on_dark']};font-weight:600;font-size:15px;")
        layout.addWidget(title)
        layout.addStretch(1)
        session = QtWidgets.QLabel(f"session: {self._controller.session.session_id}")
        session.setStyleSheet(f"color:{DESIGN['on_dark']};font-size:11px;")
        layout.addWidget(session)
        return nav

    def _build_brand_strip(self) -> QtWidgets.QWidget:
        strip = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(strip)
        layout.setContentsMargins(28, 18, 28, 4)
        layout.setSpacing(10)
        brand = QtWidgets.QLabel("LitTrace")
        brand.setObjectName("brand")
        tagline = QtWidgets.QLabel("·  Materials & Chemistry Research")
        tagline.setObjectName("tagline")
        layout.addWidget(brand)
        layout.addWidget(tagline)
        layout.addStretch(1)
        return strip

    def _build_subnav(self) -> QtWidgets.QWidget:
        subnav = QtWidgets.QFrame()
        subnav.setObjectName("subnav")
        subnav.setFixedHeight(56)
        layout = QtWidgets.QHBoxLayout(subnav)
        layout.setContentsMargins(28, 0, 28, 0)
        layout.setSpacing(8)

        def add_btn(text: str, slot, primary: bool = False) -> QtWidgets.QPushButton:
            btn = QtWidgets.QPushButton(text)
            btn.setObjectName("subnav_btn_primary" if primary else "subnav_btn")
            btn.clicked.connect(slot)
            layout.addWidget(btn)
            return btn

        add_btn("文献上下文", self._open_context_popup)
        self._context_toggle_btn = add_btn("隐藏上下文", self._toggle_context)
        self._parse_strategy_btn = add_btn("文本层解析", self._toggle_parse_strategy, primary=True)
        add_btn("Setup browser", self._on_setup_browser)
        add_btn("Doctor", self._on_doctor)
        add_btn("使用说明", self._open_help_popup)
        layout.addStretch(1)
        return subnav

    def _build_trace_panel(self) -> QtWidgets.QWidget:
        self._trace_panel = TracePanel()
        return self._trace_panel

    def _build_chat_panel(self) -> QtWidgets.QWidget:
        self._chat_panel = ChatPanel(self._controller)
        return self._chat_panel

    def _build_right_column(self) -> QtWidgets.QWidget:
        right = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._context_panel = ContextPanel()
        layout.addWidget(self._context_panel, stretch=2)

        self._rag_panel = RAGPanel(on_run_daily=self._on_run_daily)
        layout.addWidget(self._rag_panel, stretch=1)

        self._browser_panel = BrowserPanel()
        layout.addWidget(self._browser_panel, stretch=4)
        return right

    # ---- Subnav actions -------------------------------------------------

    def _open_context_popup(self) -> None:
        papers = list(self._controller.list_active_papers())
        QtWidgets.QMessageBox.information(
            self,
            "文献上下文",
            "\n".join(
                f"• {p.title} ({p.year or 'n.d.'}, {p.access_type})"
                for p in papers
            ) or "暂无激活文献",
        )

    def _toggle_context(self) -> None:
        visible = self._context_panel.isVisible()
        self._context_panel.setVisible(not visible)
        self._context_toggle_btn.setText("显示上下文" if not visible else "隐藏上下文")

    def _toggle_parse_strategy(self) -> None:
        current = self._parse_strategy_btn.text()
        if current == "文本层解析":
            self._parse_strategy_btn.setText("OCR 解析")
        else:
            self._parse_strategy_btn.setText("文本层解析")
        self._status_bar.showMessage(f"解析模式：{self._parse_strategy_btn.text()}", 3000)

    def _on_setup_browser(self) -> None:
        cmd, env, cwd = _littrace_cmd("setup-browser")
        self._run_subprocess_action(cmd, env, cwd, "Chrome 配置完成")

    def _on_doctor(self) -> None:
        cmd, env, cwd = _littrace_cmd("doctor")
        self._run_subprocess_action(cmd, env, cwd, None)

    def _run_subprocess_action(
        self,
        cmd: list[str],
        env: dict[str, str],
        cwd: str,
        success_label: str | None,
    ) -> None:
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
                cwd=cwd,
                env=env,
            )
        except subprocess.TimeoutExpired:
            self._status_bar.showMessage(f"{' '.join(cmd)} 超时", 5000)
            return
        except Exception as exc:
            self._status_bar.showMessage(f"{' '.join(cmd)} 失败: {exc}", 5000)
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(" ".join(cmd))
        dialog.resize(720, 480)
        layout = QtWidgets.QVBoxLayout(dialog)
        view = QtWidgets.QPlainTextEdit()
        view.setPlainText((completed.stdout or "") + (completed.stderr or ""))
        view.setReadOnly(True)
        layout.addWidget(view)
        if success_label:
            layout.addWidget(QtWidgets.QLabel(success_label))
        dialog.exec()
        self._status_bar.showMessage(f"{' '.join(cmd)} 完成", 3000)

    def _open_help_popup(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "使用说明",
            "LitTrace Qt shell\n"
            "\n"
            "• 左侧：执行 Trace + 历史 Session（点击切换）\n"
            "• 中间：对话窗口，输入 / 触发 slash 命令自动完成\n"
            "• 右上：文献上下文 / RAG / Daily 操作\n"
            "• 右下：嵌入式 Chromium 浏览器（用于 publisher 页面）\n"
            "\n"
            "Slash 命令示例： /papers  /parse  /table  /storyline  /quit\n"
            "右键 chat 区域可复制 / 全选 / 清屏。",
        )

    def _on_run_daily(self) -> None:
        # The button opens ``DailyConfigDialog`` first. The dialog
        # blocks the chat thread until the user either accepts or
        # cancels; both branches return immediately, this method just
        # kicks off the worker (or the browser, if the user picked
        # "🌐 打开 publisher 登录") once we have a config.
        existing_topic = "mxene_sensor"
        dialog = DailyConfigDialog(
            self,
            default_topic=existing_topic,
            default_year_min=getattr(
                self._controller.config.literature_context, "default_year_min", 2023
            ),
            default_year_max=getattr(
                self._controller.config.literature_context, "default_recent_year_min", 2026
            ),
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        topic = dialog.topic()
        keywords = dialog.keywords()
        year_min = dialog.year_min()
        year_max = dialog.year_max()
        min_papers = dialog.min_papers()
        open_login = dialog.open_login_after()

        if open_login:
            self._launch_publisher_login_browser()
            return

        self._status_bar.showMessage(
            f"检索启动中…主题：{topic}（{year_min}-{year_max}，≥{min_papers} 篇）",
            3000,
        )
        self._rag_panel.set_status(f"运行中…主题 {topic}")
        import threading

        def _worker():
            try:
                cmd, env, cwd = _littrace_cmd(
                    "sentinel", "run",
                    "--watchlist", topic,
                    "--topic", keywords or topic,
                )
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=cwd,
                    env=env,
                )
                summary = _summarise_sentinel_output(
                    completed.stdout or ""
                )
                downloaded = _summary_value(summary, "downloaded:")
                candidates = _summary_value(summary, "new_candidates:")
                headline = (
                    f"检索完成（exit={completed.returncode}）· {summary}"
                )
                shortfall = []
                if downloaded is not None and downloaded < min_papers:
                    shortfall.append(
                        f"只下了 {downloaded} 篇，少于目标 {min_papers} 篇"
                    )
                if candidates is not None and candidates < 3:
                    shortfall.append(
                        f"只检索到 {candidates} 个候选——考虑扩大关键词或时间区间"
                    )
                if shortfall:
                    headline += " · ⚠️ " + "；".join(shortfall)
                self._post_status(headline)
            except subprocess.TimeoutExpired:
                self._post_status("检索超时（10 分钟）")
            except Exception as exc:  # pragma: no cover - defensive
                self._post_status(f"检索失败: {type(exc).__name__}: {exc}")

        threading.Thread(target=_worker, daemon=True, name="littrace-daily").start()

    def _launch_publisher_login_browser(self) -> None:
        # ``littrace setup-browser --launch`` boots the LitTrace-private
        # Chromium with ``--remote-debugging-port=19222`` against
        # ``./data/chrome-cdp`` (independent of the user's day-to-day
        # Chrome). Once the browser is up, the embedded BrowserPanel
        # can be pointed at each publisher's sign-in page so the user
        # can authenticate; the next sentinel run then has access to
        # gated PDFs.
        self._status_bar.showMessage("启动 publisher 登录浏览器…", 3000)
        self._rag_panel.set_status("等待 publisher 登录…")
        import threading

        def _worker():
            try:
                cmd, env, cwd = _littrace_cmd("setup-browser", "--launch")
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=cwd,
                    env=env,
                )
                if completed.returncode == 0:
                    self._post_status("publisher 浏览器已就绪（CDP 19222）")
                else:
                    tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-1]
                    self._post_status(
                        f"启动 publisher 浏览器失败（exit={completed.returncode}）：{tail}"
                    )
            except subprocess.TimeoutExpired:
                self._post_status("启动 publisher 浏览器超时（30 秒）")
            except Exception as exc:  # pragma: no cover - defensive
                self._post_status(
                    f"启动 publisher 浏览器失败: {type(exc).__name__}: {exc}"
                )

        threading.Thread(target=_worker, daemon=True, name="littrace-setup-browser").start()

    # ---- Event wiring ----------------------------------------------------

    def _wire_events(self) -> None:
        controller = self._controller

        # Qt widgets must only be touched from the GUI thread, but the
        # controller emits events on its asyncio worker thread. Earlier
        # iterations tried ``QTimer.singleShot(0, lambda)`` from the
        # worker thread — that bound the timer to a thread with no Qt
        # event loop, so the lambda never fired. The current bridge
        # declares ``@Slot``-decorated methods on ``self`` (so they live
        # in the Qt meta-object system) and then dispatches every event
        # via ``QMetaObject.invokeMethod(self, "...", QueuedConnection,
        # ...)`` from the worker thread. ``QueuedConnection`` posts the
        # call onto the receiver's thread (the GUI thread), where the
        # actual widget mutation runs. The ``@Slot`` registration is
        # what makes the method invokable — plain Python methods are
        # silently dropped by ``invokeMethod`` even with
        # ``QueuedConnection``.
        def post(handler_name: str) -> "callable":
            def _wrapper(event: ShellEvent) -> None:
                QtCore.QMetaObject.invokeMethod(
                    self,
                    handler_name,
                    QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG("QVariant", event),
                )

            return _wrapper

        controller.bus.subscribe(post("_qt_on_message_event"))
        controller.bus.subscribe(post("_qt_on_status_event"))
        controller.bus.subscribe(post("_qt_on_thinking_event"))
        controller.bus.subscribe(post("_qt_on_workspace_event"))
        controller.bus.subscribe(post("_qt_on_error_event"))

    @QtCore.Slot("QVariant")
    def _qt_on_message_event(self, event: ShellEvent) -> None:
        if event.kind != self._controller.EVENT_MESSAGE_APPENDED:
            return
        role = event.payload.get("role", "system")
        text = event.payload.get("text", "")
        extras = {k: v for k, v in event.payload.items() if k in ("action", "warnings")}
        self._chat_panel.append_message(role, text, **extras)

    @QtCore.Slot("QVariant")
    def _qt_on_status_event(self, event: ShellEvent) -> None:
        if event.kind != self._controller.EVENT_STATUS_CHANGED:
            return
        self._status_bar.showMessage(event.payload.get("text", ""))

    @QtCore.Slot("QVariant")
    def _qt_on_thinking_event(self, event: ShellEvent) -> None:
        if event.kind != self._controller.EVENT_THINKING:
            return
        self._chat_panel.set_thinking(
            active=bool(event.payload.get("active")),
            label=str(event.payload.get("label", "思考中")),
        )

    @QtCore.Slot("QVariant")
    def _qt_on_workspace_event(self, event: ShellEvent) -> None:
        if event.kind != self._controller.EVENT_WORKSPACE_REFRESHED:
            return
        self._context_panel.refresh(list(self._controller.list_active_papers()))
        self._trace_panel.render_workflow_trace(
            ["工作区刷新"]
            + [
                f"{i + 1}. {p.title}"
                for i, p in enumerate(self._controller.list_active_papers())
            ]
        )

    @QtCore.Slot("QVariant")
    def _qt_on_error_event(self, event: ShellEvent) -> None:
        if event.kind != self._controller.EVENT_ERROR:
            return
        self._chat_panel.append_message(
            "system", f"⚠️ {event.payload.get('message', 'error')}"
        )

    # ---- Initial state ---------------------------------------------------

    def _refresh_initial_state(self) -> None:
        self._context_panel.refresh(list(self._controller.list_active_papers()))
        self._trace_panel.render_workflow_trace(["等待任务…"])
        try:
            sessions = list_chat_sessions(self._controller.config)
            self._trace_panel.set_sessions(
                sessions, current_session_id=self._controller.session.session_id
            )
        except Exception:
            pass
        self._status_bar.showMessage("就绪")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_argv(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="littrace-qt",
        description="LitTrace desktop shell (Qt + QtWebEngine).",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("LITTRACE_CONFIG", "config.yaml"),
        help="Path to config.yaml (also honors LITTRACE_CONFIG).",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Optional URL to load in the embedded browser pane on start.",
    )
    parser.add_argument(
        "--headless-smoke",
        action="store_true",
        help="Build the window, fire a chat turn, wait for the reply, then "
        "exit cleanly. Used by CI to confirm QtWebEngine + ShellController "
        "work without a display.",
    )
    return parser.parse_args(argv)


def _await_status_idle(
    controller: ShellController,
    app: QtWidgets.QApplication,
    state: dict[str, bool],
    timeout_ms: int = 45000,
) -> None:
    """Quit ``app`` once the controller emits status='就绪' *and* we have
    seen an assistant message. Falls back to ``timeout_ms`` if the chat
    turn never completes.
    """

    def _on_message(event: ShellEvent) -> None:
        if event.kind == controller.EVENT_MESSAGE_APPENDED and event.payload.get("role") == "assistant":
            state["got_reply"] = True
            print(
                f"[smoke] reply action={event.payload.get('action')!r} "
                f"text={event.payload.get('text', '')[:120]!r}",
                flush=True,
            )

    def _on_status(event: ShellEvent) -> None:
        if event.kind == controller.EVENT_STATUS_CHANGED and state["got_reply"]:
            if event.payload.get("text") == "就绪":
                state["done"] = True
                app.quit()

    def _on_error(event: ShellEvent) -> None:
        if event.kind == controller.EVENT_ERROR:
            print(f"[smoke] ERROR: {event.payload}", flush=True)
            state["done"] = True
            app.quit()

    controller.bus.subscribe(_on_message)
    controller.bus.subscribe(_on_status)
    controller.bus.subscribe(_on_error)
    QtCore.QTimer.singleShot(timeout_ms, app.quit)


def main(argv: list[str] | None = None) -> int:
    args = _parse_argv(argv if argv is not None else sys.argv[1:])

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"littrace-qt: {exc}", file=sys.stderr)
        return 2

    QtCore.QCoreApplication.setOrganizationName("LitTrace")
    QtCore.QCoreApplication.setApplicationName("LitTrace Qt")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    controller = ShellController(config)
    controller.start()
    window = LitTraceQtWindow(controller)

    if args.url:
        window._browser_panel.open_url(args.url)

    if args.headless_smoke:
        state: dict[str, bool] = {"got_reply": False, "done": False}
        _await_status_idle(controller, app, state)
        window.show()
        controller.submit_user_message("ping")
        rc = app.exec()
        controller.stop()
        if not state["got_reply"]:
            print("[smoke] FAIL: no assistant reply within timeout", flush=True)
            return 1
        print("[smoke] OK", flush=True)
        return rc

    window.show()
    rc = app.exec()
    controller.stop()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())