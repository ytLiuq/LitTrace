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
import json
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


# Map the structured ``reason`` strings the controller emits on
# ``EVENT_AUTH_REQUIRED`` to user-friendly Chinese one-liners shown
# above the device-auth steps in ``_show_auth_dialog``. Keeping this
# in the Qt module (not in the controller) lets the controller stay
# GUI-agnostic while the dialog text matches the shell's voice.
_AUTH_REASON_TEXT = {
    "missing_auth_file": "找不到 Codex 登录凭据文件",
    "auth_file_unreadable": "Codex 凭据文件解析失败",
    "no_tokens": "Codex 凭据文件没有 tokens 字段",
    "no_id_token": "Codex 凭据文件缺少 id_token",
    "unparseable_jwt": "Codex id_token 不是合法 JWT",
    "token_expired": "Codex 登录已过期",
}


def _reason_to_text(reason: str, detail: str) -> str:
    base = _AUTH_REASON_TEXT.get(reason, "Codex 登录状态异常")
    if detail:
        return f"{base}：{detail}"
    return base


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
    body = re.sub(r"^###### ([^\n]+)$", heading("6"), body, flags=re.MULTILINE)
    body = re.sub(r"^##### ([^\n]+)$", heading("5"), body, flags=re.MULTILINE)
    body = re.sub(r"^#### ([^\n]+)$", heading("5"), body, flags=re.MULTILINE)
    body = re.sub(r"^### ([^\n]+)$", heading("4"), body, flags=re.MULTILINE)
    body = re.sub(r"^## ([^\n]+)$", heading("4"), body, flags=re.MULTILINE)
    body = re.sub(r"^# ([^\n]+)$", heading("3"), body, flags=re.MULTILINE)
    body = re.sub(r"^> ([^\n]+)$", bq, body, flags=re.MULTILINE)
    body = re.sub(r"^(?:[-*] )([^\n]+)$", li_dash, body, flags=re.MULTILINE)
    body = re.sub(r"^\d+\. ([^\n]+)$", li_num, body, flags=re.MULTILINE)
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
#
# Each entry is matched as a ``re.search`` with a single trailing
# pattern (no leading ``[\s\S]*?``) so a greedy quantifier on the
# prefix doesn't eat the whole reply. The pattern is anchored to the
# end of the string with ``$`` (with ``re.DOTALL`` so the optional
# whitespace can span line breaks). If the last match is at position
# 0 the entire reply would have been narration and we drop it; in
# practice codex only emits the trailing sentence once the rest of
# the answer is already on the page, so position 0 is unusual.
_TRAILING_NARRATION_PATTERNS = [
    r"用户不关注[^。]{0,80}\s*$",
    r"用户不需要[^。]{0,80}\s*$",
    r"不.{0,4}直接展示给用户\s*$",
    r"不.{0,4}展示给用户\s*$",
    r"不.{0,8}说给用户\s*$",
    r"不.{0,8}用.{0,4}看\s*$",
    r"I['’]?ll keep the rest concise\.?\s*$",
    r"Keep the rest concise\.?\s*$",
    r"I['’]?ll be concise\.?\s*$",
    r"I'll skip the rest\.?\s*$",
    r"I['’]?ll skip the internal details\.?\s*$",
    r"Skipping internal details\.?\s*$",
    r"The rest is internal\.?\s*$",
    r"Internal notes removed\.?\s*$",
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

    Implementation note: ``re.search`` + a leading ``[\\s\\S]*?`` is
    wrong — the lazy quantifier still expands to cover the whole
    prefix and the match consumes the entire reply. ``re.finditer``
    enumerates all matches in left-to-right order; the **last** match
    is the one that touches the trailing sentence, so the start of
    that last match is the right cut point.
    """
    for pattern in _TRAILING_NARRATION_PATTERNS:
        matches = list(re.finditer(pattern, html))
        if matches:
            return html[: matches[-1].start()].rstrip()
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

COMMAND_CATALOG: list[tuple[str, str, bool, str]] = [
    # Group: 论文库
    ("context", "显示 / 隐藏当前文献上下文", True, "论文库"),
    ("papers", "列出当前上下文文献", True, "论文库"),
    ("全部下载", "选择当前上下文中全部待下载文献", True, "论文库"),
    ("选择第 N 篇下载", "选择第 N 篇进入下载计划", True, "论文库"),
    ("取消选择第 N 篇", "从下载计划中移除第 N 篇", True, "论文库"),
    ("check-downloads", "检查当前下载计划", True, "论文库"),
    ("resume-downloads", "恢复下载（等待用户授权）", True, "论文库"),
    # Group: 解析与抽取
    ("parse", "按当前解析模式处理 PDF", False, "解析"),
    ("parse --ocr", "强制使用 OCR 解析", False, "解析"),
    ("parse --text", "强制使用文本层解析", False, "解析"),
    ("table", "抽取性能指标并生成对比表", True, "解析"),
    ("storyline", "梳理论文回应关系", True, "解析"),
    ("storyline-report", "导出 storyline 报告", True, "解析"),
    ("storyline-review", "Reviewer 审阅 storyline", True, "解析"),
    # Group: 工具
    ("dashboard", "打开 RAG / Daily 仪表盘", True, "工具"),
    ("quality", "运行质量门", True, "工具"),
    ("agents", "列出可用 agents", False, "工具"),
    ("workflow", "显示当前 workflow trace", True, "工具"),
    ("quality-audits", "运行质量审计", True, "工具"),
    ("plan", "显示当前执行计划", False, "工具"),
    ("init-config", "运行 config wizard", False, "工具"),
    ("login", "打开授权登录弹窗", True, "工具"),
    ("attach", "手动附加本地 PDF", False, "工具"),
    ("attach-si", "附加 SI / 补充材料", False, "工具"),
    ("full-text", "构建 full-text context", True, "工具"),
    ("backfill-dois", "回填 DOI", False, "工具"),
    ("publisher-retrieve", "按 publisher 抓取", False, "工具"),
    ("benchmark", "运行评测基准", False, "工具"),
    ("golden-eval", "运行 golden set 评估", False, "工具"),
    ("export", "导出当前 session", False, "工具"),
    ("quit", "关闭窗口", False, "工具"),
    ("取消选择第 N 篇", "从下载计划中移除第 N 篇", True, "论文库"),
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
        history_title = QtWidgets.QLabel("历史 Session（点击切换）")
        history_title.setObjectName("pane_title")
        layout.addWidget(history_title)

        self._sessions = QtWidgets.QListWidget()
        self._sessions.setObjectName("sessions")
        self._sessions.setMaximumHeight(220)
        # Round 17: surface the click-to-switch affordance. Both
        # ``itemClicked`` and ``itemActivated`` are wired because
        # Qt's accessibility / keyboard default-activation path
        # uses ``itemActivated``, while the mouse path uses
        # ``itemClicked``. Connecting both keeps the behaviour
        # consistent regardless of input method.
        self._sessions.itemClicked.connect(self._on_session_clicked)
        self._sessions.itemActivated.connect(self._on_session_clicked)
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
            item.setToolTip(
                f"session_id: {session.session_id}\n"
                f"创建时间: {session.created_at}\n"
                "点击切换到该 session"
            )
            if session.session_id == current_session_id:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setForeground(QtGui.QColor("#3a8a8c"))
            self._sessions.addItem(item)

    def _on_session_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        session_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not session_id:
            return
        # Hand off to the parent LitTraceQtWindow which owns the
        # controller. ``TracePanel`` stays GUI-only and never
        # imports the controller, so the parent binds the click
        # to ``controller.switch_session``.
        window = self.window()
        if window is not None and hasattr(window, "_on_session_switch_requested"):
            window._on_session_switch_requested(session_id)

    # ---- Session switching ---------------------------------------------

    def _on_session_switch_requested(self, session_id: str) -> None:
        # Round 17: ``TracePanel`` hands session clicks here so the
        # controller-driven switch + UI refresh happens in one
        # place. ``controller.switch_session`` emits
        # ``SESSION_HISTORY_REFRESHED`` on success, which our
        # already-wired slot re-renders against.
        if session_id == self._controller.session.session_id:
            self._status_bar.showMessage(
                f"已在 session {session_id}", 3000
            )
            return
        ok = self._controller.switch_session(session_id)
        if ok:
            # Re-render the session list so the new active row is
            # bolded; refresh the context pane to match the new
            # session's active papers; clear the chat scrollback so
            # the user doesn't see a confusing mix of two
            # sessions' messages.
            try:
                sessions = self._controller.list_sessions()
                self._trace_panel.set_sessions(
                    sessions,
                    current_session_id=self._controller.session.session_id,
                )
            except Exception:
                pass
            self._context_panel.refresh(
                list(self._controller.list_active_papers())
            )
            self._chat_panel.clear()

    def _on_paper_deactivate_requested(self, paper: Any) -> None:
        # Round 17: ``ContextPanel`` hands the right-click
        # "取消激活" action here. ``controller.deactivate_paper``
        # already emits ``EVENT_WORKSPACE_REFRESHED`` so the
        # context list re-renders against the new active set.
        paper_id = getattr(paper, "paper_id", None)
        if not paper_id:
            return
        ok = self._controller.deactivate_paper(paper_id)
        if ok:
            self._status_bar.showMessage(
                f"已取消激活：{paper.title[:40]}",
                3000,
            )
        else:
            self._status_bar.showMessage(
                f"该 paper 不在激活列表中：{paper.title[:40]}",
                3000,
            )


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
        self._view.setOpenExternalLinks(False)
        self._view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._on_chat_context_menu)
        # Round 17: surface "查看技术细节" inline error links. The
        # link target uses the ``littrace:show-error-detail`` scheme
        # so a click never opens an external browser; the parent
        # window's ``_on_chat_anchor_clicked`` slot pops the raw
        # ``TypeError: ...`` payload into a non-modal dialog.
        # ``setOpenExternalLinks(False)`` keeps http(s) links from
        # auto-opening — those still go through the parent slot
        # and we re-emit them via QDesktopServices so the user's
        # default browser handles them.
        # Round 17 bug fix: ``_on_chat_anchor_clicked`` lives on
        # the main window, not on ChatPanel; the previous
        # ``self._view.anchorClicked.connect(self._on_chat_anchor_clicked)``
        # silently referenced a non-existent method on the
        # panel. Forward through ``self.window()`` so the main
        # window's slot handles the click.
        self._view.anchorClicked.connect(
            lambda url, _panel=self: _panel.window()._on_chat_anchor_clicked(url)
        )
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
        self._input.setPlaceholderText("输入研究问题，Enter 发送，Shift+Enter 换行…")
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
        # Build the popup with one separator row per group so the user
        # can scan the available commands by category. The
        # ``_is_separator`` flag on each item tells the filter logic
        # to hide separators when a query is non-empty.
        self._popup_meta: dict[int, str] = {}  # row -> "separator:<group>"
        last_group: str | None = None
        for name, desc, _ctx, group in COMMAND_CATALOG:
            if group != last_group:
                sep = QtWidgets.QListWidgetItem(group)
                sep.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                sep.setData(QtCore.Qt.ItemDataRole.UserRole, None)
                sep.setBackground(QtGui.QColor("#f5f6f6"))
                sep.setForeground(QtGui.QColor("#a4a7ad"))
                font = sep.font()
                font.setBold(True)
                font.setPointSize(font.pointSize() - 1)
                sep.setFont(font)
                self._popup.addItem(sep)
                self._popup_meta[self._popup.count() - 1] = (
                    f"separator:{group}"
                )
                last_group = group
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
        elapsed = getattr(self, "_thinking_elapsed", None)
        if elapsed is not None:
            # Show one decimal place up to 10 s, then integer —
            # past 10 s the user mostly cares about "is it still
            # alive" rather than sub-second precision.
            if elapsed < 10:
                elapsed_text = f"{elapsed:.1f}s"
            else:
                elapsed_text = f"{int(elapsed)}s"
            self._thinking.setText(
                f"{self._thinking_label}  {dots}  ({elapsed_text})"
            )
        else:
            self._thinking.setText(f"{self._thinking_label}  {dots}")

    def set_thinking(self, active: bool, label: str = "") -> None:
        if active:
            self._thinking_label = label or "思考中"
            self._thinking_elapsed: float | None = None
            self._thinking.show()
            self._tick_thinking()
            self._thinking_timer.start()
        else:
            self._thinking_timer.stop()
            self._thinking.hide()
            self._thinking_label = ""
            self._thinking_elapsed = None

    def set_thinking_elapsed(self, elapsed_seconds: float) -> None:
        """Round 17: append the elapsed-seconds counter so the user
        sees "思考中… 3.2s" instead of a frozen spinner. The label
        and the dots are kept in sync via ``_tick_thinking`` so the
        elapsed seconds never overwrite the dots (or vice versa).
        """
        self._thinking_elapsed = elapsed_seconds
        if self._thinking.isVisible():
            self._tick_thinking()

    # ---- Slash popup logic ----------------------------------------------

    def _next_command_matches(self, separator_row: int) -> bool:
        """Return True if the next non-separator row after
        ``separator_row`` is visible under the current filter. Used by
        the per-row visibility logic to decide whether a separator
        should stay on screen.
        """
        for row in range(separator_row + 1, self._popup.count()):
            item = self._popup.item(row)
            if self._popup_meta.get(row, "").startswith("separator:"):
                return False
            if not item.isHidden():
                return True
        return False

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
        # With a non-empty query we also hide the group separator rows
        # so the filtered list reads as a single flat menu.
        hide_separators = bool(query)
        visible_rows: list[int] = []
        for row in range(self._popup.count()):
            item = self._popup.item(row)
            meta = self._popup_meta.get(row, "")
            is_separator = meta.startswith("separator:")
            if is_separator and hide_separators:
                item.setHidden(True)
                continue
            name = item.data(QtCore.Qt.ItemDataRole.UserRole) or ""
            # Fuzzy: command name or description contains the query as a
            # substring. ``startswith`` is too strict — ``/par`` would
            # miss ``parse --ocr``.
            text_match = (
                not query
                or name.lower().find(query) >= 0
                or item.text().lower().find(query) >= 0
            )
            if is_separator:
                # A separator stays visible only if the next command
                # in the catalog is a match (so groups don't appear
                # empty once their entries are filtered out).
                item.setHidden(not self._next_command_matches(row))
                if not item.isHidden():
                    visible_rows.append(row)
            else:
                item.setHidden(not text_match)
                if text_match:
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
        # ``ensureCursorVisible`` is not reliable enough on
        # ``QTextBrowser`` once the cursor is at the very end (the
        # last paragraph can be the same height as the viewport and
        # nothing scrolls). Set the vertical scroll bar to its
        # maximum directly so the new bubble always appears at the
        # bottom of the scrollback.
        sb = self._view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_chat_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        copy_action = menu.addAction("复制")
        copy_action.setEnabled(self._view.textCursor().hasSelection())
        copy_action.triggered.connect(self._view.copy)
        select_all = menu.addAction("全选")
        select_all.triggered.connect(self._view.selectAll)
        clear_action = menu.addAction("清屏")
        clear_action.triggered.connect(self.clear)
        menu.exec(self._view.mapToGlobal(pos))

    def clear(self) -> None:
        # Round 17: clear the scrollback AND the streaming anchor
        # so the next chat turn starts with a fresh bubble. The
        # right-click "清屏" menu item and the session switch path
        # both go through here.
        self._view.clear()
        self._streaming_anchor = None

    # ---- Streaming bubble -----------------------------------------------

    def open_streaming_bubble(self) -> None:
        """Open a new empty assistant bubble and record its block
        position so subsequent ``append_delta`` calls append at its
        tail regardless of where new blocks land in the document.

        Round 17: ``_view.textCursor()`` returns a cursor at position
        0 on first read, so a naive ``insertText`` from a delta
        callback would insert at the TOP of the document — which is
        why an earlier attempt at streaming showed the assistant's
        reply inside the user's bubble. The contract is:

          * ``open_streaming_bubble`` appends a new block to the
            document end and records the new block's character
            position as ``self._streaming_anchor``.
          * ``append_delta`` moves the cursor to that anchor and
            then to ``EndOfBlock`` before inserting, so even if the
            user fires off another turn (which appends a new block
            *after* the streaming bubble), the delta still lands at
            the streaming bubble's tail.

        ``_streaming_anchor`` is ``None`` when no streaming bubble
        is open; ``append_delta`` is a no-op in that state.
        """
        from PySide6.QtGui import QTextBlockFormat

        bubble_color = "#ffffff"
        text_color = DESIGN["ink"]
        wrap = "12px 12px 12px 4px"  # assistant tail bottom-left
        max_width = "78%"
        cursor = self._view.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        block_fmt.setTopMargin(8)
        block_fmt.setBottomMargin(8)
        cursor.insertBlock(block_fmt)
        # Remember the block start position before ``insertHtml``
        # moves the cursor past the inserted span. The anchor is
        # the character offset of the new block's first character;
        # ``append_delta`` calls ``setPosition`` then ``EndOfBlock``
        # to land at the tail.
        doc = self._view.document()
        anchor = doc.lastBlock().position()
        cursor.insertHtml(
            f'<span style="display:inline-block;max-width:{max_width};'
            f"background:{bubble_color};color:{text_color};"
            f"border:1px solid {DESIGN['hairline']};"
            f"border-radius:{wrap};padding:8px 12px;"
            f"line-height:1.4;"
            f'"></span>'
        )
        self._streaming_anchor = anchor
        self._view.setTextCursor(cursor)
        sb = self._view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def append_delta(self, delta: str) -> None:
        """Append ``delta`` to the streaming bubble.

        ``append_delta`` is called many times per turn (codex ships
        deltas as small as a few characters), so it MUST be cheap.
        The implementation jumps the cursor to the streaming
        anchor's block, moves to the end of that block, and calls
        ``insertText`` — no markdown re-render, no block-format
        rewrite. Falls back to a no-op if the streaming bubble
        isn't open yet (race: a stray delta arrived before
        ``open_streaming_bubble``).
        """
        if not delta:
            return
        anchor = getattr(self, "_streaming_anchor", None)
        if anchor is None:
            return
        cursor = self._view.textCursor()
        cursor.setPosition(anchor)
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfBlock)
        cursor.insertText(delta)
        self._view.setTextCursor(cursor)
        sb = self._view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def finalize_streaming(self, full_text: str = "") -> None:
        """Close the streaming bubble.

        Round 17: the chat path emits ``EVENT_MESSAGE_APPENDED`` after
        every assistant turn. If we left the streaming bubble open
        AND appended a new ``assistant`` bubble for the same reply,
        the user would see the streamed text + the same text rendered
        as a new bubble below it. So at finalize time we have two
        choices:

          (a) drop the streaming bubble entirely and let
              ``EVENT_MESSAGE_APPENDED`` render the final reply as
              a fresh bubble (current behaviour when streaming is
              disabled).
          (b) keep the streaming bubble, replace its content with
              the markdown-rendered final reply, and skip the new
              bubble from ``EVENT_MESSAGE_APPENDED``.

        We do (b): the streamed text is the same text the user will
        see in the final bubble, but with raw ``\n`` separators
        instead of ``<br>``. Replacing the bubble body with the
        rendered markdown (bold / code / lists / …) lifts the
        fidelity in one step at the end of the turn, without
        re-rendering every delta mid-stream.

        ``full_text`` is the controller's authoritative final reply;
        if the streaming bubble never opened (e.g. deltas arrived
        after ``completed``), ``full_text`` is the only content the
        user sees — and ``replace_streaming_bubble`` falls back to
        ``append_message`` in that case.
        """
        anchor = getattr(self, "_streaming_anchor", None)
        if anchor is None:
            # Streaming bubble never opened (race: deltas arrived
            # after ``completed`` without a preceding ``open``).
            # Render the final text as a normal assistant bubble
            # so the user still sees the reply.
            if full_text:
                self.append_message("assistant", full_text)
            return
        if full_text:
            self.replace_streaming_bubble(full_text)
        self._streaming_anchor = None

    def replace_streaming_bubble(self, full_text: str) -> None:
        """Swap the streaming bubble's body for the markdown-rendered
        final reply. Used at the end of a streaming turn so the user
        sees bold / code / lists instead of the raw streamed text.
        """
        anchor = getattr(self, "_streaming_anchor", None)
        if anchor is None:
            return
        body = _strip_leading_narration(_render_message_html(full_text))
        body = _strip_trailing_narration(body)
        cursor = self._view.textCursor()
        cursor.setPosition(anchor)
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfBlock)
        # ``cursor.hasSelection()`` is False — we are collapsed at the
        # end of the block, which contains the streaming span. The
        # span is the LAST element in the block, so a forward
        # selection to the block end + a backward selection to the
        # span start picks up everything inside the span. Using
        # ``StartOfBlock`` → ``EndOfBlock`` would also work but
        # would erase any whitespace or block-level markers we want
        # to keep — the block itself only contains the streaming
        # span in practice.
        cursor.movePosition(
            QtGui.QTextCursor.MoveOperation.StartOfBlock,
            QtGui.QTextCursor.MoveMode.MoveAnchor,
        )
        cursor.movePosition(
            QtGui.QTextCursor.MoveOperation.EndOfBlock,
            QtGui.QTextCursor.MoveMode.KeepAnchor,
        )
        # Replace the selection (which is the entire streaming span)
        # with the freshly rendered HTML.
        cursor.insertHtml(
            f'<span style="display:inline-block;max-width:78%;'
            f"background:#ffffff;color:{DESIGN['ink']};"
            f"border:1px solid {DESIGN['hairline']};"
            f"border-radius:12px 12px 12px 4px;padding:8px 12px;"
            f"line-height:1.4;"
            f'">{body}</span>'
        )
        self._view.setTextCursor(cursor)


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

        # Round 17: one-line hint so the user knows the right-click
        # menu exists. Without it the only way to discover the
        # "取消激活" action was trial and error.
        hint = QtWidgets.QLabel("右键单条可取消激活")
        hint.setObjectName("status")
        hint.setStyleSheet(
            f"color:{DESIGN['ink_subtle']};font-size:11px;"
        )
        layout.addWidget(hint)

        self._list = QtWidgets.QListWidget()
        self._list.setObjectName("context")
        # Right-click context menu for the per-paper actions
        # ("取消激活" / "查看详情"). Custom menu policy keeps the
        # menu off when the user clicks empty space.
        self._list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(
            self._on_context_menu
        )
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
            # Round 17: hover tooltip surfaces the metadata that
            # doesn't fit on the list row (DOI, full author list,
            # access type, citation count). The 5-line wrap is
            # enough for typical paper metadata; longer
            # abstracts / methods would need a separate dialog
            # but the user can grep the digest for those.
            tooltip_parts = [
                f"标题：{paper.title}",
                f"年份：{year}",
                f"来源：{source}",
            ]
            if getattr(paper, "doi", None):
                tooltip_parts.append(f"DOI：{paper.doi}")
            if getattr(paper, "authors", None):
                authors = paper.authors or []
                if authors:
                    shown = "、".join(authors[:3])
                    if len(authors) > 3:
                        shown += f" 等 {len(authors)} 位"
                    tooltip_parts.append(f"作者：{shown}")
            if getattr(paper, "access_type", None):
                tooltip_parts.append(
                    f"访问类型：{paper.access_type.value if hasattr(paper.access_type, 'value') else paper.access_type}"
                )
            if getattr(paper, "citation_count", None):
                tooltip_parts.append(f"引用数：{paper.citation_count}")
            item.setToolTip("\n".join(tooltip_parts))
            self._list.addItem(item)

    def _on_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        paper = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if paper is None:
            return
        menu = QtWidgets.QMenu(self)
        deactivate = menu.addAction("取消激活")
        deactivate.triggered.connect(
            lambda: self._request_deactivate(paper)
        )
        menu.exec(self._list.mapToGlobal(pos))

    def _request_deactivate(self, paper: Any) -> None:
        # Delegate to the parent LitTraceQtWindow which owns the
        # controller. The window's slot calls
        # ``controller.deactivate_paper`` and emits the
        # ``WORKSPACE_REFRESHED`` event our existing slot already
        # handles, so the list re-renders against the new state.
        window = self.window()
        if window is not None and hasattr(window, "_on_paper_deactivate_requested"):
            window._on_paper_deactivate_requested(paper)


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

        run_btn = QtWidgets.QPushButton("🔍 搜索研究主题")
        run_btn.setObjectName("subnav_btn_primary")
        run_btn.clicked.connect(on_run_daily)
        layout.addWidget(run_btn)

        self._status = QtWidgets.QLabel("尚未启动")
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

    # The embedded Chromium starts on a small data-URL welcome page
    # so the user sees actionable guidance instead of a blank
    # ``about:blank``. The data URL carries the same publisher
    # shortcut rail as the toolbar below it, plus a one-line note
    # about ``auto_launch_chrome`` having just brought the private
    # Chrome up for them.
    HOME_URL = (
        "data:text/html;charset=utf-8,"
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>LitTrace Publisher Browser</title>"
        "<style>"
        "  body { font-family: -apple-system, 'Helvetica Neue', sans-serif;"
        "         margin: 24px; color: #0b0c0e; background: #fbfbfc; }"
        "  h1 { font-size: 18px; margin: 0 0 12px; }"
        "  p { color: #5c6068; line-height: 1.55; margin: 8px 0; }"
        "  code { background: #f7f8f8; padding: 1px 6px; border-radius: 3px;"
        "         font-family: Menlo, Consolas, monospace; font-size: 12px; }"
        "  a { color: #3a8a8c; text-decoration: none; }"
        "  .ok { color: #2a7a3a; }"
        "</style></head><body>"
        "<h1>LitTrace Publisher Browser</h1>"
        "<p class='ok'>✓ LitTrace private Chrome is up on "
        "<code>http://127.0.0.1:19222</code>.</p>"
        "<p>Pick a publisher above to open its sign-in page here. Once "
        "you sign in, the session cookie is stored in LitTrace's "
        "private profile so the next <em>检索并补全文献</em> run can "
        "reach the gated PDFs.</p>"
        "<p>Need a publisher that's not on the row? Paste its URL in "
        "the bar above to navigate manually.</p>"
        "</body></html>"
    )

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

    def __init__(self, parent: QtWidgets.QWidget | None = None, config=None) -> None:
        super().__init__(parent)
        self.setObjectName("browser_tile")
        # ``config`` is optional so the panel can be used standalone
        # (e.g. unit tests). When None the cookie-status strip renders
        # an explicit "(no config)" placeholder instead of crashing.
        self._config = config

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

        # Per-publisher cookie status strip — sits between the shortcut
        # row and the URL bar. The strip lists the same publishers
        # as the shortcut row, but renders ``✓ logged in`` or ``✗ not
        # logged in`` based on what ``_detect_publisher_cookie_domains``
        # finds in ``./data/chrome-cdp``. The check runs once on
        # construction; it is cheap (reads the cookie SQLite files
        # directly, no network). Clicking a status pill reopens the
        # publisher's sign-in page so the user can recover a stale
        # session without leaving the panel.
        self._cookie_status = QtWidgets.QLabel("")
        self._cookie_status.setObjectName("cookie_status")
        self._cookie_status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._cookie_status.setWordWrap(False)
        # Treat the ✗ markers as in-label links so clicking them jumps
        # the embedded Chromium to the publisher's sign-in page. The
        # full HTML string is re-set on every refresh; the slot only
        # needs to be connected once.
        self._cookie_status.linkActivated.connect(self.open_url)
        self._cookie_status.setStyleSheet(
            f"color:{DESIGN['ink_muted']};font-size:11px;padding:2px 10px;"
        )
        self._cookie_status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
            | QtCore.Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        cookie_strip = QtWidgets.QHBoxLayout()
        cookie_strip.setContentsMargins(8, 0, 8, 0)
        cookie_strip.addWidget(self._cookie_status, stretch=1)
        self._cookie_refresh_btn = QtWidgets.QPushButton("↻")
        self._cookie_refresh_btn.setObjectName("cookie_refresh")
        self._cookie_refresh_btn.setFixedSize(24, 20)
        self._cookie_refresh_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._cookie_refresh_btn.setToolTip("重新读取 cookie 状态")
        self._cookie_refresh_btn.clicked.connect(self._refresh_cookie_status)
        cookie_strip.addWidget(self._cookie_refresh_btn)
        layout.addLayout(cookie_strip)
        self._refresh_cookie_status()

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

    def _refresh_cookie_status(self) -> None:
        # Map each publisher shortcut to a list of cookie domains that
        # the littrace config knows about. The cookie detector searches
        # the user's private Chrome profile (``./data/chrome-cdp``)
        # for any of those domains; if at least one matches, the
        # publisher is "logged in".
        from littrace.chrome_profiles import (
            _detect_publisher_cookie_domains,
        )
        # Group domains by publisher (using the same shorthand names
        # the shortcut row uses).
        publisher_domains: list[tuple[str, list[str]]] = [
            ("Wiley", ["wiley.com", "onlinelibrary.wiley.com"]),
            ("ACS", ["acs.org", "pubs.acs.org"]),
            ("Springer", ["springer.com"]),
            ("Nature", ["nature.com"]),
        ]
        # Resolve the LitTrace-private Chrome profile path. The
        # detector only looks at the directory; we don't require the
        # file to exist (a fresh install simply reads as "all
        # unlogged in").
        if self._config is None:
            self._cookie_status.setText("(no config)")
            return
        from pathlib import Path
        profile_root = (
            self._config.cdp_downloader.chrome_user_data_dir
        ).expanduser()
        try:
            present = set(_detect_publisher_cookie_domains(Path(profile_root)))
        except Exception:
            present = set()
        bits: list[str] = []
        # Map the short label back to the publisher sign-in URL so the
        # ✗ marker becomes a clickable shortcut: clicking ✗ fires
        # ``BrowserPanel.open_url`` for that publisher, which is the
        # exact behaviour the user asked for ("主动弹出来就可以了").
        any_unlogged = False
        for label, domains in publisher_domains:
            logged = any(d in present for d in domains)
            signin_url = next(
                (u for btn_label, u in self.PUBLISHER_LINKS if btn_label.endswith(label)),
                None,
            )
            # Round 17: render each ✓ / ✗ marker with a hover
            # tooltip so the user can hover before clicking.
            # ``title=`` is plain text only, no HTML, so the
            # tooltip strings are rendered as-is by Qt.
            if logged:
                color = "#2a7a3a"
                mark = "✓"
                bits.append(
                    f'<span title="{label} 已登录（cookie 来自 {", ".join(domains)}）" '
                    f'style="color:{color};">{label} {mark}</span>'
                )
            else:
                any_unlogged = True
                # ✗ is a clickable shortcut: the click fires
                # ``open_url(signin_url)`` so the embedded Chromium
                # navigates straight to the publisher's sign-in page.
                tooltip = f"{label} 未登录，点 ✗ 打开登录页"
                bits.append(
                    f'<a href="{signin_url}" title="{tooltip}" '
                    f'style="color:#cc785c;text-decoration:none;">'
                    f"{label} ✗</a>"
                )
        # arXiv doesn't need login (open access) — mark it as always
        # ready so the user doesn't have to wonder.
        bits.append(
            f'<span title="arXiv 是开放获取，无需登录" '
            f'style="color:#2a7a3a;">arXiv ✓</span>'
        )
        # The first time the user opens the panel, the strip is a wall
        # of red ✗ and they have no idea what to do. Drop a tiny hint
        # so they know the ✗ markers are clickable sign-in shortcuts.
        if any_unlogged:
            bits.append(
                '<span style="color:#a4a7ad;font-size:11px;">'
                "&nbsp;· 点 ✗ 一键登录</span>"
            )
        self._cookie_status.setText("  ".join(bits))

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
        # ``auto_launch_chrome`` is true by default but no caller in the
        # project actually invokes ``launch_chrome_for_cdp`` — so the
        # private Chromium on ``./data/chrome-cdp`` only came up when
        # the user clicked "🌐 打开 publisher 登录" in the daily dialog.
        # Do it here too, in a background thread, so the embedded
        # BrowserPanel's publisher buttons actually work the first time
        # the user opens one. The thread is daemon, the work is gated
        # by ``check_cdp_status`` (no-op if Chrome is already up), and
        # failures are swallowed so an auto-launch miss never blocks
        # the chat thread.
        if getattr(self._controller.config.cdp_downloader, "auto_launch_chrome", False):
            self._auto_launch_chrome()

    def _auto_launch_chrome(self) -> None:
        import threading
        from littrace.chrome_profiles import launch_chrome_for_cdp

        def _worker():
            try:
                result = launch_chrome_for_cdp(self._controller.config)
                if result.launched:
                    self._post_status("publisher Chrome 已启动（CDP 19222）")
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True, name="littrace-auto-chrome").start()

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

        add_btn("论文库", self._open_context_popup)
        self._context_toggle_btn = add_btn("隐藏论文库", self._toggle_context)
        self._parse_strategy_btn = add_btn("解析：文本层", self._toggle_parse_strategy, primary=True)
        add_btn("设置浏览器", self._on_setup_browser)
        add_btn("诊断", self._on_doctor)
        add_btn("帮助", self._open_help_popup)
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

        self._browser_panel = BrowserPanel(config=self._controller.config)
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
            # Multi-round retrieval. The user's "最少检索数目" is a
            # target, not a hard cap. Each round queries sentinel with
            # a different topic variation so the underlying API
            # returns fresh candidates. Sentinel dedupes on paper id
            # (DOI), so the cumulative ``downloaded`` counter across
            # rounds is what really matters. We cap at ``MAX_ROUNDS``
            # and use a per-round ``timeout`` because a single sentinel
            # run can take ~2 min on a cold OpenAlex cache; three
            # rounds would otherwise blow past the user wait budget.
            MAX_ROUNDS = 2
            PER_ROUND_TIMEOUT = 180
            base_keywords = keywords or topic
            queries = [
                base_keywords,
                f"{base_keywords} review",
            ]
            cumulative_downloaded = 0
            cumulative_candidates = 0
            rounds_done = 0
            last_summary = ""
            last_warnings: list[str] = []
            for round_idx, q in enumerate(queries, start=1):
                round_start = time.monotonic()
                self._post_status(
                    f"第 {round_idx}/{MAX_ROUNDS} 轮检索 · query='{q}'"
                )
                # Sentinel does not emit mid-run progress lines, so
                # ``subprocess.run`` blocks silently for ~2 min. Park a
                # background timer that re-posts the same status with
                # the elapsed time appended so the user sees the run is
                # alive. The timer is cancelled as soon as the round
                # finishes (see ``_stop_progress_timer`` below).
                progress_state = {"stop": False}
                _stop_progress_timer = lambda: progress_state.update(stop=True)
                self._start_progress_timer(
                    round_idx, MAX_ROUNDS, q, round_start, progress_state
                )
                # Round 17: pass the user-selected year range and
                # target count through to the sentinel subprocess.
                # Previously only ``--watchlist`` / ``--topic`` were
                # forwarded, so the dialog's "年份区间" and "最少
                # 检索数目" fields were silently dropped and the
                # pipeline ran with the watchlist's persisted
                # defaults (or the hardcoded 2024 lower bound).
                cmd, env, cwd = _littrace_cmd(
                    "sentinel", "run",
                    "--watchlist", topic,
                    "--topic", q,
                    "--year-min", str(year_min),
                    "--year-max", str(year_max),
                    "--target-papers", str(min_papers),
                )
                try:
                    completed = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=PER_ROUND_TIMEOUT,
                        cwd=cwd,
                        env=env,
                    )
                finally:
                    progress_state["stop"] = True
                if completed is None:
                    continue
                if completed.returncode != 0:
                    tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-1] or "(no output)"
                    self._post_status(
                        f"第 {round_idx} 轮 sentinel 退出 {completed.returncode}：{tail}"
                    )
                    continue
                rounds_done += 1
                summary = _summarise_sentinel_output(completed.stdout or "")
                last_summary = summary
                # Round 17: surface download-stage warnings that the
                # sentinel run folded into ``warnings:``. Without
                # this the user only saw the run summary and had no
                # way to know that, say, the download step failed
                # silently. The last warning wins because the most
                # recent round's diagnostics are the freshest
                # signal of an ongoing problem.
                for line in (completed.stdout or "").splitlines():
                    if line.startswith("warnings:"):
                        last_warnings = [
                            chunk.strip() for chunk in line[len("warnings:"):].split("；") if chunk.strip()
                        ]
                downloaded = _summary_value(summary, "downloaded:")
                candidates = _summary_value(summary, "new_candidates:")
                if downloaded is not None:
                    cumulative_downloaded += downloaded
                if candidates is not None:
                    cumulative_candidates += candidates
                # Stop early once we have enough downloads. Sentinel
                # dedupes by DOI so accumulated downloaded is the
                # real "we found N new papers" count.
                if cumulative_downloaded >= min_papers:
                    break
            headline = (
                f"检索完成（{rounds_done} 轮）· {last_summary}"
            )
            shortfall = []
            if (
                cumulative_downloaded is not None
                and cumulative_downloaded < min_papers
            ):
                shortfall.append(
                    f"{rounds_done} 轮累计下了 {cumulative_downloaded} 篇，"
                    f"少于目标 {min_papers} 篇"
                )
            if cumulative_candidates is not None and cumulative_candidates < 5:
                shortfall.append(
                    f"累计检索到 {cumulative_candidates} 个候选"
                )
            if last_warnings:
                shortfall.append("；".join(last_warnings))
            if shortfall:
                headline += " · ⚠️ " + "；".join(shortfall)
            self._post_status(headline)

        threading.Thread(target=_worker, daemon=True, name="littrace-daily").start()

    def _start_progress_timer(
        self,
        round_idx: int,
        max_rounds: int,
        query: str,
        start_monotonic: float,
        state: dict,
    ) -> None:
        """Update the RAG status line every 10 s while a sentinel round
        is running, appending the elapsed time so the user can see the
        run is alive (sentinel does not emit any mid-run progress on
        its own). ``state["stop"]`` is set by the caller when the
        round finishes; the timer exits as soon as it notices.
        """
        def _tick():
            if state.get("stop"):
                return
            elapsed = int(time.monotonic() - start_monotonic)
            self._post_status(
                f"第 {round_idx}/{max_rounds} 轮检索中 · 已跑 {elapsed} 秒 · "
                f"query='{query}'"
            )
            QtCore.QTimer.singleShot(10000, _tick)

        QtCore.QTimer.singleShot(10000, _tick)

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
        # Round 17: route every ``ShellEvent`` through the shared
        # ``EventBridge`` instead of hand-rolling one
        # ``@Slot``/``invokeMethod`` pair per event kind. The
        # bridge installs a single dynamically-generated slot per
        # event kind, JSON-decodes the payload on the GUI thread,
        # and dispatches by ``kind``. Per-event handlers below
        # take a plain ``body: dict`` — no more
        # ``_decode_event`` + ``if kind !=`` boilerplate at the
        # top of every slot.
        #
        # ``self._event_bridge`` is parented to ``self`` so Qt
        # tears it down with the window.
        from littrace.qt_shell import EventBridge, install_subscriptions

        if not hasattr(self, "_event_bridge"):
            self._event_bridge = EventBridge(self, self._controller)
        install_subscriptions(
            self._controller,
            self._event_bridge,
            {
                self._controller.EVENT_MESSAGE_APPENDED: self._on_message_event,
                self._controller.EVENT_STATUS_CHANGED: self._on_status_event,
                self._controller.EVENT_THINKING: self._on_thinking_event,
                self._controller.EVENT_WORKSPACE_REFRESHED: self._on_workspace_event,
                self._controller.EVENT_ERROR: self._on_error_event,
                self._controller.EVENT_WARMUP_STARTED: self._on_warmup_event,
                self._controller.EVENT_WARMUP_DONE: self._on_warmup_event,
                self._controller.EVENT_ASSISTANT_STREAM_OPEN: self._on_stream_open_event,
                self._controller.EVENT_ASSISTANT_DELTA: self._on_stream_delta_event,
                self._controller.EVENT_AUTH_REQUIRED: self._on_auth_event,
                self._controller.EVENT_AUTH_OK: self._on_auth_event,
                self._controller.EVENT_THINKING_PROGRESS: self._on_thinking_progress_event,
            },
        )

    def _on_message_event(self, body: dict) -> None:
        # Round 17: routed through ``EventBridge`` so the
        # ``controller -> bus -> bridge -> slot`` plumbing is no
        # longer hand-rolled. The handler now receives the
        # decoded payload dict directly; the JSON decode +
        # ``if kind != EVENT_X: return`` boilerplate moved into
        # the bridge's per-kind slot.
        role = body.get("role", "system")
        text = body.get("text", "")
        extras = {k: v for k, v in body.items() if k in ("action", "warnings")}
        # Round 17: when an assistant reply lands and a streaming
        # bubble is open, swap its content for the markdown-rendered
        # final reply and skip the duplicate ``append_message``. The
        # streamed text is the same string the controller emits here;
        # re-rendering it as a new bubble would put two copies of the
        # same reply on screen. ``finalize_streaming`` is a no-op
        # when no streaming bubble is open (legacy / non-streaming
        # path), in which case ``append_message`` runs as before.
        if role == "assistant" and getattr(
            self._chat_panel, "_streaming_anchor", None
        ) is not None:
            self._chat_panel.finalize_streaming(text)
            return
        self._chat_panel.append_message(role, text, **extras)

    def _on_status_event(self, body: dict) -> None:
        self._status_bar.showMessage(body.get("text", ""))

    def _on_thinking_event(self, body: dict) -> None:
        self._chat_panel.set_thinking(
            active=bool(body.get("active")),
            label=str(body.get("label", "思考中")),
        )

    def _on_workspace_event(self, body: dict) -> None:
        self._context_panel.refresh(list(self._controller.list_active_papers()))
        self._trace_panel.render_workflow_trace(
            ["工作区刷新"]
            + [
                f"{i + 1}. {p.title}"
                for i, p in enumerate(self._controller.list_active_papers())
            ]
        )

    def _on_error_event(self, body: dict) -> None:
        # Round 17: the controller now sends a structured error
        # payload with ``error_code`` (one of the ``CodexErrorCode``
        # string values, or ``"other"``), a user-facing ``message``,
        # and an optional ``suggestion``. Render the message into
        # the chat scrollback as a system bubble, then surface the
        # suggestion in the status bar so the user can act on it
        # without scrolling. The raw ``raw`` payload (containing
        # ``TypeError: ...``) is hidden behind a small "details"
        # link — it used to be dumped into the chat as the main
        # text, which made every error look like a Python traceback.
        message = body.get("message", "对话出错")
        suggestion = body.get("suggestion", "")
        error_code = body.get("error_code", "other")
        # Inline message + suggestion + a details link. The link
        # target is the raw ``TypeError: ...`` string, which we
        # stash on the dialog as plain text (no HTML escaping
        # needed because QPlainTextEdit handles escaping itself).
        raw = body.get("raw", "")
        html_parts = [f"⚠️ {message}"]
        if suggestion:
            html_parts.append(
                f'<br><span style="color:#5c6068;font-size:11px;">'
                f"{suggestion}</span>"
            )
        if raw:
            html_parts.append(
                f'<br><a href="littrace:show-error-detail" '
                f'style="color:#3a8a8c;font-size:11px;">查看技术细节</a>'
            )
        self._chat_panel.append_message(
            "system", "".join(html_parts)
        )
        # Status bar: condensed one-liner. The full detail is in
        # the chat bubble + the click-through dialog.
        if suggestion:
            # First line of the suggestion, no newlines.
            short = suggestion.splitlines()[0].strip()
            self._status_bar.showMessage(f"⚠️ {message} — {short}", 10000)
        else:
            self._status_bar.showMessage(f"⚠️ {message}", 5000)
        # Remember the most recent raw error text so the link
        # click can pop it up.
        if raw:
            self._last_error_detail = (error_code, message, raw)

    def _on_warmup_event(self, body: dict) -> None:
        # Translate ``WARMUP_STARTED`` / ``WARMUP_DONE`` into the
        # status bar strip. The two-phase progression
        # "正在启动 codex…(spawning)" → "...(initializing)" →
        # "就绪" tells the user the app isn't frozen between window
        # paint and the first turn. ``WARMUP_DONE`` with phase
        # "failed" lands in the chat as a non-fatal warning — the
        # first user turn will retry the spawn automatically.
        # The bridge tags ``body`` with ``__kind`` so this single
        # handler can branch on which warmup event fired without
        # inspecting the bridge's internal ``_installed`` map.
        kind = body.get("__kind", "")
        phase = str(body.get("phase", ""))
        detail = str(body.get("detail", "") or "")
        if kind == self._controller.EVENT_WARMUP_STARTED:
            label = {
                "spawning": "正在启动 codex…(spawning)",
                "initializing": "正在启动 codex…(initializing)",
            }.get(phase, "正在启动 codex…")
            self._status_bar.showMessage(label)
            return
        # WARMUP_DONE.
        if phase == "ready":
            self._status_bar.showMessage("就绪（codex 已预热）", 4000)
        elif phase == "failed":
            msg = "codex 预热失败；首轮对话将自动重试"
            if detail:
                msg = f"{msg}（{detail}）"
            self._status_bar.showMessage(msg, 8000)
            self._chat_panel.append_message(
                "system", f"⚠️ {msg}"
            )

    def _on_stream_open_event(self, body: dict) -> None:
        # Open a streaming bubble. ``EVENT_ASSISTANT_STREAM_OPEN`` is
        # guaranteed by the controller to fire before the first delta,
        # so the bubble is ready by the time ``append_delta`` lands.
        self._chat_panel.open_streaming_bubble()

    def _on_stream_delta_event(self, body: dict) -> None:
        # Append one delta frame to the streaming bubble. ``delta`` is
        # the raw text from codex's ``item/agentMessage/delta``
        # notification; ``ChatPanel.append_delta`` is a no-op when no
        # streaming bubble is open, so a stray frame arriving after
        # ``completed`` is silently dropped.
        delta = body.get("delta", "")
        if not isinstance(delta, str):
            return
        self._chat_panel.append_delta(delta)

    def _on_thinking_progress_event(self, body: dict) -> None:
        # Append the elapsed-seconds counter to the thinking strip
        # so the user sees the turn is still alive. ``elapsed_seconds``
        # is rounded to one decimal so the label doesn't twitch
        # between 3.142 and 3.149.
        elapsed = body.get("elapsed_seconds", 0)
        try:
            elapsed_f = float(elapsed)
        except (TypeError, ValueError):
            return
        self._chat_panel.set_thinking_elapsed(elapsed_f)

    def _on_auth_event(self, body: dict) -> None:
        # Round 17: OAuth lifecycle. ``auth_required`` opens a
        # dialog with the device-auth instructions and a "Re-check"
        # button that calls ``controller.check_codex_auth(force=True)``
        # after the user runs ``codex login`` in their terminal.
        # ``auth_ok`` closes any open dialog so the user can keep
        # chatting immediately. The dialog is modeless so the chat
        # input stays usable while the warning is on screen.
        # The bridge tags ``body`` with ``__kind`` so this single
        # handler can branch on which auth event fired.
        kind = body.get("__kind", "")
        if kind == self._controller.EVENT_AUTH_REQUIRED:
            self._show_auth_dialog(
                reason=str(body.get("reason", "")),
                detail=str(body.get("detail", "")),
            )
            return
        # AUTH_OK.
        self._dismiss_auth_dialog()
        # Show a brief status confirmation so the user knows
        # the warning cleared because their login worked, not
        # because of a bug.
        detail = str(body.get("detail", ""))
        self._status_bar.showMessage(
            f"codex 已认证（{detail}）" if detail else "codex 已认证",
            4000,
        )

    def _show_auth_dialog(self, reason: str, detail: str) -> None:
        # Tear down any stale dialog from a previous check before
        # opening a fresh one — a re-check that still fails
        # shouldn't stack two dialogs on the user's screen.
        existing = getattr(self, "_auth_dialog", None)
        if existing is not None:
            try:
                existing.close()
                existing.deleteLater()
            except RuntimeError:
                pass
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("需要 Codex 登录")
        dialog.setModal(False)
        dialog.resize(540, 320)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("⚠️ Codex 未登录或登录已过期")
        title.setStyleSheet(
            f"font-size:16px;font-weight:600;color:{DESIGN['accent_coral']};"
        )
        layout.addWidget(title)

        reason_label = QtWidgets.QLabel(_reason_to_text(reason, detail))
        reason_label.setWordWrap(True)
        reason_label.setStyleSheet(
            f"font-size:13px;color:{DESIGN['ink']};"
        )
        layout.addWidget(reason_label)

        steps = QtWidgets.QLabel(
            "请在终端运行：\n\n"
            "    codex login --device-auth\n\n"
            "按提示在浏览器里完成登录后，回到这里点 "
            "「重新检查」，对话就会恢复。"
        )
        steps.setStyleSheet(
            f"font-size:12px;color:{DESIGN['ink_muted']};"
            "font-family:Menlo,Consolas,monospace;"
        )
        steps.setWordWrap(True)
        layout.addWidget(steps)

        layout.addStretch(1)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(8)
        button_row.addStretch(1)
        recheck_btn = QtWidgets.QPushButton("🔄 重新检查")
        recheck_btn.setObjectName("subnav_btn_primary")
        recheck_btn.setDefault(True)
        recheck_btn.clicked.connect(
            lambda: self._controller.check_codex_auth(force=True)
        )
        button_row.addWidget(recheck_btn)
        close_btn = QtWidgets.QPushButton("稍后再说")
        close_btn.setObjectName("subnav_btn")
        close_btn.clicked.connect(dialog.close)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        dialog.show()
        self._auth_dialog = dialog

    def _dismiss_auth_dialog(self) -> None:
        existing = getattr(self, "_auth_dialog", None)
        if existing is None:
            return
        try:
            existing.close()
            existing.deleteLater()
        except RuntimeError:
            pass
        self._auth_dialog = None

    def _on_chat_anchor_clicked(self, url: QtCore.QUrl) -> None:
        # Round 17: error-detail link in the chat bubble. The link
        # target is the ``littrace:show-error-detail`` scheme
        # followed by an opaque token (the most recent raw error
        # string). We don't actually validate the token — the
        # dialog only ever shows the cached ``_last_error_detail``
        # payload, so a forged link can at most show the user's
        # own last error. http(s) links fall through to the
        # system's default browser via QDesktopServices.
        from PySide6.QtGui import QDesktopServices

        text = url.toString()
        if text.startswith("littrace:show-error-detail"):
            detail = getattr(self, "_last_error_detail", None)
            if detail is None:
                return
            error_code, message, raw = detail
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle(f"技术细节 · {error_code}")
            dlg.resize(640, 320)
            layout = QtWidgets.QVBoxLayout(dlg)
            layout.setContentsMargins(16, 12, 16, 12)
            heading = QtWidgets.QLabel(message)
            heading.setStyleSheet(
                f"font-size:13px;font-weight:600;color:{DESIGN['ink']};"
            )
            layout.addWidget(heading)
            view = QtWidgets.QPlainTextEdit(raw)
            view.setReadOnly(True)
            view.setStyleSheet(
                f"font-family:Menlo,Consolas,monospace;font-size:11px;"
                f"color:{DESIGN['ink_muted']};"
            )
            layout.addWidget(view, stretch=1)
            ok_btn = QtWidgets.QPushButton("关闭")
            ok_btn.clicked.connect(dlg.accept)
            layout.addWidget(ok_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
            dlg.exec()
            return
        if text.startswith(("http://", "https://")):
            QDesktopServices.openUrl(url)

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
        # Drop a one-shot onboarding banner the first time the user
        # opens LitTrace. The flag file lives next to the LitTrace
        # config so it survives across sessions; closing the dialog
        # is the user's explicit acknowledgement. This is the answer to
        # the "no onboarding, new users don't know what to do" pain
        # point — three explicit steps in plain Chinese, with a
        # pointer to the right panel for each.
        from pathlib import Path
        marker = Path.home() / ".littrace_welcomed"
        if not marker.exists():
            QtCore.QTimer.singleShot(
                400, lambda: self._show_welcome_banner(marker)
            )

    def _show_welcome_banner(self, marker_path: "Path") -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("欢迎使用 LitTrace")
        dialog.setModal(True)
        dialog.resize(540, 380)

        outer = QtWidgets.QVBoxLayout(dialog)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(14)

        title = QtWidgets.QLabel("欢迎使用 LitTrace")
        title.setStyleSheet(
            f"font-size:18px;font-weight:600;color:{DESIGN['ink']};"
        )
        outer.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "3 步把 LitTrace 用起来："
        )
        subtitle.setStyleSheet(
            f"font-size:13px;color:{DESIGN['ink_muted']};"
        )
        outer.addWidget(subtitle)

        steps = [
            (
                "1. 在右下角点登录 publisher",
                "首次检索时需要 Wiley/ACS/Springer/Nature 的访问权限。"
                "Cookie 状态显示在浏览器面板顶部，"
                "点 ✗ 一键跳到登录页。",
            ),
            (
                "2. 点 \"🔍 搜索研究主题\"",
                "弹窗里填主题 / 关键词 / 年份区间 / 最少下载数。"
                "Sentinel 会按主题检索并解析 PDF。",
            ),
            (
                "3. 在中间对话框里对话",
                "输 / 弹 slash 命令；assistant 边生成边显示。"
                "聊天记录自动保存到 session。",
            ),
        ]
        for heading, body in steps:
            step_label = QtWidgets.QLabel(f"<b>{heading}</b>")
            step_label.setStyleSheet(
                f"font-size:13px;color:{DESIGN['ink']};"
            )
            body_label = QtWidgets.QLabel(body)
            body_label.setWordWrap(True)
            body_label.setStyleSheet(
                f"font-size:12px;color:{DESIGN['ink_muted']};"
            )
            outer.addWidget(step_label)
            outer.addWidget(body_label)

        outer.addStretch(1)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(8)
        button_row.addStretch(1)
        ok_btn = QtWidgets.QPushButton("明白了，不再显示")
        ok_btn.setObjectName("subnav_btn_primary")
        ok_btn.setDefault(True)

        def _on_close() -> None:
            try:
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.touch()
            except OSError as exc:
                print(f"[littrace] warning: failed to write {marker_path}: {exc}")
            # ``close()`` only hides a modeless dialog; the widget stays
            # in the ``QApplication``'s top-level list. ``deleteLater()``
            # is what actually tears it down so the next-start check
            # (``Path.exists``) sees no stale dialog.
            dialog.close()
            dialog.deleteLater()

        ok_btn.clicked.connect(_on_close)
        button_row.addWidget(ok_btn)
        outer.addLayout(button_row)

        # Show non-blocking so the controller's ``singleShot`` timer
        # callback can return and the main loop can keep spinning.
        dialog.show()


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