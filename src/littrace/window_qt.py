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
import threading
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("QT_LOGGING_RULES", "qt.webengine.*=false")

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401  # retained for
                                                                       # test imports that
                                                                       # monkey-patch the
                                                                       # BrowserPanel; we no
                                                                       # longer instantiate it.

from littrace.config import load_config
from littrace.models import PaperMetadata
from littrace.publisher_catalog import (
    PUBLISHERS as PUBLISHER_CATALOG,
    sign_in_shortlinks,
)
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

    Sentinel prints ``run_id:`` / ``new_candidates:`` / ``candidates_total:``
    / ``candidates_seen:`` / ``downloaded:`` / ``parsed:`` /
    ``access_tasks:`` lines near the top of its run summary block,
    followed by a closing ``warnings:`` blob whose individual items
    ("Dataset is missing for all rows; comparison may be unfair.")
    are useful to the parse pipeline but pure noise on a status badge.
    Extract just the run_id + the counter lines.
    """
    keys = (
        "run_id:", "new_candidates:", "candidates_total:",
        "candidates_seen:", "downloaded:", "parsed:", "access_tasks:",
    )
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
    """Left column — workflow trace + session history.

    Round 19: each section is a collapsible group (tool-button
    header + an expandable body widget). Default state is expanded;
    users collapse the workflow trace when they're not actively
    debugging a run, or collapse the session list when they only
    have one session. The QSettings key ``trace/group:<name>``
    remembers the per-section state across launches so the panel
    re-opens the way the user left it.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trace_tile")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # Round 19: QSettings-backed per-group collapse persistence.
        # We keep it scoped to the same org/app as ``DailyConfigDialog``
        # so a single ``~/.config/LitTrace/littrace-qt.conf`` file
        # owns all GUI prefs.
        self._settings = QtCore.QSettings("LitTrace", "littrace-qt")

        # Section 1: workflow trace.
        self._workflow_toggle, self._workflow_body = self._build_collapsible(
            layout,
            title="执行 Trace",
            settings_key="trace/group:workflow",
            default_expanded=True,
            stretch=3,
        )

        self._view = QtWidgets.QTextBrowser()
        self._view.setObjectName("trace_view")
        self._view.setOpenExternalLinks(False)
        self._workflow_body.layout().addWidget(self._view)

        # Section 2: session history.
        self._sessions_toggle, self._sessions_body = self._build_collapsible(
            layout,
            title="历史 Session（点击切换）",
            settings_key="trace/group:sessions",
            default_expanded=True,
            stretch=2,
        )

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
        self._sessions_body.layout().addWidget(self._sessions)

    def _build_collapsible(
        self,
        parent_layout: QtWidgets.QVBoxLayout,
        *,
        title: str,
        settings_key: str,
        default_expanded: bool,
        stretch: int,
    ) -> tuple[QtWidgets.QToolButton, QtWidgets.QWidget]:
        """Create a header tool-button + a body container that
        toggles visibility together. Returns ``(toggle, body)`` so
        the caller can populate the body with whatever widget(s) it
        needs.

        ``settings_key`` (if provided) makes the state survive
        across launches — the user collapses the workflow trace on
        day 1, reopens the app on day 2, and the trace stays
        collapsed. ``default_expanded`` is used only when the key
        is missing (first launch).
        """
        toggle = QtWidgets.QToolButton()
        toggle.setObjectName("group_toggle")
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setStyleSheet(
            "QToolButton#group_toggle{"
            f"font-size:13px;font-weight:600;color:{DESIGN['ink']};"
            "border:none;background:transparent;text-align:left;"
            "padding:4px 0;}"
            "QToolButton#group_toggle:hover{color:#3a8a8c;}"
        )
        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 4, 0, 0)
        body_layout.setSpacing(4)

        def _apply(expanded: bool) -> None:
            toggle.setChecked(expanded)
            toggle.setArrowType(
                QtCore.Qt.ArrowType.DownArrow
                if expanded
                else QtCore.Qt.ArrowType.RightArrow
            )
            body.setVisible(expanded)

        # Restore previous state, or fall back to default.
        stored = self._settings.value(settings_key, default_expanded, type=bool)
        _apply(bool(stored))

        def _on_toggled(checked: bool) -> None:
            _apply(checked)
            self._settings.setValue(settings_key, checked)

        toggle.toggled.connect(_on_toggled)

        parent_layout.addWidget(toggle)
        parent_layout.addWidget(body, stretch=stretch)
        return toggle, body

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

    def _on_paper_pin_toggle_requested(self, paper_id: str) -> None:
        # Round 19: ``ContextPanel`` hands the right-click "Pin" /
        # "取消 pin" action here. ``controller.toggle_paper_pin``
        # flips the pin state and emits ``EVENT_WORKSPACE_REFRESHED``
        # so the panel re-renders against the new pinned list. The
        # status bar lets the user know whether they pinned or
        # unpinned.
        new_state = self._controller.toggle_paper_pin(paper_id)
        action = "已 pin" if new_state else "已取消 pin"
        self._status_bar.showMessage(
            f"{action} · {paper_id}", 3000,
        )

    def _on_paper_importance_requested(
        self, paper_id: str, level: int
    ) -> None:
        # Round 19: ``ContextPanel`` hands the importance submenu
        # (普通 / 重要 / 核心 / 清除) here. Level=0 clears the
        # marker; 2 = important; 3 = critical. ``EVENT_WORKSPACE_REFRESHED``
        # re-renders the panel so the marker flips.
        ok = self._controller.set_paper_importance(paper_id, level)
        if ok:
            label = {
                0: "已清除重要性标记",
                2: "已标记为 ⭐ 重要",
                3: "已标记为 🔥 核心",
            }.get(level, "重要性已更新")
            self._status_bar.showMessage(f"{label} · {paper_id}", 3000)
        else:
            self._status_bar.showMessage(
                f"该 paper 不在激活列表中：{paper_id}", 3000,
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
        #
        # Round 19: ``_popup_base_text`` stores the ``"/<name>    — <desc>"``
        # form for each command row so ``_refresh_popup_counters``
        # can rebuild the displayed text with a fresh counter (e.g.
        # ``/papers    — 列出当前上下文文献    · 3 篇``) every time
        # the popup becomes visible.
        self._popup_meta: dict[int, str] = {}  # row -> "separator:<group>"
        self._popup_base_text: dict[int, str] = {}  # row -> base "/<name> — <desc>"
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
            base = f"/{name}    — {desc}"
            item = QtWidgets.QListWidgetItem(base)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            self._popup.addItem(item)
            self._popup_base_text[self._popup.count() - 1] = base
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

    def _popup_counter_for(self, name: str) -> str:
        """Return a short live counter string for ``name`` (e.g.
        ``"3 篇"`` or ``"RAG 2h 前 · 1.2k chunks"``) to append after
        the slash-command description. Empty string means "no
        counter — render the row as before".

        The user sees this every time they type ``/`` and the popup
        opens, so the counter doubles as a glanceable workspace
        status — they don't have to open the context panel to know
        how many papers are active.
        """
        try:
            workspace = self._controller.workspace
            active = list(getattr(workspace.context, "active_papers", []))
        except Exception:
            return ""
        # Group "论文库" commands — show the same active-paper count
        # next to whichever command the user is hovering over. Cheap
        # to recompute (it's a ``len()``), so we don't cache.
        if name in {"context", "papers", "全部下载", "check-downloads", "resume-downloads"}:
            n = len(active)
            return f"{n} 篇"
        # ``dashboard`` surfaces RAG freshness — use the controller's
        # refresh tracker (also read by the RAG panel) so the popup
        # agrees with whatever the user last saw in the right column.
        if name == "dashboard":
            try:
                status = self._controller.get_rag_refresh_status()
            except Exception:
                return ""
            ts = status.get("timestamp")
            chunks = int(status.get("indexed_chunks") or 0)
            if not ts:
                return "RAG 未刷新"
            age = max(0, int(time.time() - float(ts)))
            age_text = _fmt_age(age)
            chunk_text = f"{chunks / 1000:.1f}k" if chunks >= 1000 else str(chunks)
            return f"RAG {age_text}前 · {chunk_text} 块"
        return ""

    def _refresh_popup_counters(self) -> None:
        """Rewrite each command row's display text with the latest
        ``_popup_counter_for(name)`` value. Called from
        ``_on_input_text_changed`` so the counter is fresh every
        keystroke (not just the first ``/``).
        """
        for row, base in self._popup_base_text.items():
            item = self._popup.item(row)
            if item is None:
                continue
            name = item.data(QtCore.Qt.ItemDataRole.UserRole) or ""
            counter = self._popup_counter_for(name)
            if counter:
                item.setText(f"{base}    · {counter}")
            else:
                item.setText(base)

    def _on_input_text_changed(self) -> None:
        text = self._input.toPlainText()
        # Keep the send button in sync with whether the user has
        # anything to send. ``CursorPosition`` updates fire this slot
        # as well, so re-read the current value each time.
        self._send.setEnabled(bool(text.strip()))
        if not text.startswith("/"):
            self._popup.hide()
            return
        # Round 19: refresh the live counters (active-paper count,
        # RAG age, etc.) on every keystroke so the popup never
        # shows stale numbers. Cheap — each counter is a single
        # attribute read on the controller's workspace.
        self._refresh_popup_counters()
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
        # Round 19: route slash commands through the controller's local
        # dispatch table instead of forwarding them to Codex (which has
        # no idea what ``/parse`` means in LitTrace context). Slash
        # commands without arguments go straight to the dispatch table;
        # ``/foo bar baz`` becomes ``name="foo", args="bar baz"``.
        if text.startswith("/"):
            stripped = text[1:].strip()
            if not stripped:
                return
            parts = stripped.split(maxsplit=1)
            name = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            self._controller.submit_slash_command(name, args)
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
    """Right column upper — active literature context.

    Round 19: the original panel exposed only a flat list and a
    right-click "取消激活" action. With sessions routinely
    accumulating 30+ active papers, users had no way to (a) search
    for a specific paper, (b) mark the most relevant ones, or
    (c) inspect a single paper's full metadata. This rewrite adds:

    * A search box that filters by title / DOI / author substring.
    * A "只看 pinned" toggle so core papers stay visible when the
      list is long.
    * Per-item markers: 📌 for pinned, ⭐ for importance=2, 🔥 for
      importance=3. The visual hierarchy lets the user spot core
      papers without scrolling.
    * Double-click → modal detail dialog with the full metadata
      table (DOI, authors, abstract, access, citation, local PDF).
    * Expanded right-click menu: pin/unpin, importance menu
      (普通 / 重要 / 核心 / 清除), deactivate, view detail.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("context_tile")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("文献上下文")
        title.setObjectName("pane_title")
        layout.addWidget(title)

        # Search row
        search_row = QtWidgets.QHBoxLayout()
        search_row.setContentsMargins(0, 0, 0, 0)
        search_row.setSpacing(6)
        self._search = QtWidgets.QLineEdit()
        self._search.setObjectName("context_search")
        self._search.setPlaceholderText("搜索标题 / 作者 / DOI…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._refresh_visible)
        search_row.addWidget(self._search, stretch=1)
        self._pinned_only = QtWidgets.QCheckBox("只看 pinned")
        self._pinned_only.setObjectName("pinned_only")
        self._pinned_only.toggled.connect(self._refresh_visible)
        search_row.addWidget(self._pinned_only)
        layout.addLayout(search_row)

        # Hint line — short so the panel doesn't waste a row on it.
        hint = QtWidgets.QLabel(
            "双击查看详情 · Ctrl/Shift 多选后点 [🔍 比较选中] · 右键菜单可 pin / 标重要性"
        )
        hint.setObjectName("status")
        hint.setStyleSheet(
            f"color:{DESIGN['ink_subtle']};font-size:11px;"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._list = QtWidgets.QListWidget()
        self._list.setObjectName("context")
        # Round 19: allow Ctrl / Shift multi-select so the user can
        # pick 2+ papers and click "🔍 比较选中" to ask Codex for a
        # side-by-side comparison without typing the paper list by
        # hand. ``ExtendedSelection`` is the standard pattern (macOS
        # Cmd+click on Linux/Win; plain click clears the selection
        # first so the user doesn't accidentally drag along old
        # picks).
        self._list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        # Right-click context menu for the per-paper actions
        # ("取消激活" / "查看详情"). Custom menu policy keeps the
        # menu off when the user clicks empty space.
        self._list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(
            self._on_context_menu
        )
        # Double-click opens the detail dialog. The single-click
        # selection stays as-is so the user can still right-click
        # without immediately invoking the modal.
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        # React to selection changes so the "🔍 比较选中" button
        # enables / disables on the fly.
        self._list.itemSelectionChanged.connect(self._refresh_compare_button)
        layout.addWidget(self._list, stretch=1)

        # Compare button row — only enabled when 2+ papers are
        # selected. We don't pin this to a global keyboard shortcut
        # (those would collide with the chat panel's text editor);
        # the click affordance is enough for the day-to-day flow.
        compare_row = QtWidgets.QHBoxLayout()
        compare_row.setContentsMargins(0, 0, 0, 0)
        compare_row.setSpacing(6)
        self._compare_btn = QtWidgets.QPushButton("🔍 比较选中")
        self._compare_btn.setObjectName("compare_btn")
        self._compare_btn.setToolTip(
            "把当前选中的文献交给 Codex，要求给出方法 / 结果 / 局限的对比。"
            "需要 ≥2 篇才生效。"
        )
        self._compare_btn.setEnabled(False)
        self._compare_btn.clicked.connect(self._on_compare_clicked)
        compare_row.addWidget(self._compare_btn)
        self._compare_count = QtWidgets.QLabel("未选中")
        self._compare_count.setObjectName("compare_count")
        self._compare_count.setStyleSheet(
            f"color:{DESIGN['ink_subtle']};font-size:11px;"
        )
        compare_row.addWidget(self._compare_count)
        compare_row.addStretch(1)
        layout.addLayout(compare_row)

        # Cached so the search box can re-filter without going back
        # to the controller on every keystroke. Refreshed by
        # ``refresh()``.
        self._cached_papers: list[PaperMetadata] = []
        self._cached_pinned: list[str] = []
        self._cached_importance: dict[str, int] = {}

    def refresh(self, papers: list[PaperMetadata]) -> None:
        # Read the pin/importance state directly from the
        # controller's workspace so this panel doesn't need a
        # controller reference of its own — the parent window
        # pushes fresh workspace data via ``refresh()``.
        window = self.window()
        pinned: list[str] = []
        importance: dict[str, int] = {}
        if window is not None and hasattr(window, "_controller"):
            ctx = window._controller.workspace.context
            pinned = list(ctx.pinned_papers)
            importance = dict(ctx.importance_levels)
        self._cached_papers = list(papers)
        self._cached_pinned = pinned
        self._cached_importance = importance
        self._render_list()

    def _refresh_visible(self) -> None:
        self._render_list()

    def _render_list(self) -> None:
        query = self._search.text().strip().lower()
        pinned_only = self._pinned_only.isChecked()
        self._list.clear()
        papers = self._cached_papers
        if not papers:
            placeholder = QtWidgets.QListWidgetItem("暂无激活文献 — 在聊天中提需求即可加入")
            placeholder.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self._list.addItem(placeholder)
            return
        pinned_set = set(self._cached_pinned)
        visible_index = 0
        for paper in papers:
            paper_id = getattr(paper, "paper_id", None)
            if pinned_only and paper_id not in pinned_set:
                continue
            if query and not self._paper_matches(paper, query):
                continue
            visible_index += 1
            importance = self._cached_importance.get(paper_id, 1) if paper_id else 1
            pin_mark = "📌 " if paper_id in pinned_set else ""
            if importance >= 3:
                imp_mark = "🔥 "
            elif importance >= 2:
                imp_mark = "⭐ "
            else:
                imp_mark = ""
            year = paper.year or "n.d."
            source = paper.journal or paper.publisher or "unknown source"
            item = QtWidgets.QListWidgetItem(
                f"{pin_mark}{imp_mark}{visible_index}. {paper.title}  "
                f"({year}, {source})"
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, paper)
            item.setToolTip(self._format_tooltip(paper))
            self._list.addItem(item)

    def _paper_matches(self, paper: Any, query: str) -> bool:
        """Substring match across title / DOI / first author. The
        search is intentionally permissive — better to show a few
        extra results than to silently hide the one the user wants.
        """
        if query in (paper.title or "").lower():
            return True
        if query in (getattr(paper, "doi", None) or "").lower():
            return True
        authors = getattr(paper, "authors", None) or []
        for author in authors[:5]:
            if query in author.lower():
                return True
        return False

    def _format_tooltip(self, paper: Any) -> str:
        year = paper.year or "n.d."
        source = paper.journal or paper.publisher or "unknown source"
        tooltip_parts = [
            f"标题：{paper.title}",
            f"年份：{year}",
            f"来源：{source}",
        ]
        if getattr(paper, "doi", None):
            tooltip_parts.append(f"DOI：{paper.doi}")
        authors = getattr(paper, "authors", None) or []
        if authors:
            shown = "、".join(authors[:3])
            if len(authors) > 3:
                shown += f" 等 {len(authors)} 位"
            tooltip_parts.append(f"作者：{shown}")
        if getattr(paper, "access_type", None):
            at = paper.access_type
            at_str = at.value if hasattr(at, "value") else str(at)
            tooltip_parts.append(f"访问类型：{at_str}")
        if getattr(paper, "citation_count", None):
            tooltip_parts.append(f"引用数：{paper.citation_count}")
        return "\n".join(tooltip_parts)

    def _on_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        paper = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if paper is None:
            return
        menu = QtWidgets.QMenu(self)
        window = self.window()
        ctrl = getattr(window, "_controller", None)
        paper_id = getattr(paper, "paper_id", None)
        # Pin toggle — label depends on current state.
        if paper_id in set(self._cached_pinned):
            pin_label = "取消 pin"
        else:
            pin_label = "📌 Pin"
        pin_action = menu.addAction(pin_label)
        pin_action.triggered.connect(
            lambda _checked=False, pid=paper_id: self._request_toggle_pin(pid)
        )
        # Importance submenu
        imp_menu = menu.addMenu("标记重要性")
        for level, label in (
            (2, "⭐ 重要"),
            (3, "🔥 核心"),
            (0, "清除标记"),
        ):
            act = imp_menu.addAction(label)
            act.triggered.connect(
                lambda _checked=False, lvl=level, pid=paper_id:
                    self._request_set_importance(pid, lvl)
            )
        menu.addSeparator()
        detail_action = menu.addAction("📄 查看详情")
        detail_action.triggered.connect(
            lambda _checked=False, p=paper: self._show_detail(p)
        )
        deactivate = menu.addAction("取消激活")
        deactivate.triggered.connect(
            lambda: self._request_deactivate(paper)
        )
        menu.exec(self._list.mapToGlobal(pos))

    def _on_item_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        paper = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if paper is None:
            return
        self._show_detail(paper)

    def _show_detail(self, paper: Any) -> None:
        """Open a non-modal detail dialog for ``paper``. The dialog
        is parented to the main window so closing the window cleans
        it up; multiple detail dialogs can co-exist (one per paper)
        without blocking the rest of the GUI.
        """
        window = self.window()
        dlg = PaperDetailDialog(paper, self._cached_importance, parent=window)
        dlg.show()

    def _request_toggle_pin(self, paper_id: str) -> None:
        window = self.window()
        if window is not None and hasattr(window, "_on_paper_pin_toggle_requested"):
            window._on_paper_pin_toggle_requested(paper_id)

    def _request_set_importance(self, paper_id: str, level: int) -> None:
        window = self.window()
        if window is not None and hasattr(window, "_on_paper_importance_requested"):
            window._on_paper_importance_requested(paper_id, level)

    def _request_deactivate(self, paper: Any) -> None:
        # Delegate to the parent LitTraceQtWindow which owns the
        # controller. The window's slot calls
        # ``controller.deactivate_paper`` and emits the
        # ``WORKSPACE_REFRESHED`` event our existing slot already
        # handles, so the list re-renders against the new state.
        window = self.window()
        if window is not None and hasattr(window, "_on_paper_deactivate_requested"):
            window._on_paper_deactivate_requested(paper)

    def _refresh_compare_button(self) -> None:
        """Toggle the compare button + label based on the current
        selection count. Wired to ``itemSelectionChanged`` so it
        fires every time the user adds / removes a row.
        """
        n = len(self._list.selectedItems())
        if n == 0:
            self._compare_btn.setEnabled(False)
            self._compare_count.setText("未选中")
        elif n == 1:
            self._compare_btn.setEnabled(False)
            self._compare_count.setText("已选 1 篇（需要 ≥2 篇）")
        else:
            self._compare_btn.setEnabled(True)
            self._compare_count.setText(f"已选 {n} 篇")

    def _on_compare_clicked(self) -> None:
        """Round 19: build a Codex comparison prompt from the
        currently selected papers and submit it through the
        controller. Format: ``"比较这 N 篇文献：1. <title>…2. <title>…"``
        — the numbering matches what the user sees in the panel so
        they can verify each pick landed in the right slot.
        """
        items = self._list.selectedItems()
        if len(items) < 2:
            return
        picked: list[tuple[int, str]] = []
        for item in items:
            paper = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if paper is None:
                continue
            # Re-use the visible numbering so the user can double-check
            # "yes, paper 3 is the one I meant" before Codex answers.
            label = item.text().strip()
            # Strip the leading pin/importance glyphs + leading number
            # so the comparison list starts at the bare title.
            cleaned = label.lstrip("📌⭐🔥 ").lstrip()
            if cleaned and cleaned[0].isdigit():
                # Drop "<num>. " prefix.
                idx = cleaned.find(". ")
                if idx >= 0:
                    cleaned = cleaned[idx + 2:]
            picked.append((len(picked) + 1, cleaned or "(no title)"))
        if not picked:
            return
        bullet_lines = "\n".join(f"{i}. {title}" for i, title in picked)
        message = (
            f"请比较这 {len(picked)} 篇文献，给出方法 / 结果 / 局限性的对比表：\n"
            f"{bullet_lines}"
        )
        window = self.window()
        if window is None:
            return
        # Hand the message to the controller via the same path the
        # chat panel's send button uses, so the user sees the bubble
        # land in the chat scrollback and Codex actually answers.
        # We don't expose ``submit_user_message`` directly because
        # the chat panel keeps an internal ``_input`` widget we'd
        # like to mirror (the user might want to edit the prompt
        # before pressing Enter).
        if hasattr(window, "_controller"):
            try:
                window._controller.submit_user_message(message)
            except Exception:
                pass
        # Clear the selection so the user can immediately pick a
        # new comparison set without having to Ctrl+click each
        # previous pick.
        self._list.clearSelection()
        self._refresh_compare_button()


class PaperDetailDialog(QtWidgets.QDialog):
    """Read-only detail dialog for a single paper.

    Shown when the user double-clicks an entry in ``ContextPanel``.
    Renders every metadata field the GUI knows about (DOI, authors,
    abstract, access, citation, local PDF path if any) in a single
    table so the user can confirm they have the right paper before
    triggering ``/parse`` or ``/table``.
    """

    def __init__(
        self,
        paper: Any,
        importance: dict[str, int],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._paper = paper
        self.setWindowTitle(f"文献详情 · {paper.title[:60]}")
        # Non-modal so the user can keep interacting with the
        # main window while reading.
        self.setModal(False)
        self.resize(520, 480)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Title — bold, larger than other rows.
        title = QtWidgets.QLabel(paper.title)
        title.setWordWrap(True)
        title.setStyleSheet(
            f"font-size:15px;font-weight:600;color:{DESIGN['ink']};"
        )
        layout.addWidget(title)

        # Importance indicator + action row
        paper_id = getattr(paper, "paper_id", None)
        level = importance.get(paper_id, 1)
        imp_label = QtWidgets.QLabel()
        imp_label.setObjectName("status")
        if level >= 3:
            imp_label.setText("🔥 核心")
        elif level >= 2:
            imp_label.setText("⭐ 重要")
        else:
            imp_label.setText("普通")
        layout.addWidget(imp_label)

        # Field grid
        grid = QtWidgets.QFormLayout()
        grid.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        def _add_field(label_text: str, value_text: str) -> None:
            if not value_text:
                return
            lab = QtWidgets.QLabel(label_text)
            lab.setStyleSheet(
                f"color:{DESIGN['ink_muted']};font-size:12px;"
            )
            val = QtWidgets.QLabel(value_text)
            val.setWordWrap(True)
            val.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectable
            )
            val.setStyleSheet(
                f"color:{DESIGN['ink']};font-size:12px;"
            )
            grid.addRow(lab, val)

        _add_field("paper_id", paper_id or "")
        _add_field("DOI", getattr(paper, "doi", None) or "")
        authors = getattr(paper, "authors", None) or []
        if authors:
            _add_field("作者", "、".join(authors))
        _add_field("年份", str(paper.year) if paper.year else "")
        _add_field(
            "期刊 / 出版商",
            paper.journal or paper.publisher or "",
        )
        at = getattr(paper, "access_type", None)
        at_str = at.value if at and hasattr(at, "value") else (str(at) if at else "")
        _add_field("访问类型", at_str)
        cc = getattr(paper, "citation_count", None)
        if cc is not None:
            _add_field("引用数", str(cc))
        abstract = getattr(paper, "abstract", None) or ""
        if abstract:
            _add_field("摘要", abstract[:500] + ("…" if len(abstract) > 500 else ""))

        # Local PDF path (if LitTrace has parsed it).
        window = self.parent()
        if window is not None and hasattr(window, "_controller"):
            local_pdf = self._resolve_local_pdf(window._controller, paper_id)
            if local_pdf:
                _add_field("本地 PDF", local_pdf)

        layout.addLayout(grid)
        layout.addStretch(1)

        # Close button — keep it cheap (no Save button: the
        # metadata is read-only and the workspace is the source
        # of truth).
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.setObjectName("subnav_btn")
        close_btn.clicked.connect(self.accept)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

    def _resolve_local_pdf(self, controller: Any, paper_id: str | None) -> str | None:
        if not paper_id:
            return None
        # ``parsed_papers`` is a dict[paper_id, ParsedPaper]. If
        # the paper has been parsed we know where the PDF lives.
        parsed = getattr(controller.workspace, "parsed_papers", {}) or {}
        entry = parsed.get(paper_id)
        if entry is None:
            return None
        # ``ParsedPaper`` may store the source PDF path under
        # several attributes depending on parser backend; probe
        # each in turn.
        for attr in ("pdf_path", "source_pdf", "path"):
            value = getattr(entry, attr, None)
            if value:
                return str(value)
        return None


class RAGPanel(QtWidgets.QFrame):
    """Right column middle — single Daily run entry point + RAG status.

    Round 19: the original panel only showed a transient
    "运行中…/尚未启动" label. After a daily run the user couldn't
    tell when the index was last refreshed or whether it had gone
    stale. The new layout adds:

    * A persistent "上次 refresh" line with timestamp + chunk
      count, fetched from the controller.
    * A staleness badge — turns red when the timestamp is > 24 h
      old, orange when > 12 h, hidden when fresh.
    * A quick "立即 refresh" button that re-uses the same
      ``on_run_daily`` slot (so QA / refresh-on-demand flow into
      the same code path).
    """

    STALE_THRESHOLD_SECONDS = 12 * 3600
    VERY_STALE_THRESHOLD_SECONDS = 24 * 3600

    def __init__(
        self,
        on_run_daily,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("tile")
        self._on_run_daily = on_run_daily

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("今日管线 / RAG")
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

        refresh_btn = QtWidgets.QPushButton("⟳ 立即 refresh")
        refresh_btn.setObjectName("subnav_btn")
        refresh_btn.setToolTip(
            "复用「搜索研究主题」流程：re-fetch → re-parse → 写 RAG"
        )
        refresh_btn.clicked.connect(on_run_daily)
        layout.addWidget(refresh_btn)

        # Staleness badge — re-rendered every time the panel is
        # refreshed. Hidden when the controller has no record of a
        # refresh yet.
        self._staleness = QtWidgets.QLabel("")
        self._staleness.setObjectName("staleness")
        self._staleness.setWordWrap(True)
        layout.addWidget(self._staleness)

        # Last-refresh line — same content as the staleness badge
        # plus chunk count. Stays visible at all times so the user can
        # see when it was.
        self._last_refresh = QtWidgets.QLabel("尚未记录 refresh")
        self._last_refresh.setObjectName("last_refresh")
        self._last_refresh.setWordWrap(True)
        self._last_refresh.setStyleSheet(
            f"color:{DESIGN['ink_muted']};font-size:11px;"
        )
        layout.addWidget(self._last_refresh)

        # Transient status — kept for backwards compatibility with
        # ``set_status()`` callers that flip text during a run.
        self._status = QtWidgets.QLabel("")
        self._status.setObjectName("status")
        self._status.setWordWrap(True)
        layout.addWidget(self._status, stretch=1)

        # Initial render — picks up any timestamp the controller
        # already had (e.g. if this panel is constructed mid-run).
        self.refresh_status()

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def refresh_status(self) -> None:
        """Re-render the staleness badge + last-refresh line from
        the controller's stored timestamp. Safe to call any time —
        e.g. after a daily run completes.
        """
        status: dict[str, Any] = {}
        window = self.window()
        if window is not None and hasattr(window, "_controller"):
            try:
                status = window._controller.get_rag_refresh_status()
            except Exception:
                status = {}
        ts = status.get("timestamp")
        chunks = status.get("indexed_chunks", 0) or 0
        if not ts:
            self._last_refresh.setText("尚未记录 refresh")
            self._staleness.hide()
            return
        # Round to the minute so the text doesn't flicker every
        # second while the timer ticks.
        import time as _t
        age = max(0, int(_t.time() - ts))
        ago_text = _fmt_age(age)
        self._last_refresh.setText(
            f"上次 refresh：{ago_text}前 · 索引 {chunks} chunks"
        )
        if age >= self.VERY_STALE_THRESHOLD_SECONDS:
            self._staleness.setText(
                f"⚠️ 索引已过期 >{int(self.VERY_STALE_THRESHOLD_SECONDS // 3600)}h，"
                "建议立即 refresh"
            )
            self._staleness.setStyleSheet(
                f"color:#b53939;font-size:11px;font-weight:600;"
            )
            self._staleness.show()
        elif age >= self.STALE_THRESHOLD_SECONDS:
            self._staleness.setText(
                f"⏰ 索引超过 {int(self.STALE_THRESHOLD_SECONDS // 3600)}h，可考虑 refresh"
            )
            self._staleness.setStyleSheet(
                f"color:#b07a17;font-size:11px;"
            )
            self._staleness.show()
        else:
            self._staleness.hide()


def _fmt_age(seconds: int) -> str:
    """Format a small duration in human terms. Used by ``RAGPanel``
    so the staleness line reads naturally in both Chinese and
    English-shaped labels. Above 24 h the formatter collapses to
    hours so the panel line stays one row.
    """
    if seconds < 60:
        return f"{seconds} 秒"
    if seconds < 3600:
        return f"{seconds // 60} 分钟"
    if seconds < 86400:
        return f"{seconds // 3600} 小时"
    return f"{seconds // 86400} 天"


class DailyConfigDialog(QtWidgets.QDialog):
    """Collect the parameters that drive ``littrace sentinel run`` from
    the user before the daily pipeline kicks off. Four fields:

      * 研究主题 (used as sentinel watchlist id, required)
      * 开始 / 结束年份 (year range for retrieval)
      * 最少检索数目 (target count; warns if the run comes up short)

    Round 19: the dialog is non-modal (``setModal(False)``) so the
    user can still browse the context panel, switch sessions, or
    check the chat scrollback while it's open. ``_on_run_daily``
    listens for ``accepted`` / ``rejected`` signals instead of
    blocking on ``exec()`` — closing the dialog (via the X button,
    Esc, or the explicit "取消" button) rejects the run.

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
        # Round 19: non-modal so the user can keep poking at the
        # rest of the window while they tweak the parameters.
        # Closing the dialog (X button, Esc, "取消") rejects the run.
        self.setModal(False)
        self.resize(540, 380)

        # Round 19: the dialog now remembers the last accepted values
        # across sessions via ``QSettings`` so the user doesn't have
        # to retype the same topic every time. Falls back to the
        # ``default_*`` parameters (or config defaults the caller
        # passes) on first run.
        self._settings = QtCore.QSettings("LitTrace", "littrace-qt")

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
        self._topic_input = QtWidgets.QLineEdit(
            self._settings.value("daily/topic", default_topic, type=str)
        )
        self._topic_input.setPlaceholderText("e.g. mxene_sensor, perovskite_solar, mof_co2")
        self._topic_input.selectAll()
        form.addRow("研究主题 *", self._topic_input)

        # 关键词
        self._keywords_input = QtWidgets.QLineEdit(
            self._settings.value("daily/keywords", "", type=str)
        )
        self._keywords_input.setPlaceholderText("可选：更精确的检索词（多个用空格分隔）")
        form.addRow("关键词", self._keywords_input)

        # 年份范围
        year_row = QtWidgets.QHBoxLayout()
        year_row.setSpacing(6)
        self._year_min_input = QtWidgets.QSpinBox()
        self._year_min_input.setRange(1990, 2030)
        self._year_min_input.setValue(int(
            self._settings.value("daily/year_min", default_year_min)
        ))
        year_row.addWidget(self._year_min_input)
        year_row.addWidget(QtWidgets.QLabel("至"))
        self._year_max_input = QtWidgets.QSpinBox()
        self._year_max_input.setRange(1990, 2030)
        self._year_max_input.setValue(int(
            self._settings.value("daily/year_max", default_year_max)
        ))
        year_row.addWidget(self._year_max_input)
        year_row.addStretch(1)
        form.addRow("年份区间", year_row)

        # 最少检索数
        self._min_papers_input = QtWidgets.QSpinBox()
        self._min_papers_input.setRange(1, 200)
        self._min_papers_input.setValue(int(
            self._settings.value("daily/min_papers", default_min_papers)
        ))
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
        # Round 19: persist the user's current choices so the next
        # dialog open pre-fills with the same values. ``sync()``
        # flushes the in-memory cache to disk immediately so a
        # crash doesn't lose the just-accepted values.
        self._settings.setValue("daily/topic", topic)
        self._settings.setValue("daily/keywords", self._keywords_input.text().strip())
        self._settings.setValue("daily/year_min", self._year_min_input.value())
        self._settings.setValue("daily/year_max", self._year_max_input.value())
        self._settings.setValue("daily/min_papers", self._min_papers_input.value())
        self._settings.sync()
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


class DailyResultDialog(QtWidgets.QDialog):
    """Round 29: deprecated.

    Round 19 introduced this dialog as a separate floating surface for the
    daily-run summary. Round 29 replaced it with an inline system bubble in
    the chat scrollback (see ``_show_daily_preview_from_any_thread``) so
    the result is colocated with the user's conversation rather than living
    in a disconnected dialog the rest of the UI ignores. The class is kept
    here as a no-op so any external callers that still import it don't
    break at import time — constructing it does nothing visible. Remove
    this stub once the next minor release ships.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        # Intentionally swallow all arguments and stay hidden.
        try:
            super().__init__(None)
            self.setWindowTitle("daily 检索结果 (deprecated)")
            self.resize(0, 0)
        except Exception:
            pass

    def _make_counter(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    def _on_open_context(self) -> None:
        return None


class BrowserPanel(QtWidgets.QFrame):

    # Round 21: surface navigation progress in the main status bar so
    # the user can tell whether a click on a publisher shortcut or a
    # URL-bar Enter actually fired. The status bar lives on the main
    # window, so the panel just emits this signal and the window
    # connects it to ``QStatusBar.showMessage``.
    navigation_message = QtCore.Signal(str)
    """Right column lower — QWebEngineView pane for publisher pages.

    The panel exposes two shortcuts that come up often when the user
    wants LitTrace to be able to pull gated PDFs:

      * A row of one-click publisher login buttons (Wiley / ACS /
        Springer / Nature / arXiv) right above the URL bar. Each opens
        that publisher's sign-in page in the embedded Chromium, so the
        user can sign in once and the session cookies live in
        ``./data/chrome-cdp`` for the next ``sentinel run`` to use.
      * A free-form URL line for ad-hoc navigation.

    Round 18: the embedded view is the **only** login surface now.
    ``main()`` redirects QtWebEngine's default profile storage to
    ``data/chrome-cdp`` (the same on-disk profile sentinel reads via
    CDP), so cookies written here are immediately reusable. The
    external ``chrome.exe`` is launched lazily only when sentinel needs
    CDP — see ``LitTraceQtWindow._acquire_external_chrome_for_sentinel``
    — and torn down again before this panel re-shows the URL the user
    was on.

    The previous round removed the back/forward/reload arrows because
    the URL bar already accepts any input; this round keeps that
    decision and only adds the publisher row.
    """

    # The embedded Chromium starts on a small data-URL welcome page
    # so the user sees actionable guidance instead of a blank
    # ``about:blank``. Round 18: drop the misleading "CDP 19222 is up"
    # line — there is no external chrome anymore at this point, and
    # logging in here is the actual mechanism that flows into
    # sentinel's gated-PDF fetcher.
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
        "<p class='ok'>✓ 登录后 cookie 自动写入 <code>data/chrome-cdp</code>，"
        "下次 sentinel 检索可直接复用。</p>"
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
    #
    # Round 18: clicking a shortcut loads the sign-in page in the
    # embedded QWebEngineView. The cookie it writes lands in
    # ``data/chrome-cdp`` and is immediately visible to sentinel's
    # next gated-PDF fetch.
    #
    # Round 19: derived from the unified ``PUBLISHERS`` catalog so
    # the GUI shortcut row, the cookie strip, and the
    # ``chrome_profiles`` detector stay in lock-step. ``arXiv`` is
    # excluded from shortcuts (``requires_login=False``).
    PUBLISHER_LINKS: list[tuple[str, str]] = list(sign_in_shortlinks())

    def __init__(self, parent: QtWidgets.QWidget | None = None, config=None) -> None:
        super().__init__(parent)
        self.setObjectName("browser_tile")
        # ``config`` is optional so the panel can be used standalone
        # (e.g. unit tests). When None the cookie-status strip renders
        # an explicit "(no config)" placeholder instead of crashing.
        self._config = config

        layout = QtWidgets.QVBoxLayout(self)
        # Round 18: store a reference so ``resume_after_external_chrome``
        # can re-attach a freshly constructed ``QWebEngineView`` to the
        # same layout after sentinel finishes. Without the reference,
        # ``self.layout()`` works but the call site is harder to grep.
        self._layout = layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Round 19: collapsible header — toggle the panel between a
        # compact "cookie strip only" state and the full publisher
        # browser. Auto-collapses once every publisher is logged in so
        # the right column doesn't waste 50 % of the window on a
        # Chromium view the user has finished interacting with.
        self._expanded = True
        self._collapse_toggle = QtWidgets.QToolButton()
        self._collapse_toggle.setObjectName("browser_collapse")
        self._collapse_toggle.setText("▼ 收起浏览器")
        self._collapse_toggle.setToolTip(
            "登录完成后点这收起，只保留 cookie 状态条"
        )
        self._collapse_toggle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._collapse_toggle.setCheckable(True)
        self._collapse_toggle.setChecked(True)  # start expanded
        self._collapse_toggle.toggled.connect(self._set_expanded)

        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.setContentsMargins(8, 4, 8, 0)
        toggle_row.addStretch(1)
        toggle_row.addWidget(self._collapse_toggle)
        layout.addLayout(toggle_row)

        # Everything below this header is bundled in a single child
        # container so we can show/hide the whole chunk when the user
        # toggles collapsed mode. Without the container, hiding
        # individual widgets leaves gaps because the outer VBoxLayout
        # still allocates their geometry.
        self._body = QtWidgets.QWidget()
        self._body_layout = QtWidgets.QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(0)
        layout.addWidget(self._body, stretch=1)

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
            # Round 23: shortcut buttons no longer load the URL into
            # the embedded QWebEngineView — QtWebEngine gets flagged
            # as a high-risk client by Cloudflare on aggressive
            # publishers like pubs.acs.org and the challenge never
            # clears. Instead, spin up a LitTrace-managed Chrome
            # window via CDP (see ``_open_publisher_in_littrace_chrome``).
            # The user completes CF + login there, cookies persist in
            # the same Chrome instance, and ``sentinel`` reuses the
            # same CDP endpoint to fetch gated PDFs — no CF prompt
            # on subsequent runs.
            #
            # ``_set_expanded(True)`` keeps the legacy behaviour of
            # expanding on click (so the cookie status strip refreshes
            # visibly), but the actual navigation target is now the
            # external Chrome, not the embedded view.
            btn.clicked.connect(
                lambda _checked=False, u=url: (
                    self._set_expanded(True),
                    self._open_publisher_in_littrace_chrome(u),
                )
            )
            publisher_row.addWidget(btn)
        publisher_row.addStretch(1)
        self._body_layout.addLayout(publisher_row)

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
        self._body_layout.addLayout(cookie_strip)
        self._refresh_cookie_status()

        # Free-form URL bar for ad-hoc navigation.
        url_row = QtWidgets.QHBoxLayout()
        url_row.setContentsMargins(8, 4, 8, 4)
        url_row.addWidget(QtWidgets.QLabel("URL"))
        self._url = QtWidgets.QLineEdit()
        self._url.setPlaceholderText("粘贴 publisher 链接（Enter 打开）…")
        self._url.returnPressed.connect(self._on_url_entered)
        url_row.addWidget(self._url, stretch=1)
        # Round 21: ↻ button comes after the view is created (see
        # below) so we can wire ``clicked`` -> ``self._view.reload``.
        self._reload_btn = QtWidgets.QToolButton()
        self._reload_btn.setText("↻")
        self._reload_btn.setObjectName("browser_reload")
        self._reload_btn.setToolTip("重新加载当前页面")
        self._reload_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        url_row.addWidget(self._reload_btn)
        self._body_layout.addLayout(url_row)

        # Round 26: drop the embedded ``QWebEngineView`` entirely.
        # It was the last live QtWebEngine instance in this process,
        # and every sentinel run tore it down + rebuilt it to release
        # the profile lock for the external Chrome. On Chromium 140
        # / PySide6 6.11 the teardown dance races with Chromium's own
        # BrowserThread bookkeeping and aborts the whole process with
        # ``Must be called on Chrome_UIThread; actually called on
        # Unknown Thread.`` (verified on the user's macOS box,
        # 2026-09). Publisher login lives in ``QtCDPBrowser`` (an
        # external Chrome that the user actually sees), so the
        # embedded QtWebEngine has no useful role left.
        #
        # We replace it with a static placeholder widget — same
        # vertical space, but it never instantiates Chromium, never
        # touches ``data/chrome-cdp``'s profile lock, and never has
        # signals to disconnect on suspend. The placeholder also
        # shows a hint about how the new publisher-login flow works
        # so the user knows to click 🌐 ACS (which opens the
        # external Chrome) rather than paste a URL here.
        self._view = None
        from PySide6.QtWidgets import QLabel
        self._placeholder = QLabel()
        self._placeholder.setObjectName("browser_placeholder")
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            f"color:{DESIGN['ink_muted']};font-size:13px;padding:24px;"
        )
        self._placeholder.setText(
            "📌 Publisher 登录请点上方 🌐 按钮\n\n"
            "LitTrace 会启动一个独立 Chrome 窗口完成 Cloudflare 验证 / "
            "SSO 登录。cookie 自动写入 ./data/chrome-cdp，下次 sentinel "
            "检索直接复用。\n\n"
            "如果需要在系统浏览器中打开任意 URL，在 URL 栏粘贴后回车即可。"
        )
        self._body_layout.addWidget(self._placeholder, stretch=1)
        # Compatibility shims: keep ``_url_before_suspend`` set so
        # the sentinel / chrome-lifecycle code that reads it doesn't
        # crash; keep the reload button wired to a no-op so the click
        # is harmless.
        self._url_before_suspend: str = self.HOME_URL
        self._reload_btn.clicked.connect(
            lambda: self.navigation_message.emit(
                "↻ 重载 publisher Chrome：再次点 🌐 按钮即可"
            )
        )

    def _set_expanded(self, expanded: bool) -> None:
        """Toggle the body container between visible and hidden.

        Round 19: hiding ``self._body`` keeps the header / toggle
        button visible so the user can re-expand without hunting
        through the layout. The publisher buttons + URL bar +
        webview all live inside ``self._body`` and are hidden
        together, so the panel collapses cleanly without leaving
        gap rows.
        """
        self._expanded = bool(expanded)
        self._collapse_toggle.setText(
            "▼ 收起浏览器" if self._expanded else "▶ 展开浏览器"
        )
        self._body.setVisible(self._expanded)
        self._collapse_toggle.setChecked(self._expanded)

    def _on_url_entered(self) -> None:
        self.open_url(self._url.text())

    def _refresh_cookie_status(self) -> None:
        # Map each publisher shortcut to a list of cookie domains that
        # the littrace config knows about. The cookie detector searches
        # the user's private Chrome profile (``./data/chrome-cdp``)
        # for any of those domains; if at least one matches, the
        # publisher is "logged in".
        #
        # Round 19: derive ``publisher_domains`` from the unified
        # ``PUBLISHERS`` catalog (same source the shortcut row and
        # ``chrome_profiles._detect_publisher_cookie_domains`` use),
        # so adding a new publisher in one place updates all three.
        from littrace.chrome_profiles import (
            _detect_publisher_cookie_domains,
        )
        if self._config is None:
            self._cookie_status.setText("(no config)")
            return
        from pathlib import Path
        profile_root = (
            self._config.cdp_downloader.chrome_user_data_dir
        ).expanduser()
        # Layer 1: ask the live Chrome via CDP. This is what
        # surfaces the macOS login state to the UI.
        present_cdp: set[str] = set()
        try:
            import json as _json
            from littrace.access_layer.cdp_core import CDPBrowser
            browser = CDPBrowser(
                self._config.cdp_downloader.cdp_url,
                reconnect_attempts=1,
                command_timeout_seconds=2.0,
            )
            browser.connect_new_tab()
            # Probe every publisher cookie domain. ``Network.getCookies``
            # matches by URL host (incl. subdomain), so feed one
            # ``https://<domain>/`` per catalog entry.
            for pub in PUBLISHER_CATALOG:
                for d in pub.cookie_domains:
                    probe = f"https://{d.lstrip('.')}/"
                    try:
                        browser.ws.send(_json.dumps({
                            "id": 8001,
                            "method": "Network.getCookies",
                            "params": {"urls": [probe]},
                        }))
                        resp = _json.loads(browser.ws.recv())
                        for c in (resp.get("result") or {}).get("cookies") or []:
                            if c.get("name") == "__cf_bm":
                                continue  # CF bot-mgmt, not a session
                            host = c.get("domain", "").lstrip(".")
                            if host and (host == d.lstrip(".") or host.endswith("." + d.lstrip("."))):
                                present_cdp.add(d)
                                break
                    except Exception:
                        pass
            try:
                browser.ws.close()
            except Exception:
                pass
        except Exception:
            # No live Chrome or CDP unreachable — fall through to the
            # SQLite path below.
            pass
        # Layer 2: SQLite file (legacy path; matches the previous
        # behavior and is the only thing that works on Linux where
        # CDP may not have any cookies until the user logs in).
        try:
            present_sqlite = set(_detect_publisher_cookie_domains(Path(profile_root)))
        except Exception:
            present_sqlite = set()
        # Union: either source reporting a domain flips that
        # publisher's status to ✓.
        present = present_cdp | present_sqlite
        bits: list[str] = []
        # Iterate the unified catalog directly — each entry already
        # carries its display name, cookie domains, sign-in URL, and
        # ``requires_login`` flag, so we don't need the brittle
        # ``endswith(label)`` lookup against ``PUBLISHER_LINKS``.
        any_unlogged = False
        for pub in PUBLISHER_CATALOG:
            label = pub.display_name
            domains = list(pub.cookie_domains)
            logged = any(d in present for d in domains)
            # Round 19: open-access publishers (arXiv) always render
            # ✓ and never trigger the "auto-collapse when all
            # logged in" branch — they're never gated.
            if not pub.requires_login:
                tooltip = f"{label} 是开放获取，无需登录"
                bits.append(
                    f'<span title="{tooltip}" '
                    f'style="color:#2a7a3a;">{label} ✓</span>'
                )
                continue
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
                    f'<a href="{pub.sign_in_url}" title="{tooltip}" '
                    f'style="color:#cc785c;text-decoration:none;">'
                    f"{label} ✗</a>"
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
        # Round 19: auto-collapse once every publisher is logged in.
        # The cookie strip stays visible so the user can still see
        # the status; the chromium view, URL bar, and shortcut row
        # all disappear until the user explicitly expands the panel
        # again. We only auto-collapse on a transition (was-unlogged
        # → now-all-logged-in) so flipping the panel back open
        # doesn't bounce it closed on the next cookie refresh.
        if not any_unlogged and self._expanded:
            self._set_expanded(False)

    def _on_url_changed(self, url: QtCore.QUrl) -> None:
        # Don't echo about:blank into the URL bar — leaves a stale-looking
        # field on startup.
        text = url.toString()
        if text == "about:blank":
            self._url.clear()
        else:
            self._url.setText(text)
            # Round 18: track the last non-blank URL so
            # ``resume_after_external_chrome`` can land the user back
            # where they were before sentinel ran.
            self._url_before_suspend = text
        # Round 21: keep a rolling history of every URL the embedded
        # view has been on during the *current* navigation. We can't
        # just inspect the URL bar at ``loadFinished`` time — QtWebEngine
        # in offscreen mode silently bounces failed navigations back
        # to the welcome page after ~50 ms, so by then the URL bar
        # shows the welcome page even though we *briefly* navigated
        # to a Cloudflare interstitial. Detecting Cloudflare from
        # the URL bar alone misses it.
        history = getattr(self, "_url_history", None)
        if history is None:
            history = []
            self._url_history = history
        history.append(text)
        # Keep memory bounded — only the most recent 8 URLs matter
        # for the Cloudflare check (one publisher click + a couple of
        # redirect hops).
        if len(history) > 8:
            del history[:-8]

    def _on_load_started(self) -> None:
        # ``loadStarted`` fires *before* ``urlChanged`` on a manual
        # ``setUrl`` call, so the URL bar may still be stale here.
        # Use ``_url_before_suspend`` as a best-effort hint; the
        # authoritative URL arrives via the next ``urlChanged``.
        url = self._url.text().strip() or self._url_before_suspend
        # Strip query string + fragment for the status hint so the
        # bar stays readable — ``wiley.com/action/login?ref=foo``
        # becomes ``wiley.com/action/login``.
        short = url.split("?", 1)[0].split("#", 1)[0]
        self.navigation_message.emit(f"⏳ 正在加载 {short} …")

    def _on_load_finished(self, ok: bool) -> None:
        url = self._url.text().strip()
        short = url.split("?", 1)[0].split("#", 1)[0] if url else "页面"
        # Round 21: Cloudflare "Verify you are human" challenge
        # detection runs BEFORE the ok=True early-return. The
        # challenge page itself loads successfully (it's an HTML
        # interstitial), so ``loadFinished`` reports ok=True even
        # though the user has not reached the publisher page. If we
        # returned early on success the user would see "✓ 已加载"
        # and a blank screen — the exact bug this branch fixes.
        #
        # Pop the dialog ONLY when the *current* state is still on
        # the challenge page (URL still has ``__cf_chl``, or the
        # page title is "Just a moment..."). Many publishers route
        # their sign-in flow through Cloudflare briefly even when
        # the user gets through — those don't need a dialog. Only
        # the *stuck-on-challenge* case needs user intervention.
        try:
            title = self._view.page().title() if self._view is not None else ""
        except Exception:
            title = ""
        if not title:
            title = ""
        is_cloudflare_stuck = (
            "__cf_chl" in url
            or "cf-chl" in url.lower()
            or "just a moment" in title.lower()
            or (
                "cloudflare" in title.lower()
                and not ok  # only pop on failure when the page
                            # title mentions Cloudflare
            )
        )
        # Surface the right terminal message based on whether the
        # browser actually got past the challenge. The Cloudflare
        # dialog handles its own user-visible status update.
        if is_cloudflare_stuck:
            self._show_cloudflare_challenge_dialog(url)
        elif ok:
            self.navigation_message.emit(f"✓ 已加载 {short}")
        else:
            # ``errorOccurred`` already posted a more specific reason;
            # this message gives the user a single confirmation that
            # the navigation itself failed (e.g. server unreachable)
            # vs. a sub-resource failure where the page still loaded.
            detail = getattr(self, "_last_error_reason", None)
            tail = f" · {detail}" if detail else ""
            self.navigation_message.emit(
                f"⚠️ 加载失败 {short}{tail}"
            )
        # Reset history so the next navigation starts with a clean
        # slate — otherwise a single Cloudflare stop on the first
        # click would falsely trigger the dialog on every subsequent
        # publisher click.
        self._url_history = []

    def _on_page_error(self, error_code: int, url: str, error_string: str) -> None:
        # Cache the most-recent reason so ``_on_load_finished`` can
        # surface it after the navigation settles. The status-bar
        # message also fires immediately so the user sees the cause
        # even if QtWebEngine doesn't deliver a matching
        # ``loadFinished(ok=False)`` (e.g. for an HTTP 5xx that
        # still completes the request).
        self._last_error_reason = (
            f"{error_string} ({error_code}) {url.split('?', 1)[0]}"
        )
        short = url.split("?", 1)[0].split("#", 1)[0]
        self.navigation_message.emit(
            f"❌ {error_string} ({error_code}) · {short}"
        )

    def _on_certificate_error(self, error) -> None:
        # ``certificateError`` fires when Chromium rejects a TLS cert
        # (corp MITM proxy, expired publisher cert, Cloudflare
        # challenge mismatch). We *defer* the decision — the default
        # is to reject the connection, but the user can override
        # once we expose the override UI. For now we just surface the
        # reason so the user knows why the page is blank.
        try:
            # ``QSslError`` carries a human-readable string in PySide6;
            # fall back to the type name if not.
            reason = error.errorString() or type(error).__name__
        except Exception:
            reason = type(error).__name__
        self._last_error_reason = f"TLS 证书错误 · {reason}"
        self.navigation_message.emit(
            f"🔒 TLS 证书错误 · {reason}（如果这是公司代理，点 ↻ 重试）"
        )

    def _on_render_process_terminated(self, termination_status: int, exit_code: int) -> None:
        # Chromium crashed. Bounce the view so the user can recover
        # by clicking any publisher shortcut again. ``deleteLater`` +
        # ``resume_after_external_chrome``-style recreate.
        self.navigation_message.emit(
            f"💥 内嵌浏览器进程崩溃 (status={termination_status}, exit={exit_code})，请重试"
        )

    def _show_cloudflare_challenge_dialog(self, url: str) -> None:
        """Pop a modal dialog explaining the Cloudflare "Verify you are
        human" interstitial and offering concrete remediation paths.

        Cloudflare's challenge page looks like a successful load —
        ``loadFinished(ok=True)`` fires, the URL bar shows the
        publisher's domain, but the visible page is the white
        "Just a moment..." spinner. Without this dialog the user
        sees an empty BrowserPanel and assumes the click "didn't
        work".

        Round 22: the previous version's only escalation was "↻
        retry", which QtWebEngine's Chromium almost never clears on
        aggressive publishers like ``pubs.acs.org`` (CF treats the
        embedded view as a high-risk client). We added a "🌐 system
        browser" handoff, but in practice users with a high-risk IP
        (corporate proxy, VPN, shared egress) find that even real
        Chrome can't pass — the issue is environmental, not the URL.

        This version is explicit about that:

          1. Open in system browser (best for normal home IPs).
          2. Copy remediation steps (lets the user route LitTrace
             through their real Chrome profile so the cf_clearance
             cookie survives across runs).
          3. Copy diagnostic snippet (URL + browser + platform) so
             the user can paste it into a bug report.
          4. ↻ in-view retry (escape hatch for sites where CF does
             clear quickly on its own).
        """
        # Non-modal-ish: stash the dialog on ``self`` so it survives
        # the GC sweep while the user reads it, but still allow
        # interaction with the rest of the window. ``setModal(False)``
        # is the default for ``QDialog`` so we don't bother.
        dlg = QtWidgets.QDialog(self.window())
        dlg.setWindowTitle("Cloudflare 验证拦截")
        dlg.resize(560, 380)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("⚠️ publisher 站被 Cloudflare 拦截了")
        heading.setStyleSheet(
            f"font-size:14px;font-weight:600;color:{DESIGN['ink']};"
        )
        layout.addWidget(heading)

        msg = QtWidgets.QLabel(
            "LitTrace 内嵌的 Chromium 命中了 Cloudflare 的「Verify you are "
            "human」验证。推荐点「🔧 在 LitTrace Chrome 登录」——会弹一个 "
            "LitTrace 自己启动的 Chrome 窗口，cookie 自动写入 LitTrace 的 profile，"
            "下次 sentinel 检索直接复用，不会再被拦。\n\n"
            "如果以上方法持续失败，根因通常是 Cloudflare 把你的 IP / 网络环境判定"
            "为高风险（公司代理、VPN、共享出口都算）。这种情况下：\n"
            "  1️⃣ 换网络：关 VPN / 公司代理，改用家庭宽带或手机热点\n"
            "  2️⃣ 复用你真 Chrome 的 profile：点「📋 复制切换步骤」照做\n"
            "  3️⃣ 临时绕过：点「🌐 系统浏览器打开」自行通过后手动下载 PDF（cookie "
            "不会自动复用，下次还要重新验证）"
        )
        msg.setWordWrap(True)
        msg.setStyleSheet(f"color:{DESIGN['ink']};line-height:1.5;")
        layout.addWidget(msg)

        url_label = QtWidgets.QLabel(f"URL：{url}")
        url_label.setStyleSheet(
            f"font-family:Menlo,Consolas,monospace;font-size:11px;"
            f"color:{DESIGN['ink_muted']};"
        )
        url_label.setWordWrap(True)
        layout.addWidget(url_label)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(8)
        # Primary action: launch a LitTrace-managed Chrome window
        # pointed at the publisher login URL. ``cf_clearance`` lands
        # in the *same* profile sentinel uses, so the next
        # ``cdp_downloader`` run inherits it automatically.
        # Round 22: this is the recommended path for repeat CF
        # encounters — only the first publisher visit per profile
        # triggers the challenge; subsequent ones reuse the cookie.
        lit_chrome_btn = QtWidgets.QPushButton("🔧 在 LitTrace Chrome 登录（推荐）")
        lit_chrome_btn.setObjectName("cf_lit_chrome")
        lit_chrome_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        lit_chrome_btn.clicked.connect(
            lambda: (
                self._open_publisher_in_littrace_chrome(url),
                dlg.accept(),
            )
        )
        button_row.addWidget(lit_chrome_btn)
        # Secondary action: open in the system default browser
        # (ChatGPT-style fallback). Cookies go to *that* browser's
        # profile, NOT LitTrace's — the user must complete the CF
        # challenge again the next time sentinel runs. Useful when
        # the user prefers their own browser and doesn't mind the
        # next-time dance.
        external_btn = QtWidgets.QPushButton("🌐 系统浏览器打开")
        external_btn.setObjectName("cf_external")
        external_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        external_btn.clicked.connect(
            lambda: (
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(url)),
                dlg.accept(),
            )
        )
        button_row.addWidget(external_btn)
        # Persistent-fix action: copy the steps that route LitTrace
        # through the user's real Chrome profile, so cf_clearance
        # survives across runs even when they used the system
        # browser button. Cross-platform via the helper below.
        steps_btn = QtWidgets.QPushButton("📋 复制切换步骤")
        steps_btn.setObjectName("cf_copy_steps")
        steps_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        steps_btn.clicked.connect(
            lambda: (
                QtWidgets.QApplication.clipboard().setText(
                    self._build_profile_switch_steps(url)
                ),
                self.navigation_message.emit(
                    "📋 切换步骤已复制到剪贴板"
                ),
            )
        )
        button_row.addWidget(steps_btn)
        # Diagnostic action: copy a one-liner with URL + browser +
        # platform, useful when filing a bug report.
        diag_btn = QtWidgets.QPushButton("🪪 复制诊断信息")
        diag_btn.setObjectName("cf_copy_diag")
        diag_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        diag_btn.clicked.connect(
            lambda: (
                QtWidgets.QApplication.clipboard().setText(
                    self._build_cf_diagnostic(url)
                ),
                self.navigation_message.emit(
                    "📋 诊断信息已复制到剪贴板"
                ),
            )
        )
        button_row.addWidget(diag_btn)
        button_row.addStretch(1)
        # Escape-hatch reload. Useful on publishers where CF clears
        # quickly on its own (Wiley, Nature) — just don't lead with
        # it on aggressive sites like pubs.acs.org.
        retry_btn = QtWidgets.QPushButton("↻ 重试")
        retry_btn.setObjectName("cf_retry")
        retry_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        retry_btn.clicked.connect(lambda: (
            self.navigation_message.emit(
                "↻ 重试 publisher Chrome：再次点 🌐 按钮即可"
            ),
            dlg.accept(),
        ))
        button_row.addWidget(retry_btn)
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.clicked.connect(dlg.accept)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

    def _open_publisher_in_littrace_chrome(self, url: str) -> None:
        """Launch a LitTrace-managed Chrome window pointed at the
        publisher sign-in page so the user can complete CF + login
        against the *same* profile ``cdp_downloader`` uses.

        Why this exists: opening ``webbrowser.open(url)`` from the
        cloudflare dialog drops the user into whichever browser the OS
        considers default (Safari on macOS, Edge on Windows). Cookies
        written there live in that browser's profile — completely
        separate from ``./data/chrome-cdp`` that the next
        ``sentinel run`` will use. The user is then stuck verifying
        Cloudflare once per publisher per run.

        The fix: spin up a Chrome process bound to LitTrace's profile
        and navigate it to the sign-in URL. ``cf_clearance`` lands in
        the same SQLite file the next ``cdp_downloader`` Chrome will
        read — *no* cookie import dance, *no* profile-share config.

        Round 22: this intentionally launches with ``headless=False``
        so the user actually sees the window and can interact with
        the publisher's login form. The window stays open until the
        user closes it (or until we detect a successful login via
        cookie polling — see ``_wait_for_login``). After that the
        process is terminated so sentinel can spawn its own headless
        Chrome against the now-unlocked profile.

        Implementation note: ``cdp_downloader.py`` already had the
        recipe for spinning up Chrome via CDP — we reuse it here
        verbatim rather than forking subprocess plumbing.
        """
        if self._config is None:
            self.navigation_message.emit("❌ 无 config，无法启动 publisher 登录")
            return
        # Round 25: the previous version launched Chrome synchronously
        # on the main thread (``launch_chrome_for_cdp`` waits up to
        # ``wait_seconds=4`` for the CDP endpoint, plus
        # ``browser.navigate(url, wait_seconds=6)`` waits another 6
        # seconds for the page to load). That was 10+ seconds of
        # frozen UI every time the user clicked a publisher button.
        #
        # Round 26: even after pushing ``launch + connect + navigate``
        # onto a background thread, the QTimer(2s) polling for cookie
        # arrival meant the user could still see a "卡顿" feel during
        # long authentications (CF + SSO round-trip can take 30+s).
        # This round replaces the polling with **event-driven** cookie
        # detection via ``QtCDPBrowser.subscribe_event``. Chrome
        # pushes ``Network.cookieChanged`` the instant any cookie
        # is written, and the main thread's event loop dispatches
        # our handler within microseconds — no polling, no race
        # conditions, no perceived freeze.
        #
        # The pipeline is now:
        #   1. Main thread: emit status, spawn a worker thread that
        #      calls ``launch_chrome_for_cdp`` (still synchronous —
        #      ``Popen`` + a few ``time.sleep`` polls — but on a
        #      background thread, so the UI doesn't notice).
        #   2. Worker thread: fetch the page-level WebSocket URL
        #      from ``GET /json/list``, hand it back to the main
        #      thread via ``QMetaObject.invokeMethod``.
        #   3. Main thread: construct ``QtCDPBrowser``, subscribe to
        #      ``Network.cookieChanged``, open the WebSocket. CDP
        #      commands and event delivery all flow through Qt's
        #      event loop from here on — zero polling.
        self.navigation_message.emit("🌐 启动 LitTrace Chrome…")
        # ``publisher_label`` is the human-readable name we look for
        # in ``cookie_changed`` callbacks. We pass it down so the
        # background thread doesn't need to import the catalog.
        publisher_domain = self._publisher_domain_for_url(url) or "publisher"
        threading.Thread(
            target=self._launch_publisher_chrome_worker,
            args=(url, publisher_domain),
            daemon=True,
            name="littrace-publisher-launcher",
        ).start()
        self.navigation_message.emit("🌐 启动 LitTrace Chrome…")
        threading.Thread(
            target=self._launch_publisher_chrome_worker,
            args=(url,),
            daemon=True,
            name="littrace-publisher-launcher",
        ).start()

    def _emit_status_threadsafe(self, message: str) -> None:
        """Push a ``navigation_message`` emit through the Qt event
        loop so it lands on the main thread regardless of which
        thread invoked this. Used by ``_launch_publisher_chrome_worker``
        (background thread) and the polling timers (already on main).
        """
        # ``invokeMethod`` with ``QueuedConnection`` schedules the
        # slot invocation onto the receiver's event loop — exactly
        # what we need for thread-safe Qt widget updates.
        QtCore.QMetaObject.invokeMethod(
            self,
            "_do_emit_navigation_message",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, message),
        )

    @QtCore.Slot(str)
    def _do_emit_navigation_message(self, message: str) -> None:
        self.navigation_message.emit(message)

    def _launch_publisher_chrome_worker(self, url: str, publisher_label: str) -> None:
        """Background-thread implementation of the
        ``_open_publisher_in_littrace_chrome`` pipeline.

        Round 26: instead of constructing a synchronous ``CDPBrowser``
        (which blocks on ``ws.send`` / ``ws.recv`` and then polling
        cookies every 2s), we hand the page-level WebSocket URL back
        to the main thread, which constructs a ``QtCDPBrowser`` and
        subscribes to ``Network.cookieChanged`` events. Chrome
        *pushes* cookie writes to us — no polling, no main-thread
        freeze.
        """
        try:
            import httpx
            from littrace.chrome_profiles import launch_chrome_for_cdp
        except Exception as exc:
            self._emit_status_threadsafe(f"❌ 依赖加载失败: {exc!r}")
            return
        launch = launch_chrome_for_cdp(self._config, headless=False)
        if launch.cdp_status is None or not launch.cdp_status.available:
            self._emit_status_threadsafe(
                f"❌ LitTrace Chrome 启动失败：{launch.error or '未知错误'}"
            )
            return
        self._emit_status_threadsafe("🌐 Chrome 已启动，正在打开 publisher 登录页…")
        # Fetch a fresh page-level WebSocket URL. ``/json/new`` opens
        # a new tab pointing at ``about:blank`` and returns its
        # ``webSocketDebuggerUrl`` — that's the endpoint our
        # ``QtCDPBrowser`` will subscribe to. We do this on the
        # worker thread because it's a synchronous HTTP call.
        try:
            response = httpx.put(
                f"{self._config.cdp_downloader.cdp_url.rstrip('/')}/json/new?about:blank",
                timeout=10.0,
            )
            response.raise_for_status()
            page_ws_url = response.json().get("webSocketDebuggerUrl")
            if not page_ws_url:
                raise RuntimeError("CDP did not return a webSocketDebuggerUrl")
        except Exception as exc:
            self._emit_status_threadsafe(f"❌ 无法获取 page WS URL: {exc!r}")
            return
        # Hand off to the main thread. ``QtCDPBrowser`` is a QObject
        # so it must be created and live on the main thread.
        QtCore.QMetaObject.invokeMethod(
            self,
            "_attach_async_publisher_browser",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, page_ws_url),
            QtCore.Q_ARG(str, url),
            QtCore.Q_ARG(str, publisher_label),
        )

    @QtCore.Slot(str, str, str)
    def _attach_async_publisher_browser(
        self, page_ws_url: str, url: str, publisher_label: str
    ) -> None:
        """Main-thread slot that wires up the ``QtCDPBrowser``.

        Called via ``QMetaObject.invokeMethod`` after the worker
        thread has Chrome ready and a page WebSocket URL in hand.
        Constructs the browser, subscribes to cookie events, and
        opens the WebSocket. ``connected`` callback kicks off the
        navigation once the socket is up.
        """
        try:
            from littrace.access_layer.qt_cdp_browser import QtCDPBrowser
        except Exception as exc:
            self.navigation_message.emit(f"❌ QtCDPBrowser 加载失败: {exc!r}")
            return
        # Tear down any previous instance — re-entry safety. The
        # previous browser's ``QWebSocket`` may still be open; close
        # it so Chrome doesn't end up with a pile of zombie tabs.
        existing = getattr(self, "_qt_cdp_browser", None)
        if existing is not None:
            try:
                existing.close()
            except Exception:
                pass
        browser = QtCDPBrowser(self)
        self._qt_cdp_browser = browser
        # Subscribe to cookie events. The handler runs on the main
        # thread (QtCDPBrowser was constructed here) so it's safe to
        # touch widgets directly — no ``invokeMethod`` needed.
        #
        # Round 26: in headless / offscreen CDP profiles
        # ``Network.cookieChanged`` is not delivered (verified
        # 2026-09: ``Network.getAllCookies`` after the same
        # navigation sees the cookie, but ``cookieChanged`` is
        # silent). On the user's real GUI Chrome the event *does*
        # fire. We subscribe either way — when it works it's
        # instant; when it doesn't we fall back to the
        # ``Network.getAllCookies`` polling loop below.
        def _on_cookie(params):
            self._handle_publisher_cookie_event(params, publisher_label, url)
        browser.cookie_changed.connect(_on_cookie)
        # Once connected: enable Network events and navigate.
        def _on_connected():
            self.navigation_message.emit(
                f"🌐 CDP 已连接，导航到 {url}…"
            )
            # Enable Network events so we *might* get cookie_changed
            # (works on headed Chrome; harmless on headless).
            browser.enable_domain(
                "Network",
                on_response=lambda _msg: None,
            )
            # Navigate. ``send`` is non-blocking — the page load
            # proceeds on Chrome's side while the main thread is
            # free to handle other events.
            browser.send("Page.navigate", {"url": url})
            # Kick off the cookie-status polling timer. Each tick
            # sends ``Network.getAllCookies`` async — the response
            # arrives later via ``on_response``, never blocking the
            # main thread. When ``cookie_changed`` is firing this
            # is mostly redundant; when it isn't (offscreen / older
            # Chrome) this is the only way we learn the user logged in.
            self._start_login_polling(
                domains=tuple(
                    d for pub in PUBLISHER_CATALOG
                    if publisher_label in pub.cookie_domains
                    for d in pub.cookie_domains
                ),
                publisher_label=publisher_label,
                timeout_seconds=300,
            )
        browser.connected.connect(_on_connected)
        # Errors surface to the status bar but don't tear down the
        # browser — a transient socket hiccup is recoverable on
        # reconnect, and we'd rather see something on screen than
        # silently swallow.
        def _on_error(_id, code, message):
            self.navigation_message.emit(f"⚠️ CDP {code}: {message}")
        browser.cdp_error.connect(_on_error)
        browser.open(page_ws_url)

    def _handle_publisher_cookie_event(
        self, params: dict, publisher_label: str, url: str,
    ) -> None:
        """``cookie_changed`` handler — fired by QtCDPBrowser the
        instant Chrome writes any cookie on any domain. We filter
        for publisher-domain / known-tracker-bypass cookies and
        refresh the cookie status strip on hit. This is what makes
        the publisher login flow event-driven (no polling).
        """
        # ``Network.cookieChanged`` envelope: ``{"cookie": {...},
        # "cause": "explicit" | "..."}``. Filter out cookies that
        # don't tell us anything about publisher login (CF bot
        # management, Google Analytics, etc.).
        cookie = (params or {}).get("cookie") or {}
        name = cookie.get("name") or ""
        host = (cookie.get("domain") or "").lstrip(".").lower()
        if not name or not host:
            return
        ignorable = {
            "__cf_bm", "__cfruid",
            "_ga", "_gid", "_gat",
            "_fbp", "_fbc",
            "IDE", "NID",
        }
        if name in ignorable:
            return
        # Get the publisher's known cookie domains.
        domains = next(
            (pub.cookie_domains for pub in PUBLISHER_CATALOG
             if publisher_label in pub.cookie_domains),
            (publisher_label,),
        )
        is_publisher = any(
            host == d.lstrip(".") or host.endswith("." + d.lstrip("."))
            for d in domains
        )
        # Non-tracker, non-publisher domains are accepted as SSO
        # signals (institutional logins land on shibboleth/etc.).
        tracker_domains = {
            ".google-analytics.com", ".googletagmanager.com",
            ".doubleclick.net", ".facebook.com", ".cloudflare.com",
            ".cloudflareinsights.com", ".hotjar.com",
            ".segment.io", ".mixpanel.com", ".newrelic.com",
        }
        is_sso = not is_publisher and not any(
            host.endswith(t.lstrip(".")) for t in tracker_domains
        )
        if not (is_publisher or is_sso):
            return
        # Found a login-bearing cookie. Refresh the UI and stop
        # listening — once we're logged in, more cookie_changed
        # events would just be noise.
        kind = "publisher" if is_publisher else "sso"
        self.navigation_message.emit(
            f"✓ 检测到 {publisher_label} 鉴权完成（{kind} cookie: {host}/{name}）"
        )
        self._refresh_cookie_status()
        # Stop the timer-based polling too, in case it was started
        # in parallel by the older code path.
        for attr in ("_login_poll_timer", "_login_heartbeat_timer"):
            t = getattr(self, attr, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
            setattr(self, attr, None)
        # Keep the WebSocket open so subsequent cookie events on
        # other publishers still surface — but tear down the handler
        # so this publisher's success doesn't get re-emitted.

    @QtCore.Slot(str, str)
    def _kick_polling_from_worker(self, url: str, publisher_label: str = "") -> None:
        """Deprecated polling-path entry. Kept for callers that
        still expect a polling-style hook (e.g. unit tests); the
        real publisher-login path now goes through the
        ``cookie_changed`` event in ``_attach_async_publisher_browser``
        and never calls this in production. The QTimer-based polling
        it would have started is harmless if no timer fires.
        """
        publisher_domain = publisher_label or self._publisher_domain_for_url(url)
        if not publisher_domain:
            self.navigation_message.emit(
                "⚠️ 不知道这是哪个 publisher，请关掉 Chrome 后手动确认 cookie 状态"
            )
            return
        domains = next(
            (pub.cookie_domains for pub in PUBLISHER_CATALOG
             if publisher_domain in pub.cookie_domains),
            (publisher_domain,),
        )
        self._start_login_polling(
            domains=domains,
            publisher_label=publisher_domain or "publisher",
            timeout_seconds=300,
        )
        # KEEP THE CHROME PROCESS RUNNING. Why: on macOS, Chrome
        # encrypts cookies with a Keychain secret that's tied to
        # the *running* browser process. A fresh headless Chrome
        # spawned by ``sentinel`` later cannot decrypt it — even
        # with ``--use-mock-keychain`` (verified empirically
        # 2026-09: cookie set via Set-Cookie response is invisible
        # to a fresh Chrome process against the same profile).
        #
        # The trick is to leave the GUI Chrome alive and let
        # sentinel REUSE the same CDP endpoint. ``cdp_downloader``
        # already does the right thing: it calls ``check_cdp_status``
        # before spawning anything, and if a Chrome is already
        # listening on 19222 it just connects.
        #
        # If the user manually closes the window, the next sentinel
        # run will start a fresh headless Chrome and have to earn
        # cf_clearance from scratch. That's an acceptable
        # degradation, not a regression.

    def _publisher_domain_for_url(self, url: str) -> str | None:
        """Best-effort: which publisher catalog entry does this URL
        belong to? Used by ``_open_publisher_in_littrace_chrome`` to
        decide which cookie domains to watch for after login.

        Note: reuses the module-level ``PUBLISHER_CATALOG`` alias
        imported at the top of this file; ``PUBLISHER_CATALOG`` is
        not itself exported from ``publisher_catalog.py``.
        """
        host = (url.split("://", 1)[-1].split("/", 1)[0] if "://" in url
                else url.split("/", 1)[0]).lower()
        for pub in PUBLISHER_CATALOG:
            for d in pub.cookie_domains:
                if d in host or host.endswith("." + d):
                    return d
        return None

    def _start_login_polling(
        self,
        *,
        domains: tuple[str, ...],
        publisher_label: str,
        timeout_seconds: float = 300.0,
    ) -> None:
        """Kick off async (QTimer-driven) cookie polling so the Qt
        window stays responsive while the user works through CF / SSO
        in the external Chrome window.

        Why async: the previous version ran ``_wait_for_login`` on the
        main thread with ``time.sleep(2.0)`` between polls, which
        froze the whole Qt UI for up to 5 minutes — the user couldn't
        even move the LitTrace window while signing in. The Qt-native
        fix is a QTimer that fires every 2s on the main thread but
        doesn't block it.

        On success or timeout the timer self-stops. We also schedule
        a ``QTimer.singleShot`` to fire ``_finish_login_polling`` with
        ``ok=False`` if the user takes too long.

        Re-entrancy: if the user clicks a second publisher button
        while the first poll is still running, we replace the existing
        timer so we only watch one publisher at a time. The cookie
        status strip already shows the union, so partial overlap is
        fine.
        """
        # Stop any existing poll (re-entry safety)
        existing = getattr(self, "_login_poll_timer", None)
        if existing is not None:
            try:
                existing.stop()
            except Exception:
                pass
        iteration = [0]  # boxed so the slot closure can mutate
        state = {"done": False, "hit": None, "reason": None,
                 "publisher_hits": [], "non_tracker_hits": []}

        def _poll() -> None:
            if state["done"]:
                return
            iteration[0] += 1
            result = self._poll_login_once(domains)
            # Result shape: (publisher_hit_list, non_tracker_hit_list)
            pub_hits, nt_hits = result
            state["publisher_hits"] = pub_hits
            state["non_tracker_hits"] = nt_hits
            if pub_hits or nt_hits:
                state["done"] = True
                state["hit"] = "publisher" if pub_hits else "sso"
                timer.stop()
                self._finish_login_polling(
                    ok=True,
                    publisher_label=publisher_label,
                    hit_kind=state["hit"],
                    publisher_hits=pub_hits,
                    non_tracker_hits=nt_hits,
                )

        timer = QtCore.QTimer(self)
        timer.setInterval(2000)
        timer.timeout.connect(_poll)
        # Periodic heartbeat so the user sees progress
        def _heartbeat() -> None:
            if state["done"]:
                return
            if iteration[0] > 0 and iteration[0] % 5 == 0:
                self.navigation_message.emit(
                    f"⏳ 等待 {publisher_label} 鉴权…"
                    f"（已轮询 {iteration[0]} 次）"
                )
        heartbeat_timer = QtCore.QTimer(self)
        heartbeat_timer.setInterval(10_000)
        heartbeat_timer.timeout.connect(_heartbeat)
        self._login_poll_timer = timer
        self._login_heartbeat_timer = heartbeat_timer
        self._login_poll_state = state
        timer.start()
        heartbeat_timer.start()
        # Hard timeout
        QtCore.QTimer.singleShot(
            int(timeout_seconds * 1000),
            lambda: self._finish_login_polling(
                ok=False,
                publisher_label=publisher_label,
                hit_kind=None,
                publisher_hits=[],
                non_tracker_hits=[],
            ),
        )

    def _poll_login_once(
        self, domains: tuple[str, ...],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Single async CDP cookie probe — fires an event-driven
        ``Network.getAllCookies`` request via ``QtCDPBrowser`` and
        never blocks the main thread. The response handler
        (``_on_get_cookies_resp``) is wired in ``_start_login_polling``
        and dispatches the same ``publisher_hits`` / ``non_tracker_hits``
        shape the synchronous version used to return.

        We keep this method name so the polling loop in
        ``_start_login_polling`` keeps firing — the *real* work happens
        in the async callback. Returns ``([], [])`` here because the
        caller can't usefully act on the result synchronously; the
        callback fires later and stops the polling timer itself.
        """
        # Prefer the async QtCDPBrowser (set up by
        # ``_attach_async_publisher_browser``); fall back to the
        # synchronous browser only when the async path wasn't used.
        async_browser = getattr(self, "_qt_cdp_browser", None)
        if async_browser is not None:
            try:
                async_browser.send(
                    "Network.getAllCookies",
                    on_response=lambda msg, d=tuple(domains):
                        self._handle_async_cookie_response(msg, d),
                )
            except Exception:
                pass
            return ([], [])
        # Legacy sync path — used by tests that drive the synchronous
        # ``CDPBrowser`` directly. Kept for backwards compatibility.
        browser = getattr(self, "_littrace_login_browser", None)
        if browser is None or getattr(browser, "ws", None) is None:
            return ([], [])
        import json
        try:
            browser.ws.send(json.dumps({
                "id": 9001, "method": "Network.getCookies",
                "params": {"urls": []},
            }))
            resp = json.loads(browser.ws.recv())
            cookies = (resp.get("result") or {}).get("cookies") or []
            return self._classify_cookies(cookies, domains)
        except Exception:
            return ([], [])

    def _classify_cookies(
        self,
        cookies: list[dict],
        domains: tuple[str, ...],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Filter a raw ``Network.getCookies`` /
        ``Network.getAllCookies`` payload into the two lists the
        polling loop cares about. Shared between the sync fallback
        in ``_poll_login_once`` and the async response handler.
        """
        ignorable = {
            "__cf_bm", "__cfruid",
            "_ga", "_gid", "_gat",
            "_fbp", "_fbc",
            "IDE", "NID",
        }
        real = [c for c in cookies if c.get("name") not in ignorable]
        publisher_hits: list[tuple[str, str]] = []
        non_tracker_hits: list[tuple[str, str]] = []
        tracker_domains = {
            ".google-analytics.com", ".googletagmanager.com",
            ".doubleclick.net", ".facebook.com", ".cloudflare.com",
            ".cloudflareinsights.com", ".hotjar.com",
            ".segment.io", ".mixpanel.com", ".newrelic.com",
        }
        for c in real:
            host = c.get("domain", "").lstrip(".").lower()
            if not host:
                continue
            matched_publisher = False
            for d in domains:
                d_clean = d.lstrip(".")
                if host == d_clean or host.endswith("." + d_clean):
                    publisher_hits.append((host, c.get("name") or ""))
                    matched_publisher = True
                    break
            if matched_publisher:
                continue
            if not any(host.endswith(t.lstrip("."))
                       for t in tracker_domains):
                non_tracker_hits.append((host, c.get("name") or ""))
        return (publisher_hits, non_tracker_hits)

    def _handle_async_cookie_response(
        self, msg: dict, domains: tuple[str, ...],
    ) -> None:
        """Callback for ``Network.getAllCookies`` invoked from
        ``_poll_login_once`` via ``QtCDPBrowser.send``. Runs on the
        main thread (QtCDPBrowser is constructed there), so it's
        safe to touch widgets directly.
        """
        cookies = (msg.get("result") or {}).get("cookies") or []
        pub_hits, nt_hits = self._classify_cookies(cookies, domains)
        if not (pub_hits or nt_hits):
            return
        # Match the synchronous behavior: post a status, refresh
        # the cookie status strip, and stop the polling timer.
        # The publisher_label is grabbed from the in-flight state.
        state = getattr(self, "_login_poll_state", None)
        if state is None or state.get("done"):
            return
        state["done"] = True
        publisher_label = state.get("publisher_label", "publisher")
        kind = "publisher" if pub_hits else "sso"
        sample = pub_hits if pub_hits else nt_hits
        self.navigation_message.emit(
            f"✓ 检测到 {publisher_label} 鉴权完成（{kind} cookie: " +
            ", ".join(f"{h}/{n}" for h, n in sample[:3]) + "）"
        )
        # Stop polling timers
        for attr in ("_login_poll_timer", "_login_heartbeat_timer"):
            t = getattr(self, attr, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
            setattr(self, attr, None)
        self._refresh_cookie_status()

    def _finish_login_polling(
        self,
        *,
        ok: bool,
        publisher_label: str,
        hit_kind: str | None,
        publisher_hits: list[tuple[str, str]],
        non_tracker_hits: list[tuple[str, str]],
    ) -> None:
        """Stop the polling timer, post the terminal status message,
        and refresh the cookie status strip.

        Idempotent: if the timer already stopped itself on a hit,
        the timeout's ``singleShot`` will still call this with
        ``ok=False``. We detect that via ``self._login_poll_state``
        and no-op when the success path already ran.
        """
        state = getattr(self, "_login_poll_state", None)
        if state is None:
            return
        if state.get("done") and not ok:
            # Already finished via success — the timeout callback is
            # firing late. Ignore.
            return
        # Stop both timers
        for attr in ("_login_poll_timer", "_login_heartbeat_timer"):
            t = getattr(self, attr, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
            setattr(self, attr, None)
        state["done"] = True
        if ok:
            if hit_kind == "publisher":
                self.navigation_message.emit(
                    f"✓ {publisher_label} 登录成功（publisher cookie: " +
                    ", ".join(f"{h}/{n}" for h, n in publisher_hits[:3]) + "）"
                )
            else:
                self.navigation_message.emit(
                    f"✓ {publisher_label} 登录成功（SSO cookie: " +
                    ", ".join(f"{h}/{n}" for h, n in non_tracker_hits[:3]) + "）"
                )
        else:
            self.navigation_message.emit(
                f"⚠️ 等待 {publisher_label} 鉴权超时，请关掉 Chrome 后再试"
            )
        # Always refresh the cookie status strip — even on timeout
        # there might still be a partial login (publisher cookie
        # alone, SSO cookie alone) and the strip should reflect it.
        self._refresh_cookie_status()

    def _wait_for_login(
        self, domains: tuple[str, ...], timeout_seconds: float,
    ) -> bool:
        """Deprecated synchronous variant — kept for compatibility
        with callers that don't need the async behavior. New code
        should use ``_start_login_polling`` instead, which keeps the
        Qt window responsive.

        Why CDP and not the SQLite file: on macOS Chrome encrypts
        cookies with a per-profile Keychain secret, so the visible
        ``Cookies`` SQLite file is just an empty shell — the actual
        cookie values never hit disk. Asking Chrome through CDP
        (``Network.getCookies``) is the only reliable way to learn
        whether the user has finished logging in.

        Round 24: institutional SSO is now common (ACS through
        Shibboleth / OpenAthens / EZproxy, Wiley through federation
        gateways, etc.). On those paths the *session* cookie lands
        on a third-party host (``shibboleth.xxx.edu``,
        ``login.ezproxy.xxx``), not on the publisher's own domain.
        So we accept any non-CF cookie whose host is *not* one of
        the obvious CF / tracker names. If the user can see a logged-in
        ACS page in the Chrome window, *some* domain has a session
        cookie — that's what we detect.
        """
        # Thin wrapper over the async path: start it and synchronously
        # block waiting for completion. Only used by the original
        # synchronous call site; the new ``_open_publisher_in_littrace_chrome``
        # uses ``_start_login_polling`` directly so the Qt UI stays
        # alive while the user logs in.
        done = {"ok": False}
        def _wrap(ok, **_):
            done["ok"] = ok
        self._start_login_polling(
            domains=domains,
            publisher_label="publisher",
            timeout_seconds=timeout_seconds,
        )
        # Hook the terminal callback
        original = self._finish_login_polling
        def _hooked(**kwargs):
            _wrap(**{k: kwargs[k] for k in ("ok",) if k in kwargs})
            original(**kwargs)
        # Simpler: just poll our own state from the existing timer
        state = getattr(self, "_login_poll_state", None)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if state is not None and state.get("done"):
                done["ok"] = True
                break
            time.sleep(0.5)
        return done["ok"]

        # Replace any prior dialog so the user doesn't end up with a
        # stack of them after clicking several publisher shortcuts in
        # a row.
        previous = getattr(self, "_cf_dialog", None)
        if previous is not None:
            try:
                previous.close()
                previous.deleteLater()
            except Exception:
                pass
        self._cf_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _build_profile_switch_steps(self, url: str) -> str:
        """Build a clipboard-ready snippet that walks the user through
        routing LitTrace through their *real* Chrome profile.

        Why: QtWebEngine and the user's Chrome use separate profile
        directories (``./data/chrome-cdp`` vs.
        ``~/Library/Application Support/Google/Chrome``). The
        ``cf_clearance`` cookie the user earns by passing Cloudflare
        in real Chrome does *not* automatically appear in LitTrace's
        embedded view. The fix is to point ``chrome_user_data_dir``
        at the system Chrome profile — but only AFTER exiting Chrome
        (Chromium holds an exclusive lock on its profile dir).

        Round 22: this is a manual two-step (edit config + restart
        LitTrace). We considered a one-click "switch now" button, but
        the live profile lock makes a graceful in-process switch
        unreliable across Chrome versions, so a copy-paste terminal
        command is more robust and easier to roll back.
        """
        import platform
        import os
        system = platform.system().lower()
        home = os.path.expanduser("~")
        if system == "darwin":
            profile_dir = (
                f"{home}/Library/Application Support/Google/Chrome"
            )
            profile_name = "Default"
        elif system == "windows":
            profile_dir = (
                f"%LOCALAPPDATA%\\Google\\Chrome\\User Data"
            )
            profile_name = "Default"
        else:
            profile_dir = f"{home}/.config/google-chrome"
            profile_name = "Default"
        # The LitTrace config file lives next to wherever the user
        # launched ``littrace-window`` from — typically
        # ``<repo>/config.yaml``. We don't have a handle to the
        # on-disk path through ``_config`` (it's a parsed model,
        # not a file ref), so the snippet just names the file the
        # user can edit in-place. If a future round wires the
        # on-disk path through, this is the spot to swap it in.
        config_path = "./config.yaml"
        return (
            "# LitTrace: 让内嵌浏览器复用真 Chrome 的 profile（解决 ACS 等站点 "
            "反复触发 Cloudflare 验证）\n"
            "#\n"
            "# 前置：先在系统 Chrome 里完成 ACS 验证，然后**完全退出 Chrome** "
            "（包括 dock/任务栏图标右键退出，不是只关窗口）。\n"
            "#\n"
            f"# 步骤 1 — 修改 config.yaml，把 chrome_user_data_dir 改成 "
            f"你的真 Chrome profile：\n"
            f"#   编辑 {config_path}，找到 cdp_downloader 段：\n"
            f"#     cdp_downloader:\n"
            f"#       chrome_user_data_dir: \"{profile_dir}\"\n"
            f"#       chrome_profile_name: \"{profile_name}\"\n"
            "\n"
            "# 步骤 2 — 完全退出 LitTrace（⌘Q / 任务栏右键退出），再重新启动 "
            "littrace-window。\n"
            "\n"
            "# 步骤 3 — 再点 🌐 ACS 按钮，cf_clearance cookie 已经继承，"
            "Cloudflare 不再触发。\n"
            "\n"
            "# 注意：LitTrace 启动后**不要再开系统 Chrome**，否则 Chrome profile "
            "lock 会冲突。检索完再开 Chrome。\n"
        )

    def _build_cf_diagnostic(self, url: str) -> str:
        """Compact diagnostic snippet — URL + platform + LitTrace
        component versions — for the user to paste into a bug report
        when the Cloudflare dialog keeps firing despite all the
        above. Read-only, never sends data anywhere."""
        import platform
        import sys
        try:
            from PySide6 import QtCore as _qc
            qt_version = _qc.qVersion()
        except Exception:
            qt_version = "unknown"
        # QtWebEngine's Chromium version is the CF-relevant signal —
        # CF fingerprints the UA which QtWebEngine builds from its
        # embedded Chromium version.
        chromium_version = "unknown"
        try:
            from PySide6.QtWebEngineCore import (
                qWebEngineChromiumVersion,
            )
            chromium_version = str(qWebEngineChromiumVersion())
        except Exception:
            try:
                # PyQt5/6 alternate name
                from PyQt5.QtWebEngineCore import (
                    qWebEngineChromiumVersion as _qv,
                )
                chromium_version = str(_qv())
            except Exception:
                pass
        bits = [
            f"url: {url}",
            f"platform: {platform.platform()}",
            f"python: {sys.version.split()[0]}",
            f"qt: {qt_version}",
            f"qtwebengine-chromium: {chromium_version}",
        ]
        try:
            from littrace import __version__ as littrace_version
            bits.append(f"littrace: {littrace_version}")
        except Exception:
            pass
        # Wrap in a code block for easy paste into GitHub issues.
        return "```\n" + "\n".join(bits) + "\n```"

    def open_url(self, url: str) -> None:
        """Open ``url`` in the LitTrace-private Chrome window so the
        cookies it writes land in ``data/chrome-cdp`` (not the
        user's daily Chrome profile). Normalises bare hosts
        (``example.com`` → ``https://example.com``) and tolerates
        missing scheme prefixes; safe to call from any caller.

        Round 29: ``QDesktopServices.openUrl`` was opening the
        URL in the user's *system default browser* — which on
        macOS is the user's day-to-day Chrome profile, not
        LitTrace's ``data/chrome-cdp``. Publisher SSO cookies then
        ended up in the wrong profile and sentinel's next gated
        PDF fetch would still trip Cloudflare. The fix is to
        route the URL through LitTrace's own CDP-managed Chrome
        instance (``QtCDPBrowser``) which always reads from /
        writes to ``data/chrome-cdp`` — the same Chrome sentinel
        drives for gated downloads. ``QtCDPBrowser`` is
        constructed lazily on first use; subsequent calls reuse
        it (and the same tab) so this also stops the previous
        behaviour of opening a fresh chrome window per click.
        """
        text = (url or "").strip()
        if not text:
            return
        if not text.startswith(("http://", "https://")):
            text = "https://" + text
        self._url_before_suspend = text
        # Lazy QtCDPBrowser launch. The instance is parented to
        # the main window so Qt tears it down on close; the
        # ``_qt_cdp_browser`` attribute also tracks the running
        # instance so we don't spawn a second one for every URL
        # click.
        win = self.window()
        if win is not None and hasattr(win, "_open_url_in_littrace_chrome"):
            try:
                self.navigation_message.emit(
                    f"🌐 在 LitTrace Chrome 打开 {text}"
                )
                win._open_url_in_littrace_chrome(text)
                return
            except Exception:
                # Fall through to the URL bar status — never let
                # a publisher login click silently no-op.
                pass
        self.navigation_message.emit(
            f"⚠️ LitTrace Chrome 未就绪，无法在隔离 profile 中打开 {text}"
        )

    # ---- Suspend / resume (Round 18: external chrome handoff) ----

    def suspend_for_external_chrome(self) -> None:
        """Round 26: no-op. The embedded ``QWebEngineView`` was
        removed in round 26 because its suspend/teardown dance raced
        with Chromium's own BrowserThread bookkeeping and aborted
        the whole GUI process (``Must be called on Chrome_UIThread;
        actually called on Unknown Thread``). ``QtCDPBrowser`` owns
        the publisher-login Chrome instead; sentinel reuses that
        same Chrome instance via ``cdp_downloader``'s
        ``check_cdp_status`` path, so the BrowserPanel doesn't need
        to do any teardown to release the profile lock — it never
        held it in the first place.
        """
        return

    def resume_after_external_chrome(self) -> None:
        """Round 26: no-op. Mirror of ``suspend_for_external_chrome``
        — the BrowserPanel no longer hosts a QWebEngineView, so
        there's nothing to recreate. Kept as a method so the sentinel
        lifecycle code (``_acquire_*`` / ``_release_*``) doesn't have
        to know the BrowserPanel internals changed.
        """
        return


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
        # Round 27: the cross-process sentinel subprocess signals
        # "I'm blocked on Cloudflare" by writing a JSON file under
        # ``data/``. We poll the same file from the GUI thread and
        # pop a modal that tells the user to clear the challenge in
        # their Chrome. When they click "我处理好了" we write the
        # acknowledgement file and the sentinel subprocess picks it up
        # immediately. ``_cf_wait_dialog`` holds the active dialog
        # (or None) so we never stack two modals on top of each other.
        self._cf_wait_dialog: QtWidgets.QDialog | None = None
        self._cf_wait_last_key: tuple = ()
        self._cf_poll_timer = QtCore.QTimer(self)
        self._cf_poll_timer.setInterval(2000)
        self._cf_poll_timer.timeout.connect(self._poll_cf_wait_flag)
        self._cf_poll_timer.start()
        self.setStyleSheet(WINDOW_QSS)

        self._build_layout()
        self._wire_events()
        self._refresh_initial_state()
        # Round 18: do NOT auto-launch the external chrome on startup.
        # Login now happens inside the embedded BrowserPanel (whose
        # default profile was redirected to ``data/chrome-cdp`` in
        # ``main()``). The external chrome is launched lazily only when
        # sentinel needs CDP — see ``_acquire_external_chrome_for_sentinel``.
        # Spawning both at once would race for the SingletonLock on
        # ``data/chrome-cdp`` and either fail to start the external
        # chrome or corrupt the on-disk Cookies SQLite.

    def _auto_launch_chrome(self) -> None:
        # Round 18: this method now lives behind
        # ``_acquire_external_chrome_for_sentinel`` (lazy, only when
        # sentinel needs CDP). The startup-time auto-launch was removed
        # because the embedded QtWebEngine view now writes cookies to
        # ``data/chrome-cdp`` directly — there is no longer a need for
        # a parallel external chrome to be alive for the publisher
        # buttons to "work".
        #
        # The work runs on a daemon thread so the GUI thread doesn't
        # block on Chrome's slow startup (~2-4 s on cold cache). The
        # ``Popen`` handle from ``launch_chrome_for_cdp`` is stashed on
        # ``self._external_chrome_proc`` so ``_release_external_chrome_for_sentinel``
        # can ``terminate()`` it later.
        import threading
        from littrace.chrome_profiles import launch_chrome_for_cdp

        def _worker():
            try:
                # Round 18: littrace-qt's sentinel CDP companion always
                # launches headless so the user never sees a second
                # chrome.exe window over the Qt shell. Operators who need
                # a visible chrome for SSO debugging flip
                # ``cdp_downloader.headless = false`` in config.yaml.
                result = launch_chrome_for_cdp(
                    self._controller.config,
                    headless=self._controller.config.cdp_downloader.headless,
                )
                if result.launched:
                    self._external_chrome_proc = result.process
                    self._post_status("publisher Chrome 已启动（CDP 19222）")
                elif getattr(result, "already_available", False):
                    # Round 26 fix: the user already opened a publisher
                    # Chrome window via the BrowserPanel shortcut and
                    # it's still alive on port 19222 — sentinel will
                    # happily reuse that CDP endpoint (see
                    # ``cdp_downloader.check_cdp_status`` which returns
                    # ``available=True`` for it). We just need to
                    # acknowledge the existing process so
                    # ``_release_external_chrome_for_sentinel`` later
                    # knows not to leave it dangling. The pre-Round-26
                    # logic fell through to the else branch and
                    # reported "external chrome 未启动" — false alarm.
                    self._post_status(
                        "publisher Chrome 已在跑（sentinel 复用）"
                    )
                elif result.process is not None:
                    # CDP endpoint never came up but Chrome is alive —
                    # kill it so we don't leak a chrome.exe holding the
                    # profile lock.
                    try:
                        result.process.terminate()
                        result.process.wait(timeout=5.0)
                    except Exception:
                        pass
                else:
                    # Round 20: chrome failed to launch entirely. The
                    # previous behaviour silently swallowed this so the
                    # user only saw a cryptic "sentinel 退出 1" line
                    # ~2 minutes later with no hint that chrome was the
                    # root cause. Surface the launcher's own error
                    # message (e.g. "Could not build a Chrome launch
                    # command", or the SingletonLock conflict message
                    # from ``chrome_profiles._explain_chrome_early_exit``)
                    # so the user knows to fix Chrome / check config.
                    msg = result.error or "external chrome 未启动（原因未知）"
                    self._post_status(f"⚠️ publisher Chrome 启动失败：{msg}")
            except Exception as exc:
                # Round 20: ``launch_chrome_for_cdp`` itself raised
                # (rather than returning a result with an error field).
                # Still surface it instead of letting sentinel fail
                # later with a non-actionable message.
                self._post_status(
                    f"⚠️ publisher Chrome 启动异常：{exc.__class__.__name__}: {exc}"
                )

        threading.Thread(target=_worker, daemon=True, name="littrace-auto-chrome").start()

    # ---- External chrome lifecycle (sentinel needs CDP) ------------------

    def _acquire_external_chrome_for_sentinel(self) -> None:
        """Round 18: called right before ``subprocess.run(['littrace',
        'sentinel', 'run', ...])`` fires. Tears down the embedded
        ``QWebEngineView`` so its Chromium instance releases the profile
        lock on ``data/chrome-cdp`` and flushes cookies to disk; then
        spawns the external ``chrome.exe`` against the same profile so
        sentinel can drive it via CDP.

        Idempotent: if the external chrome is already alive, the call is
        a no-op. The sentinel subprocess inherits any log-in the user did
        in the embedded view because both targets the same on-disk
        Cookies SQLite.

        Round 26: don't tear down QtWebEngine if the publisher-login
        Chrome (``QtCDPBrowser``-managed) is still alive on CDP 19222.
        Sentinel can drive that same Chrome for PDF downloads — keeping
        the QtWebEngine view intact avoids a reproducible QtWebEngine
        native abort (``Must be called on Chrome_UIThread``) that
        fires when the view's signal handlers disconnect races with
        Chromium's own BrowserThread bookkeeping during shutdown.
        """
        # Already up? nothing to do.
        proc = getattr(self, "_external_chrome_proc", None)
        if proc is not None and proc.poll() is None:
            return
        # If the user opened a publisher Chrome via QtCDPBrowser
        # (still alive, talking to CDP 19222), sentinel will reuse
        # it via the same already_available=True branch in
        # ``launch_chrome_for_cdp``. Keep QtWebEngine running so we
        # don't trip the QtWebEngine BrowserThread crash documented
        # in 2026-09. The cookie state is fine to leave alone —
        # both processes point at the same on-disk SQLite file.
        qt_cdp = getattr(self, "_qt_cdp_browser", None)
        if qt_cdp is not None and getattr(qt_cdp, "_connected", False):
            self._post_status(
                "publisher Chrome 已在跑，sentinel 复用（不重启 QtWebEngine）"
            )
            return
        # No publisher Chrome alive: need a fresh external Chrome, which
        # requires the QtWebEngine view to release the profile lock
        # first. ``_auto_launch_chrome`` will spawn the headless
        # companion against the same profile.
        self._browser_panel.suspend_for_external_chrome()
        self._external_chrome_proc = None
        self._auto_launch_chrome()

    def _release_external_chrome_for_sentinel(self) -> None:
        """Round 18: called once sentinel finishes. Kill the external
        chrome so the embedded QtWebEngine view can grab the profile
        lock again; then re-show the BrowserPanel so the user lands
        back on the publisher page they were on before.

        Round 26: also tear down any QtCDPBrowser / CF-dialog state
        left over from a prior publisher-login attempt. Sentinel
        ``terminate()``s the external chrome — if that chrome was
        the same one our ``QtCDPBrowser`` was talking to, the
        ``QWebSocket`` is now in a half-closed state and any timer
        still firing against it would crash Qt on the next callback.
        Cleaning these up before resume keeps the QtWebEngine
        restart safe.
        """
        # 1. Stop any login-polling timer that might still be firing.
        for attr in ("_login_poll_timer", "_login_heartbeat_timer"):
            t = getattr(self, attr, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
            setattr(self, attr, None)
        # 2. Close the QtCDPBrowser instance + its QWebSocket so the
        # GC doesn't trip on a dangling reference mid-resume.
        qt_cdp = getattr(self, "_qt_cdp_browser", None)
        if qt_cdp is not None:
            try:
                qt_cdp.close()
            except Exception:
                pass
            self._qt_cdp_browser = None
        # 3. Drop the synchronous CDPBrowser handle too — its websocket
        # may be in an open state but its process is dead.
        sync_cdp = getattr(self, "_littrace_login_browser", None)
        if sync_cdp is not None:
            self._littrace_login_browser = None
        # 4. Dismiss any lingering CF challenge popup from before resume.
        cf_dlg = getattr(self, "_cf_dialog", None)
        if cf_dlg is not None:
            try:
                cf_dlg.close()
                cf_dlg.deleteLater()
            except Exception:
                pass
            self._cf_dialog = None
        # 5. Terminate the external chrome.
        proc = getattr(self, "_external_chrome_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
            except Exception:
                pass
        self._external_chrome_proc = None
        # 6. Resume the embedded QtWebEngine view (lands on HOME_URL;
        # see ``resume_after_external_chrome`` round-26 note).
        self._browser_panel.resume_after_external_chrome()

    def _cf_data_dir(self) -> "Path":
        # Round 28: match the absolute-path derivation in
        # ``cdp_downloader._cf_data_dir()`` so GUI and sentinel
        # subprocess both write to the same file regardless of cwd.
        from littrace.access_layer.cdp_downloader import _cf_data_dir
        return _cf_data_dir()

    def _poll_cf_wait_flag(self) -> None:
        """Round 27 + 28: 2-second poll for
        ``data/sentinel_cf_wait.json``. When present and we don't
        already have a dialog up for it, pop a modal pointing the
        user at the publisher URL.

        Round 28: the file's schema is now
        ``{"dois": [{doi, url, publisher, added_at}, ...],
         "oldest_added_at": <float>}`` so 45 gated papers surface
        all at once in one dialog rather than bouncing through
        45 separate modal pops. The dialog key is the *set* of
        DOIs currently waiting, not a single DOI — so a sentinel
        run that adds paper N+1 to the list updates the existing
        dialog in place rather than opening a new one.

        Stale flag handling: if the oldest entry is older than
        1 hour, assume the sentinel crashed mid-run and ignore
        the file. The next sentinel run will write a fresh flag.
        """
        import json
        wait_path = self._cf_data_dir() / "sentinel_cf_wait.json"
        ack_path = self._cf_data_dir() / "sentinel_cf_ack.json"
        if not wait_path.exists():
            return
        try:
            payload = json.loads(wait_path.read_text(encoding="utf-8"))
        except Exception:
            return
        # Stale check — a sentinel that SIGKILL'd mid-wait can leave
        # the flag forever; don't keep popping a modal for a run the
        # user already forgot about.
        oldest = payload.get("oldest_added_at")
        if isinstance(oldest, (int, float)) and (time.time() - oldest) > 3600:
            return
        # Normalize schema: accept both new ({"dois": [...]}) and old
        # ({"doi": ..., "started_at": ...}) formats.
        entries: list = payload.get("dois")
        if not entries:
            old_doi = payload.get("doi")
            if old_doi:
                entries = [{
                    "doi": old_doi,
                    "url": payload.get("url", "?"),
                    "publisher": payload.get("publisher", "?"),
                    "added_at": payload.get("started_at", 0.0),
                }]
            else:
                return
        # Build a dedup key — a sentinel run that pushes the same
        # DOI a second time is no-op for us.
        key = tuple(sorted(
            (e.get("doi") or "?") for e in entries
        ))
        if self._cf_wait_dialog is not None and self._cf_wait_last_key == key:
            return
        self._cf_wait_last_key = key
        self._open_cf_wait_dialog(entries=entries, ack_path=ack_path)

    def _open_cf_wait_dialog(
        self, *, entries: list, ack_path: "Path",
    ) -> None:
        """Modal listing all DOIs currently blocked on CF — the user
        clears the challenge once, clicks the button, and all 45
        gated papers can download in sequence."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Cloudflare 验证等待中")
        dlg.setModal(True)
        dlg.resize(560, 380)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        heading = QtWidgets.QLabel("⚠️ sentinel 卡在 Cloudflare 验证")
        heading.setStyleSheet(
            f"font-size:14px;font-weight:600;color:{DESIGN['ink']};"
        )
        layout.addWidget(heading)
        # List each gated paper blocked on CF. Cap the visible
        # list at 8 to avoid a 45-line dialog when the user has a
        # huge batch queued; show "...(N more)" for the rest.
        sample = entries[:8]
        more = max(0, len(entries) - len(sample))
        lines: list[str] = []
        for e in sample:
            doi = e.get("doi", "?")
            pub = e.get("publisher", "?")
            lines.append(f"  • {pub:>10s}  {doi}")
        if more:
            lines.append(f"  …还有 {more} 篇")
        body = "sentinel 正在等你完成 Cloudflare 的人机验证。\n"
        body += "请到 LitTrace 启动的 Chrome 窗口里点完验证（约 5 秒），\n"
        body += "然后点下面的「我处理好了」按钮。sentinel 会继续下剩余文献。\n\n"
        body += f"等待中（{len(entries)} 篇）：\n" + "\n".join(lines)
        info = QtWidgets.QLabel(body)
        info.setWordWrap(True)
        info.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        info.setStyleSheet(
            f"font-family:Menlo,Consolas,monospace;font-size:11px;"
            f"color:{DESIGN['ink']};line-height:1.5;"
        )
        layout.addWidget(info)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)

        def _on_done() -> None:
            # Write the ack file — sentinel's
            # ``_wait_for_user_acknowledgement`` polls for this and
            # returns True as soon as it appears, no GUI round-trip
            # needed. We use ``Path.touch()`` rather than write
            # anything substantial so the sentinel doesn't need to
            # parse JSON.
            try:
                ack_path.parent.mkdir(parents=True, exist_ok=True)
                ack_path.touch()
            except Exception:
                pass
            dlg.accept()

        done_btn = QtWidgets.QPushButton("✅ 我处理好了")
        done_btn.setObjectName("cf_ack_btn")
        done_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        done_btn.clicked.connect(_on_done)
        button_row.addWidget(done_btn)
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.setObjectName("cf_close_btn")
        close_btn.clicked.connect(dlg.reject)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        # Replace any prior dialog (e.g. the sentinel just hit a
        # second gated paper and re-wrote the wait file while the
        # user was looking at the first one).
        previous = getattr(self, "_cf_wait_dialog", None)
        if previous is not None:
            try:
                previous.close()
                previous.deleteLater()
            except Exception:
                pass
        self._cf_wait_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_stop_daily_clicked(self) -> None:
        """Round 20: kill switch for the daily sentinel run. Sets the
        cancel event so the worker's wait-loop notices and breaks out
        of the current round (instead of waiting out
        ``PER_ROUND_TIMEOUT``); also terminate()s the live subprocess
        so the user doesn't see the next round start while the stop
        button is "processing". The actual chrome teardown happens
        in the worker's ``finally`` block — this method only flips
        the flag and nudges the subprocess.
        """
        if not self._daily_cancel_event.is_set():
            self._daily_cancel_event.set()
            self._status_bar.showMessage("正在停止 daily 检索…", 0)
            # Disable the button so a second click doesn't try to
            # re-terminate an already-terminated process (harmless
            # but produces a noisy traceback on Windows if the
            # subprocess already exited).
            self._daily_stop_btn.setEnabled(False)
            proc = getattr(self, "_sentinel_proc", None)
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    # Process already gone — the worker's wait loop
                    # will see ``returncode`` on the next poll and
                    # exit cleanly on its own.
                    pass

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

    @QtCore.Slot("QString")
    def _show_daily_preview_from_any_thread(self, payload_json: str) -> None:
        """Slot target for ``_post_daily_preview`` — renders the
        daily-run summary as a system bubble in the chat scrollback
        so the user sees it inline with their conversation, not in a
        separate floating dialog that the rest of the UI ignores.

        Round 29: dropped the floating ``DailyResultDialog`` (it
        was a stale modality — the user already chats with the
        assistant in the main panel and a separate dialog keeps
        the result disconnected from the rest of the workflow).
        Now we render an HTML bubble into the chat panel and stash
        the structured payload on ``self._last_daily_payload`` so
        the context panel's "📚 打开上下文" button can still jump to
        the new papers.

        JSON-encoded because ``Q_ARG`` only carries primitives +
        ``QString``.
        """
        import json
        try:
            payload = json.loads(payload_json)
        except Exception:
            return
        topic = payload.get("topic", "")
        keywords = payload.get("keywords", "")
        year_min = int(payload.get("year_min") or 0)
        year_max = int(payload.get("year_max") or 0)
        target_papers = int(payload.get("target_papers") or 0)
        rounds_done = int(payload.get("rounds_done") or 0)
        cumulative_downloaded = int(payload.get("cumulative_downloaded") or 0)
        cumulative_candidates = int(payload.get("cumulative_candidates") or 0)
        new_candidates = int(payload.get("new_candidates") or 0)
        seen_candidates = int(payload.get("seen_candidates") or 0)
        warnings = list(payload.get("warnings") or [])
        summary_lines = list(payload.get("summary_lines") or [])
        # Cache so the context panel can still highlight the topic.
        self._last_daily_payload = dict(payload)
        # Build the bubble. ``summary_lines`` carries the parsed
        # per-round counter lines so the user can scroll through
        # them at their own pace; the headline mirrors the same
        # numbers the status strip shows.
        lines: list[str] = []
        lines.append("📚 <b>daily 检索完成</b>")
        lines.append(f"&nbsp;&nbsp;主题：<b>{_render_message_html(topic or '—')}</b>"
                     f"&nbsp;·&nbsp;{year_min}–{year_max}"
                     f"&nbsp;·&nbsp;目标 ≥ {target_papers} 篇")
        if keywords:
            lines.append(
                f"&nbsp;&nbsp;关键词：{_render_message_html(keywords)}"
            )
        lines.append(
            f"&nbsp;&nbsp;完成 {rounds_done} 轮"
            f"&nbsp;·&nbsp;<b style='color:#2a7a3a'>新增下载 "
            f"{cumulative_downloaded}</b> 篇"
            f"&nbsp;·&nbsp;候选 {cumulative_candidates} 篇"
            f"（新 {new_candidates} · 已纳入上下文 {seen_candidates}）"
        )
        if summary_lines:
            lines.append("&nbsp;&nbsp;<b>每轮明细</b>")
            for sline in summary_lines:
                lines.append(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;{_render_message_html(sline)}"
                )
        # Shortfall warnings (mirrors the old dialog's
        # ``累计下载 X 篇，少于目标 Y 篇`` / ``仅检索到 X 个候选``
        # hints). Color-coded so the user notices immediately.
        if cumulative_downloaded < target_papers and target_papers > 0:
            lines.append(
                "<br>&nbsp;&nbsp;⚠️ "
                f"<span style='color:#cc785c'>累计下载 "
                f"{cumulative_downloaded} 篇，少于目标 {target_papers} 篇。"
                "可以再跑一轮或放宽关键词。</span>"
            )
        if cumulative_candidates < 5:
            if seen_candidates > 0 and new_candidates == 0:
                lines.append(
                    "<br>&nbsp;&nbsp;ℹ️ "
                    "<span style='color:#5c6068'>本次检索到 "
                    f"{cumulative_candidates} 个新候选，另有 "
                    f"{seen_candidates} 篇已在之前的检索中纳入上下文。"
                    "如需新文献请修改主题、年份或关键词再试。</span>"
                )
            else:
                lines.append(
                    "<br>&nbsp;&nbsp;⚠️ "
                    f"<span style='color:#cc785c'>仅检索到 "
                    f"{cumulative_candidates} 个候选，主题可能过于冷门或过新。"
                    "</span>"
                )
        if warnings:
            warn_text = _render_message_html("；".join(warnings))
            lines.append(
                "<br>&nbsp;&nbsp;⚠️ "
                f"<span style='color:#cc785c'>{warn_text}</span>"
            )
        html = "<br>".join(lines)
        if hasattr(self, "_chat_panel") and self._chat_panel is not None:
            self._chat_panel.append_message("system", html)

    def _post_daily_preview(self, payload: dict) -> None:
        """Worker-thread helper. JSON-encodes ``payload`` and routes
        it onto the GUI thread so the user sees the structured
        ``DailyResultDialog`` instead of just the headline status
        line."""
        import json
        try:
            encoded = json.dumps(payload, ensure_ascii=False)
        except Exception:
            encoded = "{}"
        QtCore.QMetaObject.invokeMethod(
            self,
            "_show_daily_preview_from_any_thread",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, encoded),
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

        # Round 20: persistent "停止 daily 检索" button in the status
        # bar. Hidden by default; shown for the duration of a daily
        # run so the user can interrupt a stuck/wrong-topic run
        # without having to wait ``MAX_ROUNDS * PER_ROUND_TIMEOUT``
        # for the worker to time out. The cancel button on the
        # ``DailyConfigDialog`` itself only fires before the worker
        # starts (the dialog closes on accept), so a separate
        # mid-run kill switch is needed.
        self._daily_stop_btn = QtWidgets.QToolButton()
        self._daily_stop_btn.setText("⏹ 停止 daily 检索")
        self._daily_stop_btn.setObjectName("daily_stop_btn")
        self._daily_stop_btn.setToolTip(
            "立刻终止正在跑的 sentinel 子进程与外部 chrome，"
            "已经下载的文件保留在 workspace"
        )
        self._daily_stop_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._daily_stop_btn.setVisible(False)
        self._daily_stop_btn.clicked.connect(self._on_stop_daily_clicked)
        self._status_bar.addPermanentWidget(self._daily_stop_btn)
        # Round 20: cancellation primitives. ``_daily_cancel_event`` is
        # set by the GUI thread when the user clicks the stop button;
        # the worker thread checks it between rounds and breaks out
        # early. ``_sentinel_proc`` is the live ``Popen`` handle so the
        # GUI thread can ``terminate()`` the subprocess directly — the
        # wait-with-timeout loop in the worker also polls the event so
        # we don't have to wait out the full ``PER_ROUND_TIMEOUT`` on a
        # cancel.
        import threading
        self._daily_cancel_event = threading.Event()
        self._sentinel_proc: subprocess.Popen | None = None

        # Round 19: 60 s tick to keep the RAG staleness badge fresh.
        # Without this the user sees the badge flip from "fresh" to
        # "stale" only on the next explicit refresh — which can be
        # many minutes apart.
        self._rag_tick_timer = QtCore.QTimer(self)
        self._rag_tick_timer.setInterval(60_000)
        self._rag_tick_timer.timeout.connect(
            lambda: getattr(self, "_rag_panel", None)
            and self._rag_panel.refresh_status()
        )
        self._rag_tick_timer.start()

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
        # Round 21: route BrowserPanel navigation events into the main
        # status bar so the user can tell whether a publisher shortcut
        # click or a URL-bar Enter actually fired. Without this the
        # embedded view swaps its content silently and the user has no
        # way to distinguish "loading…" from "nothing happened".
        self._browser_panel.navigation_message.connect(
            lambda msg: self._status_bar.showMessage(msg, 4000)
        )
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
        # Round 19: ``DailyConfigDialog`` is now non-modal — show it
        # instead of exec'ing, and route the accepted / rejected
        # signals to handler slots. This lets the user keep poking
        # at the rest of the window (context panel, chat scrollback,
        # etc.) while they tweak the parameters, and the X button /
        # Esc key still cleanly cancels the run.
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
        # Wire one-shot handlers so the dialog doesn't have to know
        # about the controller. ``accepted`` carries the chosen
        # values; ``rejected`` is a no-op (user closed the dialog
        # without running anything).
        dialog.accepted.connect(
            lambda dlg=dialog: self._on_daily_dialog_accepted(dlg)
        )
        dialog.rejected.connect(
            lambda: self._status_bar.showMessage("已取消 daily 检索", 3000)
        )
        # Cache on the window so the user can close + reopen via
        # the toolbar button without losing their input. A second
        # open reuses the existing dialog and just ``raise_()``s it.
        previous = getattr(self, "_daily_config_dialog", None)
        if previous is not None:
            try:
                previous.close()
                previous.deleteLater()
            except Exception:
                pass
        self._daily_config_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_daily_dialog_accepted(self, dialog) -> None:
        """Round 19: handler for the non-modal ``DailyConfigDialog``'s
        ``accepted`` signal. Reads the chosen values, kicks off the
        sentinel worker (or the publisher-login flow), and lets the
        dialog close itself."""
        topic = dialog.topic()
        keywords = dialog.keywords()
        year_min = dialog.year_min()
        year_max = dialog.year_max()
        min_papers = dialog.min_papers()
        open_login = dialog.open_login_after()

        if open_login:
            self._launch_publisher_login_browser()
            return

        self._start_daily_run(
            topic=topic,
            keywords=keywords,
            year_min=year_min,
            year_max=year_max,
            min_papers=min_papers,
        )

    def _start_daily_run(
        self,
        *,
        topic: str,
        keywords: str,
        year_min: int,
        year_max: int,
        min_papers: int,
    ) -> None:
        """Round 19: the actual sentinel-subprocess kickoff. Extracted
        from ``_on_run_daily`` so both the dialog-accepted path and
        any future entrypoint (e.g. a slash command like
        ``/run-daily mxene_sensor``) can share it. ``_on_run_daily``
        handles the dialog lifecycle; this method assumes the
        parameters are already validated and only does the launch.
        """
        self._status_bar.showMessage(
            f"检索启动中…主题：{topic}（{year_min}-{year_max}，≥{min_papers} 篇）",
            3000,
        )
        self._rag_panel.set_status(f"运行中…主题 {topic}")
        # Round 20: reset the cancellation primitive so a previous run's
        # leftover "cancelled" flag doesn't immediately abort this run,
        # then show the stop button so the user knows they have a kill
        # switch for this run.
        self._daily_cancel_event.clear()
        self._daily_stop_btn.setEnabled(True)
        self._daily_stop_btn.setVisible(True)
        # Round 18: take the BrowserPanel offline and bring up the
        # external chrome so sentinel can drive it via CDP. Both
        # processes must NOT be alive simultaneously (SingletonLock on
        # ``data/chrome-cdp``); ``_acquire`` suspends the embedded
        # view first, then spawns the external one.
        self._acquire_external_chrome_for_sentinel()
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
            #
            # Round 26: a single broad try/except wraps the whole
            # worker. Previously an exception in any of the helper
            # methods (``_post_daily_preview``, ``_start_progress_timer``,
            # ``_release_external_chrome_for_sentinel``, the parser
            # that builds ``last_summary``, …) would propagate up to
            # the worker thread's unhandled-exception handler and tear
            # down the Qt main process. Catch everything, surface it
            # in the status bar, and keep going so the user at least
            # sees what went wrong instead of losing the window.
            worker_error: list[str] = []
            try:
                MAX_ROUNDS = 2
                # Round 22: 600s (was 180). The previous value was a
                # frequent source of "dialog showed 0/0/0" — sentinel
                # was SIGTERM'd mid-run (e.g. on a slow OpenAlex cache
                # or while fetching 10 PDFs), the buffered counters
                # never flushed, and the user's daily run came back
                # looking empty. The flush=True we added in cli.py is
                # belt-and-braces; the longer timeout is the actual
                # fix for users who hit the timeout wall.
                PER_ROUND_TIMEOUT = 600
                base_keywords = keywords or topic
                queries = [
                    base_keywords,
                    f"{base_keywords} review",
                ]
                cumulative_downloaded = 0
                cumulative_candidates = 0
                # Round 28: track the new-vs-seen split so the dialog
                # can show the "0 new — all already in seen" dedup hint.
                cumulative_new_candidates = 0
                cumulative_seen_candidates = 0
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
                        # Round 24: tell sentinel to write paper
                        # metadata, RAG chunks, and artifacts into
                        # the user's *current* chat session instead
                        # of a separate ``sentinel:<watchlist>``
                        # session. Without this, the GUI's main
                        # workspace stays empty and "打开上下文"
                        # shows nothing.
                        "--main-session-id",
                        self._controller.session.session_id,
                    )
                    # Round 20: Popen instead of subprocess.run so the
                    # GUI thread can ``terminate()`` the subprocess when
                    # the user clicks the stop button. The wait loop
                    # polls the cancellation event every second so we
                    # don't sit on the full ``PER_ROUND_TIMEOUT`` after
                    # a cancel. ``self._sentinel_proc`` is read by the
                    # GUI thread (only ``.poll()`` + ``.terminate()``)
                    # — both calls are documented thread-safe on
                    # ``Popen``, no lock required.
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=cwd,
                        env=env,
                    )
                    self._sentinel_proc = proc
                    try:
                        completed: subprocess.CompletedProcess | None = None
                        deadline = time.monotonic() + PER_ROUND_TIMEOUT
                        while True:
                            # ``Popen.wait(timeout=...)`` returns None
                            # while alive and raises ``TimeoutExpired``
                            # on timeout — we want neither; we want a
                            # loop that also bails on the cancel event.
                            try:
                                rc = proc.wait(timeout=1.0)
                            except subprocess.TimeoutExpired:
                                rc = None
                            if rc is not None:
                                stdout = proc.stdout.read() if proc.stdout else ""
                                stderr = proc.stderr.read() if proc.stderr else ""
                                completed = subprocess.CompletedProcess(
                                    args=cmd,
                                    returncode=rc,
                                    stdout=stdout,
                                    stderr=stderr,
                                )
                                break
                            if self._daily_cancel_event.is_set():
                                proc.terminate()
                                try:
                                    proc.wait(timeout=5.0)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                stdout = proc.stdout.read() if proc.stdout else ""
                                stderr = proc.stderr.read() if proc.stderr else ""
                                completed = subprocess.CompletedProcess(
                                    args=cmd,
                                    returncode=-1,
                                    stdout=stdout,
                                    stderr=stderr or "(user cancelled)",
                                )
                                break
                            if time.monotonic() > deadline:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=5.0)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                stdout = proc.stdout.read() if proc.stdout else ""
                                stderr = proc.stderr.read() if proc.stderr else ""
                                completed = subprocess.CompletedProcess(
                                    args=cmd,
                                    returncode=-1,
                                    stdout=stdout,
                                    stderr=stderr or "(round timeout)",
                                )
                                break
                    finally:
                        self._sentinel_proc = None
                        progress_state["stop"] = True
                    if self._daily_cancel_event.is_set():
                        # User cancelled — break out of the round loop
                        # entirely instead of trying the next round.
                        self._post_status(f"已在第 {round_idx} 轮后停止 daily 检索")
                        break
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
                    new_candidates = _summary_value(summary, "new_candidates:")
                    total_candidates = _summary_value(summary, "candidates_total:")
                    seen_candidates = _summary_value(summary, "candidates_seen:")
                    if downloaded is not None:
                        cumulative_downloaded += downloaded
                    # Round 23: prefer ``candidates_total`` (every
                    # paper this run surfaced) over ``new_candidates``
                    # (only previously-unseen). If the CLI predates
                    # Round 23, ``candidates_total`` will be None and
                    # we fall back to ``new_candidates`` for backward
                    # compat with older sentinel subprocess output.
                    if total_candidates is not None:
                        cumulative_candidates += total_candidates
                    elif new_candidates is not None:
                        cumulative_candidates += new_candidates
                    # Round 28: same split for the dialog — track
                    # the new-vs-seen tally so the user can tell
                    # "found 0 new (all already seen)" from "0
                    # found at all".
                    if new_candidates is not None:
                        cumulative_new_candidates += new_candidates
                    if seen_candidates is not None:
                        cumulative_seen_candidates += seen_candidates
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
                # Round 19: pop the non-modal daily-result dialog so
                # the user sees a structured summary instead of
                # trying to parse a 200-character status-bar line.
                # The dialog is informational only — by this point
                # sentinel has already written to the workspace;
                # ``mark_rag_refresh`` below keeps the staleness
                # badge in sync with the same data the dialog
                # surfaces. ``summary_lines`` carries the parsed
                # per-round counter lines so the user can scroll
                # through them at their own pace.
                self._post_daily_preview(
                    {
                        "topic": topic,
                        "keywords": keywords,
                        "year_min": year_min,
                        "year_max": year_max,
                        "target_papers": min_papers,
                        "rounds_done": rounds_done,
                        "cumulative_downloaded": cumulative_downloaded,
                        "cumulative_candidates": cumulative_candidates,
                        # Round 28: pass the new-vs-seen split so
                        # the dialog can disambiguate "this run
                        # found 0 NEW papers" (every paper was
                        # already in seen_paper_ids from a prior
                        # run) from "0 papers found at all". The
                        # dedup UX hint lives in
                        # ``DailyResultDialog.__init__``.
                        "new_candidates": cumulative_new_candidates,
                        "seen_candidates": cumulative_seen_candidates,
                        "warnings": list(last_warnings),
                        "summary_lines": [
                            f"{i + 1}. {line}"
                            for i, line in enumerate(last_summary.split(" · "))
                            if line
                        ],
                    }
                )
                # Round 19: stamp the controller's RAG refresh
                # timestamp so the staleness badge updates from
                # "stale" to "fresh" without the user having to
                # restart the window. ``cumulative_candidates`` is
                # the closest stand-in for chunk count the
                # sentinel summary gives us; real chunk counts
                # arrive via the dedicated rag_refresh tool which
                # is run elsewhere in the pipeline.
                self._controller.mark_rag_refresh(
                    indexed_chunks=cumulative_candidates
                )
            except Exception as exc:
                # Round 26: a broad catch around the whole sentinel
                # driver. Before this, any uncaught exception in
                # ``_post_daily_preview`` / the summary parser / the
                # progress-timer path propagated up the worker thread
                # and tore down the Qt main process — the user saw
                # the window vanish mid-retrieval with no recovery
                # path. Now we surface the error in the status bar
                # and keep the window alive so the user can read the
                # message and try again.
                worker_error.append(
                    f"{exc.__class__.__name__}: {exc}"
                )
                logger.exception(
                    "daily_run_worker_crash",
                    extra={"topic": topic, "year_min": year_min},
                )
                self._post_status(
                    f"⚠️ 检索异常中断：{worker_error[-1]}"
                )
            finally:
                # Round 18: even on exception, kill the external chrome
                # and re-show the embedded view. Without this, a
                # crashed sentinel run leaves the BrowserPanel blank
                # and the user has to restart ``littrace-qt``.
                try:
                    self._release_external_chrome_for_sentinel()
                except Exception as inner_exc:
                    logger.exception(
                        "release_external_chrome_failed",
                        extra={"error": str(inner_exc)},
                    )
                # Round 20: hide the stop button once the run is done
                # (whether it finished, errored out, or was cancelled).
                # The button would otherwise stick around on the status
                # bar after every daily run and confuse the user.
                try:
                    self._daily_stop_btn.setVisible(False)
                except Exception:
                    pass
                # Defensive: if the worker exits with the subprocess
                # handle still on the window (shouldn't happen — the
                # ``finally`` above clears it — but worth guarding
                # against a programmer error that would leak the
                # Popen handle).
                self._sentinel_proc = None
            # If we hit an error mid-run, surface a dialog so the
            # user gets the same structured "what happened" view as
            # a successful run — without this the status-bar message
            # is easy to miss.
            if worker_error:
                self._post_daily_preview(
                    {
                        "topic": topic,
                        "keywords": keywords,
                        "year_min": year_min,
                        "year_max": year_max,
                        "target_papers": min_papers,
                        "rounds_done": 0,
                        "cumulative_downloaded": 0,
                        "cumulative_candidates": 0,
                        "warnings": worker_error,
                        "summary_lines": [
                            f"检索中断：{worker_error[0]}",
                            "LitTrace 窗口已恢复，可重新发起检索。",
                        ],
                    }
                )

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
        # Round 18: login now happens inside the embedded BrowserPanel
        # because ``main()`` redirected QtWebEngine's default profile to
        # ``data/chrome-cdp`` — the same directory sentinel reads via
        # CDP. There is no longer a separate external chrome to launch;
        # pointing the embedded view at the first publisher's sign-in
        # page is enough. The user can then click the other shortcuts
        # (``🌐 ACS`` / ``🌐 Springer`` / ...) on the same view to log
        # into the rest.
        self._browser_panel.open_url(self._browser_panel.PUBLISHER_LINKS[0][1])
        self._status_bar.showMessage(
            "在 LitTrace Chrome 里登录各 publisher 即可，cookie 自动写入 data/chrome-cdp",
            5000,
        )

    def _open_url_in_littrace_chrome(self, url: str) -> None:
        """Round 29: route a publisher sign-in URL through LitTrace's
        OWN Chrome (the one launched by ``_acquire_external_chrome_for_sentinel``
        against ``data/chrome-cdp``) instead of the user's system
        browser. Cookies written by this Chrome land in
        ``data/chrome-cdp`` so the next sentinel gated-PDF fetch
        can reuse them; cookies in the user's system Chrome were
        silently invisible to sentinel (different profile, different
        Keychain on macOS).

        Implementation: reuse the existing ``_external_chrome_proc``
        / ``launch_chrome_for_cdp`` machinery to make sure a
        LitTrace Chrome is alive on port 19222, then drive it via
        the synchronous ``CDPBrowser`` (the same client sentinel
        uses). We don't depend on QtCDPBrowser because the URL
        bar + publisher-shortcut clicks happen on the main thread
        and we don't need the WebSocket-event plumbing for a
        single ``Page.navigate`` call. The CDPBrowser is created
        on a worker thread to avoid blocking the Qt event loop.
        """
        # 1. Make sure a LitTrace Chrome is alive on CDP 19222.
        # ``_acquire_external_chrome_for_sentinel`` already handles
        # this (it also refuses to use a stale QtCDPBrowser pointed
        # at the wrong profile, which is the bug we are fixing).
        self._acquire_external_chrome_for_sentinel()
        # 2. Sanity check: did Chrome actually come up?
        from littrace.chrome_profiles import check_cdp_status

        status = check_cdp_status(self._controller.config)
        if not status.available or not status.web_socket_debugger_url:
            self._post_status(
                f"⚠️ LitTrace Chrome 未在 19222 上响应，无法在隔离 profile 中打开 {url}"
            )
            return
        # 3. Drive the navigate on a worker thread so we never
        # block the Qt event loop. Reuse a single CDPBrowser
        # across calls so the same tab persists (no new about:blank
        # on every click).
        import threading
        from littrace.access_layer.cdp_core import CDPBrowser

        def _worker():
            try:
                browser = getattr(self, "_littrace_login_browser", None)
                if browser is None:
                    browser = CDPBrowser(
                        status.web_socket_debugger_url,
                        reconnect_attempts=2,
                        command_timeout_seconds=15.0,
                    )
                    browser.connect_new_tab()
                    self._littrace_login_browser = browser
                browser.navigate(url, wait_seconds=8.0)
            except Exception as exc:
                self._post_status(
                    f"⚠️ LitTrace Chrome 导航失败：{exc.__class__.__name__}: {exc}"
                )

        threading.Thread(target=_worker, daemon=True, name="littrace-login-nav").start()

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
                self._controller.EVENT_SLASH_RESULT: self._on_slash_result_event,
                self._controller.EVENT_RAG_PANEL_REFRESHED: self._on_rag_panel_event,
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
        # Round 19: stash the most recent user message so the
        # error bubble's "🔁 重试" link can re-submit it without
        # making the user scroll up to copy/paste. Empty / slash
        # commands are skipped (they don't make sense to retry
        # through the same handler that just failed).
        if role == "user" and text and not text.startswith("/"):
            self._last_user_message = text
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

    def _on_slash_result_event(self, body: dict) -> None:
        """Round 19: render a slash command result as a system bubble
        in the chat scrollback. The format mirrors the CLI's plain
        text (no markdown) so the user sees the same text they'd see
        in a terminal.
        """
        name = body.get("name", "")
        text = body.get("text", "")
        label = f"/{name}" if name else "command"
        self._chat_panel.append_message(
            "system",
            f"<b>{label}</b><br><pre style='white-space:pre-wrap;"
            "font-family:Menlo,Consolas,monospace;font-size:12px;'>"
            f"{_render_message_html(text)}</pre>",
        )
        if self._status_bar is not None:
            self._status_bar.showMessage(f"执行 /{name}", 3000)

    def _on_rag_panel_event(self, body: dict) -> None:
        """Round 19: the controller emits this every time the RAG
        panel needs to re-render — after a daily run finishes, when
        ``mark_rag_refresh`` stamps a new timestamp, etc. We just
        forward to ``RAGPanel.refresh_status``.
        """
        self._rag_panel.refresh_status()

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
        #
        # Round 19: inline action buttons under the error so the
        # user doesn't have to scroll up, retype the message, or
        # open the controller log to act:
        #   * [🔁 重试]   re-submits the last user message
        #   * [查看技术细节]   pops the raw stack trace
        #   * For ``unauthorized`` errors, also surface [🔑 重新登录]
        #     so the user doesn't have to hunt for the login button.
        message = body.get("message", "对话出错")
        suggestion = body.get("suggestion", "")
        error_code = body.get("error_code", "other")
        raw = body.get("raw", "")
        html_parts = [f"⚠️ {message}"]
        if suggestion:
            html_parts.append(
                f'<br><span style="color:#5c6068;font-size:11px;">'
                f"{suggestion}</span>"
            )
        # Build the inline action row. Each link uses the
        # ``littrace:`` URI scheme so we can route the activation
        # through ``_on_chat_anchor_clicked`` without opening a
        # browser.
        actions: list[str] = []
        last_msg = getattr(self, "_last_user_message", None)
        if last_msg:
            actions.append(
                '<a href="littrace:retry-last" '
                'style="color:#3a8a8c;font-size:11px;'
                'text-decoration:none;margin-right:10px;">'
                '🔁 重试</a>'
            )
        if raw:
            actions.append(
                '<a href="littrace:show-error-detail" '
                'style="color:#3a8a8c;font-size:11px;'
                'text-decoration:none;margin-right:10px;">'
                '查看技术细节</a>'
            )
            # Stash for the click handler.
            self._last_error_detail = (error_code, message, raw)
        if error_code == "unauthorized":
            actions.append(
                '<a href="littrace:relogin" '
                'style="color:#3a8a8c;font-size:11px;'
                'text-decoration:none;margin-right:10px;">'
                '🔑 重新登录</a>'
            )
        if actions:
            html_parts.append(
                '<br><div style="margin-top:6px;font-size:11px;">'
                + "".join(actions)
                + "</div>"
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
        # Round 19: inline retry action on chat errors. Re-submits
        # the most recent user message (tracked in
        # ``_on_message_event``) so the user doesn't have to scroll
        # up + copy/paste after a transient failure.
        if text.startswith("littrace:retry-last"):
            last = getattr(self, "_last_user_message", None)
            if not last:
                self._status_bar.showMessage("没有可重试的上一条消息", 4000)
                return
            self._controller.submit_user_message(last)
            return
        # Round 19: one-click re-login when the error is an
        # ``unauthorized``. Reuses the same login dialog the user
        # sees on cold-start.
        if text.startswith("littrace:relogin"):
            self._show_auth_dialog("reauth", "请重新登录 ChatGPT 后重试")
            return
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

    # Round 18: redirect QtWebEngine's default profile storage to the
    # same ``data/chrome-cdp`` directory sentinel uses via CDP, so
    # cookies written by the embedded BrowserPanel (when the user logs
    # into Wiley / ACS / Springer / Nature) are immediately visible to
    # the next ``sentinel run`` without a separate "open external
    # browser and log in again" step. The env-var
    # ``QTWEBENGINE_USER_DATA_DIR`` is documented as the standard knob
    # but is silently ignored on PySide6 6.7.2 / Windows (verified
    # 2026-09: the default profile still resolves to
    # ``%LOCALAPPDATA%\\...\\QtWebEngine\\OffTheRecord``); the API
    # path is the only thing that actually moves cookies.
    from PySide6.QtWebEngineCore import QWebEngineProfile

    profile_dir = config.cdp_downloader.chrome_user_data_dir.expanduser()
    profile_dir.mkdir(parents=True, exist_ok=True)
    default_profile = QWebEngineProfile.defaultProfile()
    default_profile.setPersistentStoragePath(str(profile_dir))
    default_profile.setCachePath(str(profile_dir))

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