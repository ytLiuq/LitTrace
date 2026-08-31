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

        input_row = QtWidgets.QHBoxLayout()
        self._input = QtWidgets.QLineEdit()
        self._input.setObjectName("input")
        self._input.setPlaceholderText("输入研究任务或 / 命令…")
        self._input.returnPressed.connect(self._on_send)
        self._input.textChanged.connect(self._on_input_changed)
        self._input.installEventFilter(self)
        layout.addLayout(input_row)
        input_row.addWidget(self._input, stretch=1)

        self._send = QtWidgets.QPushButton("发送")
        self._send.setObjectName("send")
        self._send.clicked.connect(self._on_send)
        input_row.addWidget(self._send)

        # Slash-command autocomplete popup, lazily shown beneath the
        # input box when the user types a leading "/".
        self._popup = QtWidgets.QListWidget()
        self._popup.setObjectName("command_popup")
        self._popup.setWindowFlags(QtCore.Qt.WindowType.Popup)
        self._popup.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._popup.itemActivated.connect(self._commit_popup_item)
        self._popup.itemClicked.connect(self._commit_popup_item)
        self._popup.hide()
        for name, desc, _ctx in COMMAND_CATALOG:
            item = QtWidgets.QListWidgetItem(f"/{name}    — {desc}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            self._popup.addItem(item)

        self._popup_index = 0

    # ---- Event filter for ↑/↓/Esc -------------------------------------

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if source is self._input and event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()
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

    def _on_input_changed(self, text: str) -> None:
        if not text.startswith("/"):
            self._popup.hide()
            return
        query = text[1:].lower()
        # Filter items whose name (without leading /) starts with query.
        self._popup_index = 0
        self._popup.clear()
        matched = 0
        for name, desc, _ctx in COMMAND_CATALOG:
            if query and not name.lower().startswith(query):
                continue
            item = QtWidgets.QListWidgetItem(f"/{name}    — {desc}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            self._popup.addItem(item)
            matched += 1
        if matched == 0:
            self._popup.hide()
            return
        self._popup.setCurrentRow(0)
        # Position popup beneath the input line edit.
        anchor = self._input.mapToGlobal(QtCore.QPoint(0, self._input.height()))
        self._popup.move(anchor)
        self._popup.setFixedWidth(max(self._input.width(), 360))
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
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._popup.hide()
        if text == "/quit":
            QtWidgets.QApplication.instance().quit()
            return
        self._controller.submit_user_message(text)

    # ---- Event handlers wired by LitTraceQtWindow ----------------------

    def append_message(self, role: str, text: str, **extras: Any) -> None:
        css_role = {
            "user": "color:#0b0c0e;font-weight:600;",
            "assistant": "color:#0b0c0e;",
            "system": "color:#a4a7ad;font-style:italic;",
        }.get(role, "color:#0b0c0e;")
        action = extras.get("action")
        action_html = (
            f'<div style="color:#5c6068;font-size:11px;">action: {action}</div>'
            if action
            else ""
        )
        warnings = extras.get("warnings") or []
        warn_html = (
            "<ul style='color:#cc785c;margin:4px 0;'>"
            + "".join(f"<li>{w}</li>" for w in warnings)
            + "</ul>"
            if warnings
            else ""
        )
        safe = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        html = (
            f'<div style="{css_role}margin:6px 0;">'
            f'<span style="color:#5c6068;font-size:11px;">{role}</span><br>'
            f"{safe}{action_html}{warn_html}</div>"
        )
        self._view.append(html)

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

        run_btn = QtWidgets.QPushButton("运行今日管线")
        run_btn.setObjectName("subnav_btn_primary")
        run_btn.clicked.connect(on_run_daily)
        layout.addWidget(run_btn)

        self._status = QtWidgets.QLabel("未运行")
        self._status.setObjectName("status")
        self._status.setWordWrap(True)
        layout.addWidget(self._status, stretch=1)

    def set_status(self, text: str) -> None:
        self._status.setText(text)


class BrowserPanel(QtWidgets.QFrame):
    """Right column lower — QWebEngineView pane for publisher pages.

    Three back/forward/reload buttons used to sit above the URL bar.
    None of them are worth their weight — the URL bar already accepts any
    input, and when LitTrace opens a publisher page automatically the
    user has no reason to navigate away. Keep just a single URL line so
    the panel stays out of the way.
    """

    HOME_URL = "about:blank"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("browser_tile")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        url_row = QtWidgets.QHBoxLayout()
        url_row.setContentsMargins(8, 8, 8, 4)
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
        text = self._url.text().strip()
        if not text:
            return
        if not text.startswith(("http://", "https://")):
            text = "https://" + text
        self._view.setUrl(QtCore.QUrl(text))

    def _on_url_changed(self, url: QtCore.QUrl) -> None:
        # Don't echo about:blank into the URL bar — leaves a stale-looking
        # field on startup.
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
        self._run_subprocess_action(
            ["littrace", "setup-browser"],
            "Chrome 配置完成",
        )

    def _on_doctor(self) -> None:
        self._run_subprocess_action(["littrace", "doctor"], None)

    def _run_subprocess_action(self, cmd: list[str], success_label: str | None) -> None:
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=20,
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
        # Single Daily run entry point. The button hands off to
        # ``littrace sentinel run --watchlist ...`` in a background thread
        # so the chat thread stays responsive. The status line is updated
        # when the run completes via a controller event.
        self._status_bar.showMessage("今日管线启动中…", 3000)
        self._rag_panel.set_status("运行中…")
        import threading

        def _worker():
            try:
                completed = subprocess.run(
                    [
                        "littrace",
                        "sentinel",
                        "run",
                        "--watchlist",
                        "mxene_sensor",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                msg = (completed.stdout or "").strip().splitlines()
                tail = " · ".join(msg[-3:]) if msg else ""
                self._post_status(f"今日管线完成（exit={completed.returncode}） · {tail}")
            except subprocess.TimeoutExpired:
                self._post_status("今日管线超时（10 分钟）")
            except Exception as exc:  # pragma: no cover - defensive
                self._post_status(f"今日管线失败: {type(exc).__name__}: {exc}")

        threading.Thread(target=_worker, daemon=True, name="littrace-daily").start()

    def _post_status(self, text: str) -> None:
        QtCore.QTimer.singleShot(0, lambda: self._rag_panel.set_status(text))

    # ---- Event wiring ----------------------------------------------------

    def _wire_events(self) -> None:
        controller = self._controller

        # Qt widgets must only be touched from the GUI thread, but the
        # controller emits events on its asyncio worker thread. Post each
        # event back to the main loop with ``QTimer.singleShot(0, ...)`` so
        # ``QTextDocument`` and friends are constructed on the right thread.
        # The headless-smoke subscribers are exceptions — they only print
        # and call ``app.quit`` which is thread-safe.
        def post_to_gui(handler):
            def _wrapper(event: ShellEvent) -> None:
                QtCore.QTimer.singleShot(0, lambda e=event: handler(e))
            return _wrapper

        def on_message(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_MESSAGE_APPENDED:
                self._chat_panel.append_message(
                    event.payload.get("role", "system"),
                    event.payload.get("text", ""),
                    **{
                        k: v
                        for k, v in event.payload.items()
                        if k in ("action", "warnings")
                    },
                )

        def on_status(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_STATUS_CHANGED:
                self._status_bar.showMessage(event.payload.get("text", ""))

        def on_thinking(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_THINKING:
                self._chat_panel.set_thinking(
                    active=bool(event.payload.get("active")),
                    label=str(event.payload.get("label", "思考中")),
                )

        def on_workspace(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_WORKSPACE_REFRESHED:
                self._context_panel.refresh(list(self._controller.list_active_papers()))
                self._trace_panel.render_workflow_trace(
                    ["工作区刷新"]
                    + [
                        f"{i + 1}. {p.title}"
                        for i, p in enumerate(self._controller.list_active_papers())
                    ]
                )

        def on_error(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_ERROR:
                self._chat_panel.append_message(
                    "system", f"⚠️ {event.payload.get('message', 'error')}"
                )

        controller.bus.subscribe(post_to_gui(on_message))
        controller.bus.subscribe(post_to_gui(on_status))
        controller.bus.subscribe(post_to_gui(on_thinking))
        controller.bus.subscribe(post_to_gui(on_workspace))
        controller.bus.subscribe(post_to_gui(on_error))

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