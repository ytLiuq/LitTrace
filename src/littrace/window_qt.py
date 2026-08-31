"""Qt-based shell for LitTrace.

This is the ``littrace-qt`` entry point. It is a deliberately small first
cut that exercises the ``ShellController`` against a Qt + QtWebEngine
front-end so we can answer the "does PySide6 actually run here?" question
before committing to a full Tk-to-Qt port. Layout is intentionally
minimal — a left chat column, a right browser pane, a status bar — but
every panel wires back to ``ShellController`` events so swapping in a
richer design later is local.

Until the Tk shell is retired, ``littrace-window`` keeps using Tk and
``littrace-qt`` runs alongside it on the same LitTrace config and
session backend.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

# QtWebEngineWidgets pulls in Chromium; the env var below silences the
# noisy "Qt WebEngine seems to be initialized from a server" warning
# when running headless for the smoke test.
os.environ.setdefault("QT_LOGGING_RULES", "qt.webengine.*=false")

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWebEngineWidgets import QWebEngineView

from littrace.config import load_config
from littrace.shell_controller import ShellController, ShellEvent


logger = logging.getLogger("littrace.window_qt")


# ---------------------------------------------------------------------------
# Design tokens — kept in sync with the Tk shell's DESIGN dict so the
# two shells look like siblings.
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
    padding: 8px 0;
}}

QLabel#tagline {{
    color: {DESIGN["ink_muted"]};
    font-size: 12px;
}}

QFrame#pane, QFrame#tile, QFrame#chat_tile {{
    background: {DESIGN["surface_1"]};
    border: 1px solid {DESIGN["hairline"]};
    border-radius: 6px;
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

QTextBrowser#chat_view {{
    background: {DESIGN["surface_1"]};
    border: none;
    color: {DESIGN["ink"]};
}}

QLabel#status {{
    color: {DESIGN["ink_subtle"]};
    padding: 4px 8px;
}}

QListWidget#context {{
    background: {DESIGN["surface_1"]};
    border: 1px solid {DESIGN["hairline"]};
    border-radius: 6px;
    color: {DESIGN["ink"]};
}}

QStatusBar {{
    background: {DESIGN["parchment"]};
    color: {DESIGN["ink_muted"]};
}}
"""


# ---------------------------------------------------------------------------
# Panel widgets
# ---------------------------------------------------------------------------


class ChatPanel(QtWidgets.QFrame):
    """Left column: scrollback + input + send button."""

    def __init__(self, controller: ShellController, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chat_tile")
        self._controller = controller

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self._view = QtWidgets.QTextBrowser()
        self._view.setObjectName("chat_view")
        self._view.setOpenExternalLinks(True)
        layout.addWidget(self._view, stretch=1)

        input_row = QtWidgets.QHBoxLayout()
        self._input = QtWidgets.QLineEdit()
        self._input.setObjectName("input")
        self._input.setPlaceholderText("输入研究任务，回车发送…")
        self._input.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input, stretch=1)

        self._send = QtWidgets.QPushButton("发送")
        self._send.setObjectName("send")
        self._send.clicked.connect(self._on_send)
        input_row.addWidget(self._send)
        layout.addLayout(input_row)

    def _on_send(self) -> None:
        text = self._input.text()
        if not text.strip():
            return
        self._input.clear()
        self._controller.submit_user_message(text)

    # ---- Event handlers wired by LitTraceQtWindow ----

    def append_message(self, role: str, text: str, **extras: Any) -> None:
        css_role = {
            "user": "color:#0b0c0e;font-weight:600;",
            "assistant": "color:#0b0c0e;",
            "system": "color:#a4a7ad;font-style:italic;",
        }.get(role, "color:#0b0c0e;")
        action = extras.get("action")
        action_html = f'<div style="color:#5c6068;font-size:11px;">{action}</div>' if action else ""
        warnings = extras.get("warnings") or []
        warn_html = (
            "<ul style='color:#cc785c;margin:4px 0;'>"
            + "".join(f"<li>{w}</li>" for w in warnings)
            + "</ul>"
            if warnings
            else ""
        )
        body = QtGui.QTextDocument().toPlainText(text)
        safe = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Preserve line breaks
        safe = safe.replace("\n", "<br>")
        html = (
            f'<div style="{css_role}margin:6px 0;">'
            f'<span style="color:#5c6068;font-size:11px;">{role}</span><br>'
            f'{safe}{action_html}{warn_html}</div>'
        )
        self._view.append(html)


class ContextPanel(QtWidgets.QFrame):
    """Right column: active literature context. Read-only summary list."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("tile")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("文献上下文")
        title.setStyleSheet(f"color:{DESIGN['ink']};font-weight:600;font-size:14px;")
        layout.addWidget(title)

        self._list = QtWidgets.QListWidget()
        self._list.setObjectName("context")
        layout.addWidget(self._list, stretch=1)

    def refresh(self, papers: list[Any]) -> None:
        self._list.clear()
        if not papers:
            placeholder = QtWidgets.QListWidgetItem("暂无激活文献 — 在聊天中提需求即可加入")
            placeholder.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self._list.addItem(placeholder)
            return
        for index, paper in enumerate(papers, start=1):
            item = QtWidgets.QListWidgetItem(
                f"{index}. {paper.title}  ({paper.year or 'n.d.'}, "
                f"{paper.journal or paper.publisher or 'unknown'})"
            )
            self._list.addItem(item)


class BrowserPanel(QtWidgets.QFrame):
    """Right column, lower half: QWebEngineView pane.

    The pane renders whatever URL the user or the controller points it
    at. LitTrace currently uses it as an embedded browser for publisher
    pages opened from the context panel; future iterations can pipe
    parsed paper Markdown through ``setHtml`` for a richer reading view.
    """

    HOME_URL = "https://duckduckgo.com/"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("tile")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        url_row = QtWidgets.QHBoxLayout()
        url_row.setContentsMargins(8, 8, 8, 4)
        self._back = QtWidgets.QPushButton("←")
        self._back.setFixedWidth(32)
        self._back.clicked.connect(self._on_back)
        url_row.addWidget(self._back)
        self._forward = QtWidgets.QPushButton("→")
        self._forward.setFixedWidth(32)
        self._forward.clicked.connect(self._on_forward)
        url_row.addWidget(self._forward)
        self._reload = QtWidgets.QPushButton("⟳")
        self._reload.setFixedWidth(32)
        self._reload.clicked.connect(self._on_reload)
        url_row.addWidget(self._reload)

        self._url = QtWidgets.QLineEdit()
        self._url.setPlaceholderText("输入 URL（Enter 打开）…")
        self._url.returnPressed.connect(self._on_url)
        url_row.addWidget(self._url, stretch=1)
        layout.addLayout(url_row)

        self._view = QWebEngineView()
        self._view.setUrl(QtCore.QUrl(self.HOME_URL))
        self._view.urlChanged.connect(self._on_url_changed)
        layout.addWidget(self._view, stretch=1)

    def _on_url(self) -> None:
        text = self._url.text().strip()
        if not text:
            return
        if not text.startswith(("http://", "https://")):
            text = "https://" + text
        self._view.setUrl(QtCore.QUrl(text))

    def _on_back(self) -> None:
        self._view.back()

    def _on_forward(self) -> None:
        self._view.forward()

    def _on_reload(self) -> None:
        self._view.reload()

    def _on_url_changed(self, url: QtCore.QUrl) -> None:
        self._url.setText(url.toString())

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

        # Initial refresh — mirrors Tk shell's first paint.
        self._context_panel.refresh(list(self._controller.list_active_papers()))
        self._status_bar.showMessage("就绪")

    # ---- UI construction -------------------------------------------------

    def _build_layout(self) -> None:
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top nav bar (dark)
        nav = QtWidgets.QFrame()
        nav.setObjectName("nav")
        nav.setFixedHeight(48)
        nav_layout = QtWidgets.QHBoxLayout(nav)
        nav_layout.setContentsMargins(20, 0, 20, 0)
        title = QtWidgets.QLabel("LitTrace")
        title.setStyleSheet(f"color:{DESIGN['on_dark']};font-weight:600;font-size:15px;")
        nav_layout.addWidget(title)
        nav_layout.addStretch(1)
        session = QtWidgets.QLabel(f"session: {self._controller.session.session_id}")
        session.setStyleSheet(f"color:{DESIGN['on_dark']};font-size:11px;")
        nav_layout.addWidget(session)
        outer.addWidget(nav)

        # Brand strip
        brand_row = QtWidgets.QHBoxLayout()
        brand_row.setContentsMargins(28, 18, 28, 6)
        brand = QtWidgets.QLabel("LitTrace")
        brand.setObjectName("brand")
        tagline = QtWidgets.QLabel("·  Materials & Chemistry Research")
        tagline.setObjectName("tagline")
        brand_row.addWidget(brand)
        brand_row.addWidget(tagline, stretch=1)
        brand_container = QtWidgets.QWidget()
        brand_container.setLayout(brand_row)
        outer.addWidget(brand_container)

        # Two-column main area: left chat, right (context above browser)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_chat_panel())
        splitter.addWidget(self._build_right_column())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([520, 720])
        outer.addWidget(splitter, stretch=1)

        self._status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self._status_bar)

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

        self._browser_panel = BrowserPanel()
        layout.addWidget(self._browser_panel, stretch=5)
        return right

    # ---- Event wiring ----------------------------------------------------

    def _wire_events(self) -> None:
        controller = self._controller

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

        def on_workspace(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_WORKSPACE_REFRESHED:
                self._context_panel.refresh(list(self._controller.list_active_papers()))

        def on_error(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_ERROR:
                self._chat_panel.append_message(
                    "system", f"⚠️ {event.payload.get('message', 'error')}"
                )

        controller.bus.subscribe(on_message)
        controller.bus.subscribe(on_status)
        controller.bus.subscribe(on_workspace)
        controller.bus.subscribe(on_error)


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
        help="Build the window, schedule a ping chat turn, then exit cleanly. "
        "Used by CI to confirm QtWebEngine + ShellController work without a display.",
    )
    return parser.parse_args(argv)


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
        # Pump one chat turn, then quit only after the controller reports
        # status back to idle. Without the idle wait the event loop tears
        # down an in-flight asyncio.Task that the Codex App Server is
        # still driving (visible as ``Task was destroyed but it is
        # pending!`` on shutdown).
        state = {"got_reply": False, "done": False}

        def _on_message(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_MESSAGE_APPENDED and event.payload.get("role") == "assistant":
                state["got_reply"] = True
                # Print the reply so CI can grep for it.
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
        # Hard ceiling so a stuck turn does not hang CI forever.
        QtCore.QTimer.singleShot(45000, app.quit)
        window.show()
        controller.submit_user_message("ping")
        rc = app.exec()
        controller.stop()
        if not state["got_reply"]:
            print("[smoke] FAIL: no assistant reply within 45s", flush=True)
            return 1
        print("[smoke] OK", flush=True)
        return rc

    window.show()
    rc = app.exec()
    controller.stop()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())