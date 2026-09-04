"""Standalone QtWebEngine health check for LitTrace.

Run this when you suspect the embedded browser pane is broken
independently of any LitTrace-specific wiring — it loads a known
non-blocked URL (example.com), waits for ``loadFinished``, prints the
final URL / title / a body-text snippet, and exits 0 on success.

This script deliberately does NOT import anything from
``littrace.window_qt`` or ``littrace.config`` — it's a pure
PySide6.QtWebEngineWidgets probe so we can isolate QtWebEngine itself
from LitTrace's BrowserPanel / cookie-domain wiring.

Usage (from the LitTrace repo root):

    PYTHONPATH=littrace/src python littrace/scripts/qtwebengine_healthcheck.py

It will open a small window showing example.com. After ~6 seconds it
prints a diagnostic line and exits. Exit code 0 means the load
succeeded; non-zero (with a clear message) means QtWebEngine is broken
at the platform level (missing system libs, wrong PySide6 version,
offscreen platform plugin, ...).

Round 22: added in response to the ACS Cloudflare investigation —
before we keep tweaking the BrowserPanel, we need to confirm
QtWebEngine itself is healthy on this machine.
"""
from __future__ import annotations

import os
import platform
import sys
import time


def _print(*args) -> None:
    print("[qt-health]", *args, flush=True)


def main() -> int:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except Exception as exc:
        _print(f"FAIL: PySide6 / QtWebEngineWidgets import error: {exc!r}")
        _print(f"  python: {sys.version.split()[0]}")
        _print(f"  platform: {platform.platform()}")
        return 2

    _print(f"Qt runtime version: {QtCore.qVersion()}")
    try:
        from PySide6.QtWebEngineCore import qWebEngineChromiumVersion
        _print(f"QtWebEngine Chromium: {qWebEngineChromiumVersion()}")
    except Exception as exc:
        _print(f"WARN: could not read QtWebEngine Chromium version: {exc!r}")

    # Force the offscreen platform plugin when no display is available
    # (CI / sandboxed runs). The user running this on macOS will have
    # a display so this is a no-op for them.
    if "--offscreen" in sys.argv:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        _print("forcing QT_QPA_PLATFORM=offscreen")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    state = {"done": False, "ok": None, "url": None, "title": None,
             "body": None, "errors": []}

    view = QWebEngineView()
    view.resize(800, 600)
    view.show()

    target = "https://example.com"

    def _on_load_finished(ok: bool) -> None:
        # QtWebEngine sometimes fires loadFinished twice (main frame +
        # an interstitial). Only act on the first so the timeout
        # below always wins eventually.
        if state["done"]:
            return
        url = view.url().toString()
        title = ""
        try:
            title = view.page().title()
        except Exception as exc:
            state["errors"].append(f"title() error: {exc!r}")
        # ``QWebEnginePage.toPlainText()`` was removed in PySide6 6.11
        # — the new API is async and takes a callback. For a quick
        # health check we just skip the body-text dump on 6.11+; if
        # we want it back we can wire ``runJavaScript('document.body
        # .innerText')`` instead.
        body = ""
        try:
            from PySide6 import __version__ as _ps
            major = int(_ps.split(".")[0])
            minor = int(_ps.split(".")[1])
            if major < 6 or (major == 6 and minor < 11):
                body = view.page().toPlainText()[:200]
        except Exception as exc:
            state["errors"].append(f"toPlainText() version gate: {exc!r}")
        state.update(done=True, ok=ok, url=url, title=title, body=body)
        _print(f"loadFinished ok={ok}")
        _print(f"  url:   {url}")
        _print(f"  title: {title!r}")
        if body:
            _print(f"  body[:200]: {body!r}")
        if state["errors"]:
            for err in state["errors"]:
                _print(f"  ERR: {err}")
        app.quit()

    def _on_load_started() -> None:
        _print(f"loadStarted: {view.url().toString()}")

    def _on_render_terminated(status: int, exit_code: int) -> None:
        _print(f"renderProcessTerminated status={status} exit={exit_code}")
        state["done"] = True
        app.quit()

    view.loadStarted.connect(_on_load_started)
    view.loadFinished.connect(_on_load_finished)
    view.renderProcessTerminated.connect(_on_render_terminated)

    # Watchdog: 15 s is plenty for example.com; if we don't hear back
    # by then, something is broken upstream (DNS, sandboxing, ...).
    watchdog = QtCore.QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(
        lambda: (
            _print("TIMEOUT: no loadFinished after 15s"),
            state.update(done=True, ok=False),
            app.quit(),
        )
    )
    watchdog.start(15_000)

    _print(f"loading {target}")
    view.setUrl(QtCore.QUrl(target))

    rc = app.exec()

    if not state["done"]:
        _print("FAIL: app exited without loadFinished or watchdog firing")
        return 3
    if not state["ok"]:
        _print("FAIL: loadFinished reported ok=False (network/DNS/proxy issue?)")
        return 4
    if "example.com" not in (state["url"] or ""):
        _print(f"FAIL: unexpected final URL {state['url']!r}")
        return 5
    _print("OK: QtWebEngine is healthy on this machine")
    return rc


if __name__ == "__main__":
    sys.exit(main())
