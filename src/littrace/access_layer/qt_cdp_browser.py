"""Asynchronous CDP browser backed by Qt's ``QWebSocket``.

Why a parallel implementation alongside the synchronous
``CDPBrowser`` in ``cdp_core.py``:

The synchronous driver uses the third-party ``websocket-client``
package and blocks the calling thread on ``ws.send`` / ``ws.recv``.
For headless PDF downloads run from a CLI that's fine — sentinel
hands it to ``asyncio.to_thread`` and walks away. For the Qt
BrowserPanel, though, blocking the main thread freezes the whole
UI (user feedback 2026-09: "Qt 窗口无法被点击" during auth).

This module wraps Chrome DevTools Protocol in Qt-native
``QWebSocket`` + ``QObject`` signals so all CDP traffic flows
through Qt's event loop. Callers subscribe to events with
``subscribe_event(method, callback)`` and Chrome *pushes* state
changes (cookie writes, page navigations, network responses) via
the ``event_received`` signal — no polling.

Comparison with the polling-based fallback the BrowserPanel was
using before:

    before: QTimer(2000ms) → Network.getCookies → filter → done
    now:    subscribe_event("Network.cookieChanged") → push filter
            → done, fires within milliseconds of the cookie write

Round 25: introduced to address the main-thread freeze reported
during ACS institutional SSO. Also incidentally fixes a race
where the user could complete a CF challenge, the 2s poll would
*just* miss it, and the cookie detector would still report "not
logged in".
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot
from PySide6.QtNetwork import QAbstractSocket
from PySide6.QtWebSockets import QWebSocket

log = logging.getLogger(__name__)


# JSON-RPC IDs grow monotonically. ``QWebSocket`` is event-driven,
# but we still need to send ``id`` on every request and match
# responses to the request that produced them.
class _IdAllocator:
    def __init__(self) -> None:
        self._next = 1

    def next(self) -> int:
        v = self._next
        self._next += 1
        return v


class QtCDPBrowser(QObject):
    """Async CDP driver for Qt UIs.

    Usage::

        browser = QtCDPBrowser()
        browser.connected.connect(on_connected)
        browser.subscribe_event(
            "Network.cookieChanged",
            lambda params: self._on_cookie(params),
        )
        browser.open(ws_url)  # ws://127.0.0.1:19222/devtools/browser/<id>

    All signals are emitted on the thread the ``QtCDPBrowser`` was
    constructed in. Subscribers don't need to worry about thread
    marshaling — Qt's queued-connection rules handle it.
    """

    # Lifecycle
    connected = Signal()
    disconnected = Signal()
    # Fires for every CDP message that has a ``method`` field. Args:
    # (method_name, params_dict). Subscribe via ``subscribe_event``
    # if you only care about a specific method.
    event_received = Signal(str, dict)
    # Convenience signal for the specific event we care about most:
    # any cookie write/delete on any domain. Fires alongside
    # ``event_received("Network.cookieChanged", params)`` but with
    # the params already filtered to the cookie dict (so subscribers
    # don't need to remember CDP's exact envelope shape).
    cookie_changed = Signal(dict)
    # Errors that prevent a normal request/response round-trip. The
    # caller can decide whether to surface these to the user.
    cdp_error = Signal(int, str, str)  # id, code, message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._ws: QWebSocket | None = None
        self._ids = _IdAllocator()
        # id -> callback (called on response)
        self._pending: dict[int, Callable[[dict], None]] = {}
        # method -> list of callbacks (called on event)
        self._handlers: dict[str, list[Callable[[dict], None]]] = {}
        self._connected = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def open(self, ws_url: str) -> None:
        """Open a CDP WebSocket. ``connected`` fires when ready;
        ``cdp_error`` fires on transport failure."""
        if self._ws is not None:
            self.close()
        ws = QWebSocket()
        self._ws = ws
        ws.connected.connect(self._on_connected)
        ws.disconnected.connect(self._on_disconnected)
        ws.textMessageReceived.connect(self._on_text)
        ws.errorOccurred.connect(self._on_error)
        # open() is async — completion via the ``connected`` signal.
        ws.open(ws_url)

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws.deleteLater()
            self._ws = None
        self._connected = False
        self._pending.clear()

    # ------------------------------------------------------------------
    # Request/response
    # ------------------------------------------------------------------

    def send(
        self,
        method: str,
        params: dict | None = None,
        *,
        on_response: Callable[[dict], None] | None = None,
        timeout_ms: int = 30_000,
    ) -> int:
        """Send a JSON-RPC request. ``on_response`` (optional) is
        invoked on the same thread once the matching response
        arrives. Returns the JSON-RPC id so callers can correlate
        responses manually if needed.

        Note: we don't enforce the timeout ourselves — ``on_response``
        is simply never called if the response never arrives.
        Callers that need a hard timeout should pair ``send`` with
        a ``QTimer.singleShot``.
        """
        if self._ws is None or not self._connected:
            raise RuntimeError("QtCDPBrowser: not connected")
        rid = self._ids.next()
        msg = {"id": rid, "method": method}
        if params:
            msg["params"] = params
        if on_response is not None:
            self._pending[rid] = on_response
        self._ws.sendTextMessage(json.dumps(msg))
        return rid

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------

    def subscribe_event(
        self,
        method: str,
        callback: Callable[[dict], None],
    ) -> Callable[[], None]:
        """Register ``callback`` for every CDP event with the given
        ``method`` name (e.g. ``"Network.cookieChanged"``). Returns
        a callable that unsubscribes — keep it around if you need to
        detach the listener later.
        """
        self._handlers.setdefault(method, []).append(callback)
        return lambda: self._handlers.get(method, []).remove(callback)

    def enable_domain(self, domain: str, on_response=None) -> None:
        """Tell CDP to start emitting events from ``domain``
        (e.g. ``"Network"``, ``"Page"``). Must be called once before
        the corresponding events fire. Idempotent on the protocol
        side — sending twice doesn't double-deliver."""
        self.send(f"{domain}.enable", on_response=on_response)

    # ------------------------------------------------------------------
    # Internal slots — only ever called on this object's thread
    # ------------------------------------------------------------------

    @Slot()
    def _on_connected(self) -> None:
        self._connected = True
        self.connected.emit()

    @Slot()
    def _on_disconnected(self) -> None:
        self._connected = False
        self.disconnected.emit()

    @Slot(QAbstractSocket.SocketError, str)
    def _on_error(
        self, _code: QAbstractSocket.SocketError, message: str
    ) -> None:
        log.warning("QWebSocket error: %s", message)
        self.cdp_error.emit(0, "websocket", message)

    @Slot(str)
    def _on_text(self, raw: str) -> None:
        import logging
        log = logging.getLogger(__name__)
        log.info("CDP raw: %r", raw[:300])
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("non-JSON CDP frame: %r", raw[:120])
            return
        # Distinguish response (has ``id``) from event (has ``method``).
        if "id" in msg and "method" not in msg:
            self._handle_response(msg)
        elif "method" in msg:
            self._handle_event(msg)
        else:
            log.debug("unrecognized CDP frame: %r", msg)

    def _handle_response(self, msg: dict) -> None:
        rid = msg.get("id")
        if rid is None:
            return
        cb = self._pending.pop(rid, None)
        if cb is None:
            log.debug("response with no pending callback: id=%s", rid)
            return
        if "error" in msg:
            err = msg["error"] or {}
            self.cdp_error.emit(
                rid,
                str(err.get("code", "?")),
                str(err.get("message", "?")),
            )
        cb(msg)

    def _handle_event(self, msg: dict) -> None:
        method = msg.get("method") or ""
        params = msg.get("params") or {}
        # Special-case ``Network.cookieChanged`` — we always emit
        # ``cookie_changed`` for it regardless of subscribers, so
        # callers don't have to remember the CDP envelope shape.
        if method == "Network.cookieChanged":
            self.cookie_changed.emit(params)
        for cb in list(self._handlers.get(method, [])):
            try:
                cb(params)
            except Exception:
                log.exception("event handler for %s raised", method)
        self.event_received.emit(method, params)