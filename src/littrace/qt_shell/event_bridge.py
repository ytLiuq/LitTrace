"""Round 17 Qt event-bridge helper.

The LitTrace ``ShellController`` runs its asyncio chat pipeline
on a background worker thread, so it cannot touch Qt widgets
directly. The previous bridge in ``window_qt._wire_events`` did
the right thing — JSON-encode the ``ShellEvent``, dispatch via
``QMetaObject.invokeMethod(..., QueuedConnection, ...)``,
decode JSON in the slot, dispatch by ``kind`` — but it was
re-implemented for every event kind the shell cared about.

``EventBridge`` collects that plumbing into one helper:

  bridge = EventBridge(receiver, controller)
  install_subscriptions(
      controller,
      bridge,
      {
          controller.EVENT_MESSAGE_APPENDED: my_widget.update_message,
          controller.EVENT_STATUS_CHANGED: my_widget.update_status,
          ...
      },
  )

Each subscription becomes one entry in the controller's
``ShellEventBus`` plus one ``@Slot``-decorated method on the
``EventBridge``. The bridge slots live on a stable Qt meta-object
so the ``invokeMethod`` call from the worker thread is matched
by Qt's signal dispatcher; the callback receives the decoded
``{kind, payload}`` dict and runs on the GUI thread, so it can
mutate widgets safely.

Implementation note: ``EventBridge`` exposes a single Qt
``Signal`` (``eventReceived``) carrying the JSON payload. The
controller's bus subscriber posts onto the signal via
``emit``; the signal's ``Qt.QueuedConnection`` semantics
guarantee the slot fires on the GUI thread. This is
functionally identical to ``QMetaObject.invokeMethod`` +
``QueuedConnection``, but uses Qt's signal-slot machinery
directly, which avoids the meta-object method-name registration
trap that bit the previous setattr-based design.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

from PySide6 import QtCore

if TYPE_CHECKING:
    from littrace.shell_controller import ShellController


EventCallback = Callable[[dict[str, Any]], None]


class EventBridge(QtCore.QObject):
    """Qt-side bridge from ``ShellEvent`` bus events to widget callbacks.

    Usage::

        bridge = EventBridge(self, controller)
        bridge.subscribe(controller.EVENT_MESSAGE_APPENDED, self._on_message)

    ``subscribe`` registers the callback; ``install_subscriptions``
    then wires the controller's bus to deliver events to the
    right callback through Qt's signal-slot system.
    """

    # Single signal that carries the JSON-encoded event
    # payload. Any number of subscribers can listen; the
    # internal dispatcher routes by ``kind`` to the right
    # callback. ``str`` is the only Q_ARG type Qt reliably
    # round-trips through its meta-system without type
    # coercion bugs.
    eventReceived = QtCore.Signal(str)

    def __init__(self, receiver: QtCore.QObject, controller: "ShellController") -> None:
        super().__init__(receiver)
        self._controller = controller
        self.setParent(receiver)
        self._installed: dict[str, EventCallback] = {}
        # Single dispatch slot connected to ``eventReceived``.
        # Decodes the JSON, looks up the callback for the
        # event's ``kind``, and runs the callback on the GUI
        # thread (because ``eventReceived`` is emitted from
        # the bus subscriber with ``Qt.QueuedConnection``).
        self.eventReceived.connect(self._dispatch)

    def subscribe(self, kind: str, callback: EventCallback) -> None:
        """Install ``callback`` as the GUI-thread handler for events
        of ``kind``.

        Idempotent: a second ``subscribe`` call with the same
        ``kind`` replaces the previous callback (no duplicate
        delivery).
        """
        self._installed[kind] = callback

    def subscribe_many(self, kinds_to_callbacks: dict[str, EventCallback]) -> None:
        """Bulk register one callback per kind.

        Round 17: lets ``window_qt`` register a single handler
        against multiple kinds (e.g. WARMUP_STARTED +
        WARMUP_DONE both routing to ``_on_warmup_event``)
        without the second-``subscribe`` call clobbering the
        first callback's ``_installed`` entry.
        """
        for kind, callback in kinds_to_callbacks.items():
            self._installed[kind] = callback

    @QtCore.Slot(str)
    def _dispatch(self, payload: str) -> None:
        """GUI-thread dispatcher. ``payload`` is the JSON-encoded
        event the controller's bus subscriber forwarded via
        ``eventReceived.emit``.

        Decodes the payload, looks up the callback registered
        for the event's ``kind``, and invokes it on the GUI
        thread with the decoded body. Tags ``body`` with
        ``__kind`` so handlers registered against multiple
        kinds can branch on which one fired.
        """
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            return
        kind = data.get("kind", "")
        body = data.get("payload") or {}
        callback = self._installed.get(kind)
        if callback is None:
            return
        body = dict(body)
        body["__kind"] = kind
        try:
            callback(body)
        except Exception:
            # A buggy callback must not stop the bridge
            # from delivering subsequent events — same
            # policy as ``ShellEventBus.emit``.
            pass


def install_subscriptions(
    controller: "ShellController",
    bridge: EventBridge,
    subscriptions: dict[str, EventCallback],
) -> None:
    """Wire ``subscriptions`` onto ``controller.bus`` via ``bridge``.

    Round 17 convenience: ``window_qt.py`` lists all of its
    event subscriptions in one place instead of scattering
    ``controller.bus.subscribe(post(...))`` calls. The function
    also registers the controller-side subscriber that fires
    the bridge's ``eventReceived`` signal — without that
    subscriber, no event ever reaches the GUI thread.

    Two kinds can share a callback (the WARMUP_*/AUTH_* pairs in
    ``window_qt`` do). ``subscribe_many`` keeps both entries in
    ``bridge._installed`` so each kind routes to the right
    callback.
    """
    bridge.subscribe_many(subscriptions)

    def _bus_subscriber(event) -> None:
        # ``Qt.QueuedConnection`` here means the
        # ``eventReceived`` signal's connected slot fires on
        # the bridge's thread (the GUI thread, since the
        # bridge is parented to the main window). The
        # controller's bus subscriber runs on the controller's
        # asyncio worker thread, so without the queued
        # connection the slot would crash trying to touch
        # Qt widgets.
        payload = json.dumps(
            {"kind": event.kind, "payload": event.payload},
            ensure_ascii=False,
        )
        bridge.eventReceived.emit(payload)

    controller.bus.subscribe(_bus_subscriber)
