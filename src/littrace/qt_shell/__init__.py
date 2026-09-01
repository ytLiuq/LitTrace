"""Helpers for the Qt shell.

Round 17: ``window_qt.py`` previously hand-rolled the
``ShellEvent`` → JSON → ``QMetaObject.invokeMethod`` → ``@Slot``
bridge for every event kind it cared about. The wiring was
duplicated for each subscriber and the slot bodies all repeated
the same JSON decode + ``kind`` dispatch. ``EventBridge`` here
collects those patterns into one helper that takes a single
mapping of ``{kind: callback}`` and turns it into:

  * a ``ShellEvent`` subscriber that the controller's
    ``ShellEventBus`` can register directly,
  * a registered Qt slot that decodes the JSON payload and
    calls the callback with the decoded body.

Round 17 also exposed ``install_subscriptions`` so ``window_qt``
can list all of its event subscriptions in one place instead of
scattering them between ``_wire_events`` and the individual
``_qt_on_xxx`` methods.
"""
from .event_bridge import EventBridge, install_subscriptions

__all__ = ["EventBridge", "install_subscriptions"]
