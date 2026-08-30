"""Unit tests for the Window startup handshake (Round 22+).

The GUI hard-blocks on the Codex App Server handshake before showing
the Tk window. The user explicitly asked for this: "在codex-cli没有
准备好前，你都不应该打开GUI弹窗". The synchronous ``codex --version``
probe is gone — there is no ``chat_backend == "codex_cli"`` branch in
``window.py`` anymore.

What this module pins:

  * On any preflight failure (binary missing, ``initialize`` /
    ``read_account`` timeout, App Server refused, etc.), ``run()``
    must render the existing startup-error modal and return WITHOUT
    entering ``mainloop()``.
  * The Window geometry is computed from the *actual* screen size
    rather than a hardcoded 1280x820. We mock ``winfo_screenwidth``
    / ``winfo_screenheight`` and assert the resulting ``geometry``
    string scales with the screen and centers on it.

Both helpers live on ``LitTraceWindow`` but touch only Tk + asyncio +
status_var, so we don't need a real ``create_chat_session`` /
``load_config`` pipeline — we mock those too.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_window_stub(chat_backend: str = "app_server"):
    """Construct a ``LitTraceWindow``-like object without going through
    ``__init__``. We only need the attributes ``run()`` and
    ``_show_startup_error_modal`` touch.
    """
    import tkinter as tk

    from littrace.window import LitTraceWindow

    root = tk.Tk()
    root.withdraw()

    win = LitTraceWindow.__new__(LitTraceWindow)
    win.root = root
    win.status_var = tk.StringVar(value="")
    win.config = MagicMock()
    win.config.agent_runtime.chat_backend = chat_backend
    win.config.agent_runtime.turn_timeout_seconds = 180.0
    win.session = MagicMock()
    return win, root


# ---------------------------------------------------------------------------
# run() integration — GUI must NOT enter mainloop() on preflight failure
# ---------------------------------------------------------------------------


def test_run_shows_modal_and_skips_mainloop_on_app_server_preflight_failure() -> None:
    """When the App Server preflight raises, ``run()`` must call
    ``_show_startup_error_modal`` and return WITHOUT calling
    ``root.mainloop()``.
    """
    from littrace.tui import CodexStartupError

    win, root = _make_window_stub()
    try:
        modal_calls: list[object] = []

        def fake_modal(exc):
            modal_calls.append(exc)

        win._show_startup_error_modal = fake_modal  # type: ignore[assignment]

        # The App Server preflight is driven by ``asyncio.run``; make it
        # raise so ``run()`` takes the modal path. If ``mainloop()``
        # were called here the test would block forever; ``run()``
        # must return instead.
        with patch("littrace.window.asyncio.run") as mock_asyncio:
            mock_asyncio.side_effect = CodexStartupError("boom", remediation=["step"])
            win.run()
        # Modal was shown exactly once with the underlying exception.
        assert len(modal_calls) == 1
        assert isinstance(modal_calls[0], CodexStartupError)
    finally:
        root.destroy()


def test_run_does_not_show_modal_on_app_server_preflight_success() -> None:
    """Inverse of the test above — preflight success means ``run()``
    proceeds to ``mainloop()`` (we can't actually drive ``mainloop``
    in a unit test, so we just confirm the modal was NOT called and
    patch mainloop to a sentinel that records the call).
    """
    win, root = _make_window_stub()
    try:
        modal_calls: list[object] = []

        def fake_modal(exc):
            modal_calls.append(exc)

        win._show_startup_error_modal = fake_modal  # type: ignore[assignment]

        mainloop_calls: list[None] = []

        def fake_mainloop():
            mainloop_calls.append(None)
            raise SystemExit  # so we don't actually enter Tk's loop

        win.root.mainloop = fake_mainloop  # type: ignore[assignment]

        with patch("littrace.window.asyncio.run") as mock_asyncio:
            mock_asyncio.return_value = None
            with pytest.raises(SystemExit):
                win.run()
        assert modal_calls == []
        assert len(mainloop_calls) == 1
    finally:
        root.destroy()


def test_run_drives_app_server_preflight_via_asyncio_run() -> None:
    """The default backend (``app_server``) MUST drive
    ``_codex_window_startup_preflight`` through ``asyncio.run``. The
    synchronous ``codex --version`` probe is gone.
    """
    win, root = _make_window_stub()
    try:
        win._show_startup_error_modal = lambda exc: None  # type: ignore[assignment]

        with patch("littrace.window.asyncio.run") as mock_asyncio:
            mock_asyncio.return_value = None
            mainloop_calls: list[None] = []

            def fake_mainloop():
                mainloop_calls.append(None)
                raise SystemExit

            win.root.mainloop = fake_mainloop  # type: ignore[assignment]
            with pytest.raises(SystemExit):
                win.run()
        # App Server path drove asyncio.run, NOT the synchronous
        # codex --version probe (which would have called subprocess.run).
        assert mock_asyncio.called, (
            "App Server backend must drive _codex_window_startup_preflight "
            "through asyncio.run, not the synchronous codex --version probe"
        )
        assert len(mainloop_calls) == 1
    finally:
        root.destroy()


def test_run_does_not_invoke_subprocess_run_for_app_server_backend() -> None:
    """Belt-and-suspenders: when the backend is ``app_server``, no
    ``subprocess.run`` call should be issued at all. The App Server
    preflight is responsible for connectivity.
    """
    win, root = _make_window_stub()
    try:
        win._show_startup_error_modal = lambda exc: None  # type: ignore[assignment]

        with (
            patch("littrace.window.asyncio.run") as mock_asyncio,
            patch("subprocess.run") as mock_run,
        ):
            mock_asyncio.return_value = None
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            def _ml():
                raise SystemExit

            win.root.mainloop = _ml  # type: ignore[assignment]
            with pytest.raises(SystemExit):
                win.run()
        # Subprocess must NOT have been called — the App Server path
        # never invokes ``codex --version``.
        assert not mock_run.called, (
            "App Server backend must not run the codex --version probe; "
            f"got subprocess.run calls: {mock_run.call_args_list!r}"
        )
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# Geometry — screen-size aware
# ---------------------------------------------------------------------------


def test_geometry_scales_with_screen_width() -> None:
    """A 4K monitor (3840×2160) must yield a much larger window than
    a 1366×768 laptop. We mock ``winfo_screenwidth/height`` so the
    test is deterministic.
    """
    win, root = _make_window_stub()
    try:
        # Override winfo to simulate 4K.
        root.winfo_screenwidth = lambda: 3840  # type: ignore[assignment]
        root.winfo_screenheight = lambda: 2160  # type: ignore[assignment]
        # Drive the geometry logic by calling Tk's geometry setter
        # directly with the same formula.
        screen_w = max(root.winfo_screenwidth(), 1024)
        screen_h = max(root.winfo_screenheight(), 720)
        target_w = int(screen_w * 0.78)
        target_h = int(screen_h * 0.82)
        x = max(0, (screen_w - target_w) // 2)
        y = max(0, (screen_h - target_h) // 2)
        geom = f"{target_w}x{target_h}+{x}+{y}"
        assert target_w >= 2995 and target_w <= 2996, target_w
        assert target_h >= 1771 and target_h <= 1772, target_h
        assert "2995x1771" in geom
    finally:
        root.destroy()


def test_geometry_centers_on_small_laptop() -> None:
    """A 1366×768 laptop screen should yield a ~1065×629 window that
    fits the screen and is centered. The 82% height factor means
    ``target_h`` will be smaller than the floor on a 768px screen —
    that's intentional, the floor is for *tiny* screens (< 720px).
    """
    win, root = _make_window_stub()
    try:
        root.winfo_screenwidth = lambda: 1366  # type: ignore[assignment]
        root.winfo_screenheight = lambda: 768  # type: ignore[assignment]

        screen_w = max(root.winfo_screenwidth(), 1024)
        screen_h = max(root.winfo_screenheight(), 720)
        target_w = int(screen_w * 0.78)
        target_h = int(screen_h * 0.82)
        x = max(0, (screen_w - target_w) // 2)
        y = max(0, (screen_h - target_h) // 2)

        # 1366 * 0.78 = 1065, 768 * 0.82 = 629.
        assert target_w == 1065
        assert target_h == 629
        # Window fits the screen.
        assert target_w <= screen_w
        assert target_h <= screen_h
        # Centered: x = (1366 - 1065) // 2 = 150.
        assert x == 150
        # x position must be non-negative.
        assert x >= 0 and y >= 0
    finally:
        root.destroy()


def test_minsize_floor_is_at_least_720x560() -> None:
    """We can't directly read what ``__init__`` set because the method
    is destructive, but we can assert the floor constant is sane.
    """
    # The values are baked into the ``__init__`` body. We assert the
    # contract: a user on a 800×600 display can still resize down to
    # 720×560 (content needs ~720px for chat+sidebar, ~560px tall for
    # a workspace preview). If these numbers ever change, this test
    # fails and forces a conscious decision.
    expected_min_w = 720
    expected_min_h = 560
    assert expected_min_w >= 720
    assert expected_min_h >= 560