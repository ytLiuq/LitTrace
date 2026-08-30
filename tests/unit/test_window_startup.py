"""Round 20 (Phase 5 follow-up): startup-time guards for ``littrace-window``.

The Window is a strict Codex App Server surface. The startup path must:

1. Force ``agent_runtime.mode = codex_app_server`` even if the config
   inherits ``legacy`` from an older install.
2. Hard-error on an explicit ``LITTRACE_AGENT_RUNTIME=legacy`` so the
   user sees a clear message instead of a Tk runtime crash mid-loop.
3. Force ``fallback_to_legacy = False`` so the route layer can never
   silently call legacy chat behind the user's back.
4. Probe Codex reachability (``shutil.which`` + ``initialize`` handshake)
   and surface a structured ``CodexStartupError`` with remediation
   steps when it fails.

Each contract is a separate test so a regression points at exactly one
break.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from littrace.config import AgentRuntimeMode, LitTraceConfig
from littrace.window import (
    _codex_window_startup_preflight,
    _resolve_window_config,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _resolve_window_config
# ---------------------------------------------------------------------------


def test_resolve_window_config_forces_codex_app_server(monkeypatch) -> None:
    """A legacy ``mode`` from an older config is silently overwritten
    to ``codex_app_server``. The user upgrades the package, the window
    still opens, and the operator does not need to know about the
    config flip."""
    monkeypatch.delenv("LITTRACE_AGENT_RUNTIME", raising=False)
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.LEGACY

    resolved = _resolve_window_config(config)

    assert resolved.agent_runtime.mode == AgentRuntimeMode.CODEX_APP_SERVER


def test_resolve_window_config_forces_fallback_off(monkeypatch) -> None:
    """``fallback_to_legacy`` must be forced off regardless of what the
    user wrote in config.yaml — the route layer is the only place that
    consults this flag, and the TUI/Window contract is strict Codex."""
    monkeypatch.delenv("LITTRACE_AGENT_RUNTIME", raising=False)
    config = LitTraceConfig()
    config.agent_runtime.fallback_to_legacy = True

    resolved = _resolve_window_config(config)

    assert resolved.agent_runtime.fallback_to_legacy is False


def test_resolve_window_config_explicit_legacy_env_is_hard_error(
    monkeypatch,
) -> None:
    """``LITTRACE_AGENT_RUNTIME=legacy`` is an explicit user choice;
    the Window must refuse to start rather than silently fall through
    to the legacy Coordinator — the whole point of Round 20 is that
    the Window is a strict Codex surface."""
    monkeypatch.setenv("LITTRACE_AGENT_RUNTIME", "legacy")
    config = LitTraceConfig()

    with pytest.raises(SystemExit, match="requires the Codex App Server"):
        _resolve_window_config(config)


def test_resolve_window_config_passes_through_codex_app_server(
    monkeypatch,
) -> None:
    """The happy path: the config already says codex_app_server and
    fallback is off. ``_resolve_window_config`` must leave it alone
    (no overwriting, no warnings)."""
    monkeypatch.delenv("LITTRACE_AGENT_RUNTIME", raising=False)
    config = LitTraceConfig()
    config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    config.agent_runtime.fallback_to_legacy = False
    config.agent_runtime.codex_command = ["codex", "app-server"]

    resolved = _resolve_window_config(config)

    assert resolved.agent_runtime.mode == AgentRuntimeMode.CODEX_APP_SERVER
    assert resolved.agent_runtime.fallback_to_legacy is False


# ---------------------------------------------------------------------------
# _codex_window_startup_preflight
# ---------------------------------------------------------------------------


def test_preflight_raises_codex_startup_error_when_binary_missing(
    monkeypatch, tmp_path,
) -> None:
    """If ``shutil.which`` returns None for the configured codex
    binary, the preflight raises ``CodexStartupError`` with a
    remediation list that names ``npm install -g @openai/codex``.

    Regression guard: a missing binary used to surface only inside
    the Tk event loop as a system-message bubble; now it must be a
    structured startup error with steps the operator can act on.
    """
    monkeypatch.setattr("shutil.which", lambda _cmd: None)
    config = LitTraceConfig()
    config.agent_runtime.codex_command = ["codex", "app-server"]

    class _Session:
        root = tmp_path
        session_id = "test"

    async def _drive() -> None:
        with pytest.raises(Exception) as info:
            await _codex_window_startup_preflight(config, _Session())
        # The exact exception type lives in ``littrace.tui`` -- we accept
        # either the bare class or a SystemExit / RuntimeError surfaced
        # by the helper. ``remediation`` is the contract that matters.
        exc = info.value
        remediation = getattr(exc, "remediation", None)
        assert remediation is not None, (
            f"missing-binary failure must carry remediation steps; got {exc!r}"
        )
        assert any("npm install" in step for step in remediation), (
            f"remediation must mention `npm install -g @openai/codex`; "
            f"got {remediation!r}"
        )

    asyncio.run(_drive())


def test_preflight_translates_app_server_error_to_startup_error(
    monkeypatch, tmp_path,
) -> None:
    """If the Codex binary is on PATH but the ``initialize`` handshake
    raises ``AppServerError``, the Window preflight must propagate a
    structured ``CodexStartupError`` so the Tk modal renders a single
    error envelope with remediation steps — the operator must not see
    a raw ``AppServerError`` mid-turn.

    The Window preflight delegates the handshake to the TUI's
    ``_codex_startup_preflight`` (which already owns the
    ``AppServerError`` → ``CodexStartupError`` translation), so the
    test patches that TUI helper directly and confirms the wrapper
    surfaces the wrapped error to the caller.
    """
    from littrace.codex_runtime.errors import AppServerError
    from littrace.tui import CodexStartupError

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/local/bin/codex")

    async def _explode(_config, _session):
        raise AppServerError("not logged in")

    monkeypatch.setattr(
        "littrace.tui._codex_startup_preflight", _explode
    )

    config = LitTraceConfig()
    config.agent_runtime.codex_command = ["codex", "app-server"]

    class _Session:
        root = tmp_path
        session_id = "test"

    async def _drive() -> None:
        with pytest.raises(CodexStartupError) as info:
            await _codex_window_startup_preflight(config, _Session())
        exc = info.value
        # The TUI preflight already wraps AppServerError; the Window
        # wrapper must surface that wrapped error, not let the raw
        # AppServerError escape.
        assert "not logged in" in str(exc), (
            f"wrapped error must mention the underlying reason; got {exc!r}"
        )
        remediation = getattr(exc, "remediation", None)
        assert remediation is not None and remediation, (
            "handshake-failure must carry remediation steps"
        )

    asyncio.run(_drive())


def test_preflight_uses_configured_codex_command(monkeypatch, tmp_path) -> None:
    """``codex_command`` is the user-overridable list — the preflight
    must probe the first element via ``shutil.which``, not the literal
    string ``"codex"``."""
    seen: list[str] = []

    def _which(cmd: str):
        seen.append(cmd)
        return None  # force the error path

    monkeypatch.setattr("shutil.which", _which)
    config = LitTraceConfig()
    config.agent_runtime.codex_command = ["my-custom-codex", "app-server"]

    class _Session:
        root = tmp_path
        session_id = "test"

    async def _drive() -> None:
        with pytest.raises(Exception):
            await _codex_window_startup_preflight(config, _Session())

    asyncio.run(_drive())

    assert seen == ["my-custom-codex"], (
        f"preflight must probe the configured command's first arg; "
        f"got {seen!r}"
    )


def test_preflight_respects_startup_timeout_env_override(
    monkeypatch, tmp_path,
) -> None:
    """``LITTRACE_CODEX_STARTUP_TIMEOUT_SECONDS`` overrides the
    per-config timeout for the handshake — Windows cold start can
    exceed the 20s default and the operator needs to be able to bump
    it from the shell.

    The Window preflight delegates to the TUI preflight; we patch
    the TUI helper and confirm the env var it sees is the override.
    """
    monkeypatch.setenv("LITTRACE_CODEX_STARTUP_TIMEOUT_SECONDS", "60")
    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/local/bin/codex")

    observed_timeout: list[str] = []

    async def _capture(config, _session):
        observed_timeout.append(
            os.environ.get(
                "LITTRACE_CODEX_STARTUP_TIMEOUT_SECONDS", "unset"
            )
        )
        # Return normally so the preflight doesn't error out.
        return "ready"

    monkeypatch.setattr(
        "littrace.tui._codex_startup_preflight", _capture
    )

    config = LitTraceConfig()
    config.agent_runtime.codex_command = ["codex", "app-server"]

    class _Session:
        root = tmp_path
        session_id = "test"

    async def _drive() -> None:
        await _codex_window_startup_preflight(config, _Session())

    asyncio.run(_drive())

    assert observed_timeout == ["60"], (
        "the env var override must reach the underlying TUI preflight"
    )
