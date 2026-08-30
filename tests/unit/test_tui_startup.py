"""Round 20 follow-up: TUI startup preflight auth-check regression.

When we drove ``_codex_startup_preflight`` against real codex 0.150
on a machine that was already logged in via ChatGPT, the preflight
raised a spurious ``CodexStartupError("Codex 未登录...")`` modal —
even though ``read_account`` returned a populated ``account`` field
(``{type: chatgpt, email: ..., planType: plus}``).

The cause: ``_app_server_initialize`` translated
``requiresOpenaiAuth: True`` straight into ``AppServerError``,
without checking that the loaded account was non-null. codex 0.150
reports ``requiresOpenaiAuth: True`` even after a successful
ChatGPT login (the flag is a hint about the in-app "trust this
device" step, not a hard auth state). The canonical check lives in
``CodexAppServerChatService._require_authentication``: only raise
if BOTH ``requiresOpenaiAuth`` is True AND ``account`` is None.

These tests pin that contract so a future refactor of the preflight
doesn't lock every ChatGPT-login operator out of the TUI again.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeClient:
    """Bare object that exposes ``read_account`` the way
    ``AppServerClient`` does — the preflight only calls that one
    method, so we can fake it without pulling in the real client +
    subprocess machinery.
    """

    def __init__(self, account: dict[str, Any]) -> None:
        self._account = account
        self.read_account_calls = 0

    async def read_account(self, *, refresh_token: bool = False) -> dict[str, Any]:
        self.read_account_calls += 1
        return self._account


# ---------------------------------------------------------------------------
# _app_server_initialize auth-check contract
# ---------------------------------------------------------------------------


async def _drive_initialize(account_payload: dict[str, Any]) -> Any:
    """Run ``_app_server_initialize`` against a fake client and
    return whatever it returns (or re-raise).
    """
    from littrace.tui import _app_server_initialize

    client = _FakeClient(account_payload)
    return await _app_server_initialize(client, timeout_seconds=10.0)


def test_initialize_passes_when_account_is_populated_with_auth_flag() -> None:
    """Regression: codex 0.150 reports ``requiresOpenaiAuth=True`` even
    after a successful ChatGPT login, but the loaded ``account`` dict
    is non-null. The preflight must NOT raise in that case — otherwise
    every ChatGPT-login operator gets a bogus "Codex 未登录" modal.
    """
    async def _drive() -> None:
        result = await _drive_initialize({
            "account": {
                "type": "chatgpt",
                "email": "liu1757388@gmail.com",
                "planType": "plus",
            },
            "requiresOpenaiAuth": True,
        })
        assert result == "ready"

    asyncio.run(_drive())


def test_initialize_passes_when_no_auth_required() -> None:
    """Happy path with API-key auth: ``requiresOpenaiAuth=False`` and
    a populated ``account`` must pass through.
    """
    async def _drive() -> None:
        result = await _drive_initialize({
            "account": {"type": "apiKey"},
            "requiresOpenaiAuth": False,
        })
        assert result == "ready"

    asyncio.run(_drive())


def test_initialize_raises_when_account_is_null_and_auth_required() -> None:
    """Truly-unauthenticated case: ``requiresOpenaiAuth=True`` AND
    ``account=None``. The preflight MUST raise ``AppServerError`` so
    the outer except arm wraps it into a structured
    ``CodexStartupError`` with auth remediation steps.
    """
    from littrace.codex_runtime.errors import AppServerError

    async def _drive() -> None:
        with pytest.raises(AppServerError, match="未登录"):
            await _drive_initialize({
                "account": None,
                "requiresOpenaiAuth": True,
            })

    asyncio.run(_drive())


def test_initialize_does_not_call_read_account_more_than_once() -> None:
    """The probe must be cheap — one ``read_account`` call per
    preflight, not a retry loop. Otherwise every TUI launch would
    spend a round-trip on a benign RPC.
    """
    async def _drive() -> None:
        client = _FakeClient({
            "account": {"type": "chatgpt", "email": "x"},
            "requiresOpenaiAuth": True,
        })
        from littrace.tui import _app_server_initialize
        await _app_server_initialize(client, timeout_seconds=10.0)
        assert client.read_account_calls == 1, (
            f"preflight should call read_account exactly once; "
            f"got {client.read_account_calls}"
        )

    asyncio.run(_drive())
