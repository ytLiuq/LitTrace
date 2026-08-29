"""Round 17 CR: cover the persistent MCP token contract.

The previous CR pass (R14) introduced a file-backed token so the
parent LitTrace and the codex subprocess agree on a single shared
secret across restarts. The contract has three parts:

  1. ``LITTRACE_MCP_TOKEN`` env var wins when set (operator override).
  2. Otherwise the persisted file at ``$CODEX_HOME/littrace-mcp.token``
     is reused if it exists; the token is stable across boots.
  3. Otherwise a fresh token is generated AND written to the file
     with 0o600 permissions so it is not world-readable on a
     multi-user box.

Round 17 also fixed a long-standing typo: ``_get_gateway``'s lazy
plugin-discovery path called ``log.warning`` but only ``logger`` was
defined. These tests assert both halves of the contract.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from littrace.mcp_server import _get_or_create_mcp_token, _token_path


pytestmark = pytest.mark.unit


@pytest.fixture
def isolated_token_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect CODEX_HOME / LITTRACE_MCP_TOKEN away from the operator's real setup.

    The module-level ``MCP_TOKEN`` constant is evaluated at import
    time, so we cannot easily reset it between tests. We work
    around that by patching both ``_token_path`` and the env vars
    the function reads, then calling the function directly (which
    is what production does for every fresh mcp_server boot).
    """
    fake_home = tmp_path / "codex-home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CODEX_HOME", str(fake_home))
    monkeypatch.delenv("LITTRACE_MCP_TOKEN", raising=False)
    return fake_home


def test_token_path_uses_codex_home_env(
    isolated_token_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(isolated_token_env))
    assert _token_path() == isolated_token_env / "littrace-mcp.token"


def test_token_path_falls_back_to_home_when_codex_home_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    # On Windows Path.home() reads USERPROFILE, not HOME.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert _token_path() == tmp_path / ".codex" / "littrace-mcp.token"


def test_explicit_env_token_wins_over_persisted_file(
    isolated_token_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pre-populate the persisted file with a known stale value.
    stale = isolated_token_env / "littrace-mcp.token"
    stale.write_text("mcp-stale-from-prior-boot", encoding="utf-8")
    monkeypatch.setenv("LITTRACE_MCP_TOKEN", "mcp-explicit-pinned")
    token = _get_or_create_mcp_token()
    assert token == "mcp-explicit-pinned"
    # Persisted file is left alone — the env-var path doesn't
    # rewrite the operator's pinned secret.
    assert stale.read_text(encoding="utf-8") == "mcp-stale-from-prior-boot"


def test_persisted_token_is_reused_across_calls(
    isolated_token_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LITTRACE_MCP_TOKEN", raising=False)
    first = _get_or_create_mcp_token()
    second = _get_or_create_mcp_token()
    third = _get_or_create_mcp_token()
    assert first == second == third
    persisted = (isolated_token_env / "littrace-mcp.token").read_text(
        encoding="utf-8"
    ).strip()
    assert persisted == first


def test_fresh_token_is_persisted_with_0600(
    isolated_token_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform == "win32":
        pytest.skip("os.chmod 0o600 is a no-op on NTFS")
    monkeypatch.delenv("LITTRACE_MCP_TOKEN", raising=False)
    token = _get_or_create_mcp_token()
    path = isolated_token_env / "littrace-mcp.token"
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip() == token
    # Stat the file and check only the owner bits — group/other
    # bits on Linux are reflected in st_mode, but the chmod call
    # strips them; the rest of the mode (file type, sticky) is
    # platform-dependent and we don't want to assert on it.
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600, f"token file mode is {oct(mode)}, expected 0o600"


def test_token_persist_failure_does_not_crash(
    isolated_token_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A read-only or sandboxed CODEX_HOME must not take down the process.

    POSIX ``chmod 0o555`` reliably blocks writes for the owner;
    NTFS ignores the read-only bit for files we own, so the test
    only runs on POSIX. The threat model on Windows is the NTFS
    ACL, which we do not try to simulate here.
    """
    if sys.platform == "win32":
        pytest.skip("chmod 0o555 does not block writes on NTFS")
    # Point CODEX_HOME at a path whose parent exists but is unwritable.
    read_only_parent = isolated_token_env / "ro"
    read_only_parent.mkdir()
    read_only_parent.chmod(0o555)
    target = read_only_parent / "littrace-mcp.token"
    monkeypatch.setenv("CODEX_HOME", str(read_only_parent))
    monkeypatch.delenv("LITTRACE_MCP_TOKEN", raising=False)
    try:
        token = _get_or_create_mcp_token()
    finally:
        read_only_parent.chmod(0o755)
    assert token.startswith("mcp-")
    assert not target.exists()
    # The warning surfaces; we don't pin the exact text because
    # the error class is platform-dependent.
    assert any("mcp_token_persist_failed" in rec.message for rec in caplog.records)


def test_get_gateway_uses_logger_when_plugin_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 17 regression: the lazy ``_get_gateway`` used to call
    ``log.warning`` in three places, but only ``logger`` was
    defined. A plugin that emitted a warning would therefore
    raise ``NameError`` and crash the MCP server on its first
    tool call. This test forces that path and asserts the call
    now resolves to the real logger.
    """
    from littrace import mcp_server

    # Reset the lazy singleton so the next call re-runs the
    # plugin-discovery path. Tests that import mcp_server also
    # import littrace.config which loads config.yaml from the
    # repo root; that's fine for the plugin-discovery branch
    # because we short-circuit the import with monkeypatch.
    monkeypatch.setattr(mcp_server, "_GATEWAY", None)
    monkeypatch.setattr(mcp_server, "_GATEWAY_PLUGINS_APPLIED", False)

    captured: list[tuple[str, tuple]] = []

    class _FakeLogger:
        def warning(self, msg: str, *args: object) -> None:
            captured.append((msg, args))

    # The _get_gateway body reads ``logger`` at call time, so
    # patch the module attribute rather than the import binding.
    monkeypatch.setattr(mcp_server, "logger", _FakeLogger())

    # Stub the gateway construction + the plugin catalog so we
    # don't drag in the real config / state store.
    class _FakeGateway:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def list_external_tool_specs(self) -> list[dict]:
            return []

    class _FakeResult:
        def apply(self, *, mcp_gateway: object) -> list[str]:
            return ["plugin warned but we'll keep going"]

    class _FakeMarketplace:
        def list_plugins(self) -> _FakeResult:  # type: ignore[override]
            return _FakeResult()

    # Patch the imports the lazy body performs.
    import littrace.marketplace as mp  # noqa: F401
    monkeypatch.setattr(mp, "list_plugins", lambda: _FakeResult())
    # state_store_from_config is called inside _get_gateway when
    # _GATEWAY is None. We don't want to open a real Postgres
    # connection in a unit test, so replace it with a stub.
    from littrace import state_db

    monkeypatch.setattr(
        state_db, "state_store_from_config", lambda _cfg: object(),
    )
    # LitTraceToolGateway.__init__ is the next call; replace it
    # with a no-op constructor that returns our fake.
    from littrace.codex_runtime.gateway import LitTraceToolGateway

    def _fake_init(self: object, _config: object, _store: object) -> None:
        pass

    monkeypatch.setattr(LitTraceToolGateway, "__init__", _fake_init)
    monkeypatch.setattr(mcp_server, "LitTraceToolGateway", _FakeGateway)

    gateway = mcp_server._get_gateway()
    assert isinstance(gateway, _FakeGateway)
    assert captured, "logger.warning was never called for the plugin warning"
    assert any("external MCP plugin load failed" in msg for msg, _ in captured)
