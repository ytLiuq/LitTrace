"""Live SSE test for ``POST /chat/stream``.

Round 6's ``test_chat_stream.py`` (under ``tests/api/``) monkeypatches
the chat service with an in-process fake. This live test goes the
whole way: it spawns ``scripts/fake_codex_server.py`` as a real
subprocess, points LitTrace at it via ``config.agent_runtime.codex_command``,
and reads the resulting SSE body through ``httpx.AsyncClient``. The
JSONL handshake (initialize / thread/start / mcpServerStatus/list /
turn/start) all happens on real asyncio subprocess pipes; the only
fake is the codex-harness side of the protocol.

Skip conditions:

  * ``LITTRACE_LIVE_TESTS`` not ``1`` (gated to opt-in runners).
  * Postgres not reachable (we exercise the full route so the
    session_state row + workspace dir must be writable).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.live


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAKE_CODEX = PROJECT_ROOT / "scripts" / "fake_codex_server.py"


def _require_live() -> None:
    if os.environ.get("LITTRACE_LIVE_TESTS") != "1":
        pytest.skip("set LITTRACE_LIVE_TESTS=1 to run live SSE tests")
    if not shutil.which("uv"):
        pytest.skip("uv is required to run the fake codex server")
    if not FAKE_CODEX.exists():
        pytest.skip(f"fake codex server script missing at {FAKE_CODEX}")
    # The shared conftest scrubs ``LITTRACE_*`` env vars on every
    # test except the three opt-in prefixes (LIVE / E2E / RAG).
    # ``LITTRACE_POSTGRES_DSN`` is not preserved, so we read the
    # RAG-prefixed DSN instead. Operators can set
    # ``LITTRACE_RAG_POSTGRES_DSN`` to point this test at the
    # same Postgres the rest of the live suite uses.
    dsn = (
        os.environ.get("LITTRACE_RAG_POSTGRES_DSN")
        or os.environ.get("LITTRACE_POSTGRES_DSN")
    )
    if not dsn:
        pytest.skip(
            "Postgres DSN not set (set LITTRACE_RAG_POSTGRES_DSN or "
            "LITTRACE_POSTGRES_DSN before running the live suite)"
        )


def _make_config(tmp_path: Path) -> dict[str, object]:
    return {
        "agent_runtime": {
            "codex_command": [sys.executable, str(FAKE_CODEX)],
            "codex_home": str(tmp_path / "codex-home"),
            "codex_home_mode": "isolated",
            "startup_timeout_seconds": 5.0,
            "request_timeout_seconds": 5.0,
            "rollout_enabled": False,
            "fallback_to_legacy": False,
            # Real Postgres under the test's tmp_path; the route
            # uses session_state.workspace_json as the canonical
            # workspace so the streaming path is identical to
            # production behaviour.
        },
        "metadata_store": {
            "backend": "postgres",
            "postgres_dsn": os.environ.get("LITTRACE_POSTGRES_DSN"),
            "schema_name": os.environ.get("LITTRACE_LIVE_SCHEMA", "littrace_e2e"),
        },
        "storage": {
            "sessions_dir": str(tmp_path / "sessions"),
            "paper_library_dir": str(tmp_path / "papers"),
            "metadata_dir": str(tmp_path / "metadata"),
        },
        "rag": {
            "enabled": False,
        },
    }


def _boot_app(tmp_path: Path, monkeypatch) -> tuple[object, dict[str, object]]:
    """Construct the FastAPI app with the fake codex server wired in.

    We bypass the ``config.yaml`` loader so the live test never picks
    up the developer's local environment. ``make_app`` keeps
    environment-driven middleware (``X-LitTrace-API-Version``) and
    the doc / OpenAPI gates intact.
    """
    from littrace.api.app import make_app
    from littrace.config import LitTraceConfig

    # Use the RAG-prefixed DSN because the shared conftest scrubs
    # LITTRACE_POSTGRES_DSN for every test.
    dsn = (
        os.environ.get("LITTRACE_RAG_POSTGRES_DSN")
        or os.environ.get("LITTRACE_POSTGRES_DSN")
    )
    cfg_dict = _make_config(tmp_path)
    cfg_dict["metadata_store"]["postgres_dsn"] = dsn
    config = LitTraceConfig.model_validate(cfg_dict)
    # ``make_app`` does not take a config; the chat route reads it
    # from ``api_app.load_config``. Patch that one symbol so the
    # route uses our test config — ``monkeypatch`` restores the
    # original after the test exits.
    import littrace.api.app as api_app_module
    monkeypatch.setattr(
        api_app_module, "load_config",
        lambda path="config.yaml": config,
    )
    return make_app(), config.model_dump(mode="json")


async def _read_sse_events(response: httpx.Response) -> list[dict[str, object]]:
    body = await response.aread()
    text = body.decode("utf-8")
    events: list[dict[str, object]] = []
    current: dict[str, object] = {}
    for raw_line in text.splitlines():
        if not raw_line or raw_line.startswith(":"):
            if current:
                events.append(current)
                current = {}
            continue
        if raw_line.startswith("event:"):
            current["event"] = raw_line[len("event:"):].strip()
        elif raw_line.startswith("data:"):
            current["data"] = raw_line[len("data:"):].strip()
    if current:
        events.append(current)
    return [
        {"event": ev.get("event"), "data": json.loads(str(ev["data"]))}
        for ev in events
    ]


@pytest.mark.anyio
async def test_chat_stream_end_to_end_against_fake_codex(
    tmp_path: Path, monkeypatch,
) -> None:
    _require_live()

    app, _config = _boot_app(tmp_path, monkeypatch)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver",
    ) as client:
        async with client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "MXene 复合材料在柔性压力传感器中的灵敏度对比",
                "session_id": "live-sse-smoke-1",
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = await _read_sse_events(response)

    delta_events = [e for e in events if e["event"] == "delta"]
    done_events = [e for e in events if e["event"] == "done"]
    error_events = [e for e in events if e["event"] == "error"]
    assert error_events == [], f"unexpected error events: {error_events}"
    assert delta_events, "no delta events received from the fake codex server"
    # The default fake delta list is "好的,按灵敏度,排序我推荐 5 篇。"
    deltas = [str(e["data"]["delta"]) for e in delta_events]
    assert deltas == ["好的", "按灵敏度", "排序我推荐 5 篇。"], deltas
    assert len(done_events) == 1
    done_data = done_events[0]["data"]
    assert done_data["reply"] == "好的按灵敏度排序我推荐 5 篇。"
    assert done_data["action"] == "codex_app_server_chat"
    assert done_data["session_id"] == "live-sse-smoke-1"
    assert "X-LitTrace-API-Version" in response.headers
    assert response.headers["X-LitTrace-API-Version"] == "0.4"
