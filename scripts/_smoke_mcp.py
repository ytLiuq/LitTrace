"""Full MCP smoke: app-server initialize → start_thread (with littrace mcp config) → call get_workspace_context."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

os.environ.setdefault("LITTRACE_CONFIG_PATH", str(PROJECT_ROOT / "config.yaml"))
os.environ.setdefault("CODEX_HOME", str(PROJECT_ROOT / "data" / "codex-home"))

from littrace.codex_runtime.client import AppServerClient
from littrace.codex_runtime.runtime import CodexAppServerRuntimeManager
from littrace.config import load_config
from littrace.codex_runtime.errors import AppServerError


async def main() -> int:
    cfg = load_config(os.environ["LITTRACE_CONFIG_PATH"])
    runtime = cfg.agent_runtime

    scratch_dir = Path(tempfile.mkdtemp(prefix="littrace-smoke-"))
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Mirror the env contract from CodexAppServerChatService._codex_environment
    env: dict[str, str] = {}
    if runtime.codex_home_mode.value == "isolated":
        env["CODEX_HOME"] = str(runtime.codex_home.expanduser().resolve())

    # Reuse the service helpers so the smoke matches the production path.
    from littrace.codex_runtime.service import CodexAppServerChatService

    svc = CodexAppServerChatService(cfg)
    thread_overrides = svc._thread_overrides(scratch_dir)
    mcp_cfg = thread_overrides.get("config", {}).get("mcp_servers", {})
    print(f"[mcp_servers in thread_overrides] keys={list(mcp_cfg.keys())}")
    if "littrace" not in mcp_cfg:
        print("[FAIL] littrace server not in mcp_servers config")
        return 1
    print(f"[littrace server] command={mcp_cfg['littrace']['command']} args={mcp_cfg['littrace']['args']}")
    print(f"[littrace server] required={mcp_cfg['littrace'].get('required')} enabled_tools_count={len(mcp_cfg['littrace'].get('enabled_tools', []))}")

    command = list(runtime.codex_command)
    client = AppServerClient(
        command,
        startup_timeout=runtime.startup_timeout_seconds,
        request_timeout=runtime.request_timeout_seconds,
        environment=env,
    )
    try:
        await client.start()
        thread = await client.start_thread(thread_overrides)
        thread_id = str(thread.get("id") or "")
        print(f"[thread.start] OK id={thread_id}")

        # Mirror service._chat_with_client: bind the codex thread to a
        # LitTrace session before calling MCP tools, otherwise the gateway
        # rejects with "Codex thread is not bound to an active LitTrace session".
        from littrace.state_db import (
            AgentThreadBindingRecord,
            SessionStateRecord,
            state_store_from_config,
        )
        state = state_store_from_config(cfg)
        # FK requires session_state row first.
        state.upsert_session_state(
            SessionStateRecord(
                session_id="smoke-session",
                status="draft",
                revision=0,
            )
        )
        binding = state.upsert_agent_thread_binding(
            AgentThreadBindingRecord(
                session_id="smoke-session",
                codex_thread_id=thread_id,
                runtime_kind="codex_app_server/isolated",
                status="active",
                workspace_revision=0,
            )
        )
        print(f"[binding] session_id={binding.session_id} thread_id={binding.codex_thread_id}")

        try:
            result = await client.call_mcp_tool(
                thread_id, "littrace", "get_workspace_context", {},
            )
            print(f"[mcp health probe] OK")
            print(f"[mcp health probe] payload={json.dumps(result, ensure_ascii=False)[:400]}")
            if isinstance(result, dict) and result.get("isError"):
                print("[FAIL] MCP health probe returned isError=True")
                return 3
        except AppServerError as exc:
            print(f"[FAIL] MCP health probe raised: {exc}")
            return 2
    finally:
        await client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))