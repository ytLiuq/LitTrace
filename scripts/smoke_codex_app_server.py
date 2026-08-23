"""Sanitized real-process smoke check for the LitTrace App Server client."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from littrace.codex_runtime.client import AppServerClient
from littrace.codex_runtime.gateway import APP_SERVER_TOOL_NAMES
from littrace.codex_runtime.service import CodexAppServerChatService
from littrace.config import LitTraceConfig


async def _run(home: Path, *, check_littrace_mcp: bool) -> None:
    resolved_home = home.expanduser().resolve()
    resolved_home.mkdir(parents=True, exist_ok=True)
    client = AppServerClient(
        ["codex", "app-server"],
        environment={"CODEX_HOME": str(resolved_home)},
    )
    try:
        initialized = await client.start()
        account = await client.read_account(refresh_token=False)
        report: dict[str, object] = {
            "codex_home_matches": initialized.get("codexHome") == str(resolved_home),
            "account_present": account.get("account") is not None,
            "requires_openai_auth": account.get("requiresOpenaiAuth"),
        }
        if check_littrace_mcp:
            global_status = await client.list_mcp_server_status()
            report["global_mcp_server_names"] = sorted(
                str(item.get("name")) for item in global_status.get("data", [])
            )
            config = LitTraceConfig()
            config.agent_runtime.codex_home = resolved_home
            service = CodexAppServerChatService(config, state_store=object())
            configured_tools = sorted(service._mcp_server_config()["enabled_tools"])
            report["configured_littrace_tools"] = configured_tools
            report["configured_littrace_tools_match"] = configured_tools == sorted(
                APP_SERVER_TOOL_NAMES
            )
            scratch = resolved_home / "smoke-workspace"
            scratch.mkdir(parents=True, exist_ok=True)
            thread = await client.start_thread(service._thread_overrides(scratch))
            probe = await client.call_mcp_tool(
                str(thread["id"]),
                config.agent_runtime.mcp_server_name,
                "get_workspace_context",
                {},
            )
            report["littrace_mcp_probe_returned"] = isinstance(probe, dict)
            report["littrace_mcp_probe_is_error"] = probe.get("isError") is True
            probe_text = json.dumps(probe, ensure_ascii=False).lower()
            report["littrace_mcp_probe_reached_gateway"] = any(
                marker in probe_text
                for marker in (
                    "postgres",
                    "connection",
                    "codex thread is not bound",
                    "littrace session",
                )
            )
            report["littrace_mcp_probe_unknown_server_or_tool"] = any(
                marker in probe_text
                for marker in ("unknown mcp server", "unknown tool", "not found")
            )
        print(json.dumps(report))
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-home", type=Path, required=True)
    parser.add_argument("--check-littrace-mcp", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args.codex_home, check_littrace_mcp=args.check_littrace_mcp))


if __name__ == "__main__":
    main()
