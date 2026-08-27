"""Real chat integration (bypass intent routing): directly call CodexAppServerChatService.chat().

This is the production code path used by /api/chat for intents that survive
agent_runtime.handle_agent_chat's legacy carve-out (currently: download selection
intents). handle_agent_chat routes almost every intent back to legacy, so we
skip that layer to see whether the App Server + MCP tool round-trip actually
works end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

os.environ.setdefault("LITTRACE_CONFIG_PATH", str(PROJECT_ROOT / "config.yaml"))
os.environ.setdefault("CODEX_HOME", str(PROJECT_ROOT / "data" / "codex-home"))
os.environ.setdefault("LITTRACE_CODEX_HOME_MODE", "isolated")

from littrace.codex_runtime.service import CodexAppServerChatService
from littrace.config import load_config
from littrace.models import ChatRequest, LiteratureWorkspace
from littrace.session import load_or_create_session


async def main() -> int:
    cfg = load_config(os.environ["LITTRACE_CONFIG_PATH"])
    rt = cfg.agent_runtime
    print(f"[config] mode={rt.mode} mcp_server={rt.mcp_server_name} rollout_enabled={rt.rollout_enabled}")
    print(f"[config] sandbox_policy={rt.sandbox_policy} approval=on-request")
    # Speed up the failure path while we debug approval flow.
    cfg.agent_runtime.turn_timeout_seconds = 30

    session = load_or_create_session(cfg, session_id=None)
    session.root.mkdir(parents=True, exist_ok=True)
    print(f"[session] id={session.session_id} root={session.root}")

    workspace = LiteratureWorkspace()

    # DEVELOPER_INSTRUCTIONS says: "Use only tools on the `littrace` MCP server
    # for facts and changes involving the active session." Plain English keeps
    # the legacy intent parser (which only fires through handle_agent_chat) out
    # of the way; we are calling the service directly.
    message = (
        "Call the littrace MCP server's get_workspace_context tool and report "
        "the workspace_revision, paper_count, and active_paper_count back to me."
    )
    request = ChatRequest(message=message, live=False)

    print(f"[chat] -> {message!r}")
    t0 = time.monotonic()
    try:
        svc = CodexAppServerChatService(cfg)
        response, workspace = await asyncio.wait_for(
            svc.chat(request, workspace, session),
            timeout=rt.turn_timeout_seconds + 30,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"\n[chat] FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
        # Walk the exception chain — service.py:381 raises a fresh
        # AttributeError while trying to read exc.error_code on a bare
        # AppServerError. The real cause is __context__/__cause__.
        cause = exc.__cause__ or exc.__context__
        while cause is not None:
            print(f"[chat]   caused by {type(cause).__name__}: {cause}")
            cause = cause.__cause__ or cause.__context__
        # Also dump the codex app-server stderr tail to see what the
        # server was complaining about right before it timed out.
        try:
            from littrace.codex_runtime.runtime import _RUNTIME_MANAGERS
            for mgr in _RUNTIME_MANAGERS.values():
                client = getattr(mgr, "_client", None)
                if client is not None:
                    tail = client.stderr_tail[-40:]
                    if tail:
                        print("\n[codex stderr tail]")
                        for ln in tail:
                            print(f"  | {ln}")
        except Exception as dump_exc:
            print(f"[stderr dump failed] {dump_exc}")
        return 1
    elapsed = time.monotonic() - t0
    print(f"\n[chat] OK in {elapsed:.1f}s action={response.action}")
    print(f"[chat] reply={response.reply[:500]!r}")
    if response.warnings:
        print(f"[chat] warnings={response.warnings}")

    rollout_path = session.root / "rollouts" / f"rollout-{session.session_id}.jsonl"
    print(f"\n[rollout] {rollout_path}")
    if rollout_path.exists():
        size = rollout_path.stat().st_size
        events: list[dict] = []
        for line in rollout_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        print(f"[rollout] size={size} bytes event_count={len(events)}")
        mcp_calls = 0
        for ev in events:
            t = ev.get("type_") or ev.get("type") or "?"
            extra_keys = {k: v for k, v in ev.items() if k not in ("type_", "type")}
            line_repr = json.dumps(extra_keys, ensure_ascii=False)
            if "mcp" in line_repr.lower() or t in ("turn_context", "tool_call", "tool_result"):
                mcp_calls += 1
            print(f"  - {t}: {line_repr[:250]}")
        print(f"\n[rollout] MCP-related events: {mcp_calls}")
    else:
        print("[rollout] FILE NOT FOUND — recorder never wrote")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))