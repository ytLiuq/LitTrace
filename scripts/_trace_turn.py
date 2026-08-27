"""Trace a real turn — print stderr and dump notifications via recorder JSONL."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

os.environ.setdefault("LITTRACE_CONFIG_PATH", str(PROJECT_ROOT / "config.yaml"))
os.environ.setdefault("CODEX_HOME", str(PROJECT_ROOT / "data" / "codex-home"))

from littrace.codex_runtime.client import AppServerClient
from littrace.config import load_config


async def main() -> int:
    cfg = load_config(os.environ["LITTRACE_CONFIG_PATH"])
    rt = cfg.agent_runtime
    cfg.agent_runtime.turn_timeout_seconds = 20

    scratch = Path(tempfile.mkdtemp(prefix="lt-trace-"))
    env: dict[str, str] = {}
    if rt.codex_home_mode.value == "isolated":
        env["CODEX_HOME"] = str(rt.codex_home.expanduser().resolve())

    from littrace.codex_runtime.service import CodexAppServerChatService
    svc = CodexAppServerChatService(cfg)
    overrides = svc._thread_overrides(scratch)

    client = AppServerClient(
        list(rt.codex_command),
        startup_timeout=rt.startup_timeout_seconds,
        request_timeout=rt.request_timeout_seconds,
        environment=env,
    )

    # Side-channel recorder that captures every server notification so we
    # can see what codex is doing even when the turn is in flight.
    from littrace.codex_runtime.rollout import RolloutRecorder
    rec = RolloutRecorder(Path("/tmp/lt-trace-rollout.jsonl"))
    rec.open()

    await client.start()
    print("[initialize OK]")
    thread_id = ""
    try:
        thread = await client.start_thread(overrides)
        thread_id = str(thread.get("id") or "")
        print(f"[thread.start OK id={thread_id}]")
        client.set_rollout_recorder(thread_id, rec)

        from littrace.state_db import (
            AgentThreadBindingRecord, SessionStateRecord, state_store_from_config,
        )
        state = state_store_from_config(cfg)
        state.upsert_session_state(SessionStateRecord(session_id="trace", status="draft", revision=0))
        state.upsert_agent_thread_binding(AgentThreadBindingRecord(
            session_id="trace", codex_thread_id=thread_id,
            runtime_kind="codex_app_server/isolated", status="active", workspace_revision=0,
        ))

        print("[health probe]")
        hp = await client.call_mcp_tool(thread_id, "littrace", "get_workspace_context", {})
        print(f"  -> {json.dumps(hp, ensure_ascii=False)[:200]}")

        print("\n[run_turn] starting...")
        t0 = time.monotonic()
        try:
            turn = await client.run_turn(
                thread_id,
                "Call get_workspace_context and report workspace_revision.",
                timeout=rt.turn_timeout_seconds,
            )
            print(f"[run_turn] OK in {time.monotonic()-t0:.1f}s status={turn.status}")
            print(f"[run_turn] reply={turn.reply[:400]!r}")
        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"[run_turn] FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
    finally:
        rec.close()
        await client.close()

    print(f"\n[codex stderr tail — last 40 lines]")
    for ln in client.stderr_tail[-40:]:
        print(f"  | {ln}")

    print(f"\n[side-channel recorder — /tmp/lt-trace-rollout.jsonl]")
    rec_path = Path("/tmp/lt-trace-rollout.jsonl")
    if rec_path.exists():
        for i, line in enumerate(rec_path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("type_") or ev.get("type") or "?"
            extra = {k: v for k, v in ev.items() if k not in ("type_", "type")}
            print(f"  {i:3d} {t}: {json.dumps(extra, ensure_ascii=False)[:240]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))