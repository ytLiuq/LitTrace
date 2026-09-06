"""Codex App Server adapter for bounded literature reranking."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from littrace.codex_runtime.service import CodexAppServerChatService
from littrace.config import LitTraceConfig


async def rank_with_codex(
    config: LitTraceConfig,
    topic: str,
    records: list[dict[str, object]],
) -> dict[int, float] | None:
    """Ask Codex for bounded relevance scores, returning ``None`` on failure."""
    if config.agent_runtime.mode.value != "codex_app_server":
        return None
    service = CodexAppServerChatService(config)
    manager = service._shared_runtime_manager()
    prompt = (
        "Rank these academic papers for the literal research topic. Return JSON only: "
        "[{\"index\":0,\"score\":0.0}]. Score materials/device relevance from 0 to 1; "
        "penalize unrelated medicine, infrastructure, control, or generic monitoring papers. "
        "Do not add or remove papers.\n"
        + json.dumps({"topic": topic, "papers": records}, ensure_ascii=False)
    )

    async def operation(client: Any) -> dict[int, float] | None:
        account = await client.read_account(refresh_token=False)
        if account.get("requiresOpenaiAuth") is True and account.get("account") is None:
            return None
        scratch = Path(config.agent_runtime.scratch_root).expanduser().resolve() / "rerank"
        scratch.mkdir(parents=True, exist_ok=True)
        thread = await client.start_thread({
            "cwd": str(scratch),
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "serviceName": "littrace-rerank",
            "developerInstructions": "Return only the requested JSON. Do not call tools.",
        })
        thread_id = str(thread.get("id") or "")
        if not thread_id:
            return None
        turn = await client.run_turn(
            thread_id,
            prompt,
            timeout=min(config.agent_runtime.turn_timeout_seconds, 60.0),
        )
        return _parse_scores(turn.reply)

    try:
        return await asyncio.wait_for(
            manager.use(operation),
            timeout=min(config.agent_runtime.turn_timeout_seconds + 10.0, 75.0),
        )
    except Exception:
        return None


def _parse_scores(text: str) -> dict[int, float] | None:
    stripped = text.strip()
    start, end = stripped.find("["), stripped.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    scores: dict[int, float] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        index, score = item.get("index"), item.get("score")
        if isinstance(index, int) and isinstance(score, int | float):
            scores[index] = max(0.0, min(1.0, float(score)))
    return scores or None
