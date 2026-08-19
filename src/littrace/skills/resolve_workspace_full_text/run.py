"""Implementation of the ``resolve_workspace_full_text`` skill."""
from __future__ import annotations

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.retrieval.full_text import resolve_workspace_full_text
from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
)

from littrace.skills._helpers import _output_or_raise, _run_async_skill


async def run(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> LiteratureWorkspace:
    result = await _run_async_skill(
        "resolve_workspace_full_text",
        lambda payload: resolve_workspace_full_text(payload["workspace"], payload["config"]),
        {"workspace": workspace, "config": config},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "resolve_workspace_full_text")