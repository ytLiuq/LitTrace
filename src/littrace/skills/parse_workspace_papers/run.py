"""Implementation of the ``parse_workspace_papers`` skill."""
from __future__ import annotations

from littrace.config import LitTraceConfig
from littrace.evidence.parsing import parse_workspace_papers
from littrace.models import LiteratureWorkspace
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
) -> tuple[LiteratureWorkspace, dict[str, object]]:
    result = await _run_async_skill(
        "parse_workspace_papers",
        lambda payload: parse_workspace_papers(payload["workspace"], payload["config"]),
        {"workspace": workspace, "config": config},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "parse_workspace_papers")