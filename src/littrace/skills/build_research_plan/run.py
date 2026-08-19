"""Implementation of the ``build_research_plan`` skill."""
from __future__ import annotations

from littrace.models import LiteratureWorkspace
from littrace.research_planner import build_research_plan
from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
)

from littrace.skills._helpers import _output_or_raise, _run_async_skill


async def run(
    topic: str,
    workspace: LiteratureWorkspace,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    result = await _run_async_skill(
        "build_research_plan",
        lambda payload: build_research_plan(payload["topic"], payload["workspace"]),
        {"topic": topic, "workspace": workspace},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "build_research_plan")