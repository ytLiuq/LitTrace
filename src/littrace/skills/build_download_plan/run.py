"""Implementation of the ``build_download_plan`` skill."""
from __future__ import annotations

from littrace.access_layer.download_planning import build_download_plan
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
)

from littrace.skills._helpers import _active_papers, _output_or_raise, _run_async_skill


async def run(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    result = await _run_async_skill(
        "build_download_plan",
        lambda payload: build_download_plan(
            payload["config"], payload["papers"], set(payload["selected_for_download"])
        ),
        {
            "config": config,
            "papers": _active_papers(workspace),
            "selected_for_download": workspace.context.selected_for_download,
        },
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "build_download_plan")