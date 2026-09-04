"""Implementation of the ``quality_report`` skill."""
from __future__ import annotations

from littrace.config import LitTraceConfig
from littrace.evaluation.quality_report import build_quality_report
from littrace.models import LiteratureWorkspace
from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
)

from littrace.skills._helpers import _output_or_raise, _run_sync_skill


def run(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
    session_id: str | None = None,
):
    return _output_or_raise(
        _run_sync_skill(
            "quality_report",
            lambda payload: build_quality_report(
                payload["config"], payload["workspace"], session_id=payload.get("session_id")
            ),
            {"config": config, "workspace": workspace, "session_id": session_id},
            context=context,
            ledger=ledger,
            policy=policy,
            idempotency_key=idempotency_key,
        ),
        "quality_report",
    )
