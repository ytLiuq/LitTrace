"""Implementation of the ``build_quality_metrics`` skill.

This skill was previously a contract without a wrapper. It returns the
``QualityReport.metrics`` dict directly (the contract advertises
``output_schema="dict[str, object]"``), avoiding a wrapper-of-wrapper
through :mod:`littrace.skill_runner` that would risk a circular import.
"""
from __future__ import annotations

from littrace.config import LitTraceConfig
from littrace.evaluation.quality_report import build_quality_report
from littrace.models import LiteratureWorkspace
from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
    run_sync_tool,
    tool_contract,
)


def run(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> dict[str, float]:
    """Compute and return the dict-shaped quality metrics.

    Returns the ``metrics`` field of the underlying :class:`QualityReport`.
    """
    result = run_sync_tool(
        tool_contract("build_quality_metrics"),
        lambda payload: build_quality_report(
            payload["config"], payload["workspace"]
        ).metrics,
        {"config": config, "workspace": workspace},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
        metadata={"skill": "build_quality_metrics"},
    )
    if not result.ok or result.output is None:
        raise RuntimeError(result.error or "build_quality_metrics failed")
    return result.output