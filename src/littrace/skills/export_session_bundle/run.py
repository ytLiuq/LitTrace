"""Implementation of the ``export_session_bundle`` skill.

The lazy import of :func:`littrace.export.export_session_bundle` is preserved
verbatim from the legacy :mod:`littrace.skill_runner` to avoid a circular
import. Do not move it to module top level.
"""
from __future__ import annotations

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
)

from littrace.skills._helpers import _output_or_raise, _run_async_skill


async def run(
    session,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> dict[str, str]:
    # Lazy import — preserved from legacy skill_runner.py to avoid
    # a circular dependency with littrace.export.
    from littrace.export import export_session_bundle

    result = await _run_async_skill(
        "export_session_bundle",
        lambda payload: export_session_bundle(
            payload["session"], payload["workspace"], payload["config"]
        ),
        {"session": session, "workspace": workspace, "config": config},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "export_session_bundle")