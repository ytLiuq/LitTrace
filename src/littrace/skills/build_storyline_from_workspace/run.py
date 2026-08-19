"""Implementation of the ``build_storyline_from_workspace`` skill."""
from __future__ import annotations

from littrace.evidence.storyline import build_storyline_from_workspace
from littrace.models import LiteratureWorkspace
from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
)

from littrace.skills._helpers import _output_or_raise, _run_sync_skill


def run(
    workspace: LiteratureWorkspace,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    result = _run_sync_skill(
        "build_storyline_from_workspace",
        build_storyline_from_workspace,
        workspace,
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "build_storyline_from_workspace")