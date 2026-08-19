"""Implementation of the ``build_research_document_report`` skill."""
from __future__ import annotations

from littrace.config import LitTraceConfig
from littrace.evidence.document_composer import build_research_document_report
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
    title: str | None = None,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    result = await _run_async_skill(
        "build_research_document_report",
        lambda payload: build_research_document_report(
            payload["workspace"], payload["config"], title=payload["title"]
        ),
        {"workspace": workspace, "config": config, "title": title},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "build_research_document_report")