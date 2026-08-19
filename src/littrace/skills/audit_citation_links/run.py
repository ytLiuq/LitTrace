"""Implementation of the ``audit_citation_links`` skill."""
from __future__ import annotations

from littrace.citations import audit_citation_links
from littrace.config import LitTraceConfig
from littrace.models import PaperMetadata
from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
)

from littrace.skills._helpers import _output_or_raise, _run_async_skill


async def run(
    papers: list[PaperMetadata],
    config: LitTraceConfig,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
):
    result = await _run_async_skill(
        "audit_citation_links",
        lambda payload: audit_citation_links(payload["papers"], payload["config"]),
        {"papers": papers, "config": config},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    return _output_or_raise(result, "audit_citation_links")