"""Implementation of the ``search_papers`` skill."""
from __future__ import annotations

from typing import Any, Callable

from littrace.config import LitTraceConfig
from littrace.models import PaperSearchRequest, PaperSearchResult
from littrace.retrieval.search import (
    LiveSearchClient,
    MockMaterialsSearchClient,
)
from littrace.retrieval.query_planner import plan_query_variants
from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
)

from littrace.skills._helpers import (
    SearchSkillResult,
    _run_async_skill,
)


async def run(
    request: PaperSearchRequest,
    config: LitTraceConfig,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> SearchSkillResult:
    use_live = (
        config.api.enable_live_search if request.live is None else request.live
    )
    client = (
        LiveSearchClient(config, progress_callback=progress_callback)
        if use_live
        else MockMaterialsSearchClient()
    )
    if use_live and (
        not request.query_variants
        or request.query_variants == [request.topic]
    ):
        variants = await plan_query_variants(request.topic, config)
        request = request.model_copy(update={"query_variants": variants})
    result = await _run_async_skill(
        "search_papers",
        client.fetch,
        request,
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
        metadata={"live": use_live},
    )
    diagnostics = client.diagnostics if use_live else None
    if not result.ok or result.output is None:
        if diagnostics:
            diagnostics.errors.append(result.error or "search_papers failed")
        return SearchSkillResult(
            result=PaperSearchResult(request=request, papers=[]),
            diagnostics=diagnostics,
            use_live=use_live,
            tool_result=result,
        )
    return SearchSkillResult(
        result=result.output,
        diagnostics=diagnostics,
        use_live=use_live,
        tool_result=result,
    )
