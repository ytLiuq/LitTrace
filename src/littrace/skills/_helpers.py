"""Internal helpers shared by every LitTrace skill.

These were originally defined inside :mod:`littrace.skill_runner` and are
extracted here so each per-skill ``run.py`` can import them without
pulling in the legacy 416-line module. New skills should use these helpers
to keep wiring consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
    ToolResult,
    run_sync_tool,
    run_tool,
    tool_contract,
)


@dataclass
class SearchSkillResult:
    """Return shape for ``search_papers.run``.

    Preserved exactly as it appeared in :mod:`littrace.skill_runner` so the
    21 callers that import it via the shim continue to work — both the
    shim and ``littrace.skills.search_papers`` re-export this class, and
    Python dataclass identity is preserved (same module path).
    """

    result: Any
    diagnostics: Any | None
    use_live: bool
    tool_result: ToolResult[Any]


async def _run_async_skill(
    name: str,
    func: Any,
    payload: Any,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolResult[Any]:
    return await run_tool(
        tool_contract(name),
        func,
        payload,
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
        metadata={"skill": name, **(metadata or {})},
    )


def _run_sync_skill(
    name: str,
    func: Any,
    payload: Any,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> ToolResult[Any]:
    return run_sync_tool(
        tool_contract(name),
        func,
        payload,
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
        metadata={"skill": name},
    )


def _output_or_raise(result: ToolResult[Any], skill_name: str) -> Any:
    if not result.ok or result.output is None:
        raise RuntimeError(result.error or f"{skill_name} failed")
    return result.output


def _active_papers(workspace: Any) -> list[Any]:
    return [
        workspace.papers[paper_id]
        for paper_id in workspace.context.active_papers
        if paper_id in workspace.papers
 ]