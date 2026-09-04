"""Implementation of the ``execute_downloads`` skill.

The lazy import of :func:`littrace.downloads.execute_downloads` is preserved
verbatim from the legacy :mod:`littrace.skill_runner` to avoid a circular
import. Do not move it to module top level.
"""
from __future__ import annotations

from littrace.config import LitTraceConfig
from littrace.models import (
    DownloadExecutionRequest,
    DownloadExecutionResult,
    LiteratureWorkspace,
)
from littrace.tool_contracts import (
    ToolCallContext,
    ToolExecutionLedger,
    ToolExecutionPolicy,
)

from littrace.skills._helpers import (
    _active_papers,
    _output_or_raise,
    _run_async_skill,
)


async def run(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    request: DownloadExecutionRequest,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
    idempotency_key: str | None = None,
) -> DownloadExecutionResult:
    # Lazy import — preserved from legacy skill_runner.py to avoid
    # a circular dependency with littrace.downloads.
    from littrace.downloads import execute_downloads

    result = await _run_async_skill(
        "execute_downloads",
        lambda payload: execute_downloads(payload["config"], payload["papers"], payload["request"]),
        {"config": config, "papers": _active_papers(workspace), "request": request},
        context=context,
        ledger=ledger,
        policy=policy,
        idempotency_key=idempotency_key,
    )
    final = _output_or_raise(result, "execute_downloads")
    # Parsing is a separate durable stage. A storage-only download does not
    # create a local PDF, so invoking the local parser here would always report
    # a false missing-PDF failure. The topic pipeline enqueues parse_job rows
    # after artifact registration and the worker materializes bytes from the
    # object store before parsing.
    return final
