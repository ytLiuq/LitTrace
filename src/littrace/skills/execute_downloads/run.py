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
    # Round 25: kick off ``parse_workspace_skill`` synchronously
    # after every download. Sentinel used to gate this on
    # ``config.sentinel.parse_on_daily`` (default False) so most
    # runs never auto-parsed; users had to either flip that flag or
    # manually call ``/parse``. The user explicitly asked for
    # "silent parse on download" — parse is now unconditional after
    # any ``execute_downloads`` that lands PDFs in the artifact
    # store. The parse skill itself is short-lived; we surface
    # progress via the status strip so the user can see
    # "已下载 N 篇，正在解析 M 篇".
    if final.downloaded_count > 0 and not request.dry_run:
        from littrace.skills.parse_workspace_papers.run import (
            run as parse_workspace_papers_skill,
        )
        try:
            _, parse_report = await parse_workspace_papers_skill(
                workspace, config,
                context=context,
                ledger=ledger,
                policy=policy,
                idempotency_key=f"parse-after-{idempotency_key or 'adhoc'}",
            )
        except Exception as exc:  # noqa: BLE001
            # Parse failures must not break the download result —
            # the PDFs are already in the object store and the user
            # can re-trigger parse manually.
            print(
                f"[execute_downloads] silent parse failed: "
                f"{exc.__class__.__name__}: {exc}",
                flush=True,
            )
    return final