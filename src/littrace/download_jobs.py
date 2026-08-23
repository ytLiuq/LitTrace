"""Durable command worker for App Server initiated downloads.

Codex only submits an immutable command. This module is the LitTrace-owned
execution boundary that claims commands from Postgres, performs domain work,
and records a terminal or retryable result on the same queue row.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.models import (
    DownloadExecutionRequest,
    DownloadExecutionResult,
    LiteratureWorkspace,
    PaperMetadata,
)
from littrace.state_db import AsyncTaskQueueReport, AsyncTaskRecord, StateStore

DownloadExecutor = Callable[
    [LitTraceConfig, LiteratureWorkspace, DownloadExecutionRequest],
    Awaitable[DownloadExecutionResult],
]


class DownloadJobBatchReport(BaseModel):
    schema_version: str = "littrace.download_job_batch_report.v1"
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    downloaded: int = 0
    requires_login: int = 0
    job_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None


async def run_pending_download_jobs(
    config: LitTraceConfig,
    *,
    limit: int = 20,
    worker_id: str | None = None,
    state_store: StateStore | None = None,
    executor: DownloadExecutor | None = None,
) -> DownloadJobBatchReport:
    """Claim and execute one batch of durable ``download_job`` commands."""

    if state_store is None:
        from littrace.state_db import state_store_from_config

        state_store = state_store_from_config(config)
    if executor is None:
        from littrace.skill_runner import execute_downloads_skill

        executor = execute_downloads_skill
    report = DownloadJobBatchReport()
    owner = worker_id or f"download:{socket.gethostname()}:{uuid4().hex[:12]}"
    jobs = state_store.claim_pending_async_tasks(
        worker_id=owner,
        kind="download_job",
        limit=max(1, limit),
        lease_seconds=max(config.download_retry.interval_seconds * 4, 120.0),
    )
    if not jobs:
        report.finished_at = datetime.now(UTC).isoformat()
        return report

    for job in jobs:
        report.job_ids.append(job.task_id)
        try:
            workspace, request = _execution_input(job)
            result = await executor(config, workspace, request)
            _mark_download_job_completed(state_store, job, result)
            report.processed += 1
            report.downloaded += result.downloaded_count
            report.requires_login += result.requires_login_count
        except Exception as exc:  # noqa: BLE001 - queue boundary must persist failures
            _mark_download_job_failed(state_store, job, exc, config)
            report.failed += 1
            report.warnings.append(
                f"download_job:{job.task_id}:{exc.__class__.__name__}: {exc}"
            )
    report.finished_at = datetime.now(UTC).isoformat()
    return report


async def run_download_job_daemon(
    config: LitTraceConfig,
    *,
    interval_seconds: float | None = None,
    limit: int | None = None,
) -> None:
    """Run the command worker as a separate long-lived process."""

    interval = max(
        interval_seconds
        if interval_seconds is not None
        else config.download_retry.interval_seconds,
        0.1,
    )
    batch_size = limit or config.download_retry.batch_size
    while True:
        await run_pending_download_jobs(config, limit=batch_size)
        await asyncio.sleep(interval)


def download_jobs_status(
    config: LitTraceConfig,
    *,
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> tuple[AsyncTaskQueueReport, list[AsyncTaskRecord]]:
    from littrace.state_db import state_store_from_config

    store = state_store_from_config(config)
    return (
        store.async_tasks_queue_report(kind="download_job"),
        store.list_async_tasks(
            kind="download_job",
            session_id=session_id,
            status=status,
            limit=limit,
        ),
    )


def requeue_dead_download_jobs(config: LitTraceConfig, *, limit: int = 20) -> int:
    from littrace.state_db import state_store_from_config

    return state_store_from_config(config).requeue_dead_async_tasks(
        kind="download_job",
        limit=limit,
    )


def _execution_input(
    job: AsyncTaskRecord,
) -> tuple[LiteratureWorkspace, DownloadExecutionRequest]:
    payload = job.result_json if isinstance(job.result_json, dict) else {}
    command = payload.get("command")
    if not isinstance(command, dict):
        raise TypeError("download job is missing its command payload")
    raw_ids = command.get("paper_ids")
    raw_papers = command.get("papers")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("download job paper_ids must be a non-empty array")
    if not isinstance(raw_papers, list):
        raise TypeError("download job papers snapshot must be an array")
    paper_ids = list(dict.fromkeys(str(item) for item in raw_ids if str(item)))
    papers = [PaperMetadata.model_validate(item) for item in raw_papers]
    by_id = {paper.paper_id: paper for paper in papers}
    missing = [paper_id for paper_id in paper_ids if paper_id not in by_id]
    if missing:
        raise ValueError(
            "download job papers snapshot is missing: " + ", ".join(missing[:10])
        )
    target = str(command.get("target") or "local_and_storage")
    request = DownloadExecutionRequest(
        paper_ids=paper_ids,
        session_id=job.session_id,
        target=target,
    )
    workspace = LiteratureWorkspace(papers={paper_id: by_id[paper_id] for paper_id in paper_ids})
    workspace.context.active_papers = paper_ids
    return workspace, request


def _mark_download_job_completed(
    state_store: StateStore,
    job: AsyncTaskRecord,
    result: DownloadExecutionResult,
) -> None:
    now = datetime.now(UTC).isoformat()
    payload = dict(job.result_json) if isinstance(job.result_json, dict) else {}
    payload["execution"] = result.model_dump(mode="json")
    job.status = "completed"
    job.result_json = payload
    job.completed_at = now
    job.updated_at = now
    job.next_attempt_at = None
    job.last_error = None
    job.lease_owner = None
    job.lease_expires_at = None
    job.last_heartbeat_at = None
    state_store.update_async_task(job)


def _mark_download_job_failed(
    state_store: StateStore,
    job: AsyncTaskRecord,
    exc: Exception,
    config: LitTraceConfig,
) -> None:
    now = datetime.now(UTC)
    job.last_error = f"{exc.__class__.__name__}: {exc}"
    job.updated_at = now.isoformat()
    job.lease_owner = None
    job.lease_expires_at = None
    job.last_heartbeat_at = None
    if job.attempt_count < config.download_retry.max_attempts:
        job.status = "failed"
        delay = min(
            config.download_retry.base_delay_seconds
            * (2 ** max(job.attempt_count - 1, 0)),
            3600,
        )
        job.next_attempt_at = (now + timedelta(seconds=delay)).isoformat()
    else:
        job.status = "dead"
        job.next_attempt_at = None
        job.completed_at = now.isoformat()
    state_store.update_async_task(job)
