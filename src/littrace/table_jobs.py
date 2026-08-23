"""Durable performance-table extraction worker."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.evidence.tables import extract_performance_cells
from littrace.models import (
    LiteratureWorkspace,
    PerformanceCell,
    coerce_parsed,
)
from littrace.state_db import AsyncTaskQueueReport, AsyncTaskRecord, StateStore


class TableExecutionOutput(BaseModel):
    performance_cells: list[PerformanceCell] = Field(default_factory=list)
    structured_artifacts: list[dict[str, object]] = Field(default_factory=list)
    parsed_sha256: dict[str, str] = Field(default_factory=dict)
    harness: dict[str, object] = Field(default_factory=dict)


class TableJobBatchReport(BaseModel):
    schema_version: str = "littrace.table_job_batch_report.v1"
    processed: int = 0
    failed: int = 0
    performance_cells: int = 0
    structured_artifacts: int = 0
    stale: int = 0
    job_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None


TableExecutor = Callable[
    [LitTraceConfig, LiteratureWorkspace],
    Awaitable[TableExecutionOutput],
]


async def run_pending_table_jobs(
    config: LitTraceConfig,
    *,
    limit: int = 10,
    worker_id: str | None = None,
    state_store: StateStore | None = None,
    executor: TableExecutor | None = None,
) -> TableJobBatchReport:
    if state_store is None:
        from littrace.state_db import state_store_from_config

        state_store = state_store_from_config(config)
    executor = executor or _execute_table_job
    report = TableJobBatchReport()
    owner = worker_id or f"table:{socket.gethostname()}:{uuid4().hex[:12]}"
    jobs = state_store.claim_pending_async_tasks(
        worker_id=owner,
        kind="table_job",
        limit=max(1, limit),
        lease_seconds=max(config.download_retry.interval_seconds * 8, 600.0),
    )
    if not jobs:
        report.finished_at = datetime.now(UTC).isoformat()
        return report

    for job in jobs:
        report.job_ids.append(job.task_id)
        try:
            paper_ids, parsed_sha256 = _table_job_input(job)
            snapshot, stale_before_execution = _load_table_snapshot(
                state_store,
                job.session_id,
                paper_ids,
                parsed_sha256,
            )
            if len(stale_before_execution) == len(paper_ids):
                output = TableExecutionOutput(parsed_sha256=parsed_sha256)
            else:
                output = await executor(config, snapshot)
                output.parsed_sha256 = parsed_sha256
            stale_ids = _commit_table_output(state_store, job, output, paper_ids)
            report.processed += 1
            report.performance_cells += len(output.performance_cells)
            report.structured_artifacts += len(output.structured_artifacts)
            report.stale += len(stale_ids)
        except Exception as exc:  # noqa: BLE001 - queue boundary persists failures
            _mark_table_job_failed(state_store, job, exc, config)
            report.failed += 1
            report.warnings.append(
                f"table_job:{job.task_id}:{exc.__class__.__name__}: {exc}"
            )
    report.finished_at = datetime.now(UTC).isoformat()
    return report


async def run_table_job_daemon(
    config: LitTraceConfig,
    *,
    interval_seconds: float | None = None,
    limit: int | None = None,
) -> None:
    interval = max(
        interval_seconds
        if interval_seconds is not None
        else config.download_retry.interval_seconds,
        0.1,
    )
    batch_size = limit or config.download_retry.batch_size
    while True:
        await run_pending_table_jobs(config, limit=batch_size)
        await asyncio.sleep(interval)


def table_jobs_status(
    config: LitTraceConfig,
    *,
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> tuple[AsyncTaskQueueReport, list[AsyncTaskRecord]]:
    from littrace.state_db import state_store_from_config

    store = state_store_from_config(config)
    return (
        store.async_tasks_queue_report(kind="table_job"),
        store.list_async_tasks(
            kind="table_job",
            session_id=session_id,
            status=status,
            limit=limit,
        ),
    )


def requeue_dead_table_jobs(config: LitTraceConfig, *, limit: int = 20) -> int:
    from littrace.state_db import state_store_from_config

    return state_store_from_config(config).requeue_dead_async_tasks(
        kind="table_job",
        limit=limit,
    )


def _table_job_input(job: AsyncTaskRecord) -> tuple[list[str], dict[str, str]]:
    payload = job.result_json if isinstance(job.result_json, dict) else {}
    command = payload.get("command")
    if not isinstance(command, dict):
        raise TypeError("table job is missing its command payload")
    raw_ids = command.get("paper_ids")
    raw_hashes = command.get("parsed_sha256")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("table job paper_ids must be a non-empty array")
    if not isinstance(raw_hashes, dict):
        raise TypeError("table job parsed_sha256 must be an object")
    paper_ids = list(dict.fromkeys(str(item) for item in raw_ids if str(item)))
    parsed_sha256 = {
        paper_id: str(raw_hashes.get(paper_id) or "") for paper_id in paper_ids
    }
    missing = [paper_id for paper_id, digest in parsed_sha256.items() if not digest]
    if missing:
        raise ValueError("table job parsed hashes are missing: " + ", ".join(missing))
    return paper_ids, parsed_sha256


def _load_table_snapshot(
    state_store: StateStore,
    session_id: str,
    paper_ids: list[str],
    parsed_sha256: dict[str, str],
) -> tuple[LiteratureWorkspace, list[str]]:
    state = state_store.get_session_state(session_id)
    if state is None:
        raise LookupError(f"LitTrace session state does not exist: {session_id}")
    current = LiteratureWorkspace.model_validate(state.workspace_json)
    stale_ids = _stale_parsed_ids(current, paper_ids, parsed_sha256)
    snapshot = LiteratureWorkspace(
        papers={
            paper_id: current.papers[paper_id]
            for paper_id in paper_ids
            if paper_id in current.papers and paper_id not in stale_ids
        },
        parsed_papers={
            paper_id: coerce_parsed(current.parsed_papers[paper_id])
            for paper_id in paper_ids
            if paper_id in current.parsed_papers and paper_id not in stale_ids
        },
    )
    snapshot.context.active_papers = [
        paper_id for paper_id in paper_ids if paper_id not in stale_ids
    ]
    return snapshot, stale_ids


async def _execute_table_job(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
) -> TableExecutionOutput:
    updated, harness = await extract_performance_cells(workspace, config)
    return TableExecutionOutput(
        performance_cells=updated.performance_cells,
        structured_artifacts=list(updated.context.filters.structured_artifacts),
        harness=(
            harness.model_dump(mode="json")
            if hasattr(harness, "model_dump")
            else {"summary": str(harness)}
        ),
    )


def _commit_table_output(
    state_store: StateStore,
    job: AsyncTaskRecord,
    output: TableExecutionOutput,
    paper_ids: list[str],
    *,
    max_cas_attempts: int = 5,
) -> list[str]:
    if not job.lease_owner:
        raise RuntimeError("table job lost its queue lease before commit")
    stale_ids: list[str] = []
    for _attempt in range(max_cas_attempts):
        state = state_store.get_session_state(job.session_id)
        if state is None:
            raise LookupError(f"LitTrace session state does not exist: {job.session_id}")
        workspace = LiteratureWorkspace.model_validate(state.workspace_json)
        stale_ids = _stale_parsed_ids(workspace, paper_ids, output.parsed_sha256)
        valid_ids = set(paper_ids) - set(stale_ids)
        workspace.performance_cells = [
            cell for cell in workspace.performance_cells if cell.paper_id not in valid_ids
        ] + [cell for cell in output.performance_cells if cell.paper_id in valid_ids]
        existing_artifacts = [
            item
            for item in workspace.context.filters.structured_artifacts
            if not isinstance(item, dict) or str(item.get("paper_id") or "") not in valid_ids
        ]
        workspace.context.filters.structured_artifacts = [
            *existing_artifacts,
            *[
                item
                for item in output.structured_artifacts
                if str(item.get("paper_id") or "") in valid_ids
            ],
        ]
        workspace.context.filters.workspace_revision = state.revision + 1
        result_json = dict(job.result_json) if isinstance(job.result_json, dict) else {}
        result_json["execution"] = {
            "performance_cell_count": sum(
                cell.paper_id in valid_ids for cell in output.performance_cells
            ),
            "structured_artifact_count": sum(
                str(item.get("paper_id") or "") in valid_ids
                for item in output.structured_artifacts
            ),
            "harness": output.harness,
            "merged_paper_ids": sorted(valid_ids),
            "stale_paper_ids": stale_ids,
            "committed_revision": state.revision + 1,
        }
        workspace_json = workspace.model_dump(mode="json")
        canonical_json = json.dumps(
            workspace_json,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            state_store.commit_async_workspace_result(
                session_id=job.session_id,
                task_id=job.task_id,
                lease_owner=job.lease_owner,
                expected_revision=state.revision,
                workspace_json=workspace_json,
                workspace_sha256=sha256(canonical_json.encode()).hexdigest(),
                result_json=result_json,
                audit_event={
                    "type": "async_workspace_job_committed",
                    "task_id": job.task_id,
                    "kind": "table_job",
                    "merged_paper_ids": sorted(valid_ids),
                    "stale_paper_ids": stale_ids,
                    "expected_revision": state.revision,
                    "committed_revision": state.revision + 1,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            return stale_ids
        except RuntimeError as exc:
            if "CAS mismatch" not in str(exc):
                raise
    raise RuntimeError(
        f"SessionState CAS mismatch persisted after {max_cas_attempts} merge attempts"
    )


def _stale_parsed_ids(
    workspace: LiteratureWorkspace,
    paper_ids: list[str],
    parsed_sha256: dict[str, str],
) -> list[str]:
    active_ids = set(workspace.context.active_papers)
    stale: list[str] = []
    for paper_id in paper_ids:
        parsed = coerce_parsed(workspace.parsed_papers.get(paper_id))
        if (
            paper_id not in active_ids
            or paper_id not in workspace.papers
            or not parsed.parsed
            or _model_sha256(parsed) != parsed_sha256.get(paper_id)
        ):
            stale.append(paper_id)
    return stale


def _model_sha256(value: object) -> str:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _mark_table_job_failed(
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
