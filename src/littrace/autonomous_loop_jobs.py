"""Durable autonomous-review worker.

Round 20. Mirrors :mod:`littrace.storyline_jobs` /
:mod:`littrace.document_jobs` so the Codex MCP command
``enqueue_autonomous_review`` can hand work to the durable Postgres
queue. The reviewer loop is ``littrace.autonomous_loop.run_review_loop``
which is async and LLM-backed — the worker simply awaits it.

Special note: ``auto_replan=True`` causes ``run_review_loop`` to
in-place mutate the workspace (``parse_workspace_skill`` /
``extract_tables_skill`` / ``build_storyline_skill`` via
``_execute_safe_replan_actions``). We accept this and rely on the
existing CAS-merge pattern: the worker lease is held for the full
review (300 s default), and ``_commit_autonomous_review_output``
re-reads canonical workspace on every CAS attempt before writing back.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from pydantic import BaseModel, Field

from littrace.autonomous_loop import run_review_loop
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace, ReviewLoopReport, coerce_parsed
from littrace.state_db import AsyncTaskQueueReport, AsyncTaskRecord, StateStore


class AutonomousReviewExecutionOutput(BaseModel):
    report: ReviewLoopReport | None = None
    auto_replan: bool = False
    source_sha256: dict[str, str] = Field(default_factory=dict)


class AutonomousReviewJobBatchReport(BaseModel):
    schema_version: str = "littrace.autonomous_review_job_batch_report.v1"
    processed: int = 0
    failed: int = 0
    rounds: int = 0
    score: float = 0.0
    passed: bool = False
    release_ready: bool = False
    stale: int = 0
    job_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None


AutonomousReviewExecutor = Callable[
    [LitTraceConfig, LiteratureWorkspace, bool],
    Awaitable[AutonomousReviewExecutionOutput],
]


async def run_pending_autonomous_review_jobs(
    config: LitTraceConfig,
    *,
    limit: int = 10,
    worker_id: str | None = None,
    state_store: StateStore | None = None,
    executor: AutonomousReviewExecutor | None = None,
) -> AutonomousReviewJobBatchReport:
    if state_store is None:
        from littrace.state_db import state_store_from_config

        state_store = state_store_from_config(config)
    executor = executor or _execute_autonomous_review_job
    report = AutonomousReviewJobBatchReport()
    owner = worker_id or f"autonomous_review:{socket.gethostname()}:{uuid4().hex[:12]}"
    jobs = state_store.claim_pending_async_tasks(
        worker_id=owner,
        kind="autonomous_review_job",
        limit=max(1, limit),
        lease_seconds=max(config.download_retry.interval_seconds * 4, 300.0),
    )
    if not jobs:
        report.finished_at = datetime.now(UTC).isoformat()
        return report

    for job in jobs:
        report.job_ids.append(job.task_id)
        try:
            command = _autonomous_review_job_input(job)
            paper_ids = command["paper_ids"]
            source_sha256 = command["source_sha256"]
            auto_replan = command["auto_replan"]
            snapshot, stale_before_execution = _load_autonomous_review_snapshot(
                state_store,
                job.session_id,
                paper_ids,
                source_sha256,
            )
            if len(stale_before_execution) == len(paper_ids):
                output = AutonomousReviewExecutionOutput(
                    auto_replan=auto_replan,
                    source_sha256=source_sha256,
                )
            else:
                output = await executor(config, snapshot, auto_replan)
                output.auto_replan = auto_replan
                output.source_sha256 = source_sha256
            (
                stale_ids,
                rounds,
                score,
                passed,
                release_ready,
            ) = _commit_autonomous_review_output(
                state_store,
                job,
                output,
                paper_ids,
            )
            report.processed += 1
            report.rounds = rounds
            report.score = score
            report.passed = passed
            report.release_ready = release_ready
            report.stale += len(stale_ids)
        except Exception as exc:  # noqa: BLE001 - queue boundary persists failures
            _mark_autonomous_review_job_failed(state_store, job, exc, config)
            report.failed += 1
            report.warnings.append(
                f"autonomous_review_job:{job.task_id}:{exc.__class__.__name__}: {exc}"
            )
    report.finished_at = datetime.now(UTC).isoformat()
    return report


async def run_autonomous_review_job_daemon(
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
        await run_pending_autonomous_review_jobs(config, limit=batch_size)
        await asyncio.sleep(interval)


def autonomous_review_jobs_status(
    config: LitTraceConfig,
    *,
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> tuple[AsyncTaskQueueReport, list[AsyncTaskRecord]]:
    from littrace.state_db import state_store_from_config

    store = state_store_from_config(config)
    return (
        store.async_tasks_queue_report(kind="autonomous_review_job"),
        store.list_async_tasks(
            kind="autonomous_review_job",
            session_id=session_id,
            status=status,
            limit=limit,
        ),
    )


def requeue_dead_autonomous_review_jobs(
    config: LitTraceConfig, *, limit: int = 20
) -> int:
    from littrace.state_db import state_store_from_config

    return state_store_from_config(config).requeue_dead_async_tasks(
        kind="autonomous_review_job",
        limit=limit,
    )


def _autonomous_review_job_input(job: AsyncTaskRecord) -> dict[str, object]:
    payload = job.result_json if isinstance(job.result_json, dict) else {}
    command = payload.get("command")
    if not isinstance(command, dict):
        raise TypeError("autonomous_review job is missing its command payload")
    raw_ids = command.get("paper_ids")
    raw_hashes = command.get("source_sha256")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("autonomous_review job paper_ids must be a non-empty array")
    if not isinstance(raw_hashes, dict):
        raise TypeError("autonomous_review job source_sha256 must be an object")
    paper_ids = list(dict.fromkeys(str(item) for item in raw_ids if str(item)))
    source_sha256 = {
        paper_id: str(raw_hashes.get(paper_id) or "") for paper_id in paper_ids
    }
    missing = [paper_id for paper_id, digest in source_sha256.items() if not digest]
    if missing:
        raise ValueError(
            "autonomous_review source hashes are missing: " + ", ".join(missing)
        )
    auto_replan_raw = command.get("auto_replan", False)
    if not isinstance(auto_replan_raw, bool):
        raise TypeError("auto_replan must be a boolean")
    return {
        "paper_ids": paper_ids,
        "source_sha256": source_sha256,
        "auto_replan": auto_replan_raw,
    }


def _load_autonomous_review_snapshot(
    state_store: StateStore,
    session_id: str,
    paper_ids: list[str],
    source_sha256: dict[str, str],
) -> tuple[LiteratureWorkspace, list[str]]:
    state = state_store.get_session_state(session_id)
    if state is None:
        raise LookupError(f"LitTrace session state does not exist: {session_id}")
    current = LiteratureWorkspace.model_validate(state.workspace_json)
    stale_ids = _stale_source_ids(current, paper_ids, source_sha256)
    valid_ids = [paper_id for paper_id in paper_ids if paper_id not in stale_ids]
    snapshot = current.model_copy(deep=True)
    snapshot.papers = {paper_id: current.papers[paper_id] for paper_id in valid_ids}
    snapshot.parsed_papers = {
        paper_id: coerce_parsed(current.parsed_papers[paper_id])
        for paper_id in valid_ids
    }
    snapshot.context.active_papers = valid_ids
    return snapshot, stale_ids


async def _execute_autonomous_review_job(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    auto_replan: bool,
) -> AutonomousReviewExecutionOutput:
    objective = (
        getattr(workspace.context.filters, "topic", None)
        or "autonomous review"
    )
    report = await run_review_loop(
        config,
        str(objective),
        workspace,
        auto_replan=auto_replan,
    )
    return AutonomousReviewExecutionOutput(report=report, auto_replan=auto_replan)


def _commit_autonomous_review_output(
    state_store: StateStore,
    job: AsyncTaskRecord,
    output: AutonomousReviewExecutionOutput,
    paper_ids: list[str],
    *,
    max_cas_attempts: int = 5,
) -> tuple[list[str], int, float, bool, bool]:
    if not job.lease_owner:
        raise RuntimeError("autonomous_review job lost its queue lease before commit")
    stale_ids: list[str] = []
    for _attempt in range(max_cas_attempts):
        state = state_store.get_session_state(job.session_id)
        if state is None:
            raise LookupError(f"LitTrace session state does not exist: {job.session_id}")
        workspace = LiteratureWorkspace.model_validate(state.workspace_json)
        stale_ids = _stale_source_ids(workspace, paper_ids, output.source_sha256)
        if output.report is None:
            rounds = 0
            score = 0.0
            passed = False
            release_ready = False
        else:
            workspace.context.filters.autonomous_loop_report = (
                output.report.model_dump(mode="json")
            )
            rounds = len(output.report.rounds)
            score = output.report.score
            passed = output.report.passed
            release_ready = output.report.release_ready
        workspace.context.filters.workspace_revision = state.revision + 1
        result_json = dict(job.result_json) if isinstance(job.result_json, dict) else {}
        result_json["execution"] = {
            "rounds": rounds,
            "score": score,
            "passed": passed,
            "release_ready": release_ready,
            "auto_replan": output.auto_replan,
            "merged_paper_ids": sorted(set(paper_ids) - set(stale_ids)),
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
                    "kind": "autonomous_review_job",
                    "auto_replan": output.auto_replan,
                    "rounds": rounds,
                    "score": score,
                    "passed": passed,
                    "release_ready": release_ready,
                    "merged_paper_ids": sorted(set(paper_ids) - set(stale_ids)),
                    "stale_paper_ids": stale_ids,
                    "expected_revision": state.revision,
                    "committed_revision": state.revision + 1,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            return stale_ids, rounds, score, passed, release_ready
        except RuntimeError as exc:
            if "CAS mismatch" not in str(exc):
                raise
    raise RuntimeError(
        f"SessionState CAS mismatch persisted after {max_cas_attempts} merge attempts"
    )


def _stale_source_ids(
    workspace: LiteratureWorkspace,
    paper_ids: list[str],
    source_sha256: dict[str, str],
) -> list[str]:
    active_ids = set(workspace.context.active_papers)
    stale: list[str] = []
    for paper_id in paper_ids:
        parsed = coerce_parsed(workspace.parsed_papers.get(paper_id))
        if (
            paper_id not in active_ids
            or paper_id not in workspace.papers
            or not parsed.parsed
            or _source_sha256(workspace, paper_id) != source_sha256.get(paper_id)
        ):
            stale.append(paper_id)
    return stale


def _source_sha256(workspace: LiteratureWorkspace, paper_id: str) -> str:
    payload = {
        "paper": workspace.papers[paper_id].model_dump(mode="json"),
        "parsed": coerce_parsed(workspace.parsed_papers[paper_id]).model_dump(mode="json"),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _mark_autonomous_review_job_failed(
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
