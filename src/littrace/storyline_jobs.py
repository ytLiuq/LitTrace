"""Durable evidence-grounded storyline worker."""

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
from littrace.evidence.storyline import build_storyline_from_workspace
from littrace.models import (
    Claim,
    ClaimKind,
    LiteratureWorkspace,
    StorylineClaim,
    coerce_parsed,
)
from littrace.state_db import AsyncTaskQueueReport, AsyncTaskRecord, StateStore


class StorylineExecutionOutput(BaseModel):
    storyline_claims: list[StorylineClaim] = Field(default_factory=list)
    source_sha256: dict[str, str] = Field(default_factory=dict)


class StorylineJobBatchReport(BaseModel):
    schema_version: str = "littrace.storyline_job_batch_report.v1"
    processed: int = 0
    failed: int = 0
    claims: int = 0
    evidence_records: int = 0
    stale: int = 0
    job_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None


StorylineExecutor = Callable[
    [LitTraceConfig, LiteratureWorkspace],
    Awaitable[StorylineExecutionOutput],
]


async def run_pending_storyline_jobs(
    config: LitTraceConfig,
    *,
    limit: int = 10,
    worker_id: str | None = None,
    state_store: StateStore | None = None,
    executor: StorylineExecutor | None = None,
) -> StorylineJobBatchReport:
    if state_store is None:
        from littrace.state_db import state_store_from_config

        state_store = state_store_from_config(config)
    executor = executor or _execute_storyline_job
    report = StorylineJobBatchReport()
    owner = worker_id or f"storyline:{socket.gethostname()}:{uuid4().hex[:12]}"
    jobs = state_store.claim_pending_async_tasks(
        worker_id=owner,
        kind="storyline_job",
        limit=max(1, limit),
        lease_seconds=max(config.download_retry.interval_seconds * 4, 300.0),
    )
    if not jobs:
        report.finished_at = datetime.now(UTC).isoformat()
        return report

    for job in jobs:
        report.job_ids.append(job.task_id)
        try:
            paper_ids, source_sha256 = _storyline_job_input(job)
            snapshot, stale_before_execution = _load_storyline_snapshot(
                state_store,
                job.session_id,
                paper_ids,
                source_sha256,
            )
            if len(stale_before_execution) == len(paper_ids):
                output = StorylineExecutionOutput(source_sha256=source_sha256)
            else:
                output = await executor(config, snapshot)
                output.source_sha256 = source_sha256
            stale_ids, claim_count, evidence_count = _commit_storyline_output(
                state_store,
                job,
                output,
                paper_ids,
            )
            report.processed += 1
            report.claims += claim_count
            report.evidence_records += evidence_count
            report.stale += len(stale_ids)
        except Exception as exc:  # noqa: BLE001 - queue boundary persists failures
            _mark_storyline_job_failed(state_store, job, exc, config)
            report.failed += 1
            report.warnings.append(
                f"storyline_job:{job.task_id}:{exc.__class__.__name__}: {exc}"
            )
    report.finished_at = datetime.now(UTC).isoformat()
    return report


async def run_storyline_job_daemon(
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
        await run_pending_storyline_jobs(config, limit=batch_size)
        await asyncio.sleep(interval)


def storyline_jobs_status(
    config: LitTraceConfig,
    *,
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> tuple[AsyncTaskQueueReport, list[AsyncTaskRecord]]:
    from littrace.state_db import state_store_from_config

    store = state_store_from_config(config)
    return (
        store.async_tasks_queue_report(kind="storyline_job"),
        store.list_async_tasks(
            kind="storyline_job",
            session_id=session_id,
            status=status,
            limit=limit,
        ),
    )


def requeue_dead_storyline_jobs(config: LitTraceConfig, *, limit: int = 20) -> int:
    from littrace.state_db import state_store_from_config

    return state_store_from_config(config).requeue_dead_async_tasks(
        kind="storyline_job",
        limit=limit,
    )


def _storyline_job_input(job: AsyncTaskRecord) -> tuple[list[str], dict[str, str]]:
    payload = job.result_json if isinstance(job.result_json, dict) else {}
    command = payload.get("command")
    if not isinstance(command, dict):
        raise TypeError("storyline job is missing its command payload")
    raw_ids = command.get("paper_ids")
    raw_hashes = command.get("source_sha256")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("storyline job paper_ids must be a non-empty array")
    if not isinstance(raw_hashes, dict):
        raise TypeError("storyline job source_sha256 must be an object")
    paper_ids = list(dict.fromkeys(str(item) for item in raw_ids if str(item)))
    source_sha256 = {
        paper_id: str(raw_hashes.get(paper_id) or "") for paper_id in paper_ids
    }
    missing = [paper_id for paper_id, digest in source_sha256.items() if not digest]
    if missing:
        raise ValueError("storyline source hashes are missing: " + ", ".join(missing))
    return paper_ids, source_sha256


def _load_storyline_snapshot(
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
    snapshot = LiteratureWorkspace(
        papers={paper_id: current.papers[paper_id] for paper_id in valid_ids},
        parsed_papers={
            paper_id: coerce_parsed(current.parsed_papers[paper_id])
            for paper_id in valid_ids
        },
        performance_cells=[
            cell for cell in current.performance_cells if cell.paper_id in valid_ids
        ],
    )
    snapshot.context.active_papers = valid_ids
    return snapshot, stale_ids


async def _execute_storyline_job(
    _config: LitTraceConfig,
    workspace: LiteratureWorkspace,
) -> StorylineExecutionOutput:
    claims = await asyncio.to_thread(build_storyline_from_workspace, workspace)
    return StorylineExecutionOutput(storyline_claims=claims)


def _commit_storyline_output(
    state_store: StateStore,
    job: AsyncTaskRecord,
    output: StorylineExecutionOutput,
    paper_ids: list[str],
    *,
    max_cas_attempts: int = 5,
) -> tuple[list[str], int, int]:
    if not job.lease_owner:
        raise RuntimeError("storyline job lost its queue lease before commit")
    stale_ids: list[str] = []
    for _attempt in range(max_cas_attempts):
        state = state_store.get_session_state(job.session_id)
        if state is None:
            raise LookupError(f"LitTrace session state does not exist: {job.session_id}")
        workspace = LiteratureWorkspace.model_validate(state.workspace_json)
        stale_ids = _stale_source_ids(workspace, paper_ids, output.source_sha256)
        valid_ids = set(paper_ids) - set(stale_ids)
        prior_evidence = dict(workspace.evidence_records)
        workspace.claims = [
            claim
            for claim in workspace.claims
            if not _is_replaced_storyline_claim(claim, prior_evidence, valid_ids)
        ]
        canonical_claims: list[Claim] = []
        added_evidence_ids: set[str] = set()
        for storyline_claim in output.storyline_claims:
            if not storyline_claim.evidence or any(
                span.paper_id not in valid_ids for span in storyline_claim.evidence
            ):
                continue
            evidence_ids: list[str] = []
            support_quotes: dict[str, str] = {}
            for span in storyline_claim.evidence:
                assert span.evidence_id is not None
                workspace.evidence_records[span.evidence_id] = span
                evidence_ids.append(span.evidence_id)
                added_evidence_ids.add(span.evidence_id)
                if span.snippet:
                    support_quotes[span.evidence_id] = span.snippet
            canonical_claims.append(
                Claim(
                    text=storyline_claim.claim,
                    claim_kind=(
                        ClaimKind.CAUSAL
                        if storyline_claim.claim_type
                        in {"later_response", "solution_limit_response_chain"}
                        else ClaimKind.QUALITATIVE
                    ),
                    evidence_ids=evidence_ids,
                    support_quotes=support_quotes,
                    claim_origin="storyline",
                )
            )
        workspace.claims.extend(canonical_claims)
        workspace.context.filters.storyline_claim_count = sum(
            claim.claim_origin == "storyline" for claim in workspace.claims
        )
        workspace.context.filters.workspace_revision = state.revision + 1
        result_json = dict(job.result_json) if isinstance(job.result_json, dict) else {}
        result_json["execution"] = {
            "storyline_claim_count": len(canonical_claims),
            "evidence_record_count": len(added_evidence_ids),
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
                    "kind": "storyline_job",
                    "merged_paper_ids": sorted(valid_ids),
                    "stale_paper_ids": stale_ids,
                    "expected_revision": state.revision,
                    "committed_revision": state.revision + 1,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            return stale_ids, len(canonical_claims), len(added_evidence_ids)
        except RuntimeError as exc:
            if "CAS mismatch" not in str(exc):
                raise
    raise RuntimeError(
        f"SessionState CAS mismatch persisted after {max_cas_attempts} merge attempts"
    )


def _is_replaced_storyline_claim(
    claim: Claim,
    evidence_records: dict[str, object],
    valid_ids: set[str],
) -> bool:
    if claim.claim_origin != "storyline":
        return False
    return any(
        getattr(evidence_records.get(evidence_id), "paper_id", None) in valid_ids
        for evidence_id in claim.evidence_ids
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


def _mark_storyline_job_failed(
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
