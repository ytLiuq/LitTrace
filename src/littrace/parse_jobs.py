"""Durable PDF parsing command worker.

The worker reads immutable PDF artifact references from a queued command,
materializes bytes from object storage for Docling/OCR, and atomically merges
the parsed result into the current Postgres workspace under CAS.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from pydantic import BaseModel, Field

from littrace.artifact_registry import (
    ArtifactRecord,
    artifact_registry_from_config,
)
from littrace.artifact_store import (
    ArtifactKeyContext,
    BlobRef,
    artifact_store_from_config,
    build_artifact_object_key,
)
from littrace.config import LitTraceConfig
from littrace.evidence.parsing import parse_workspace_papers
from littrace.models import LiteratureWorkspace, PaperMetadata, ParsedPaper
from littrace.state_db import (
    AsyncTaskQueueReport,
    AsyncTaskRecord,
    StateStore,
    state_store_from_config,
)


class ParseExecutionOutput(BaseModel):
    parsed_papers: dict[str, ParsedPaper] = Field(default_factory=dict)
    docling_quality_reports: dict[str, dict[str, object]] = Field(default_factory=dict)
    source_sha256: dict[str, str] = Field(default_factory=dict)
    report: dict[str, object] = Field(default_factory=dict)


class ParseJobBatchReport(BaseModel):
    schema_version: str = "littrace.parse_job_batch_report.v1"
    processed: int = 0
    failed: int = 0
    parsed: int = 0
    parse_failed: int = 0
    stale: int = 0
    job_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None


ParseExecutor = Callable[
    [LitTraceConfig, str, list[PaperMetadata], list[ArtifactRecord], str],
    Awaitable[ParseExecutionOutput],
]
SourceShaLookup = Callable[[str, str], str | None]


def enqueue_parse_job(
    config: LitTraceConfig,
    session,
    workspace: LiteratureWorkspace,
    paper_ids: list[str],
    *,
    parse_strategy: str = "auto",
) -> AsyncTaskRecord:
    """Create a durable parse task for already-stored PDF artifacts.

    The task snapshots both metadata and immutable artifact records so the
    worker can materialize PDFs from object storage without relying on the
    user's local paper directory.
    """
    ids = list(dict.fromkeys(paper_ids))
    if not ids:
        raise ValueError("paper_ids must not be empty")
    registry = artifact_registry_from_config(config)
    papers = []
    sources = []
    for paper_id in ids:
        paper = workspace.papers.get(paper_id)
        if paper is None:
            raise ValueError(f"Unknown paper for parse job: {paper_id}")
        source = registry.find_in_session(
            f"paper_pdf:{paper_id}",
            session_id=session.session_id,
        )
        if source is None or not source.sha256:
            raise ValueError(f"PDF artifact is not registered: {paper_id}")
        papers.append(paper.model_dump(mode="json"))
        sources.append(source.model_dump(mode="json"))
    now = datetime.now(UTC).isoformat()
    source_fingerprint = sorted(
        (str(source.get("paper_id") or ""), str(source.get("sha256") or ""))
        for source in sources
    )
    digest = sha256(
        json.dumps(
            {
                "session_id": session.session_id,
                "paper_ids": ids,
                "strategy": parse_strategy,
                "sources": source_fingerprint,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:32]
    task = AsyncTaskRecord(
        task_id=f"parse:{digest}",
        session_id=session.session_id,
        kind="parse_job",
        artifact_id=f"parse_batch:{digest}",
        event_type="parse_requested",
        source_revision=str(workspace.context.filters.workspace_revision),
        result_json={
            "schema_version": "littrace.parse_job.v1",
            "command": {
                "paper_ids": ids,
                "parse_strategy": parse_strategy,
                "papers": papers,
                "sources": sources,
            },
        },
        created_at=now,
        updated_at=now,
    )
    state_store_from_config(config).enqueue_async_task(task)
    return task


async def run_pending_parse_jobs(
    config: LitTraceConfig,
    *,
    limit: int = 10,
    worker_id: str | None = None,
    session_id: str | None = None,
    state_store: StateStore | None = None,
    executor: ParseExecutor | None = None,
    source_sha_lookup: SourceShaLookup | None = None,
) -> ParseJobBatchReport:
    """Claim, execute, and atomically merge one batch of ``parse_job`` rows."""

    if state_store is None:
        from littrace.state_db import state_store_from_config

        state_store = state_store_from_config(config)
    executor = executor or _execute_parse_job
    if source_sha_lookup is None:
        registry = artifact_registry_from_config(config)

        def source_sha_lookup(session_id: str, paper_id: str) -> str | None:
            record = registry.find_in_session(
                f"paper_pdf:{paper_id}",
                session_id=session_id,
            )
            return record.sha256 if record is not None else None

    report = ParseJobBatchReport()
    owner = worker_id or f"parse:{socket.gethostname()}:{uuid4().hex[:12]}"
    claim_kwargs = {
        "worker_id": owner,
        "kind": "parse_job",
        "limit": max(1, limit),
        "lease_seconds": max(config.download_retry.interval_seconds * 8, 600.0),
    }
    if session_id is not None:
        claim_kwargs["session_id"] = session_id
    jobs = state_store.claim_pending_async_tasks(**claim_kwargs)
    if not jobs:
        report.finished_at = datetime.now(UTC).isoformat()
        return report

    for job in jobs:
        report.job_ids.append(job.task_id)
        try:
            papers, sources, strategy = _parse_job_input(job)
            output = await executor(config, job.session_id, papers, sources, strategy)
            stale_ids = _commit_parse_output(
                state_store,
                job,
                output,
                source_sha_lookup=source_sha_lookup,
            )
            report.processed += 1
            report.parsed += int(output.report.get("parsed_count") or 0)
            report.parse_failed += int(output.report.get("failed_count") or 0)
            report.stale += len(stale_ids)
        except Exception as exc:  # noqa: BLE001 - queue boundary persists failures
            _mark_parse_job_failed(state_store, job, exc, config)
            report.failed += 1
            report.warnings.append(
                f"parse_job:{job.task_id}:{exc.__class__.__name__}: {exc}"
            )
    report.finished_at = datetime.now(UTC).isoformat()
    return report


async def run_parse_job_daemon(
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
        await run_pending_parse_jobs(config, limit=batch_size)
        await asyncio.sleep(interval)


def parse_jobs_status(
    config: LitTraceConfig,
    *,
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> tuple[AsyncTaskQueueReport, list[AsyncTaskRecord]]:
    from littrace.state_db import state_store_from_config

    store = state_store_from_config(config)
    return (
        store.async_tasks_queue_report(kind="parse_job"),
        store.list_async_tasks(
            kind="parse_job",
            session_id=session_id,
            status=status,
            limit=limit,
        ),
    )


def requeue_dead_parse_jobs(config: LitTraceConfig, *, limit: int = 20) -> int:
    from littrace.state_db import state_store_from_config

    return state_store_from_config(config).requeue_dead_async_tasks(
        kind="parse_job",
        limit=limit,
    )


def _parse_job_input(
    job: AsyncTaskRecord,
) -> tuple[list[PaperMetadata], list[ArtifactRecord], str]:
    payload = job.result_json if isinstance(job.result_json, dict) else {}
    command = payload.get("command")
    if not isinstance(command, dict):
        raise TypeError("parse job is missing its command payload")
    raw_ids = command.get("paper_ids")
    raw_papers = command.get("papers")
    raw_sources = command.get("sources")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("parse job paper_ids must be a non-empty array")
    if not isinstance(raw_papers, list):
        raise TypeError("parse job papers snapshot must be an array")
    if not isinstance(raw_sources, list):
        raise TypeError("parse job sources snapshot must be an array")
    paper_ids = list(dict.fromkeys(str(item) for item in raw_ids if str(item)))
    by_id = {
        paper.paper_id: paper
        for item in raw_papers
        if (paper := PaperMetadata.model_validate(item)).paper_id in paper_ids
    }
    source_by_paper = {
        source.paper_id: source
        for item in raw_sources
        if (source := ArtifactRecord.model_validate(item)).paper_id in paper_ids
    }
    missing_papers = [paper_id for paper_id in paper_ids if paper_id not in by_id]
    missing_sources = [paper_id for paper_id in paper_ids if paper_id not in source_by_paper]
    if missing_papers:
        raise ValueError("parse job papers snapshot is missing: " + ", ".join(missing_papers))
    if missing_sources:
        raise ValueError("parse job sources snapshot is missing: " + ", ".join(missing_sources))
    strategy = str(command.get("parse_strategy") or "auto")
    if strategy not in {"auto", "text_only", "ocr"}:
        raise ValueError("parse job has an invalid parse_strategy")
    return (
        [by_id[paper_id] for paper_id in paper_ids],
        [source_by_paper[paper_id] for paper_id in paper_ids],
        strategy,
    )


async def _execute_parse_job(
    config: LitTraceConfig,
    session_id: str,
    papers: list[PaperMetadata],
    sources: list[ArtifactRecord],
    strategy: str,
) -> ParseExecutionOutput:
    store = artifact_store_from_config(config)
    source_by_paper = {source.paper_id: source for source in sources}
    with TemporaryDirectory(prefix="littrace-parse-") as temporary:
        parse_config = config.model_copy(deep=True)
        parse_config.parsing.parse_strategy = strategy
        parse_config.storage.paper_library_dir = Path(temporary) / "papers"
        workspace = LiteratureWorkspace(papers={paper.paper_id: paper for paper in papers})
        workspace.context.active_papers = [paper.paper_id for paper in papers]
        source_sha256: dict[str, str] = {}
        from littrace.access_layer.paths import target_pdf_path

        for paper in papers:
            source = source_by_paper.get(paper.paper_id)
            if source is None or not source.sha256:
                raise ValueError(f"Missing immutable PDF source for {paper.paper_id}")
            data = store.get_bytes(
                BlobRef(
                    backend=source.backend,
                    bucket=source.bucket,
                    object_key=source.object_key,
                    sha256=source.sha256,
                    size_bytes=source.size_bytes,
                    content_type=source.content_type,
                )
            )
            actual_sha = sha256(data).hexdigest()
            if actual_sha != source.sha256:
                raise ValueError(
                    f"PDF artifact checksum mismatch for {paper.paper_id}: "
                    f"expected {source.sha256}, got {actual_sha}"
                )
            target = target_pdf_path(parse_config, paper)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            source_sha256[paper.paper_id] = actual_sha

        parsed_workspace, parse_report = await asyncio.to_thread(
            parse_workspace_papers,
            workspace,
            parse_config,
        )
        _store_figure_artifacts(config, session_id, parsed_workspace)
        for parsed in parsed_workspace.parsed_papers.values():
            parsed.pdf_path = None
        return ParseExecutionOutput(
            parsed_papers=parsed_workspace.parsed_papers,
            docling_quality_reports=(
                parsed_workspace.context.filters.docling_quality_reports
            ),
            source_sha256=source_sha256,
            report=parse_report,
        )


def _store_figure_artifacts(
    config: LitTraceConfig,
    session_id: str,
    workspace: LiteratureWorkspace,
) -> None:
    store = artifact_store_from_config(config)
    registry = artifact_registry_from_config(config)
    for paper_id, parsed in workspace.parsed_papers.items():
        stored_refs: list[dict[str, object]] = []
        for index, figure in enumerate(parsed.figures, start=1):
            if not isinstance(figure, dict):
                continue
            path_value = figure.get("asset_path")
            path = Path(str(path_value)) if path_value else None
            if path is None or not path.is_file():
                continue
            figure_id = str(figure.get("figure_id") or f"figure-{index}")
            artifact_id = f"figure_image:{paper_id}:{figure_id}"
            object_key = build_artifact_object_key(
                config,
                ArtifactKeyContext(
                    session_id=session_id,
                    kind="figure_image",
                    artifact_id=artifact_id,
                    filename=f"{figure_id}.png",
                    paper_id=paper_id,
                ),
            )
            ref = store.put_bytes(
                object_key,
                path.read_bytes(),
                content_type="image/png",
                metadata={
                    "session_id": session_id,
                    "paper_id": paper_id,
                    "kind": "figure_image",
                    "figure_id": figure_id,
                },
            )
            registry.upsert(
                ArtifactRecord.from_blob_ref(
                    ref,
                    artifact_id=artifact_id,
                    session_id=session_id,
                    kind="figure_image",
                    paper_id=paper_id,
                    metadata={"figure_id": figure_id},
                )
            )
            serialized_ref = ref.model_dump(mode="json")
            figure["storage_ref"] = serialized_ref
            figure["asset_path"] = None
            stored_refs.append(serialized_ref)
        if stored_refs:
            parsed.structured_document["figure_assets"] = stored_refs
            parsed.structured_document["figures"] = parsed.figures


def _commit_parse_output(
    state_store: StateStore,
    job: AsyncTaskRecord,
    output: ParseExecutionOutput,
    *,
    source_sha_lookup: SourceShaLookup,
    max_cas_attempts: int = 5,
) -> list[str]:
    if not job.lease_owner:
        raise RuntimeError("parse job lost its queue lease before commit")
    stale_ids: list[str] = []
    for _attempt in range(max_cas_attempts):
        state = state_store.get_session_state(job.session_id)
        if state is None:
            raise LookupError(f"LitTrace session state does not exist: {job.session_id}")
        workspace = LiteratureWorkspace.model_validate(state.workspace_json)
        stale_ids = []
        merged_ids: list[str] = []
        active_ids = set(workspace.context.active_papers)
        for paper_id, parsed in output.parsed_papers.items():
            expected_sha = output.source_sha256.get(paper_id)
            current_sha = source_sha_lookup(job.session_id, paper_id)
            if (
                paper_id not in active_ids
                or paper_id not in workspace.papers
                or not expected_sha
                or current_sha != expected_sha
            ):
                stale_ids.append(paper_id)
                continue
            workspace.parsed_papers[paper_id] = parsed
            quality = output.docling_quality_reports.get(paper_id)
            if quality is not None:
                workspace.context.filters.docling_quality_reports[paper_id] = quality
            merged_ids.append(paper_id)
        workspace.context.filters.parsed_full_text_count = sum(
            parsed.parsed for parsed in workspace.parsed_papers.values()
        )
        workspace.context.filters.workspace_revision = state.revision + 1
        result_json = dict(job.result_json) if isinstance(job.result_json, dict) else {}
        result_json["execution"] = {
            "report": output.report,
            "merged_paper_ids": merged_ids,
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
                    "kind": "parse_job",
                    "merged_paper_ids": merged_ids,
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


def _mark_parse_job_failed(
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
