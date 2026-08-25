from __future__ import annotations

import asyncio
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.artifact_ops import reconcile_session_artifacts
from littrace.lifecycle import dispatch_embedding_outbox
from littrace.context import add_ranked_candidate_papers
from littrace.models import AccessType, DownloadExecutionRequest, PaperSearchRequest, coerce_parsed
from littrace.figure_enrichment import enrich_parsed_figures
from littrace.retrieval.search import filter_papers_by_retrieval_policy
from littrace.retrieval.rag_refresh import RagRefreshReport, refresh_session_rag_index
from littrace.retrieval.rag_profile import load_session_rag_profile
from littrace.sentinel.cli import run_sentinel
from littrace.session import load_or_create_session, load_workspace, save_workspace
from littrace.skill_runner import (
    execute_downloads_skill,
    parse_workspace_skill,
    search_papers_skill,
)
from littrace.state_db import AsyncTaskRecord, state_store_from_config


class RagDailyJobReport(BaseModel):
    schema_version: str = "littrace.rag_daily_job_report.v1"
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str | None = None
    sentinel_watchlists: int = 0
    sentinel_failed: int = 0
    sessions_refreshed: int = 0
    sessions_failed: int = 0
    sessions_skipped: int = 0
    embedding_jobs_processed: int = 0
    embedding_jobs_failed: int = 0
    figures_enriched: int = 0
    figures_rejected: int = 0
    figure_enrichment_failed: int = 0
    outbox_dispatched: int = 0
    outbox_failed: int = 0
    artifacts_reconciled: int = 0
    missing_artifacts: int = 0
    embedding_requeued: int = 0
    session_reports: list[RagRefreshReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RagEmbeddingJobBatchReport(BaseModel):
    schema_version: str = "littrace.rag_embedding_job_batch_report.v1"
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    outbox_dispatched: int = 0
    outbox_failed: int = 0
    figures_enriched: int = 0
    figures_rejected: int = 0
    figure_enrichment_failed: int = 0
    warnings: list[str] = Field(default_factory=list)
    job_ids: list[str] = Field(default_factory=list)


class ResearchBackgroundSyncReport(BaseModel):
    schema_version: str = "littrace.research_background_sync_report.v1"
    session_id: str
    topic: str | None = None
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    searched_count: int = 0
    open_access_count: int = 0
    policy_rejected_count: int = 0
    candidate_filter_reasons: dict[str, str] = Field(default_factory=dict)
    candidate_filter_counts: dict[str, int] = Field(default_factory=dict)
    downloaded_count: int = 0
    requires_login_count: int = 0
    parsed_count: int = 0
    outbox_dispatched: int = 0
    embedding_jobs_processed: int = 0
    embedding_jobs_failed: int = 0
    rag_refreshed: bool = False
    rag_report: RagRefreshReport | None = None
    warnings: list[str] = Field(default_factory=list)


async def run_daily_rag_maintenance(config: LitTraceConfig) -> RagDailyJobReport:
    report = RagDailyJobReport()
    # Reconciliation precedes dispatch so any download-path crash can be repaired.
    for session_id in iter_rag_session_ids(config):
        reconciliation = reconcile_session_artifacts(config, session_id)
        report.artifacts_reconciled += reconciliation.checked
        report.missing_artifacts += reconciliation.missing
        report.embedding_requeued += reconciliation.requeued
        report.warnings.extend(reconciliation.warnings)
    embedding_report = await run_pending_embedding_jobs(config)
    report.outbox_dispatched = embedding_report.outbox_dispatched
    report.outbox_failed = embedding_report.outbox_failed
    report.embedding_jobs_processed = embedding_report.processed
    report.embedding_jobs_failed = embedding_report.failed
    report.figures_enriched = embedding_report.figures_enriched
    report.figures_rejected = embedding_report.figures_rejected
    report.figure_enrichment_failed = embedding_report.figure_enrichment_failed
    report.warnings.extend(embedding_report.warnings)
    for watchlist_id in iter_sentinel_watchlist_ids(config.storage.sessions_dir):
        try:
            await run_sentinel(config, watchlist_id)
            report.sentinel_watchlists += 1
        except Exception as exc:
            report.sentinel_failed += 1
            report.warnings.append(
                f"sentinel:{watchlist_id}:{exc.__class__.__name__}: {exc}"
            )

    for session_id in iter_rag_session_ids(config):
        try:
            session = load_or_create_session(config, session_id)
            workspace = load_workspace(session)
            background_report = await run_session_research_background_sync(
                config,
                session,
                workspace,
            )
            if not background_report.skipped:
                report.session_reports.extend(
                    [background_report.rag_report]
                    if background_report.rag_report is not None
                    else []
                )
                report.sessions_refreshed += 1 if background_report.rag_refreshed else 0
                report.sessions_skipped += 0 if background_report.rag_refreshed else 1
                report.warnings.extend(background_report.warnings)
                continue
            profile = load_session_rag_profile(session, config=config)
            skip_reason = _rag_refresh_skip_reason(profile)
            if skip_reason is not None:
                report.sessions_skipped += 1
                continue
            _, refresh_report = await refresh_session_rag_index(config, session, workspace)
            save_workspace(session, workspace, config=config)
            report.session_reports.append(refresh_report)
            report.sessions_refreshed += 1
        except Exception as exc:
            report.sessions_failed += 1
            report.warnings.append(
                f"session:{session_id}:{exc.__class__.__name__}: {exc}"
            )
    report.finished_at = datetime.now().isoformat(timespec="seconds")
    return report


async def run_session_research_background_sync(
    config: LitTraceConfig,
    session,
    workspace=None,
    progress_callback=None,
) -> ResearchBackgroundSyncReport:
    if workspace is None:
        workspace = load_workspace(session)
    filters = workspace.context.filters
    policy = filters.research_retrieval_policy
    topic = ((policy.canonical_topic if policy else None) or filters.topic or filters.research_background or "").strip()
    report = ResearchBackgroundSyncReport(
        session_id=session.session_id,
        topic=topic or None,
    )
    if not topic:
        return _skip_background_sync(report, "no_research_background")
    if filters.research_background_status == "rejected":
        return _skip_background_sync(report, "research_background_rejected")
    if policy is None:
        return _skip_background_sync(report, "missing_retrieval_policy")
    if not config.rag.auto_download_open_access:
        return _skip_background_sync(report, "auto_download_open_access_disabled")

    # Daily background sync: pull any new open-access PDFs that the user has
    # not yet downloaded locally. Bytes go to the object store only — never
    # to paper_library_dir — so the user's working directory stays clean.
    request = PaperSearchRequest(
        topic=topic,
        discipline=filters.discipline or "materials chemistry",
        year_min=filters.year_min or config.literature_context.default_recent_year_min,
        limit=max(config.literature_context.active_context_limit, 10),
        wants_recent=True,
        live=config.api.enable_live_search,
        query_variants=(policy.query_variants if policy and policy.query_variants else [topic]),
    )
    search = await search_papers_skill(request, config, progress_callback=progress_callback)
    papers = list(search.result.papers)
    report.searched_count = len(papers)
    if not papers:
        return _skip_background_sync(report, "no_source_results")
    accepted_papers, rejected = filter_papers_by_retrieval_policy(papers, policy)
    report.policy_rejected_count = len(rejected)
    report.candidate_filter_counts["policy_rejected"] = len(rejected)
    for paper_id in rejected:
        report.candidate_filter_reasons[paper_id] = "policy_rejected"
    for paper_id in rejected:
        if paper_id not in workspace.context.excluded_papers:
            workspace.context.excluded_papers.append(paper_id)
    workspace = add_ranked_candidate_papers(
        workspace,
        papers,
        request,
        active_limit=config.literature_context.active_context_limit,
    )
    workspace.context.filters.research_background_last_sync_at = datetime.now(UTC).isoformat()
    open_access_ids = [
        paper.paper_id
        for paper in workspace.papers.values()
        if paper.paper_id in {paper.paper_id for paper in accepted_papers}
        and paper.access_type == AccessType.OPEN_ACCESS
    ]
    report.open_access_count = len(open_access_ids)
    parse_report = {"parsed_count": 0}
    report.parsed_count = 0
    if open_access_ids:
        download_result = await execute_downloads_skill(
            config,
            workspace,
            DownloadExecutionRequest(
                paper_ids=open_access_ids,
                session_id=session.session_id,
                target="storage_only",
            ),
        )
        report.downloaded_count = download_result.downloaded_count
        report.requires_login_count = download_result.requires_login_count
        if report.downloaded_count:
            workspace, parse_report = await parse_workspace_skill(workspace, config)
            report.parsed_count = int(parse_report.get("parsed_count", 0) or 0)
    # Persist parsed artifacts before consuming their durable outbox records.
    # The worker owns pgvector writes and embedding-job completion, so freshness
    # reflects the same path that actually produced the vectors.
    save_workspace(session, workspace, config=config)
    embedding_report = await run_pending_embedding_jobs(
        config,
        limit=max(config.literature_context.active_context_limit * 2, 20),
    )
    report.outbox_dispatched = embedding_report.outbox_dispatched
    report.embedding_jobs_processed = embedding_report.processed
    report.embedding_jobs_failed = embedding_report.failed
    report.warnings.extend(embedding_report.warnings)
    workspace = load_workspace(session)
    filters = workspace.context.filters
    persisted_rag_report = workspace.context.filters.rag_refresh_report
    if persisted_rag_report:
        report.rag_report = RagRefreshReport.model_validate(persisted_rag_report)
        report.rag_refreshed = not report.rag_report.skipped
    filters.research_background_last_downloaded_count = report.downloaded_count
    filters.research_background_last_parsed_count = report.parsed_count
    filters.research_background_last_sync_at = datetime.now(UTC).isoformat()
    save_workspace(session, workspace, config=config)
    report.finished_at = datetime.now(UTC).isoformat()
    return report


def _skip_background_sync(
    report: ResearchBackgroundSyncReport,
    reason: str,
) -> ResearchBackgroundSyncReport:
    report.skipped = True
    report.skip_reason = reason
    report.finished_at = datetime.now(UTC).isoformat()
    return report


async def run_pending_embedding_jobs(
    config: LitTraceConfig,
    *,
    limit: int = 20,
) -> RagEmbeddingJobBatchReport:
    report = RagEmbeddingJobBatchReport()
    state_store = state_store_from_config(config)
    if state_store is None:
        report.skipped += 1
        report.finished_at = datetime.now(UTC).isoformat()
        return report
    dispatched, outbox_failed, outbox_warnings = dispatch_embedding_outbox(config, limit=limit)
    report.outbox_dispatched = dispatched
    report.outbox_failed = outbox_failed
    report.warnings.extend(outbox_warnings)
    if outbox_failed:
        report.failed += outbox_failed
    # A dispatcher accepting an event is progress, but embedding is only counted
    # as processed after the pgvector refresh succeeds below.
    worker_id = f"{socket.gethostname()}:{uuid4().hex[:12]}"
    jobs = state_store.claim_pending_async_tasks(
        worker_id=worker_id,
        kind="embedding_job",
        limit=limit,
        lease_seconds=max(config.download_retry.interval_seconds * 4, 120.0),
    )
    if not jobs:
        report.finished_at = datetime.now(UTC).isoformat()
        return report
    jobs_by_session: dict[str, list[AsyncTaskRecord]] = {}
    for job in jobs:
        jobs_by_session.setdefault(job.session_id, []).append(job)
    for session_id, session_jobs in jobs_by_session.items():
        artifact_ids = {
            paper_id
            for job in session_jobs
            if (paper_id := _paper_id_from_artifact_id(job.artifact_id)) is not None
        }
        try:
            session = load_or_create_session(config, session_id)
            workspace = load_workspace(session)
            for paper_id in artifact_ids or workspace.parsed_papers.keys():
                parsed = workspace.parsed_papers.get(paper_id)
                if parsed is None:
                    continue
                parsed = coerce_parsed(parsed)
                enrichment = await enrich_parsed_figures(config, parsed)
                workspace.parsed_papers[paper_id] = parsed
                report.figures_enriched += enrichment.enriched
                report.figures_rejected += enrichment.rejected
                report.figure_enrichment_failed += enrichment.failed
                report.warnings.extend(
                    f"figure_enrichment:{paper_id}:{warning}"
                    for warning in enrichment.warnings
                )
            _, refresh_report = await refresh_session_rag_index(
                config,
                session,
                workspace,
                artifact_ids=artifact_ids or None,
            )
            save_workspace(session, workspace, config=config)
            result_json = refresh_report.model_dump(mode="json")
            for job in session_jobs:
                report.job_ids.append(job.task_id)
                _mark_embedding_job_completed(state_store, job, result_json)
                report.processed += 1
        except Exception as exc:
            for job in session_jobs:
                report.job_ids.append(job.task_id)
                _mark_embedding_job_failed(state_store, job, exc, config)
                report.failed += 1
                report.warnings.append(
                    f"embedding_job:{job.task_id}:{exc.__class__.__name__}: {exc}"
                )
    report.finished_at = datetime.now(UTC).isoformat()
    return report


async def run_daily_rag_daemon(
    config: LitTraceConfig,
    *,
    interval_hours: float = 24.0,
    run_immediately: bool = True,
) -> None:
    if run_immediately:
        await run_daily_rag_maintenance(config)
    sleep_seconds = max(interval_hours, 0.1) * 3600
    while True:
        await asyncio.sleep(sleep_seconds)
        await run_daily_rag_maintenance(config)


def iter_workspace_session_ids(sessions_dir: Path) -> list[str]:
    if not sessions_dir.exists():
        return []
    session_ids: list[str] = []
    store = state_store_from_config(config)
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir() or session_dir.name == "sentinel":
            continue
        # Round 3 topic B: Postgres is the source of truth. Only
        # sessions with a session_state row are live; pre-migration
        # disk-only leftovers are skipped.
        if store.get_session_state(session_dir.name) is None:
            continue
        session_ids.append(session_dir.name)
    return session_ids


def iter_rag_session_ids(config: LitTraceConfig, *, limit: int = 500) -> list[str]:
    state_store = state_store_from_config(config)
    if state_store is not None:
        return [record.session_id for record in state_store.list_session_states(limit=limit)]
    return iter_workspace_session_ids(config.storage.sessions_dir)


def _paper_id_from_artifact_id(artifact_id: str) -> str | None:
    if artifact_id.startswith("paper_pdf:"):
        return artifact_id.split(":", 1)[1] or None
    if artifact_id.startswith("supplementary:"):
        parts = artifact_id.split(":")
        return parts[1] if len(parts) >= 2 else None
    if artifact_id:
        return artifact_id
    return None


def iter_sentinel_watchlist_ids(sessions_dir: Path) -> list[str]:
    root = sessions_dir / "sentinel"
    if not root.exists():
        return []
    watchlist_ids: list[str] = []
    for watchlist_dir in sorted(root.iterdir()):
        if not watchlist_dir.is_dir():
            continue
        if (watchlist_dir / "watchlist.yaml").exists():
            watchlist_ids.append(watchlist_dir.name)
    return watchlist_ids


def _rag_refresh_skip_reason(profile) -> str | None:
    if profile is None:
        return "no_rag_profile"
    if not getattr(profile, "auto_refresh_enabled", False):
        return "auto_refresh_disabled"
    frequency = str(getattr(profile, "refresh_frequency", "daily") or "daily").lower()
    if frequency in {"manual", "off", "disabled", "never"}:
        return f"refresh_frequency:{frequency}"
    interval = _refresh_interval_for_frequency(frequency)
    if interval is None:
        return None
    refreshed_at = getattr(profile, "last_refreshed_at", None)
    if not refreshed_at:
        return None
    try:
        last = datetime.fromisoformat(refreshed_at)
    except ValueError:
        return None
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    return None if now - last >= interval else f"not_due:{frequency}"


def _refresh_interval_for_frequency(frequency: str) -> timedelta | None:
    if frequency == "hourly":
        return timedelta(hours=1)
    if frequency == "daily":
        return timedelta(days=1)
    if frequency == "weekly":
        return timedelta(days=7)
    return None


def _mark_embedding_job_completed(
    state_store,
    job: AsyncTaskRecord,
    result_json: dict[str, object],
) -> None:
    job.status = "completed"
    job.result_json = result_json
    job.completed_at = datetime.now(UTC).isoformat()
    job.updated_at = job.completed_at
    job.last_error = None
    job.lease_owner = None
    job.lease_expires_at = None
    job.last_heartbeat_at = None
    state_store.update_async_task(job)


def _mark_embedding_job_failed(
    state_store,
    job: AsyncTaskRecord,
    exc: Exception,
    config: LitTraceConfig,
) -> None:
    job.last_error = f"{exc.__class__.__name__}: {exc}"
    job.updated_at = datetime.now(UTC).isoformat()
    job.lease_owner = None
    job.lease_expires_at = None
    job.last_heartbeat_at = None
    if job.attempt_count < config.download_retry.max_attempts:
        job.status = "failed"
        delay = min(config.download_retry.base_delay_seconds * (2 ** max(job.attempt_count - 1, 0)), 3600)
        job.next_attempt_at = (datetime.now(UTC) + timedelta(seconds=delay)).isoformat()
    else:
        job.status = "dead"
        job.next_attempt_at = None
        job.completed_at = datetime.now(UTC).isoformat()
    state_store.update_async_task(job)
