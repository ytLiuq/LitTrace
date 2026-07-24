from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.context import add_ranked_candidate_papers
from littrace.models import AccessType, DownloadExecutionRequest, PaperSearchRequest
from littrace.retrieval.rag_refresh import RagRefreshReport, refresh_session_rag_index
from littrace.retrieval.rag_profile import load_session_rag_profile
from littrace.sentinel.cli import run_sentinel
from littrace.session import load_or_create_session, load_workspace, save_workspace
from littrace.skill_runner import execute_downloads_skill, parse_workspace_skill, search_papers_skill
from littrace.state_db import EmbeddingJobRecord, state_store_from_config


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
    session_reports: list[RagRefreshReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RagEmbeddingJobBatchReport(BaseModel):
    schema_version: str = "littrace.rag_embedding_job_batch_report.v1"
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    warnings: list[str] = Field(default_factory=list)
    job_ids: list[str] = Field(default_factory=list)


class ResearchBackgroundSyncReport(BaseModel):
    schema_version: str = "littrace.research_background_sync_report.v1"
    user_id: str
    session_id: str
    topic: str | None = None
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    searched_count: int = 0
    open_access_count: int = 0
    downloaded_count: int = 0
    requires_login_count: int = 0
    parsed_count: int = 0
    rag_refreshed: bool = False
    rag_report: RagRefreshReport | None = None
    warnings: list[str] = Field(default_factory=list)


async def run_daily_rag_maintenance(config: LitTraceConfig) -> RagDailyJobReport:
    report = RagDailyJobReport()
    embedding_report = await run_pending_embedding_jobs(config)
    report.embedding_jobs_processed = embedding_report.processed
    report.embedding_jobs_failed = embedding_report.failed
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

    for session_id in iter_workspace_session_ids(config.storage.sessions_dir):
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
            profile = load_session_rag_profile(session)
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
) -> ResearchBackgroundSyncReport:
    if workspace is None:
        workspace = load_workspace(session)
    filters = workspace.context.filters
    topic = (filters.research_background or filters.topic or "").strip()
    report = ResearchBackgroundSyncReport(
        user_id=getattr(session, "user_id", config.storage.default_user_id),
        session_id=session.session_id,
        topic=topic or None,
    )
    if not topic:
        return _skip_background_sync(report, "no_research_background")
    if filters.research_background_status == "rejected":
        return _skip_background_sync(report, "research_background_rejected")
    if not config.rag.auto_download_open_access:
        return _skip_background_sync(report, "auto_download_open_access_disabled")

    request = PaperSearchRequest(
        topic=topic,
        discipline=filters.discipline or "materials chemistry",
        year_min=filters.year_min or config.literature_context.default_recent_year_min,
        limit=max(config.literature_context.active_context_limit, 10),
        wants_recent=True,
        live=config.api.enable_live_search,
    )
    search = await search_papers_skill(request, config)
    papers = list(search.result.papers)
    report.searched_count = len(papers)
    if not papers:
        return _skip_background_sync(report, "no_source_results")
    workspace = add_ranked_candidate_papers(
        workspace,
        papers,
        request,
        active_limit=config.literature_context.active_context_limit,
    )
    workspace.context.filters.research_background_last_sync_at = datetime.now(UTC).isoformat()
    open_access_ids = [
        paper.paper_id
        for paper in papers
        if paper.access_type == AccessType.OPEN_ACCESS
    ]
    report.open_access_count = len(open_access_ids)
    if open_access_ids:
        download_result = await execute_downloads_skill(
            config,
            workspace,
            DownloadExecutionRequest(
                paper_ids=open_access_ids,
                user_id=getattr(session, "user_id", config.storage.default_user_id),
                session_id=session.session_id,
            ),
        )
        report.downloaded_count = download_result.downloaded_count
        report.requires_login_count = download_result.requires_login_count
    if report.downloaded_count:
        workspace, parse_report = await parse_workspace_skill(workspace, config)
        report.parsed_count = int(parse_report.get("parsed_count", 0) or 0)
    _, rag_report = await refresh_session_rag_index(config, session, workspace)
    report.rag_refreshed = not rag_report.skipped
    report.rag_report = rag_report
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
    jobs = state_store.list_pending_embedding_jobs(limit=limit)
    if not jobs:
        report.finished_at = datetime.now(UTC).isoformat()
        return report
    refreshed_sessions: set[tuple[str, str]] = set()
    for job in jobs:
        report.job_ids.append(job.job_id)
        job.status = "running"
        job.attempt_count += 1
        job.updated_at = datetime.now(UTC).isoformat()
        state_store.update_embedding_job(job)
        try:
            session_key = (job.user_id, job.session_id)
            if session_key not in refreshed_sessions:
                session = load_or_create_session(config, job.session_id)
                workspace = load_workspace(session)
                _, refresh_report = await refresh_session_rag_index(config, session, workspace)
                save_workspace(session, workspace, config=config)
                refreshed_sessions.add(session_key)
                result_json = refresh_report.model_dump(mode="json")
            else:
                result_json = {"skipped": True, "skip_reason": "session_already_refreshed"}
            _mark_embedding_job_completed(state_store, job, result_json)
            report.processed += 1
        except Exception as exc:
            _mark_embedding_job_failed(state_store, job, exc, config)
            report.failed += 1
            report.warnings.append(
                f"embedding_job:{job.job_id}:{exc.__class__.__name__}: {exc}"
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
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir() or session_dir.name == "sentinel":
            continue
        if (session_dir / "workspace.json").exists():
            session_ids.append(session_dir.name)
    return session_ids


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
    job: EmbeddingJobRecord,
    result_json: dict[str, object],
) -> None:
    job.status = "completed"
    job.result_json = result_json
    job.completed_at = datetime.now(UTC).isoformat()
    job.updated_at = job.completed_at
    job.last_error = None
    state_store.update_embedding_job(job)


def _mark_embedding_job_failed(
    state_store,
    job: EmbeddingJobRecord,
    exc: Exception,
    config: LitTraceConfig,
) -> None:
    job.status = "failed"
    job.last_error = f"{exc.__class__.__name__}: {exc}"
    job.updated_at = datetime.now(UTC).isoformat()
    if job.attempt_count < config.download_retry.max_attempts:
        delay = min(config.download_retry.base_delay_seconds * (2 ** max(job.attempt_count - 1, 0)), 3600)
        job.next_attempt_at = (datetime.now(UTC) + timedelta(seconds=delay)).isoformat()
    else:
        job.completed_at = datetime.now(UTC).isoformat()
    state_store.update_embedding_job(job)
