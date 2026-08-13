from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from littrace.artifact_ops import ArtifactAuditReport, audit_session_artifacts
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.session import load_existing_session, load_workspace
from littrace.state_db import EmbeddingJobRecord, state_store_from_config


MetricStatus = Literal["measured", "estimated", "not_measured"]


class SessionMetric(BaseModel):
    name: str
    status: MetricStatus = "measured"
    value: float | int | str | None = None
    numerator: int | None = None
    denominator: int | None = None
    stale_count: int | None = None
    missing_count: int | None = None
    detail: str | None = None


class SessionKnowledgeMetricsReport(BaseModel):
    """The four operational health metrics for one research session."""

    schema_version: str = "littrace.session_health.v1"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str
    readiness: str = "not_ready"
    discovery: SessionMetric
    acquisition: SessionMetric
    rag: SessionMetric
    consistency: SessionMetric
    artifact_audit: ArtifactAuditReport | None = None
    warnings: list[str] = Field(default_factory=list)


EMBEDDABLE_ARTIFACT_KINDS = {"paper_pdf", "supplementary", "structured_document"}


def build_session_knowledge_metrics(
    config: LitTraceConfig,
    session_id: str,
    *,
    artifact_limit: int = 200,
) -> SessionKnowledgeMetricsReport:
    session = load_existing_session(config, session_id)
    if session is None:
        return _empty_report(session_id, warnings=["session_not_found"])
    workspace = load_workspace(session)
    artifact_audit = audit_session_artifacts(config, session.session_id, limit=artifact_limit)
    state_store = state_store_from_config(config)
    try:
        lifecycle_events = state_store.list_paper_lifecycle_events(session.session_id)
    except Exception:
        lifecycle_events = []
    relevant_count, _, _ = _relevant_paper_count(workspace)
    discovery_count, discovery_status, discovery_detail = _today_discovery_count(workspace, lifecycle_events)
    stored_pdf_count = _stored_pdf_count(artifact_audit)
    embeddable_count = _embeddable_artifact_count(artifact_audit)
    fresh_count, freshness_status, freshness_detail = _fresh_embedding_count(
        config,
        session.session_id,
        artifact_audit,
    )
    readiness = _readiness(
        workspace,
        relevant_count=relevant_count,
        stored_pdf_count=stored_pdf_count,
        embeddable_count=embeddable_count,
        fresh_count=fresh_count,
    )
    consistency_passed, consistency_total, consistency_status, consistency_detail = (
        _consistency_components(artifact_audit)
    )
    stale_count = max(embeddable_count - fresh_count, 0)
    acquisition, acquisition_status = _acquisition_metric(lifecycle_events, stored_pdf_count, relevant_count, artifact_audit.truncated)
    if artifact_audit.truncated and freshness_status == "measured":
        freshness_status = "estimated"
    return SessionKnowledgeMetricsReport(
        session_id=session.session_id,
        readiness=readiness,
        discovery=SessionMetric(
            name="discovery",
            status=discovery_status,
            value=discovery_count,
            numerator=discovery_count,
            detail=discovery_detail,
        ),
        acquisition=SessionMetric(
            name="acquisition",
            status=acquisition_status,
            value=acquisition[0], numerator=acquisition[1], denominator=acquisition[2], detail=acquisition[3],
        ),
        rag=SessionMetric(
            name="rag",
            status=freshness_status,
            value=_rate(fresh_count, embeddable_count),
            numerator=fresh_count,
            denominator=embeddable_count,
            stale_count=stale_count,
            detail=f"{freshness_detail}; stale={stale_count}",
        ),
        consistency=SessionMetric(
            name="consistency",
            status=consistency_status,
            value=_rate(consistency_passed, consistency_total),
            numerator=consistency_passed,
            denominator=consistency_total,
            missing_count=artifact_audit.missing_object_count,
            detail=f"{consistency_detail}; missing={artifact_audit.missing_object_count}",
        ),
        artifact_audit=artifact_audit,
        warnings=[
            *artifact_audit.warnings,
            *(["artifact_audit_truncated"] if artifact_audit.truncated else []),
            *(["rag_freshness_not_backed_by_embedding_jobs"] if freshness_status == "estimated" else []),
        ],
    )


def _empty_report(session_id: str, *, warnings: list[str]) -> SessionKnowledgeMetricsReport:
    empty = SessionMetric(name="unavailable", status="not_measured", value=None)
    return SessionKnowledgeMetricsReport(
        session_id=session_id,
        discovery=empty.model_copy(update={"name": "discovery"}),
        acquisition=empty.model_copy(update={"name": "acquisition"}),
        rag=empty.model_copy(update={"name": "rag"}),
        consistency=empty.model_copy(update={"name": "consistency"}),
        warnings=warnings,
    )


def _today_discovery_count(workspace: LiteratureWorkspace, lifecycle_events=()) -> tuple[int, MetricStatus, str]:
    """Count papers first recorded by this session during the current UTC day."""

    today = datetime.now(UTC).date()
    discovered_lifecycle = {
        event.paper_id for event in lifecycle_events
        if event.event_type == "discovered_relevant" and _is_today(event.occurred_at, today)
    }
    if discovered_lifecycle:
        return len(discovered_lifecycle), "measured", "unique discovered_relevant lifecycle events today (UTC)"
    relevant_ids = _relevant_paper_ids(workspace)
    discovered: set[str] = set()
    has_today_event = False
    for event in workspace.source_events:
        if _is_today(event.retrieved_at, today):
            has_today_event = True
    for record in workspace.source_records.values():
        if record.failure_class is None and _is_today(record.retrieved_at, today) and (
            not relevant_ids or record.paper_id in relevant_ids
        ):
            discovered.add(record.paper_id)
    if discovered or has_today_event:
        return (
            len(discovered),
            "measured",
            "unique related papers first recorded by source retrievals today (UTC)",
        )
    if workspace.source_events or workspace.source_records:
        return (0, "measured", "no related papers were first recorded today (UTC)")
    return (0, "not_measured", "no source retrieval ledger is available for this session")


def _relevant_paper_ids(workspace: LiteratureWorkspace) -> set[str]:
    filters = workspace.context.filters
    if filters.candidate_pool_ids:
        return set(filters.candidate_pool_ids)
    if workspace.context.active_papers:
        return set(workspace.context.active_papers)
    return set(workspace.papers)


def _is_today(value: str, today) -> bool:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).date() == today


def _relevant_paper_count(workspace: LiteratureWorkspace) -> tuple[int, MetricStatus, str]:
    filters = workspace.context.filters
    if filters.valid_candidate_count:
        return (
            filters.valid_candidate_count,
            "measured",
            "valid_candidate_count from the latest search/context pipeline",
        )
    if filters.candidate_pool_ids:
        return (
            len(filters.candidate_pool_ids),
            "estimated",
            "candidate_pool_ids used because explicit relevance labels are unavailable",
        )
    if workspace.context.active_papers:
        return (
            len(workspace.context.active_papers),
            "estimated",
            "active_papers used because explicit relevance labels are unavailable",
        )
    return (len(workspace.papers), "estimated", "workspace paper count used as fallback")


def _acquisition_metric(events, stored_pdf_count: int, relevant_count: int, truncated: bool):
    terminal = {"acquisition_verified", "acquisition_auth_required", "acquisition_failed_terminal"}
    terminal_tasks: dict[str, str] = {}
    retryable = 0
    for event in events:
        if event.event_type == "acquisition_failed_retryable":
            retryable += 1
        if event.event_type in terminal:
            terminal_tasks[event.task_id or event.paper_id] = event.event_type
    if terminal_tasks:
        verified = sum(event == "acquisition_verified" for event in terminal_tasks.values())
        auth = sum(event == "acquisition_auth_required" for event in terminal_tasks.values())
        failed = sum(event == "acquisition_failed_terminal" for event in terminal_tasks.values())
        return (
            _rate(verified, len(terminal_tasks)), verified, len(terminal_tasks),
            f"verified / terminal acquisition attempts; auth_required={auth}; failed_terminal={failed}; retryable_events={retryable}",
        ), "measured"
    status: MetricStatus = "not_measured" if relevant_count <= 0 else "estimated" if truncated else "measured"
    return (
        _rate(stored_pdf_count, relevant_count), stored_pdf_count, relevant_count,
        "PDFs present in object storage / related papers in this session (lifecycle unavailable)",
    ), status


def _stored_pdf_count(artifact_audit: ArtifactAuditReport) -> int:
    return sum(1 for item in artifact_audit.items if item.kind == "paper_pdf" and item.exists)


def _embeddable_artifact_count(artifact_audit: ArtifactAuditReport) -> int:
    return sum(
        1
        for item in artifact_audit.items
        if item.kind in EMBEDDABLE_ARTIFACT_KINDS and item.exists
    )


def _fresh_embedding_count(
    config: LitTraceConfig,
    session_id: str,
    artifact_audit: ArtifactAuditReport,
) -> tuple[int, MetricStatus, str]:
    embeddable = [
        item for item in artifact_audit.items
        if item.kind in EMBEDDABLE_ARTIFACT_KINDS and item.exists
    ]
    if not embeddable:
        return (0, "measured", "no embeddable artifacts")
    store = state_store_from_config(config)
    if store is None:
        return (
            sum(1 for item in embeddable if item.kind == "structured_document"),
            "estimated",
            "metadata_store is not Postgres; using structured documents as a conservative freshness proxy",
        )
    jobs = store.list_embedding_jobs(session_id=session_id, limit=max(len(embeddable) * 4, 100))
    latest_by_artifact: dict[str, EmbeddingJobRecord] = {}
    for job in jobs:
        existing = latest_by_artifact.get(job.artifact_id)
        if existing is None or job.updated_at > existing.updated_at:
            latest_by_artifact[job.artifact_id] = job
    fresh = 0
    for item in embeddable:
        job = latest_by_artifact.get(item.artifact_id)
        if job is None or job.status != "completed":
            continue
        if item.sha256 and job.content_sha256 and item.sha256 != job.content_sha256:
            continue
        fresh += 1
    return (fresh, "measured", "completed embedding jobs matching current artifact sha256")


def _readiness(
    workspace: LiteratureWorkspace,
    *,
    relevant_count: int,
    stored_pdf_count: int,
    embeddable_count: int,
    fresh_count: int,
) -> str:
    filters = workspace.context.filters
    if not filters.research_background or filters.research_background_status != "accepted":
        return "not_ready"
    if relevant_count <= 0:
        return "not_ready"
    if stored_pdf_count <= 0:
        return "search_ready"
    if embeddable_count <= 0 or fresh_count <= 0:
        return "pdf_ready"
    if workspace.performance_cells or workspace.claims or filters.document_report:
        return "analysis_ready"
    return "rag_ready"


def _consistency_components(
    artifact_audit: ArtifactAuditReport,
) -> tuple[int, int, MetricStatus, str]:
    if artifact_audit.warnings:
        return (0, 0, "not_measured", "artifact registry audit is unavailable")
    total = artifact_audit.checked_count
    passed = max(artifact_audit.checked_count - artifact_audit.missing_object_count, 0)
    if total == 0:
        return (0, 0, "not_measured", "no registered artifacts to audit")
    status: MetricStatus = "estimated" if artifact_audit.truncated else "measured"
    return (
        passed,
        total,
        status,
        "registered artifacts present in object storage / audited artifacts",
    )


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)
