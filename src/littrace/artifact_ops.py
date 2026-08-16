from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from littrace.artifact_registry import ArtifactRecord, artifact_registry_from_config
from littrace.artifact_store import BlobRef, artifact_store_from_config
from littrace.config import LitTraceConfig
from littrace.session import load_existing_session
from littrace.lifecycle import enqueue_embedding_outbox, record_lifecycle_event
from littrace.state_db import state_store_from_config


class ArtifactAuditItem(BaseModel):
    artifact_id: str
    kind: str
    object_key: str
    backend: str
    bucket: str | None = None
    exists: bool = False
    size_bytes: int | None = None
    sha256: str | None = None
    error: str | None = None


class ArtifactAuditReport(BaseModel):
    schema_version: str = "littrace.artifact_audit_report.v1"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str
    artifact_count: int = 0
    checked_count: int = 0
    missing_object_count: int = 0
    total_size_bytes: int = 0
    truncated: bool = False
    items: list[ArtifactAuditItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ArtifactReconciliationReport(BaseModel):
    schema_version: str = "littrace.artifact_reconciliation.v1"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str
    checked: int = 0
    missing: int = 0
    requeued: int = 0
    warnings: list[str] = Field(default_factory=list)


def audit_session_artifacts(
    config: LitTraceConfig,
    session_id: str,
    *,
    limit: int = 200,
) -> ArtifactAuditReport:
    session = load_existing_session(config, session_id)
    if session is None:
        return ArtifactAuditReport(
            session_id=session_id,
            warnings=["session_not_found"],
        )
    report = ArtifactAuditReport(session_id=session.session_id)
    try:
        records = artifact_registry_from_config(config).list_for_session(
            session_id=session.session_id,
        )
    except Exception as exc:
        report.warnings.append(f"artifact_registry_error:{exc.__class__.__name__}: {exc}")
        return report

    report.artifact_count = len(records)
    report.truncated = len(records) > limit
    store = artifact_store_from_config(config)
    for record in records[:limit]:
        item = _audit_artifact_record(store, record)
        report.items.append(item)
        report.checked_count += 1
        if not item.exists:
            report.missing_object_count += 1
        if item.size_bytes:
            report.total_size_bytes += item.size_bytes
    return report


def _audit_artifact_record(store, record: ArtifactRecord) -> ArtifactAuditItem:
    item = ArtifactAuditItem(
        artifact_id=record.artifact_id,
        kind=record.kind,
        object_key=record.object_key,
        backend=record.backend,
        bucket=record.bucket,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
    )
    ref = BlobRef(
        backend=record.backend,
        bucket=record.bucket,
        object_key=record.object_key,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        content_type=record.content_type,
        metadata={},
    )
    try:
        item.exists = bool(store.exists(ref))
    except Exception as exc:
        item.exists = False
        item.error = f"{exc.__class__.__name__}: {exc}"
    return item


def reconcile_session_artifacts(
    config: LitTraceConfig,
    session_id: str,
    *,
    limit: int = 200,
) -> ArtifactReconciliationReport:
    """Repair durable hand-offs after crashes or object-store/index divergence."""
    report = ArtifactReconciliationReport(session_id=session_id)
    session = load_existing_session(config, session_id)
    if session is None:
        report.warnings.append("session_not_found")
        return report
    try:
        records = artifact_registry_from_config(config).list_for_session(session_id=session_id)
        store = artifact_store_from_config(config)
        state_store = state_store_from_config(config)
        lifecycle = state_store.list_chat_events(session_id)
    except Exception as exc:
        report.warnings.append(f"reconciliation_setup:{exc.__class__.__name__}: {exc}")
        return report
    last_event = {(event.paper_id, event.artifact_id): event.event_type for event in lifecycle}
    completed = {
        (job.artifact_id, job.content_sha256)
        for job in state_store.list_async_tasks(
            session_id=session_id,
            status="completed",
            kind="embedding_job",
            limit=max(limit * 5, 1000),
        )
    }
    for record in records[:limit]:
        report.checked += 1
        item = _audit_artifact_record(store, record)
        paper_id = record.paper_id or record.artifact_id
        if not item.exists:
            report.missing += 1
            if last_event.get((paper_id, record.artifact_id)) != "artifact_missing":
                record_lifecycle_event(
                    config, session_id=session_id, paper_id=paper_id,
                    event_type="artifact_missing", artifact_id=record.artifact_id,
                    payload={"object_key": record.object_key, "sha256": record.sha256},
                )
            continue
        if record.kind not in {"paper_pdf", "supplementary", "structured_document"}:
            continue
        if (record.artifact_id, record.sha256) not in completed:
            enqueue_embedding_outbox(
                config, session_id=session_id, artifact_id=record.artifact_id,
                content_sha256=record.sha256, payload={"source_revision": record.revision, "reason": "reconciliation"},
            )
            if last_event.get((paper_id, record.artifact_id)) != "embedding_requeued":
                record_lifecycle_event(
                    config, session_id=session_id, paper_id=paper_id,
                    event_type="embedding_requeued", artifact_id=record.artifact_id,
                    payload={"sha256": record.sha256, "reason": "reconciliation"},
                )
            report.requeued += 1
    return report
