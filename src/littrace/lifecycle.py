from __future__ import annotations

from datetime import UTC, datetime, timedelta

from littrace.config import LitTraceConfig
from littrace.state_db import (
    ArtifactOutboxRecord,
    EmbeddingJobRecord,
    PaperLifecycleEventRecord,
    state_store_from_config,
)


def record_lifecycle_event(
    config: LitTraceConfig,
    *,
    session_id: str,
    paper_id: str,
    event_type: str,
    task_id: str | None = None,
    artifact_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> PaperLifecycleEventRecord:
    record = PaperLifecycleEventRecord(
        session_id=session_id,
        paper_id=paper_id,
        event_type=event_type,
        task_id=task_id,
        artifact_id=artifact_id,
        payload=payload or {},
    )
    return state_store_from_config(config).append_paper_lifecycle_event(record)


def enqueue_embedding_outbox(
    config: LitTraceConfig,
    *,
    session_id: str,
    artifact_id: str,
    content_sha256: str | None,
    payload: dict[str, object] | None = None,
) -> ArtifactOutboxRecord:
    return state_store_from_config(config).enqueue_artifact_outbox(
        ArtifactOutboxRecord(
            session_id=session_id,
            artifact_id=artifact_id,
            content_sha256=content_sha256,
            payload=payload or {},
        )
    )


def dispatch_embedding_outbox(
    config: LitTraceConfig,
    *,
    limit: int = 20,
) -> tuple[int, int, list[str]]:
    """Accept durable outbox records into the embedding queue.

    The outbox is marked complete only after the idempotent embedding job upsert
    succeeds.  A crashed dispatcher leaves its lease to expire for another worker.
    """
    store = state_store_from_config(config)
    worker_id = f"outbox:{datetime.now(UTC).timestamp():.6f}"
    records = store.claim_artifact_outbox(worker_id=worker_id, limit=limit)
    dispatched = failed = 0
    warnings: list[str] = []
    for record in records:
        try:
            revision = record.payload.get("source_revision")
            source_revision = str(revision) if revision is not None else None
            store.enqueue_embedding_job(
                EmbeddingJobRecord(
                    job_id=_embedding_job_id(
                        record.session_id, record.artifact_id,
                        record.content_sha256, source_revision,
                    ),
                    profile_id=f"pending:{record.session_id}",
                    session_id=record.session_id,
                    artifact_id=record.artifact_id,
                    source_revision=source_revision,
                    content_sha256=record.content_sha256,
                )
            )
            record.status = "completed"
            record.completed_at = datetime.now(UTC).isoformat()
            record.last_error = None
            record.next_attempt_at = None
            record.lease_owner = None
            record.lease_expires_at = None
            record.updated_at = record.completed_at
            store.update_artifact_outbox(record)
            dispatched += 1
        except Exception as exc:
            failed += 1
            record.last_error = f"{exc.__class__.__name__}: {exc}"
            record.lease_owner = None
            record.lease_expires_at = None
            record.updated_at = datetime.now(UTC).isoformat()
            if record.attempt_count >= config.download_retry.max_attempts:
                record.status = "dead"
                record.next_attempt_at = None
                record.completed_at = record.updated_at
            else:
                record.status = "failed"
                delay = min(
                    config.download_retry.base_delay_seconds * (2 ** max(record.attempt_count - 1, 0)),
                    3600,
                )
                record.next_attempt_at = (datetime.now(UTC) + timedelta(seconds=delay)).isoformat()
            store.update_artifact_outbox(record)
            warnings.append(f"outbox:{record.outbox_id}:{record.last_error}")
    return dispatched, failed, warnings


def _embedding_job_id(
    session_id: str,
    artifact_id: str,
    content_sha256: str | None,
    source_revision: str | None,
) -> str:
    from hashlib import sha256

    digest = sha256("\0".join((session_id, artifact_id, content_sha256 or "", source_revision or "")).encode()).hexdigest()[:24]
    return f"emb:{digest}"
