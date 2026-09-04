from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256

from littrace.config import LitTraceConfig
from littrace.state_db import AsyncTaskRecord, state_store_from_config


def record_lifecycle_event(
    config: LitTraceConfig,
    *,
    session_id: str,
    paper_id: str,
    event_type: str,
    task_id: str | None = None,
    artifact_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> None:
    """Append an immutable paper-lifecycle observation to ``chat_trail.events``.

    The old standalone ``paper_lifecycle_events`` table is gone; events
    live as JSONB inside ``chat_trail`` and are append-only via
    ``StateStore.append_chat_event``.
    """
    state_store_from_config(config).append_chat_event(
        session_id,
        {
            "event_id": sha256(
                "\0".join(
                    (session_id, paper_id, event_type, datetime.now(UTC).isoformat())
                ).encode()
            ).hexdigest()[:16],
            "paper_id": paper_id,
            "event_type": event_type,
            "occurred_at": datetime.now(UTC).isoformat(),
            "task_id": task_id,
            "artifact_id": artifact_id,
            "payload": payload or {},
        },
    )


def enqueue_embedding_outbox(
    config: LitTraceConfig,
    *,
    session_id: str,
    artifact_id: str,
    content_sha256: str | None,
    payload: dict[str, object] | None = None,
) -> AsyncTaskRecord:
    """Enqueue an outbox-style async task. Bytes go to object store; this row
    is the durable hand-off to the embedding dispatcher."""
    now = datetime.now(UTC).isoformat()
    outbox_id = sha256(
        "\0".join((session_id, artifact_id, content_sha256 or "", now)).encode()
    ).hexdigest()
    return state_store_from_config(config).enqueue_async_task(
        AsyncTaskRecord(
            task_id=outbox_id,
            session_id=session_id,
            kind="artifact_outbox",
            artifact_id=artifact_id,
            event_type="embedding_requested",
            content_sha256=content_sha256 or "",
            status="queued",
            created_at=now,
            updated_at=now,
            result_json=payload or {},
        )
    )


def dispatch_embedding_outbox(
    config: LitTraceConfig,
    *,
    limit: int = 20,
    session_id: str | None = None,
) -> tuple[int, int, list[str]]:
    """Promote ``artifact_outbox`` rows into ``embedding_job`` rows in the
    consolidated ``async_tasks`` table.

    The outbox row is marked complete only after the idempotent embedding
    job upsert succeeds. A crashed dispatcher leaves its lease to expire
    for another worker.
    """
    store = state_store_from_config(config)
    worker_id = f"outbox:{datetime.now(UTC).timestamp():.6f}"
    claim_kwargs = {
        "worker_id": worker_id,
        "kind": "artifact_outbox",
        "limit": limit,
    }
    if session_id is not None:
        claim_kwargs["session_id"] = session_id
    records = store.claim_pending_async_tasks(**claim_kwargs)
    dispatched = failed = 0
    warnings: list[str] = []
    for record in records:
        try:
            revision = record.result_json.get("source_revision") if isinstance(record.result_json, dict) else None
            source_revision = str(revision) if revision is not None else None
            job_id = _embedding_job_id(
                record.session_id, record.artifact_id, record.content_sha256, source_revision
            )
            store.enqueue_async_task(
                AsyncTaskRecord(
                    task_id=job_id,
                    session_id=record.session_id,
                    kind="embedding_job",
                    artifact_id=record.artifact_id,
                    profile_id=f"pending:{record.session_id}",
                    source_revision=source_revision or "",
                    content_sha256=record.content_sha256,
                    status="queued",
                )
            )
            record.status = "completed"
            record.completed_at = datetime.now(UTC).isoformat()
            record.last_error = None
            record.next_attempt_at = None
            record.lease_owner = None
            record.lease_expires_at = None
            record.updated_at = record.completed_at
            store.update_async_task(record)
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
            store.update_async_task(record)
            warnings.append(f"outbox:{record.task_id}:{record.last_error}")
    return dispatched, failed, warnings


def _embedding_job_id(
    session_id: str,
    artifact_id: str,
    content_sha256: str | None,
    source_revision: str | None,
) -> str:
    digest = sha256(
        "\0".join((session_id, artifact_id, content_sha256 or "", source_revision or "")).encode()
    ).hexdigest()[:24]
    return f"emb:{digest}"
