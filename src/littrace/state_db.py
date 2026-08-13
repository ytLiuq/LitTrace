from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from littrace.config import LitTraceConfig


class SessionStateRecord(BaseModel):
    session_id: str
    root_path: str
    workspace_sha256: str | None = None
    workspace_json: dict[str, object] = Field(default_factory=dict)
    manifest_json: dict[str, object] = Field(default_factory=dict)
    artifact_index_json: dict[str, object] = Field(default_factory=dict)
    memory_json: dict[str, object] = Field(default_factory=dict)
    rag_profile_json: dict[str, object] = Field(default_factory=dict)
    revision: int = 0
    structured_document_count: int = 0
    workspace_snapshot_count: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SessionSummaryRecord(BaseModel):
    session_id: str
    root_path: str
    updated_at: str
    workspace_sha256: str | None = None
    revision: int = 0
    structured_document_count: int = 0
    workspace_snapshot_count: int = 0


class SessionMessageRecord(BaseModel):
    message_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    role: str
    content_json: dict[str, object] = Field(default_factory=dict)
    content_text: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SessionSnapshotRecord(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    revision: int
    workspace_sha256: str
    workspace_json: dict[str, object] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class EmbeddingJobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid4().hex)
    profile_id: str
    session_id: str
    artifact_id: str
    source_revision: str | None = None
    content_sha256: str | None = None
    status: str = "queued"
    attempt_count: int = 0
    next_attempt_at: str | None = None
    last_error: str | None = None
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    last_heartbeat_at: str | None = None
    result_json: dict[str, object] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None


class PaperLifecycleEventRecord(BaseModel):
    """Immutable observation of one paper's acquisition/indexing lifecycle."""

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    paper_id: str
    event_type: str
    occurred_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    task_id: str | None = None
    artifact_id: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class ArtifactOutboxRecord(BaseModel):
    """Durable hand-off from a stored artifact to the embedding queue."""

    outbox_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    artifact_id: str
    event_type: str = "embedding_requested"
    content_sha256: str | None = None
    status: str = "pending"
    attempt_count: int = 0
    next_attempt_at: str | None = None
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    last_error: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None


class EmbeddingJobQueueReport(BaseModel):
    schema_version: str = "littrace.embedding_job_queue_report.v1"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    queued: int = 0
    running: int = 0
    failed: int = 0
    dead: int = 0
    completed: int = 0
    total: int = 0
    reclaimable_running: int = 0
    ready_to_claim: int = 0
    oldest_ready_at: str | None = None
    latest_error: str | None = None


class StateStore(Protocol):
    def upsert_session(self, record: SessionStateRecord) -> SessionStateRecord:
        ...

    def get_session(self, session_id: str) -> SessionStateRecord | None:
        ...

    def list_sessions(self, *, limit: int = 20) -> list[SessionSummaryRecord]:
        ...

    def upsert_message(self, record: SessionMessageRecord) -> SessionMessageRecord:
        ...

    def list_messages(self, session_id: str) -> list[SessionMessageRecord]:
        ...

    def upsert_memory(
        self,
        session_id: str,
        *,
        memory_json: dict[str, object],
    ) -> dict[str, object]:
        ...

    def load_memory(
        self,
        session_id: str,
    ) -> dict[str, object] | None:
        ...

    def upsert_snapshot(self, record: SessionSnapshotRecord) -> SessionSnapshotRecord:
        ...

    def enqueue_embedding_job(self, record: EmbeddingJobRecord) -> EmbeddingJobRecord:
        ...

    def list_pending_embedding_jobs(self, *, limit: int = 20) -> list[EmbeddingJobRecord]:
        ...

    def list_embedding_jobs(
        self,
        *,
        status: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[EmbeddingJobRecord]:
        ...

    def embedding_job_queue_report(self) -> EmbeddingJobQueueReport:
        ...

    def requeue_dead_embedding_jobs(
        self,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> int:
        ...

    def claim_pending_embedding_jobs(
        self,
        *,
        worker_id: str,
        limit: int = 20,
        lease_seconds: float = 300.0,
    ) -> list[EmbeddingJobRecord]:
        ...

    def update_embedding_job(self, record: EmbeddingJobRecord) -> EmbeddingJobRecord:
        ...

    def append_paper_lifecycle_event(
        self, record: PaperLifecycleEventRecord
    ) -> PaperLifecycleEventRecord:
        ...

    def list_paper_lifecycle_events(
        self, session_id: str, *, paper_id: str | None = None, since: str | None = None
    ) -> list[PaperLifecycleEventRecord]:
        ...

    def enqueue_artifact_outbox(self, record: ArtifactOutboxRecord) -> ArtifactOutboxRecord:
        ...

    def claim_artifact_outbox(
        self, *, worker_id: str, limit: int = 20, lease_seconds: float = 300.0
    ) -> list[ArtifactOutboxRecord]:
        ...

    def update_artifact_outbox(self, record: ArtifactOutboxRecord) -> ArtifactOutboxRecord:
        ...

    def delete_session(self, session_id: str) -> int:
        ...


@dataclass
class PostgresStateStore:
    dsn: str
    schema_name: str = "littrace"
    _initialized: bool = False

    def upsert_session(self, record: SessionStateRecord) -> SessionStateRecord:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.sessions (
                    session_id, root_path, workspace_sha256, workspace_json,
                    manifest_json, artifact_index_json, memory_json, rag_profile_json,
                    revision, structured_document_count, workspace_snapshot_count,
                    created_at, updated_at, payload
                )
                VALUES (
                    %(session_id)s, %(root_path)s, %(workspace_sha256)s,
                    %(workspace_json)s, %(manifest_json)s, %(artifact_index_json)s,
                    %(memory_json)s, %(rag_profile_json)s, %(revision)s,
                    %(structured_document_count)s, %(workspace_snapshot_count)s,
                    %(created_at)s, %(updated_at)s, %(payload)s
                )
                ON CONFLICT (session_id) DO UPDATE SET
                    root_path = EXCLUDED.root_path,
                    workspace_sha256 = EXCLUDED.workspace_sha256,
                    workspace_json = EXCLUDED.workspace_json,
                    manifest_json = EXCLUDED.manifest_json,
                    artifact_index_json = EXCLUDED.artifact_index_json,
                    memory_json = EXCLUDED.memory_json,
                    rag_profile_json = EXCLUDED.rag_profile_json,
                    revision = EXCLUDED.revision,
                    structured_document_count = EXCLUDED.structured_document_count,
                    workspace_snapshot_count = EXCLUDED.workspace_snapshot_count,
                    updated_at = EXCLUDED.updated_at,
                    payload = EXCLUDED.payload
                """,
                _session_row(record),
            )
            conn.commit()
        return record

    def get_session(self, session_id: str) -> SessionStateRecord | None:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.sessions
                WHERE session_id = %(session_id)s
                """,
                {"session_id": session_id},
            ).fetchone()
        if row is None:
            return None
        return _record_from_payload(row[0], SessionStateRecord)

    def list_sessions(
        self,
        *,
        limit: int = 20,
    ) -> list[SessionSummaryRecord]:
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.sessions
                ORDER BY updated_at DESC
                LIMIT %(limit)s
                """,
                {"limit": limit},
            ).fetchall()
        return [
            SessionSummaryRecord(
                session_id=record.session_id,
                root_path=record.root_path,
                updated_at=record.updated_at,
                workspace_sha256=record.workspace_sha256,
                revision=record.revision,
                structured_document_count=record.structured_document_count,
                workspace_snapshot_count=record.workspace_snapshot_count,
            )
            for row in rows
            if (record := _record_from_payload(row[0], SessionStateRecord)) is not None
        ]

    def upsert_message(self, record: SessionMessageRecord) -> SessionMessageRecord:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.session_messages (
                    message_id, session_id, role, content_json, content_text,
                    created_at, updated_at, payload
                )
                VALUES (
                    %(message_id)s, %(session_id)s, %(role)s,
                    %(content_json)s, %(content_text)s, %(created_at)s, %(updated_at)s,
                    %(payload)s
                )
                ON CONFLICT (message_id) DO UPDATE SET
                    role = EXCLUDED.role,
                    content_json = EXCLUDED.content_json,
                    content_text = EXCLUDED.content_text,
                    updated_at = EXCLUDED.updated_at,
                    payload = EXCLUDED.payload
                """,
                _message_row(record),
            )
            conn.commit()
        return record

    def list_messages(self, session_id: str) -> list[SessionMessageRecord]:
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.session_messages
                WHERE session_id = %(session_id)s
                ORDER BY created_at ASC
                """,
                {"session_id": session_id},
            ).fetchall()
        return [
            record
            for row in rows
            if (record := _record_from_payload(row[0], SessionMessageRecord)) is not None
        ]

    def upsert_memory(
        self,
        session_id: str,
        *,
        memory_json: dict[str, object],
    ) -> dict[str, object]:
        self._ensure_schema()
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.session_memory (
                    session_id, memory_json, updated_at
                )
                VALUES (
                    %(session_id)s, %(memory_json)s, %(updated_at)s
                )
                ON CONFLICT (session_id) DO UPDATE SET
                    memory_json = EXCLUDED.memory_json,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "session_id": session_id,
                    "memory_json": _jsonb(memory_json),
                    "updated_at": now,
                },
            )
            conn.commit()
        return memory_json

    def load_memory(
        self,
        session_id: str,
    ) -> dict[str, object] | None:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT memory_json
                FROM {self.schema_name}.session_memory
                WHERE session_id = %(session_id)s
                """,
                {"session_id": session_id},
            ).fetchone()
        if row is None:
            return None
        record = _record_from_payload(row[0], dict)
        return record if isinstance(record, dict) else None

    def upsert_snapshot(self, record: SessionSnapshotRecord) -> SessionSnapshotRecord:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.session_snapshots (
                    snapshot_id, session_id, revision, workspace_sha256,
                    workspace_json, created_at, payload
                )
                VALUES (
                    %(snapshot_id)s, %(session_id)s, %(revision)s,
                    %(workspace_sha256)s, %(workspace_json)s, %(created_at)s, %(payload)s
                )
                ON CONFLICT (snapshot_id) DO UPDATE SET
                    revision = EXCLUDED.revision,
                    workspace_sha256 = EXCLUDED.workspace_sha256,
                    workspace_json = EXCLUDED.workspace_json,
                    payload = EXCLUDED.payload
                """,
                _snapshot_row(record),
            )
            conn.commit()
        return record

    def enqueue_embedding_job(self, record: EmbeddingJobRecord) -> EmbeddingJobRecord:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.embedding_jobs (
                    job_id, profile_id, session_id, artifact_id, source_revision,
                    content_sha256, status, attempt_count, next_attempt_at, last_error,
                    lease_owner, lease_expires_at, last_heartbeat_at,
                    result_json, created_at, updated_at, completed_at, payload
                )
                VALUES (
                    %(job_id)s, %(profile_id)s, %(session_id)s, %(artifact_id)s,
                    %(source_revision)s, %(content_sha256)s, %(status)s, %(attempt_count)s,
                    %(next_attempt_at)s, %(last_error)s, %(lease_owner)s, %(lease_expires_at)s,
                    %(last_heartbeat_at)s, %(result_json)s, %(created_at)s,
                    %(updated_at)s, %(completed_at)s, %(payload)s
                )
                ON CONFLICT (job_id) DO UPDATE SET
                    status = CASE
                        WHEN {self.schema_name}.embedding_jobs.status = 'completed'
                         AND {self.schema_name}.embedding_jobs.content_sha256 IS NOT DISTINCT FROM EXCLUDED.content_sha256
                         AND {self.schema_name}.embedding_jobs.source_revision IS NOT DISTINCT FROM EXCLUDED.source_revision
                        THEN {self.schema_name}.embedding_jobs.status
                        ELSE EXCLUDED.status
                    END,
                    attempt_count = CASE
                        WHEN {self.schema_name}.embedding_jobs.status = 'completed'
                         AND {self.schema_name}.embedding_jobs.content_sha256 IS NOT DISTINCT FROM EXCLUDED.content_sha256
                         AND {self.schema_name}.embedding_jobs.source_revision IS NOT DISTINCT FROM EXCLUDED.source_revision
                        THEN {self.schema_name}.embedding_jobs.attempt_count
                        ELSE EXCLUDED.attempt_count
                    END,
                    next_attempt_at = CASE
                        WHEN {self.schema_name}.embedding_jobs.status = 'completed'
                         AND {self.schema_name}.embedding_jobs.content_sha256 IS NOT DISTINCT FROM EXCLUDED.content_sha256
                         AND {self.schema_name}.embedding_jobs.source_revision IS NOT DISTINCT FROM EXCLUDED.source_revision
                        THEN {self.schema_name}.embedding_jobs.next_attempt_at
                        ELSE EXCLUDED.next_attempt_at
                    END,
                    last_error = CASE
                        WHEN {self.schema_name}.embedding_jobs.status = 'completed'
                         AND {self.schema_name}.embedding_jobs.content_sha256 IS NOT DISTINCT FROM EXCLUDED.content_sha256
                         AND {self.schema_name}.embedding_jobs.source_revision IS NOT DISTINCT FROM EXCLUDED.source_revision
                        THEN {self.schema_name}.embedding_jobs.last_error
                        ELSE EXCLUDED.last_error
                    END,
                    lease_owner = EXCLUDED.lease_owner,
                    lease_expires_at = EXCLUDED.lease_expires_at,
                    last_heartbeat_at = EXCLUDED.last_heartbeat_at,
                    result_json = CASE
                        WHEN {self.schema_name}.embedding_jobs.status = 'completed'
                         AND {self.schema_name}.embedding_jobs.content_sha256 IS NOT DISTINCT FROM EXCLUDED.content_sha256
                         AND {self.schema_name}.embedding_jobs.source_revision IS NOT DISTINCT FROM EXCLUDED.source_revision
                        THEN {self.schema_name}.embedding_jobs.result_json
                        ELSE EXCLUDED.result_json
                    END,
                    updated_at = EXCLUDED.updated_at,
                    completed_at = CASE
                        WHEN {self.schema_name}.embedding_jobs.status = 'completed'
                         AND {self.schema_name}.embedding_jobs.content_sha256 IS NOT DISTINCT FROM EXCLUDED.content_sha256
                         AND {self.schema_name}.embedding_jobs.source_revision IS NOT DISTINCT FROM EXCLUDED.source_revision
                        THEN {self.schema_name}.embedding_jobs.completed_at
                        ELSE EXCLUDED.completed_at
                    END,
                    payload = EXCLUDED.payload
                """,
                _embedding_job_row(record),
            )
            conn.commit()
        return record

    def claim_pending_embedding_jobs(
        self,
        *,
        worker_id: str,
        limit: int = 20,
        lease_seconds: float = 300.0,
    ) -> list[EmbeddingJobRecord]:
        self._ensure_schema()
        lease_seconds = max(lease_seconds, 1.0)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.embedding_jobs
                WHERE (
                    status IN ('queued', 'failed')
                    AND (next_attempt_at IS NULL OR next_attempt_at <= now())
                )
                OR (
                    status = 'running'
                    AND lease_expires_at IS NOT NULL
                    AND lease_expires_at <= now()
                )
                ORDER BY updated_at ASC
                LIMIT %(limit)s
                FOR UPDATE SKIP LOCKED
                """,
                {"limit": limit},
            ).fetchall()
            claimed: list[EmbeddingJobRecord] = []
            for row in rows:
                record = _record_from_payload(row[0], EmbeddingJobRecord)
                if record is None:
                    continue
                now = datetime.now(UTC)
                record.status = "running"
                record.attempt_count += 1
                record.lease_owner = worker_id
                record.lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
                record.last_heartbeat_at = now.isoformat()
                record.updated_at = now.isoformat()
                conn.execute(
                    f"""
                    UPDATE {self.schema_name}.embedding_jobs
                    SET status = %(status)s,
                        attempt_count = %(attempt_count)s,
                        lease_owner = %(lease_owner)s,
                        lease_expires_at = %(lease_expires_at)s,
                        last_heartbeat_at = %(last_heartbeat_at)s,
                        updated_at = %(updated_at)s,
                        payload = %(payload)s
                    WHERE job_id = %(job_id)s
                    """,
                    _embedding_job_row(record),
                )
                claimed.append(record)
            conn.commit()
        return claimed

    def list_pending_embedding_jobs(self, *, limit: int = 20) -> list[EmbeddingJobRecord]:
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.embedding_jobs
                WHERE status IN ('queued', 'failed')
                  AND (next_attempt_at IS NULL OR next_attempt_at <= now())
                ORDER BY updated_at ASC
                LIMIT %(limit)s
                """,
                {"limit": limit},
            ).fetchall()
        return [
            record
            for row in rows
            if (record := _record_from_payload(row[0], EmbeddingJobRecord)) is not None
        ]

    def list_embedding_jobs(
        self,
        *,
        status: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[EmbeddingJobRecord]:
        self._ensure_schema()
        params: dict[str, object] = {"limit": limit}
        clauses: list[str] = []
        if status is not None:
            clauses.append("status = %(status)s")
            params["status"] = status
        if session_id is not None:
            clauses.append("session_id = %(session_id)s")
            params["session_id"] = session_id
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.embedding_jobs
                {where_clause}
                ORDER BY updated_at DESC
                LIMIT %(limit)s
                """,
                params,
            ).fetchall()
        return [
            record
            for row in rows
            if (record := _record_from_payload(row[0], EmbeddingJobRecord)) is not None
        ]

    def embedding_job_queue_report(self) -> EmbeddingJobQueueReport:
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT status, COUNT(*)
                FROM {self.schema_name}.embedding_jobs
                GROUP BY status
                """
            ).fetchall()
            ready_row = conn.execute(
                f"""
                SELECT COUNT(*), MIN(updated_at)
                FROM {self.schema_name}.embedding_jobs
                WHERE (
                    status IN ('queued', 'failed')
                    AND (next_attempt_at IS NULL OR next_attempt_at <= now())
                )
                """
            ).fetchone()
            reclaimable_row = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM {self.schema_name}.embedding_jobs
                WHERE status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= now()
                """
            ).fetchone()
            error_row = conn.execute(
                f"""
                SELECT last_error
                FROM {self.schema_name}.embedding_jobs
                WHERE last_error IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()
        report = EmbeddingJobQueueReport()
        for status, count in rows:
            value = int(count)
            if status == "queued":
                report.queued = value
            elif status == "running":
                report.running = value
            elif status == "failed":
                report.failed = value
            elif status == "dead":
                report.dead = value
            elif status == "completed":
                report.completed = value
            report.total += value
        if ready_row is not None:
            report.ready_to_claim = int(ready_row[0] or 0)
            oldest = ready_row[1]
            if isinstance(oldest, datetime):
                report.oldest_ready_at = oldest.isoformat()
            elif oldest is not None:
                report.oldest_ready_at = str(oldest)
        if reclaimable_row is not None:
            report.reclaimable_running = int(reclaimable_row[0] or 0)
        if error_row is not None:
            report.latest_error = error_row[0]
        return report

    def requeue_dead_embedding_jobs(
        self,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> int:
        self._ensure_schema()
        params: dict[str, object] = {"limit": limit}
        session_clause = ""
        if session_id is not None:
            session_clause = "AND session_id = %(session_id)s"
            params["session_id"] = session_id
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.embedding_jobs
                WHERE status = 'dead'
                {session_clause}
                ORDER BY updated_at ASC
                LIMIT %(limit)s
                FOR UPDATE SKIP LOCKED
                """,
                params,
            ).fetchall()
            count = 0
            for row in rows:
                record = _record_from_payload(row[0], EmbeddingJobRecord)
                if record is None:
                    continue
                record.status = "queued"
                record.next_attempt_at = None
                record.last_error = None
                record.lease_owner = None
                record.lease_expires_at = None
                record.last_heartbeat_at = None
                record.completed_at = None
                record.updated_at = now
                conn.execute(
                    f"""
                    UPDATE {self.schema_name}.embedding_jobs
                    SET status = %(status)s,
                        next_attempt_at = %(next_attempt_at)s,
                        last_error = %(last_error)s,
                        lease_owner = %(lease_owner)s,
                        lease_expires_at = %(lease_expires_at)s,
                        last_heartbeat_at = %(last_heartbeat_at)s,
                        updated_at = %(updated_at)s,
                        completed_at = %(completed_at)s,
                        payload = %(payload)s
                    WHERE job_id = %(job_id)s
                    """,
                    _embedding_job_row(record),
                )
                count += 1
            conn.commit()
        return count

    def update_embedding_job(self, record: EmbeddingJobRecord) -> EmbeddingJobRecord:
        return self.enqueue_embedding_job(record)

    def append_paper_lifecycle_event(
        self, record: PaperLifecycleEventRecord
    ) -> PaperLifecycleEventRecord:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.paper_lifecycle_events (
                    event_id, session_id, paper_id, event_type, occurred_at,
                    task_id, artifact_id, payload
                ) VALUES (
                    %(event_id)s, %(session_id)s, %(paper_id)s, %(event_type)s,
                    %(occurred_at)s, %(task_id)s, %(artifact_id)s, %(payload)s
                ) ON CONFLICT (event_id) DO NOTHING
                """,
                _lifecycle_event_row(record),
            )
            conn.commit()
        return record

    def list_paper_lifecycle_events(
        self,
        session_id: str,
        *,
        paper_id: str | None = None,
        since: str | None = None,
    ) -> list[PaperLifecycleEventRecord]:
        self._ensure_schema()
        params: dict[str, object] = {"session_id": session_id}
        clauses = ["session_id = %(session_id)s"]
        if paper_id is not None:
            clauses.append("paper_id = %(paper_id)s")
            params["paper_id"] = paper_id
        if since is not None:
            clauses.append("occurred_at >= %(since)s")
            params["since"] = since
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT event_id, session_id, paper_id, event_type, occurred_at,
                       task_id, artifact_id, payload
                FROM {self.schema_name}.paper_lifecycle_events
                WHERE {' AND '.join(clauses)}
                ORDER BY occurred_at ASC, event_id ASC
                """,
                params,
            ).fetchall()
        return [_lifecycle_event_from_row(row) for row in rows]

    def enqueue_artifact_outbox(self, record: ArtifactOutboxRecord) -> ArtifactOutboxRecord:
        """Idempotently enqueue an artifact revision for embedding."""
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.artifact_outbox (
                    outbox_id, session_id, artifact_id, event_type, content_sha256,
                    status, attempt_count, next_attempt_at, lease_owner,
                    lease_expires_at, last_error, payload, created_at, updated_at,
                    completed_at
                ) VALUES (
                    %(outbox_id)s, %(session_id)s, %(artifact_id)s, %(event_type)s,
                    %(content_sha256)s, %(status)s, %(attempt_count)s,
                    %(next_attempt_at)s, %(lease_owner)s, %(lease_expires_at)s,
                    %(last_error)s, %(payload)s, %(created_at)s, %(updated_at)s,
                    %(completed_at)s
                ) ON CONFLICT (session_id, artifact_id, event_type, content_sha256)
                DO UPDATE SET
                    updated_at = EXCLUDED.updated_at,
                    payload = EXCLUDED.payload,
                    status = CASE
                        WHEN {self.schema_name}.artifact_outbox.status IN ('completed', 'running')
                        THEN {self.schema_name}.artifact_outbox.status
                        ELSE 'pending'
                    END,
                    next_attempt_at = CASE
                        WHEN {self.schema_name}.artifact_outbox.status IN ('completed', 'running')
                        THEN {self.schema_name}.artifact_outbox.next_attempt_at
                        ELSE NULL
                    END
                """,
                _outbox_row(record),
            )
            conn.commit()
        return record

    def claim_artifact_outbox(
        self,
        *,
        worker_id: str,
        limit: int = 20,
        lease_seconds: float = 300.0,
    ) -> list[ArtifactOutboxRecord]:
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT outbox_id, session_id, artifact_id, event_type, content_sha256,
                       status, attempt_count, next_attempt_at, lease_owner,
                       lease_expires_at, last_error, payload, created_at, updated_at,
                       completed_at
                FROM {self.schema_name}.artifact_outbox
                WHERE (
                    status IN ('pending', 'failed')
                    AND (next_attempt_at IS NULL OR next_attempt_at <= now())
                ) OR (
                    status = 'running' AND lease_expires_at IS NOT NULL
                    AND lease_expires_at <= now()
                )
                ORDER BY updated_at ASC
                LIMIT %(limit)s
                FOR UPDATE SKIP LOCKED
                """,
                {"limit": limit},
            ).fetchall()
            claimed: list[ArtifactOutboxRecord] = []
            for row in rows:
                record = _outbox_from_row(row)
                now = datetime.now(UTC)
                record.status = "running"
                record.attempt_count += 1
                record.lease_owner = worker_id
                record.lease_expires_at = (now + timedelta(seconds=max(lease_seconds, 1))).isoformat()
                record.updated_at = now.isoformat()
                conn.execute(
                    f"""UPDATE {self.schema_name}.artifact_outbox SET
                        status = %(status)s, attempt_count = %(attempt_count)s,
                        next_attempt_at = %(next_attempt_at)s, lease_owner = %(lease_owner)s,
                        lease_expires_at = %(lease_expires_at)s, last_error = %(last_error)s,
                        payload = %(payload)s, updated_at = %(updated_at)s,
                        completed_at = %(completed_at)s
                    WHERE outbox_id = %(outbox_id)s""",
                    _outbox_row(record),
                )
                claimed.append(record)
            conn.commit()
        return claimed

    def update_artifact_outbox(self, record: ArtifactOutboxRecord) -> ArtifactOutboxRecord:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""UPDATE {self.schema_name}.artifact_outbox SET
                    status = %(status)s, attempt_count = %(attempt_count)s,
                    next_attempt_at = %(next_attempt_at)s, lease_owner = %(lease_owner)s,
                    lease_expires_at = %(lease_expires_at)s, last_error = %(last_error)s,
                    payload = %(payload)s, updated_at = %(updated_at)s,
                    completed_at = %(completed_at)s
                WHERE outbox_id = %(outbox_id)s""",
                _outbox_row(record),
            )
            conn.commit()
        return record

    def delete_session(self, session_id: str) -> int:
        self._ensure_schema()
        deleted = 0
        with self._connect() as conn:
            for table in (
                "artifact_outbox",
                "paper_lifecycle_events",
                "embedding_jobs",
                "session_snapshots",
                "session_messages",
                "session_memory",
                "sessions",
            ):
                result = conn.execute(
                    f"""
                    DELETE FROM {self.schema_name}.{table}
                    WHERE session_id = %(session_id)s
                    """,
                    {"session_id": session_id},
                )
                deleted += result.rowcount if result.rowcount >= 0 else 0
            conn.commit()
        return deleted

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._connect() as conn:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.sessions (
                    session_id TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL,
                    workspace_sha256 TEXT,
                    workspace_json JSONB NOT NULL,
                    manifest_json JSONB NOT NULL,
                    artifact_index_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    memory_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    rag_profile_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    revision INTEGER NOT NULL DEFAULT 0,
                    structured_document_count INTEGER NOT NULL DEFAULT 0,
                    workspace_snapshot_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    payload JSONB NOT NULL
                )
                """
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.sessions ADD COLUMN IF NOT EXISTS payload JSONB"
            )
            conn.execute(
                f"""
                UPDATE {self.schema_name}.sessions
                SET payload = jsonb_build_object(
                    'session_id', session_id,
                    'root_path', root_path,
                    'workspace_sha256', workspace_sha256,
                    'workspace_json', workspace_json,
                    'manifest_json', manifest_json,
                    'artifact_index_json', artifact_index_json,
                    'memory_json', memory_json,
                    'rag_profile_json', rag_profile_json,
                    'revision', revision,
                    'structured_document_count', structured_document_count,
                    'workspace_snapshot_count', workspace_snapshot_count,
                    'created_at', created_at,
                    'updated_at', updated_at
                )
                WHERE payload IS NULL
                """
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.sessions ALTER COLUMN payload SET NOT NULL"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.session_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_json JSONB NOT NULL,
                    content_text TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    payload JSONB NOT NULL
                )
                """
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.session_messages ADD COLUMN IF NOT EXISTS payload JSONB"
            )
            conn.execute(
                f"""
                UPDATE {self.schema_name}.session_messages
                SET payload = jsonb_build_object(
                    'message_id', message_id,
                    'session_id', session_id,
                    'role', role,
                    'content_json', content_json,
                    'content_text', content_text,
                    'created_at', created_at,
                    'updated_at', updated_at
                )
                WHERE payload IS NULL
                """
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.session_messages ALTER COLUMN payload SET NOT NULL"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.session_memory (
                    session_id TEXT PRIMARY KEY,
                    memory_json JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.session_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    workspace_sha256 TEXT NOT NULL,
                    workspace_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    payload JSONB NOT NULL
                )
                """
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.session_snapshots ADD COLUMN IF NOT EXISTS payload JSONB"
            )
            conn.execute(
                f"""
                UPDATE {self.schema_name}.session_snapshots
                SET payload = jsonb_build_object(
                    'snapshot_id', snapshot_id,
                    'session_id', session_id,
                    'revision', revision,
                    'workspace_sha256', workspace_sha256,
                    'workspace_json', workspace_json,
                    'created_at', created_at
                )
                WHERE payload IS NULL
                """
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.session_snapshots ALTER COLUMN payload SET NOT NULL"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.embedding_jobs (
                    job_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    source_revision TEXT,
                    content_sha256 TEXT,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TIMESTAMPTZ,
                    last_error TEXT,
                    lease_owner TEXT,
                    lease_expires_at TIMESTAMPTZ,
                    last_heartbeat_at TIMESTAMPTZ,
                    result_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ,
                    payload JSONB NOT NULL
                )
                """
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.embedding_jobs ADD COLUMN IF NOT EXISTS payload JSONB"
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.embedding_jobs ADD COLUMN IF NOT EXISTS lease_owner TEXT"
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.embedding_jobs ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ"
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.embedding_jobs ADD COLUMN IF NOT EXISTS last_heartbeat_at TIMESTAMPTZ"
            )
            conn.execute(
                f"""
                UPDATE {self.schema_name}.embedding_jobs
                SET payload = jsonb_build_object(
                    'job_id', job_id,
                    'profile_id', profile_id,
                    'session_id', session_id,
                    'artifact_id', artifact_id,
                    'source_revision', source_revision,
                    'content_sha256', content_sha256,
                    'status', status,
                    'attempt_count', attempt_count,
                    'next_attempt_at', next_attempt_at,
                    'last_error', last_error,
                    'lease_owner', lease_owner,
                    'lease_expires_at', lease_expires_at,
                    'last_heartbeat_at', last_heartbeat_at,
                    'result_json', result_json,
                    'created_at', created_at,
                    'updated_at', updated_at,
                    'completed_at', completed_at
                )
                WHERE payload IS NULL
                """
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.embedding_jobs ADD COLUMN IF NOT EXISTS lease_owner TEXT"
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.embedding_jobs ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ"
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.embedding_jobs ADD COLUMN IF NOT EXISTS last_heartbeat_at TIMESTAMPTZ"
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.embedding_jobs ALTER COLUMN payload SET NOT NULL"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.paper_lifecycle_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at TIMESTAMPTZ NOT NULL,
                    task_id TEXT,
                    artifact_id TEXT,
                    payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.artifact_outbox (
                    outbox_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TIMESTAMPTZ,
                    lease_owner TEXT,
                    lease_expires_at TIMESTAMPTZ,
                    last_error TEXT,
                    payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ
                )
                """
            )
            conn.execute(
                f"UPDATE {self.schema_name}.artifact_outbox SET content_sha256 = '' WHERE content_sha256 IS NULL"
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.artifact_outbox ALTER COLUMN content_sha256 SET DEFAULT ''"
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.artifact_outbox ALTER COLUMN content_sha256 SET NOT NULL"
            )
            for table in (
                "sessions",
                "session_messages",
                "session_memory",
                "session_snapshots",
                "embedding_jobs",
                "artifact_outbox",
                "paper_lifecycle_events",
            ):
                conn.execute(
                    f"ALTER TABLE {self.schema_name}.{table} DROP COLUMN IF EXISTS user_id CASCADE"
                )
            conn.execute(f"DROP TABLE IF EXISTS {self.schema_name}.session_permissions")
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS session_messages_session_idx ON {self.schema_name}.session_messages (session_id, created_at ASC)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS embedding_jobs_pending_idx ON {self.schema_name}.embedding_jobs (status, next_attempt_at, updated_at)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS embedding_jobs_lease_idx ON {self.schema_name}.embedding_jobs (lease_expires_at, updated_at)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS paper_lifecycle_session_paper_idx ON {self.schema_name}.paper_lifecycle_events (session_id, paper_id, occurred_at)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS paper_lifecycle_session_event_idx ON {self.schema_name}.paper_lifecycle_events (session_id, event_type, occurred_at)"
            )
            conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS artifact_outbox_idempotency_idx ON {self.schema_name}.artifact_outbox (session_id, artifact_id, event_type, content_sha256)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS artifact_outbox_pending_idx ON {self.schema_name}.artifact_outbox (status, next_attempt_at, updated_at)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS artifact_outbox_lease_idx ON {self.schema_name}.artifact_outbox (lease_expires_at, updated_at)"
            )
            conn.commit()
        self._initialized = True

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "Postgres state storage requires the optional rag extra: "
                "pip install -e '.[rag]'"
            ) from exc
        return psycopg.connect(self.dsn)


def state_store_from_config(config: LitTraceConfig) -> PostgresStateStore:
    if config.metadata_store.backend != "postgres":
        raise ValueError("metadata_store.backend must be 'postgres' for state storage.")
    if not config.metadata_store.postgres_dsn:
        raise ValueError("metadata_store.postgres_dsn is required for Postgres state storage.")
    return PostgresStateStore(
        config.metadata_store.postgres_dsn,
        schema_name=config.metadata_store.schema_name,
    )


def _session_row(record: SessionStateRecord) -> dict[str, object]:
    row = record.model_dump(mode="json")
    row["workspace_json"] = _jsonb(record.workspace_json)
    row["manifest_json"] = _jsonb(record.manifest_json)
    row["artifact_index_json"] = _jsonb(record.artifact_index_json)
    row["memory_json"] = _jsonb(record.memory_json)
    row["rag_profile_json"] = _jsonb(record.rag_profile_json)
    row["payload"] = _jsonb(record.model_dump(mode="json"))
    return row


def _message_row(record: SessionMessageRecord) -> dict[str, object]:
    row = record.model_dump(mode="json")
    row["content_json"] = _jsonb(record.content_json)
    row["payload"] = _jsonb(record.model_dump(mode="json"))
    return row


def _snapshot_row(record: SessionSnapshotRecord) -> dict[str, object]:
    row = record.model_dump(mode="json")
    row["workspace_json"] = _jsonb(record.workspace_json)
    row["payload"] = _jsonb(record.model_dump(mode="json"))
    return row


def _embedding_job_row(record: EmbeddingJobRecord) -> dict[str, object]:
    row = record.model_dump(mode="json")
    row["result_json"] = _jsonb(record.result_json)
    row["payload"] = _jsonb(record.model_dump(mode="json"))
    return row


def _lifecycle_event_row(record: PaperLifecycleEventRecord) -> dict[str, object]:
    row = record.model_dump(mode="json")
    row["payload"] = _jsonb(record.payload)
    return row


def _lifecycle_event_from_row(row: object) -> PaperLifecycleEventRecord:
    values = list(row)  # psycopg's default tuple row factory
    payload = _record_from_payload(values[7], dict) or {}
    occurred_at = values[4]
    return PaperLifecycleEventRecord(
        event_id=str(values[0]), session_id=str(values[1]), paper_id=str(values[2]),
        event_type=str(values[3]),
        occurred_at=occurred_at.isoformat() if isinstance(occurred_at, datetime) else str(occurred_at),
        task_id=values[5], artifact_id=values[6], payload=payload,
    )


def _outbox_row(record: ArtifactOutboxRecord) -> dict[str, object]:
    row = record.model_dump(mode="json")
    row["content_sha256"] = record.content_sha256 or ""
    row["payload"] = _jsonb(record.payload)
    return row


def _outbox_from_row(row: object) -> ArtifactOutboxRecord:
    values = list(row)
    payload = _record_from_payload(values[11], dict) or {}

    def timestamp(index: int) -> str | None:
        value = values[index]
        if value is None:
            return None
        return value.isoformat() if isinstance(value, datetime) else str(value)

    return ArtifactOutboxRecord(
        outbox_id=str(values[0]), session_id=str(values[1]), artifact_id=str(values[2]),
        event_type=str(values[3]), content_sha256=str(values[4]) or None,
        status=str(values[5]), attempt_count=int(values[6]), next_attempt_at=timestamp(7),
        lease_owner=values[8], lease_expires_at=timestamp(9), last_error=values[10],
        payload=payload, created_at=timestamp(12) or datetime.now(UTC).isoformat(),
        updated_at=timestamp(13) or datetime.now(UTC).isoformat(), completed_at=timestamp(14),
    )


def _jsonb(value: Any) -> Any:
    try:
        from psycopg.types.json import Jsonb
    except ImportError:
        return json.dumps(value)
    return Jsonb(value)


def _record_from_payload(payload: object, model: type[BaseModel] | type[dict[str, object]]):
    try:
        if model is dict:
            if isinstance(payload, str):
                return json.loads(payload)
            return payload if isinstance(payload, dict) else None
        if isinstance(payload, str):
            return model.model_validate_json(payload)  # type: ignore[attr-defined]
        return model.model_validate(payload)  # type: ignore[attr-defined]
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
        return None
