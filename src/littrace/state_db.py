from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from littrace.config import LitTraceConfig


class SessionStateRecord(BaseModel):
    session_id: str
    user_id: str
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
    user_id: str
    root_path: str
    updated_at: str
    workspace_sha256: str | None = None
    revision: int = 0
    structured_document_count: int = 0
    workspace_snapshot_count: int = 0


class SessionMessageRecord(BaseModel):
    message_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    user_id: str
    role: str
    content_json: dict[str, object] = Field(default_factory=dict)
    content_text: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SessionSnapshotRecord(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    user_id: str
    revision: int
    workspace_sha256: str
    workspace_json: dict[str, object] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SessionPermissionGrant(BaseModel):
    grant_id: str = Field(default_factory=lambda: uuid4().hex)
    scope: str
    resource_key: str
    owner_user_id: str
    grantee_user_id: str
    role: str = "viewer"
    granted_by: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    expires_at: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class EmbeddingJobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid4().hex)
    profile_id: str
    user_id: str
    session_id: str
    artifact_id: str
    source_revision: str | None = None
    content_sha256: str | None = None
    status: str = "queued"
    attempt_count: int = 0
    next_attempt_at: str | None = None
    last_error: str | None = None
    result_json: dict[str, object] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None


class StateStore(Protocol):
    def upsert_session(self, record: SessionStateRecord) -> SessionStateRecord:
        ...

    def get_session(self, session_id: str, *, user_id: str | None = None) -> SessionStateRecord | None:
        ...

    def list_sessions(self, *, user_id: str | None = None, limit: int = 20) -> list[SessionSummaryRecord]:
        ...

    def upsert_message(self, record: SessionMessageRecord) -> SessionMessageRecord:
        ...

    def list_messages(self, session_id: str, *, user_id: str | None = None) -> list[SessionMessageRecord]:
        ...

    def upsert_memory(
        self,
        session_id: str,
        *,
        user_id: str,
        memory_json: dict[str, object],
    ) -> dict[str, object]:
        ...

    def load_memory(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, object] | None:
        ...

    def upsert_snapshot(self, record: SessionSnapshotRecord) -> SessionSnapshotRecord:
        ...

    def grant_permission(self, grant: SessionPermissionGrant) -> SessionPermissionGrant:
        ...

    def has_access(
        self,
        *,
        scope: str,
        resource_key: str,
        user_id: str,
        allowed_roles: tuple[str, ...] = ("owner", "editor", "viewer", "admin"),
    ) -> bool:
        ...

    def enqueue_embedding_job(self, record: EmbeddingJobRecord) -> EmbeddingJobRecord:
        ...

    def list_pending_embedding_jobs(self, *, limit: int = 20) -> list[EmbeddingJobRecord]:
        ...

    def update_embedding_job(self, record: EmbeddingJobRecord) -> EmbeddingJobRecord:
        ...

    def delete_session(self, session_id: str, *, user_id: str | None = None) -> int:
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
                    session_id, user_id, root_path, workspace_sha256, workspace_json,
                    manifest_json, artifact_index_json, memory_json, rag_profile_json,
                    revision, structured_document_count, workspace_snapshot_count,
                    created_at, updated_at, payload
                )
                VALUES (
                    %(session_id)s, %(user_id)s, %(root_path)s, %(workspace_sha256)s,
                    %(workspace_json)s, %(manifest_json)s, %(artifact_index_json)s,
                    %(memory_json)s, %(rag_profile_json)s, %(revision)s,
                    %(structured_document_count)s, %(workspace_snapshot_count)s,
                    %(created_at)s, %(updated_at)s, %(payload)s
                )
                ON CONFLICT (session_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
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

    def get_session(self, session_id: str, *, user_id: str | None = None) -> SessionStateRecord | None:
        self._ensure_schema()
        params: dict[str, object] = {"session_id": session_id}
        user_clause = ""
        if user_id is not None:
            user_clause = " AND user_id = %(user_id)s"
            params["user_id"] = user_id
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.sessions
                WHERE session_id = %(session_id)s{user_clause}
                """,
                params,
            ).fetchone()
        if row is None:
            return None
        return _record_from_payload(row[0], SessionStateRecord)

    def list_sessions(
        self,
        *,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[SessionSummaryRecord]:
        self._ensure_schema()
        params: dict[str, object] = {"limit": limit}
        user_clause = ""
        if user_id is not None:
            user_clause = "WHERE user_id = %(user_id)s"
            params["user_id"] = user_id
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.sessions
                {user_clause}
                ORDER BY updated_at DESC
                LIMIT %(limit)s
                """,
                params,
            ).fetchall()
        return [
            SessionSummaryRecord(
                session_id=record.session_id,
                user_id=record.user_id,
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
                    message_id, session_id, user_id, role, content_json, content_text,
                    created_at, updated_at, payload
                )
                VALUES (
                    %(message_id)s, %(session_id)s, %(user_id)s, %(role)s,
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

    def list_messages(self, session_id: str, *, user_id: str | None = None) -> list[SessionMessageRecord]:
        self._ensure_schema()
        params: dict[str, object] = {"session_id": session_id}
        user_clause = ""
        if user_id is not None:
            user_clause = " AND user_id = %(user_id)s"
            params["user_id"] = user_id
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.session_messages
                WHERE session_id = %(session_id)s{user_clause}
                ORDER BY created_at ASC
                """,
                params,
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
        user_id: str,
        memory_json: dict[str, object],
    ) -> dict[str, object]:
        self._ensure_schema()
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.session_memory (
                    session_id, user_id, memory_json, updated_at
                )
                VALUES (
                    %(session_id)s, %(user_id)s, %(memory_json)s, %(updated_at)s
                )
                ON CONFLICT (session_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    memory_json = EXCLUDED.memory_json,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "memory_json": _jsonb(memory_json),
                    "updated_at": now,
                },
            )
            conn.commit()
        return memory_json

    def load_memory(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, object] | None:
        self._ensure_schema()
        params: dict[str, object] = {"session_id": session_id}
        user_clause = ""
        if user_id is not None:
            user_clause = " AND user_id = %(user_id)s"
            params["user_id"] = user_id
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT memory_json
                FROM {self.schema_name}.session_memory
                WHERE session_id = %(session_id)s{user_clause}
                """,
                params,
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
                    snapshot_id, session_id, user_id, revision, workspace_sha256,
                    workspace_json, created_at, payload
                )
                VALUES (
                    %(snapshot_id)s, %(session_id)s, %(user_id)s, %(revision)s,
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

    def grant_permission(self, grant: SessionPermissionGrant) -> SessionPermissionGrant:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.session_permissions (
                    grant_id, scope, resource_key, owner_user_id, grantee_user_id, role,
                    granted_by, created_at, expires_at, metadata
                )
                VALUES (
                    %(grant_id)s, %(scope)s, %(resource_key)s, %(owner_user_id)s,
                    %(grantee_user_id)s, %(role)s, %(granted_by)s, %(created_at)s,
                    %(expires_at)s, %(metadata)s
                )
                ON CONFLICT (scope, resource_key, grantee_user_id) DO UPDATE SET
                    role = EXCLUDED.role,
                    granted_by = EXCLUDED.granted_by,
                    expires_at = EXCLUDED.expires_at,
                    metadata = EXCLUDED.metadata
                """,
                _permission_row(grant),
            )
            conn.commit()
        return grant

    def has_access(
        self,
        *,
        scope: str,
        resource_key: str,
        user_id: str,
        allowed_roles: tuple[str, ...] = ("owner", "editor", "viewer", "admin"),
    ) -> bool:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT owner_user_id, role, expires_at
                FROM {self.schema_name}.session_permissions
                WHERE scope = %(scope)s
                  AND resource_key = %(resource_key)s
                  AND grantee_user_id = %(user_id)s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                {"scope": scope, "resource_key": resource_key, "user_id": user_id},
            ).fetchone()
        if row is None:
            return False
        role = str(row[1])
        if role not in allowed_roles:
            return False
        expires_at = row[2]
        if isinstance(expires_at, datetime) and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if isinstance(expires_at, datetime) and expires_at <= datetime.now(UTC):
            return False
        return True

    def enqueue_embedding_job(self, record: EmbeddingJobRecord) -> EmbeddingJobRecord:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.embedding_jobs (
                    job_id, profile_id, user_id, session_id, artifact_id, source_revision,
                    content_sha256, status, attempt_count, next_attempt_at, last_error,
                    result_json, created_at, updated_at, completed_at, payload
                )
                VALUES (
                    %(job_id)s, %(profile_id)s, %(user_id)s, %(session_id)s, %(artifact_id)s,
                    %(source_revision)s, %(content_sha256)s, %(status)s, %(attempt_count)s,
                    %(next_attempt_at)s, %(last_error)s, %(result_json)s, %(created_at)s,
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

    def update_embedding_job(self, record: EmbeddingJobRecord) -> EmbeddingJobRecord:
        return self.enqueue_embedding_job(record)

    def delete_session(self, session_id: str, *, user_id: str | None = None) -> int:
        self._ensure_schema()
        params: dict[str, object] = {"session_id": session_id}
        user_clause = ""
        if user_id is not None:
            user_clause = " AND user_id = %(user_id)s"
            params["user_id"] = user_id
        deleted = 0
        with self._connect() as conn:
            for table in (
                "embedding_jobs",
                "session_snapshots",
                "session_messages",
                "session_memory",
                "sessions",
            ):
                result = conn.execute(
                    f"""
                    DELETE FROM {self.schema_name}.{table}
                    WHERE session_id = %(session_id)s{user_clause}
                    """,
                    params,
                )
                deleted += result.rowcount if result.rowcount >= 0 else 0
            conn.execute(
                f"""
                DELETE FROM {self.schema_name}.session_permissions
                WHERE (
                    scope = 'session'
                    AND resource_key = %(session_id)s
                )
                OR (
                    scope = 'artifact'
                    AND resource_key LIKE %(artifact_prefix)s
                    {user_clause}
                )
                """,
                {
                    **params,
                    "artifact_prefix": f"{session_id}:%",
                },
            )
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
                    user_id TEXT NOT NULL,
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
                    'user_id', user_id,
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
                    user_id TEXT NOT NULL,
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
                    'user_id', user_id,
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
                    user_id TEXT NOT NULL,
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
                    user_id TEXT NOT NULL,
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
                    'user_id', user_id,
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
                CREATE TABLE IF NOT EXISTS {self.schema_name}.session_permissions (
                    grant_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    resource_key TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    grantee_user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    granted_by TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    expires_at TIMESTAMPTZ,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    UNIQUE (scope, resource_key, grantee_user_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.embedding_jobs (
                    job_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    source_revision TEXT,
                    content_sha256 TEXT,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TIMESTAMPTZ,
                    last_error TEXT,
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
                f"""
                UPDATE {self.schema_name}.embedding_jobs
                SET payload = jsonb_build_object(
                    'job_id', job_id,
                    'profile_id', profile_id,
                    'user_id', user_id,
                    'session_id', session_id,
                    'artifact_id', artifact_id,
                    'source_revision', source_revision,
                    'content_sha256', content_sha256,
                    'status', status,
                    'attempt_count', attempt_count,
                    'next_attempt_at', next_attempt_at,
                    'last_error', last_error,
                    'result_json', result_json,
                    'created_at', created_at,
                    'updated_at', updated_at,
                    'completed_at', completed_at
                )
                WHERE payload IS NULL
                """
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.embedding_jobs ALTER COLUMN payload SET NOT NULL"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS sessions_user_updated_idx ON {self.schema_name}.sessions (user_id, updated_at DESC)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS session_messages_session_idx ON {self.schema_name}.session_messages (session_id, created_at ASC)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS session_permissions_scope_idx ON {self.schema_name}.session_permissions (scope, resource_key, grantee_user_id)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS embedding_jobs_pending_idx ON {self.schema_name}.embedding_jobs (status, next_attempt_at, updated_at)"
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


def state_store_from_config(config: LitTraceConfig) -> PostgresStateStore | None:
    if config.metadata_store.backend != "postgres":
        return None
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


def _permission_row(record: SessionPermissionGrant) -> dict[str, object]:
    row = record.model_dump(mode="json")
    row["metadata"] = _jsonb(record.metadata)
    return row


def _embedding_job_row(record: EmbeddingJobRecord) -> dict[str, object]:
    row = record.model_dump(mode="json")
    row["result_json"] = _jsonb(record.result_json)
    row["payload"] = _jsonb(record.model_dump(mode="json"))
    return row


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
