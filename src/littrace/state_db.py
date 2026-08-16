"""Postgres-backed state store for the consolidated 3-table memory layout.

Memory layers (post-consolidation):
  L1 session_state : workspace truth + memory view + artifact index + rag profile
  L2 chat_trail    : append-only messages, snapshot refs, lifecycle events
  L3 async_tasks   : durable queue for embedding jobs + outbox events
  L4 RAGIndex      : pgvector collection (separate schema, not this module)

Replaces the previous 9-table layout (sessions, session_messages,
session_memory, session_snapshots, embedding_jobs, artifact_outbox,
paper_lifecycle_events, download_tasks, artifact_registry).
download_tasks lives in ``download_tasks.py`` with its own connection.

Public API is a ``StateStore`` Protocol so tests and callers depend on the
abstract surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig


def _safe_identifier(value: str) -> str:
    """Coerce ``value`` into a Postgres-safe schema/table identifier.

    Postgres caps unquoted identifiers at 63 bytes; the only legal
    characters are ``[A-Za-z0-9_]`` (with a leading digit rule). We
    rewrite anything else to ``_`` and prefix a ``littrace_`` tag if
    the result is empty or starts with a digit.
    """
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in value)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"littrace_{cleaned}"
    return cleaned[:63]


def require_postgres_metadata(metadata_store) -> tuple[str, str]:
    """Validate that ``metadata_store`` is a Postgres config and return
    ``(dsn, schema_name)``.

    Centralises the ``backend must be 'postgres'`` + ``postgres_dsn is
    required`` checks previously duplicated across 5 ``*_from_config``
    factories. Raises :class:`ValueError` with a caller-specific suffix
    so the message still tells the operator which subsystem failed.
    """
    if getattr(metadata_store, "backend", None) != "postgres":
        raise ValueError("metadata_store.backend must be 'postgres'.")
    dsn = getattr(metadata_store, "postgres_dsn", None)
    schema_name = getattr(metadata_store, "schema_name", "littrace")
    if not dsn:
        raise ValueError("metadata_store.postgres_dsn is required.")
    return dsn, schema_name


# ---------------------------------------------------------------------------
# Pydantic records
# ---------------------------------------------------------------------------


class SessionStateRecord(BaseModel):
    """L1: workspace truth + memory view + artifact index + rag profile (1 row / session)."""

    session_id: str
    workspace_sha256: str | None = None
    workspace_json: dict[str, object] = Field(default_factory=dict)
    manifest_json: dict[str, object] = Field(default_factory=dict)
    artifact_index_json: dict[str, object] = Field(default_factory=dict)
    memory_view_json: dict[str, object] = Field(default_factory=dict)
    rag_profile_json: dict[str, object] = Field(default_factory=dict)
    revision: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SessionSummaryRecord(BaseModel):
    """Compact view returned by list_session_states (kept for chat sidebar)."""

    session_id: str
    updated_at: str
    workspace_sha256: str | None = None
    revision: int = 0


class AsyncTaskRecord(BaseModel):
    """L3: durable queue row, kind discriminates embedding_job vs artifact_outbox."""

    task_id: str
    session_id: str
    kind: Literal["embedding_job", "artifact_outbox"]
    artifact_id: str = ""
    event_type: str = ""
    profile_id: str = ""
    source_revision: str = ""
    content_sha256: str = ""
    status: Literal["queued", "running", "completed", "failed", "dead"] = "queued"
    attempt_count: int = 0
    next_attempt_at: str | None = None
    last_heartbeat_at: str | None = None
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    last_error: str | None = None
    result_json: dict[str, object] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None


class AsyncTaskQueueReport(BaseModel):
    schema_version: str = "littrace.async_task_queue_report.v1"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    kind: str | None = None
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


# ---------------------------------------------------------------------------
# Protocol — 3-table consolidated surface
# ---------------------------------------------------------------------------


class StateStore(Protocol):
    """All chat-path / RAG-path / daily-path state persistence in one place.

    The chat path reaches this Protocol indirectly via session.py +
    runtime/memory.py. The RAG and daily paths reach it directly via
    rag_jobs.py + lifecycle.py + rag_ops.py.
    """

    # ---- L1: session working state -----------------------------------------

    def upsert_session_state(self, state: SessionStateRecord) -> SessionStateRecord: ...
    def get_session_state(self, session_id: str) -> SessionStateRecord | None: ...
    def list_session_states(self, *, limit: int = 20) -> list[SessionSummaryRecord]: ...
    def delete_session_state(self, session_id: str) -> int: ...

    # ---- L2: chat trail (messages + snapshot refs + lifecycle events) -------

    def append_chat_message(self, session_id: str, record: dict[str, object]) -> None: ...
    def list_chat_messages(self, session_id: str) -> list[dict[str, object]]: ...
    def append_chat_event(self, session_id: str, event: dict[str, object]) -> None: ...
    def list_chat_events(
        self, session_id: str, *, paper_id: str | None = None
    ) -> list[dict[str, object]]: ...

    # ---- L3: async tasks (embedding_job + artifact_outbox) ---------------

    def enqueue_async_task(self, task: AsyncTaskRecord) -> AsyncTaskRecord: ...
    def claim_pending_async_tasks(
        self,
        *,
        worker_id: str,
        kind: str | None = None,
        limit: int = 20,
        lease_seconds: float = 300.0,
    ) -> list[AsyncTaskRecord]: ...
    def update_async_task(self, task: AsyncTaskRecord) -> AsyncTaskRecord: ...
    def list_async_tasks(
        self,
        *,
        status: str | None = None,
        session_id: str | None = None,
        kind: str | None = None,
        limit: int = 20,
    ) -> list[AsyncTaskRecord]: ...
    def async_tasks_queue_report(self, *, kind: str | None = None) -> AsyncTaskQueueReport: ...
    def requeue_dead_async_tasks(
        self, *, kind: str | None = None, limit: int = 20
    ) -> int: ...


# ---------------------------------------------------------------------------
# Postgres implementation
# ---------------------------------------------------------------------------


_LEASE_RECLAIM_SQL = """
    (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= now())
"""


@dataclass
class PostgresStateStore:
    """Postgres-backed implementation of StateStore for the 3-table layout."""

    dsn: str
    schema_name: str = "littrace"
    allow_schema_reset: bool = False
    _initialized: bool = False

    # --- connection helpers -------------------------------------------------

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "Postgres state storage requires the optional rag extra: "
                "pip install -e '.[rag]'"
            ) from exc
        return psycopg.connect(self.dsn)

    # --- schema bootstrap ---------------------------------------------------

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")

            # DROP legacy tables ONLY when the operator has explicitly
            # enabled schema reset. Default off — protects production /
            # shared DSNs from accidental data loss on every boot.
            if self.allow_schema_reset:
                for old in (
                    "artifact_outbox",
                    "embedding_jobs",
                    "paper_lifecycle_events",
                    "session_messages",
                    "session_memory",
                    "session_snapshots",
                    "sessions",
                ):
                    cur.execute(f"DROP TABLE IF EXISTS {s}.{old} CASCADE")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {s}.session_state (
                    session_id TEXT PRIMARY KEY,
                    workspace_sha256 TEXT,
                    workspace_json JSONB NOT NULL,
                    manifest_json JSONB NOT NULL,
                    artifact_index_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    memory_view_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    rag_profile_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    revision INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {s}.chat_trail (
                    session_id TEXT PRIMARY KEY REFERENCES {s}.session_state(session_id)
                        ON DELETE CASCADE,
                    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
                    snapshot_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                    events JSONB NOT NULL DEFAULT '[]'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL,
                    CONSTRAINT chat_trail_messages_cap CHECK (jsonb_array_length(messages) <= 5000),
                    CONSTRAINT chat_trail_events_cap   CHECK (jsonb_array_length(events)   <= 5000)
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS chat_trail_messages_idx ON {s}.chat_trail USING GIN (messages)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS chat_trail_events_idx ON {s}.chat_trail USING GIN (events)"
            )

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {s}.async_tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    artifact_id TEXT NOT NULL DEFAULT '',
                    event_type TEXT NOT NULL DEFAULT '',
                    profile_id TEXT NOT NULL DEFAULT '',
                    source_revision TEXT NOT NULL DEFAULT '',
                    content_sha256 TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TIMESTAMPTZ,
                    last_heartbeat_at TIMESTAMPTZ,
                    lease_owner TEXT,
                    lease_expires_at TIMESTAMPTZ,
                    last_error TEXT,
                    result_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ
                )
                """
            )
            cur.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS async_tasks_idempotency_idx "
                f"ON {s}.async_tasks (session_id, artifact_id, kind, content_sha256)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS async_tasks_pending_idx "
                f"ON {s}.async_tasks (status, next_attempt_at)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS async_tasks_lease_idx "
                f"ON {s}.async_tasks (lease_expires_at)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS async_tasks_session_idx "
                f"ON {s}.async_tasks (session_id, status)"
            )

            conn.commit()
        self._initialized = True

    # --- L1: session_state --------------------------------------------------

    def upsert_session_state(
        self, state: SessionStateRecord, *, expected_revision: int | None = None
    ) -> SessionStateRecord:
        """Upsert a session row. If ``expected_revision`` is set, treat the
        UPDATE as compare-and-set: only overwrite if the row is at the
        given revision. Returns the new record on success, raises
        ``RuntimeError`` on CAS mismatch so the caller can retry.

        On CONFLICT (new session), the CAS check is bypassed and the
        initial revision is used.
        """
        self._ensure_schema()
        s = self.schema_name
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn, conn.cursor() as cur:
            if expected_revision is not None:
                cur.execute(
                    f"""
                    UPDATE {s}.session_state
                    SET workspace_sha256 = %s,
                        workspace_json = %s,
                        manifest_json = %s,
                        artifact_index_json = %s,
                        memory_view_json = %s,
                        rag_profile_json = %s,
                        revision = %s,
                        updated_at = now()
                    WHERE session_id = %s AND revision = %s
                    RETURNING created_at, updated_at
                    """,
                    (
                        state.workspace_sha256,
                        _jsonb(state.workspace_json),
                        _jsonb(state.manifest_json),
                        _jsonb(state.artifact_index_json),
                        _jsonb(state.memory_view_json),
                        _jsonb(state.rag_profile_json),
                        state.revision,
                        state.session_id,
                        expected_revision,
                    ),
                )
                if cur.rowcount == 0:
                    current = self.get_session_state(state.session_id)
                    if current is None:
                        # Row vanished between read and CAS; do an INSERT.
                        self.upsert_session_state(state, expected_revision=None)
                        return state
                    raise RuntimeError(
                        f"SessionState CAS mismatch for {state.session_id}: "
                        f"expected revision {expected_revision}, got {current.revision}"
                    )
                row = cur.fetchone()
                conn.commit()
                return state.model_copy(
                    update={
                        "created_at": str(row[0]),
                        "updated_at": str(row[1]),
                    }
                )
            cur.execute(
                f"""
                INSERT INTO {s}.session_state (
                    session_id, workspace_sha256, workspace_json, manifest_json,
                    artifact_index_json, memory_view_json, rag_profile_json,
                    revision, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, now()), now())
                ON CONFLICT (session_id) DO UPDATE SET
                    workspace_sha256 = EXCLUDED.workspace_sha256,
                    workspace_json = EXCLUDED.workspace_json,
                    manifest_json = EXCLUDED.manifest_json,
                    artifact_index_json = EXCLUDED.artifact_index_json,
                    memory_view_json = EXCLUDED.memory_view_json,
                    rag_profile_json = EXCLUDED.rag_profile_json,
                    revision = EXCLUDED.revision,
                    updated_at = now()
                RETURNING created_at, updated_at
                """,
                (
                    state.session_id,
                    state.workspace_sha256,
                    _jsonb(state.workspace_json),
                    _jsonb(state.manifest_json),
                    _jsonb(state.artifact_index_json),
                    _jsonb(state.memory_view_json),
                    _jsonb(state.rag_profile_json),
                    state.revision,
                    state.created_at,
                ),
            )
            row = cur.fetchone()
            conn.commit()
        return state.model_copy(
            update={
                "created_at": str(row[0]),
                "updated_at": str(row[1]),
            }
        )

    def get_session_state(self, session_id: str) -> SessionStateRecord | None:
        self._ensure_schema()
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT session_id, workspace_sha256, workspace_json, manifest_json,
                       artifact_index_json, memory_view_json, rag_profile_json,
                       revision, structured_document_count, workspace_snapshot_count,
                       created_at, updated_at
                FROM {s}.session_state
                WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return SessionStateRecord(
            session_id=row[0],
            workspace_sha256=row[1],
            workspace_json=_from_jsonb(row[2]),
            manifest_json=_from_jsonb(row[3]),
            artifact_index_json=_from_jsonb(row[4]),
            memory_view_json=_from_jsonb(row[5]),
            rag_profile_json=_from_jsonb(row[6]),
            revision=row[7],
            structured_document_count=row[8],
            workspace_snapshot_count=row[9],
            created_at=row[10].isoformat() if hasattr(row[10], "isoformat") else str(row[10]),
            updated_at=row[11].isoformat() if hasattr(row[11], "isoformat") else str(row[11]),
        )

    def list_session_states(self, *, limit: int = 20) -> list[SessionSummaryRecord]:
        self._ensure_schema()
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT session_id, workspace_sha256, revision,
                       structured_document_count, workspace_snapshot_count, updated_at
                FROM {s}.session_state
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [
            SessionSummaryRecord(
                session_id=row[0],
                workspace_sha256=row[1],
                revision=row[2],
                structured_document_count=row[3],
                workspace_snapshot_count=row[4],
                updated_at=row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
            )
            for row in rows
        ]

    def update_memory_view(self, session_id: str, payload: dict[str, object]) -> bool:
        """Single-field update of ``session_state.memory_view_json``.

        Use this instead of ``upsert_session_state`` for partial writes so
        concurrent ``save_workspace`` calls don't lose their full-row
        overwrite to a stale read-modify-write on memory.
        """
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {s}.session_state
                SET memory_view_json = %s,
                    updated_at = now()
                WHERE session_id = %s
                """,
                (_jsonb(payload), session_id),
            )
            updated = cur.rowcount
            conn.commit()
        if updated:
            return True
        # Row doesn't exist — synthesize a minimal record so a memory-only
        # save still has a home before the first save_workspace.
        self.upsert_session_state(
            SessionStateRecord(
                session_id=session_id,
                memory_view_json=payload,
            )
        )
        return True

    def delete_session_state(self, session_id: str) -> int:
        self._ensure_schema()
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {s}.session_state WHERE session_id = %s", (session_id,))
            deleted = cur.rowcount
            conn.commit()
        return deleted

    # --- L2: chat_trail -----------------------------------------------------

    def _ensure_chat_trail(self, conn, session_id: str) -> None:
        s = self.schema_name
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {s}.chat_trail (session_id, updated_at)
                VALUES (%s, now())
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id,),
            )

    def append_chat_message(self, session_id: str, record: dict[str, object]) -> None:
        self._ensure_schema()
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            self._ensure_chat_trail(conn, session_id)
            # Roll-over: if we are within 100 entries of the SQL CHECK cap
            # (5000), drop the oldest 20% so a long session can keep
            # appending without tripping the constraint.
            cur.execute(
                f"SELECT jsonb_array_length(messages) FROM {s}.chat_trail WHERE session_id = %s",
                (session_id,),
            )
            length_row = cur.fetchone()
            current_len = int(length_row[0]) if length_row and length_row[0] is not None else 0
            if current_len >= 5000 - 100:
                drop_count = max(1, current_len // 5)
                cur.execute(
                    f"""
                    UPDATE {s}.chat_trail
                    SET messages = (
                        SELECT COALESCE(jsonb_agg(elem ORDER BY ord), '[]'::jsonb)
                        FROM (
                            SELECT elem, ord
                            FROM jsonb_array_elements(messages) WITH ORDINALITY AS t(elem, ord)
                            ORDER BY ord DESC
                            LIMIT %s
                        ) recent
                    ),
                    updated_at = now()
                    WHERE session_id = %s
                    """,
                    (5000 - drop_count, session_id),
                )
            cur.execute(
                f"""
                UPDATE {s}.chat_trail
                SET messages = messages || %s::jsonb,
                    updated_at = now()
                WHERE session_id = %s
                """,
                (_jsonb([record]), session_id),
            )
            # Keep the session_state.updated_at column in sync so the
            # sidebar session list (ordered by session_state.updated_at)
            # surfaces new chat activity even when no save_workspace ran.
            cur.execute(
                f"UPDATE {s}.session_state SET updated_at = now() WHERE session_id = %s",
                (session_id,),
            )
            conn.commit()

    def list_chat_messages(self, session_id: str) -> list[dict[str, object]]:
        self._ensure_schema()
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT messages FROM {s}.chat_trail WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
        if row is None or row[0] is None:
            return []
        return list(row[0])

    def append_chat_event(self, session_id: str, event: dict[str, object]) -> None:
        self._ensure_schema()
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            self._ensure_chat_trail(conn, session_id)
            cur.execute(
                f"SELECT jsonb_array_length(events) FROM {s}.chat_trail WHERE session_id = %s",
                (session_id,),
            )
            length_row = cur.fetchone()
            current_len = int(length_row[0]) if length_row and length_row[0] is not None else 0
            if current_len >= 5000 - 100:
                drop_count = max(1, current_len // 5)
                cur.execute(
                    f"""
                    UPDATE {s}.chat_trail
                    SET events = (
                        SELECT COALESCE(jsonb_agg(elem ORDER BY ord), '[]'::jsonb)
                        FROM (
                            SELECT elem, ord
                            FROM jsonb_array_elements(events) WITH ORDINALITY AS t(elem, ord)
                            ORDER BY ord DESC
                            LIMIT %s
                        ) recent
                    ),
                    updated_at = now()
                    WHERE session_id = %s
                    """,
                    (5000 - drop_count, session_id),
                )
            cur.execute(
                f"""
                UPDATE {s}.chat_trail
                SET events = events || %s::jsonb,
                    updated_at = now()
                WHERE session_id = %s
                """,
                (_jsonb([event]), session_id),
            )
            cur.execute(
                f"UPDATE {s}.session_state SET updated_at = now() WHERE session_id = %s",
                (session_id,),
            )
            conn.commit()

    def list_chat_events(
        self, session_id: str, *, paper_id: str | None = None
    ) -> list[dict[str, object]]:
        self._ensure_schema()
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT events FROM {s}.chat_trail WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
        if row is None or row[0] is None:
            return []
        events = list(row[0])
        if paper_id is not None:
            events = [e for e in events if e.get("paper_id") == paper_id]
        return events

    # --- L3: async_tasks ----------------------------------------------------

    def enqueue_async_task(self, task: AsyncTaskRecord) -> AsyncTaskRecord:
        self._ensure_schema()
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {s}.async_tasks (
                    task_id, session_id, kind, artifact_id, event_type, profile_id,
                    source_revision, content_sha256, status, attempt_count,
                    next_attempt_at, last_heartbeat_at, lease_owner,
                    lease_expires_at, last_error, result_json, created_at,
                    updated_at, completed_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, now(), %s
                )
                ON CONFLICT (session_id, artifact_id, kind, content_sha256)
                DO UPDATE SET
                    profile_id = EXCLUDED.profile_id,
                    source_revision = EXCLUDED.source_revision,
                    event_type = EXCLUDED.event_type,
                    updated_at = now()
                RETURNING task_id, created_at, updated_at
                """,
                (
                    task.task_id,
                    task.session_id,
                    task.kind,
                    task.artifact_id,
                    task.event_type,
                    task.profile_id,
                    task.source_revision,
                    task.content_sha256,
                    task.status,
                    task.attempt_count,
                    task.next_attempt_at,
                    task.last_heartbeat_at,
                    task.lease_owner,
                    task.lease_expires_at,
                    task.last_error,
                    _jsonb(task.result_json),
                    task.created_at,
                    task.completed_at,
                ),
            )
            row = cur.fetchone()
            conn.commit()
        # RETURNING includes the actual persisted task_id, which differs from
        # the caller's value when ON CONFLICT DO UPDATE preserved the first
        # enqueue's row. Use the persisted id so downstream references
        # (lifecycle events keyed by task_id) match the real row.
        return task.model_copy(
            update={
                "task_id": str(row[0]),
                "created_at": str(row[1]),
                "updated_at": str(row[2]),
            }
        )

    def claim_pending_async_tasks(
        self,
        *,
        worker_id: str,
        kind: str | None = None,
        limit: int = 20,
        lease_seconds: float = 300.0,
    ) -> list[AsyncTaskRecord]:
        self._ensure_schema()
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            if kind is not None:
                cur.execute(
                    f"""
                    SELECT task_id FROM {s}.async_tasks
                    WHERE kind = %s
                      AND (status = 'queued'
                           OR (status = 'failed' AND next_attempt_at IS NOT NULL AND next_attempt_at <= now())
                           OR {_LEASE_RECLAIM_SQL})
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT %s
                    """,
                    (kind, limit),
                )
            else:
                cur.execute(
                    f"""
                    SELECT task_id FROM {s}.async_tasks
                    WHERE (status = 'queued'
                           OR (status = 'failed' AND next_attempt_at IS NOT NULL AND next_attempt_at <= now())
                           OR {_LEASE_RECLAIM_SQL})
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT %s
                    """,
                    (limit,),
                )
            ids = [row[0] for row in cur.fetchall()]
            if not ids:
                conn.commit()
                return []
            cur.execute(
                f"""
                UPDATE {s}.async_tasks
                SET status = 'running',
                    attempt_count = attempt_count + 1,
                    lease_owner = %s,
                    lease_expires_at = now() + (%s * interval '1 second'),
                    last_heartbeat_at = now(),
                    updated_at = now()
                WHERE task_id = ANY(%s)
                RETURNING task_id, session_id, kind, artifact_id, event_type,
                          profile_id, source_revision, content_sha256, status,
                          attempt_count, next_attempt_at, last_heartbeat_at,
                          lease_owner, lease_expires_at, last_error,
                          result_json, created_at, updated_at, completed_at
                """,
                (worker_id, lease_seconds, ids),
            )
            rows = cur.fetchall()
            conn.commit()
        return [_row_to_async_task(row) for row in rows]

    def update_async_task(self, task: AsyncTaskRecord) -> AsyncTaskRecord:
        self._ensure_schema()
        s = self.schema_name
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {s}.async_tasks SET
                    status = %s,
                    attempt_count = %s,
                    next_attempt_at = %s,
                    last_heartbeat_at = %s,
                    lease_owner = %s,
                    lease_expires_at = %s,
                    last_error = %s,
                    result_json = %s,
                    completed_at = %s,
                    updated_at = now()
                WHERE task_id = %s
                """,
                (
                    task.status,
                    task.attempt_count,
                    task.next_attempt_at,
                    task.last_heartbeat_at,
                    task.lease_owner,
                    task.lease_expires_at,
                    task.last_error,
                    _jsonb(task.result_json),
                    task.completed_at,
                    task.task_id,
                ),
            )
            conn.commit()
        return task

    def list_async_tasks(
        self,
        *,
        status: str | None = None,
        session_id: str | None = None,
        kind: str | None = None,
        limit: int = 20,
    ) -> list[AsyncTaskRecord]:
        self._ensure_schema()
        s = self.schema_name
        clauses = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        if session_id is not None:
            clauses.append("session_id = %s")
            params.append(session_id)
        if kind is not None:
            clauses.append("kind = %s")
            params.append(kind)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        sql = f"""
            SELECT task_id, session_id, kind, artifact_id, event_type,
                   profile_id, source_revision, content_sha256, status,
                   attempt_count, next_attempt_at, last_heartbeat_at,
                   lease_owner, lease_expires_at, last_error,
                   result_json, created_at, updated_at, completed_at
            FROM {s}.async_tasks
            {where}
            ORDER BY created_at DESC
            LIMIT %s
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        return [_row_to_async_task(row) for row in rows]

    def async_tasks_queue_report(self, *, kind: str | None = None) -> AsyncTaskQueueReport:
        self._ensure_schema()
        s = self.schema_name
        clauses: list[str] = []
        params: list[object] = []
        if kind is not None:
            clauses.append("kind = %s")
            params.append(kind)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    status,
                    count(*) AS n,
                    min(created_at) FILTER (WHERE status IN ('queued','failed')) AS oldest,
                    max(last_error) AS last_error
                FROM {s}.async_tasks
                {where}
                GROUP BY status
                """,
                tuple(params),
            )
            by_status = {row[0]: row for row in cur.fetchall()}
            cur.execute(
                f"""
                SELECT
                    count(*) FILTER (
                        WHERE (status = 'running'
                              AND lease_expires_at IS NOT NULL
                              AND lease_expires_at <= now())
                    ) AS reclaimable,
                    count(*) FILTER (
                        WHERE (status IN ('queued','failed')
                              AND (next_attempt_at IS NULL OR next_attempt_at <= now()))
                    ) AS ready
                FROM {s}.async_tasks
                {where}
                """,
                tuple(params),
            )
            counts_row = cur.fetchone()
        report = AsyncTaskQueueReport(kind=kind)
        for status_name, key in [
            ("queued", "queued"),
            ("running", "running"),
            ("failed", "failed"),
            ("dead", "dead"),
            ("completed", "completed"),
        ]:
            value = int(by_status.get(status_name, (None, 0))[1])
            setattr(report, key, value)
        report.total = report.queued + report.running + report.failed + report.dead + report.completed
        report.reclaimable_running = int(counts_row[0]) if counts_row else 0
        report.ready_to_claim = int(counts_row[1]) if counts_row else 0
        oldest_rows = [r for r in by_status.values() if r[2] is not None]
        if oldest_rows:
            report.oldest_ready_at = min(r[2] for r in oldest_rows).isoformat()
        last_errors = [r[3] for r in by_status.values() if r[3] is not None]
        if last_errors:
            report.latest_error = last_errors[0]
        return report

    def requeue_dead_async_tasks(
        self, *, kind: str | None = None, limit: int = 20
    ) -> int:
        self._ensure_schema()
        s = self.schema_name
        clauses = ["status = 'dead'"]
        params: list[object] = []
        if kind is not None:
            clauses.append("kind = %s")
            params.append(kind)
        params.append(limit)
        sql = f"""
            UPDATE {s}.async_tasks
            SET status = 'queued',
                last_error = NULL,
                lease_owner = NULL,
                lease_expires_at = NULL,
                updated_at = now()
            WHERE {' AND '.join(clauses)}
            RETURNING task_id
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            conn.commit()
        return len(rows)


def _row_to_async_task(row: tuple) -> AsyncTaskRecord:
    """Convert a raw row from async_tasks SELECTs into an AsyncTaskRecord."""
    return AsyncTaskRecord(
        task_id=row[0],
        session_id=row[1],
        kind=row[2],
        artifact_id=row[3],
        event_type=row[4],
        profile_id=row[5],
        source_revision=row[6],
        content_sha256=row[7],
        status=row[8],
        attempt_count=row[9],
        next_attempt_at=row[10].isoformat() if row[10] is not None and hasattr(row[10], "isoformat") else row[10],
        last_heartbeat_at=row[11].isoformat() if row[11] is not None and hasattr(row[11], "isoformat") else row[11],
        lease_owner=row[12],
        lease_expires_at=row[13].isoformat() if row[13] is not None and hasattr(row[13], "isoformat") else row[13],
        last_error=row[14],
        result_json=_from_jsonb(row[15]),
        created_at=row[16].isoformat() if hasattr(row[16], "isoformat") else str(row[16]),
        updated_at=row[17].isoformat() if hasattr(row[17], "isoformat") else str(row[17]),
        completed_at=row[18].isoformat() if row[18] is not None and hasattr(row[18], "isoformat") else row[18],
    )


def _jsonb(value: object) -> str:
    """Serialize for a JSONB cast. Uses ``ensure_ascii=False`` so non-ASCII
    (Chinese, emoji, etc.) round-trips at native size and GIN-indexed
    JSONB queries match the same string the caller searched for. Falls
    back to ``str()`` for non-JSON-serializable values (``datetime``,
    ``UUID``, ``Path``, ``set`` …) so a stray non-serializable in
    ``workspace_json`` cannot crash a save_workspace."""
    import json as _json

    def _default(value):
        return str(value)

    return _json.dumps(value, ensure_ascii=False, default=_default)


def _from_jsonb(value) -> object:
    """Decode a JSONB column value returned by psycopg (already a Python dict/list)."""
    return value if value is not None else {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def state_store_from_config(config: LitTraceConfig) -> PostgresStateStore:
    dsn, schema_name = require_postgres_metadata(config.metadata_store)
    return PostgresStateStore(
        dsn=dsn,
        schema_name=schema_name,
        allow_schema_reset=config.metadata_store.allow_schema_reset,
    )