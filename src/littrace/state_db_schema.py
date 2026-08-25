"""Postgres schema DDL for LitTrace (round 7 cleanup extraction).

Extracted from ``PostgresStateStore._ensure_schema`` so the
DDL list lives in one declarative place instead of being
inlined inside a 213-line method. The shape of this module
is intentionally pure: every public function returns a list
of SQL statements (or executes them against a cursor). The
host class keeps the connection / transaction lifecycle.

Schema map::

    session_state                  -- one row per LitTrace session
    session_state_snapshots        -- per-revision workspace history
    agent_thread_bindings          -- codex-runtime thread mapping
    agent_tool_calls               -- CAS-protected replay log
    chat_trail                     -- conversation / event / snapshot-ref history
    async_tasks                    -- generic background-job queue

Legacy tables (``artifact_outbox``, ``embedding_jobs``,
``paper_lifecycle_events``, ``session_messages``,
``session_memory``, ``session_snapshots``, ``sessions``) are
dropped only when ``allow_schema_reset`` is true. Default is
false so a production re-boot never destroys data.
"""

from __future__ import annotations

from typing import Any

# Legacy table names. Dropped under ``allow_schema_reset``
# only. Kept as a module-level constant so the schema file is
# the single source of truth for which tables are considered
# obsolete.
_LEGACY_TABLES: tuple[str, ...] = (
    "artifact_outbox",
    "embedding_jobs",
    "paper_lifecycle_events",
    "session_messages",
    "session_memory",
    "session_snapshots",
    "sessions",
)


def schema_statements(
    schema_name: str,
    *,
    allow_schema_reset: bool = False,
) -> list[str]:
    """Return the full ordered DDL list for ``schema_name``.

    The list is intentionally flat so the caller can decide
    whether to execute it inside one transaction (the
    current behaviour) or split into chunks. Order matters:
    CREATE SCHEMA must precede the table statements; the
    session_state CREATE must precede tables that
    REFERENCE it (FK constraints).
    """
    s = schema_name
    statements: list[str] = [f"CREATE SCHEMA IF NOT EXISTS {s}"]

    if allow_schema_reset:
        for old in _LEGACY_TABLES:
            statements.append(f"DROP TABLE IF EXISTS {s}.{old} CASCADE")

    statements.extend([
        # session_state: one row per LitTrace session.
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
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('draft', 'idle', 'active', 'systemError', 'archived')),
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        # Migration: older deployments where ``status`` did
        # not exist yet. ADD COLUMN IF NOT EXISTS makes this
        # a no-op on fresh DBs.
        f"ALTER TABLE {s}.session_state "
        f"ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active'",
        # The CHECK constraint lives in the CREATE TABLE
        # above; pre-status deployments did not enforce it.
        # Drop the old (3-value) constraint and add the new
        # (5-value) one idempotently. The DO block swallows
        # ``undefined_object`` so a fresh deployment never
        # trips over the missing old constraint.
        f"DO $$ BEGIN "
        f"  ALTER TABLE {s}.session_state "
        f"    DROP CONSTRAINT session_state_status_check; "
        f"  ALTER TABLE {s}.session_state "
        f"    ADD CONSTRAINT session_state_status_check "
        f"    CHECK (status IN ('draft', 'idle', 'active', 'systemError', 'archived')); "
        f"EXCEPTION WHEN undefined_object OR duplicate_object THEN NULL; "
        f"END $$",
        # agent_thread_bindings: codex-runtime thread mapping.
        f"""
        CREATE TABLE IF NOT EXISTS {s}.agent_thread_bindings (
            session_id TEXT PRIMARY KEY REFERENCES {s}.session_state(session_id)
                ON DELETE CASCADE,
            codex_thread_id TEXT UNIQUE NOT NULL,
            runtime_kind TEXT NOT NULL,
            runtime_version TEXT,
            workspace_revision INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            last_error TEXT,
            turn_count INTEGER NOT NULL DEFAULT 0,
            last_total_tokens INTEGER NOT NULL DEFAULT 0,
            last_compacted_at TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        f"CREATE INDEX IF NOT EXISTS agent_thread_bindings_status_idx "
        f"ON {s}.agent_thread_bindings (status, updated_at)",
        # Round 5: migration for older deployments where
        # agent_thread_bindings does not yet carry the
        # compaction fields.
        f"DO $$ BEGIN "
        f"  ALTER TABLE {s}.agent_thread_bindings "
        f"    ADD COLUMN turn_count INTEGER NOT NULL DEFAULT 0; "
        f"  ALTER TABLE {s}.agent_thread_bindings "
        f"    ADD COLUMN last_total_tokens INTEGER NOT NULL DEFAULT 0; "
        f"  ALTER TABLE {s}.agent_thread_bindings "
        f"    ADD COLUMN last_compacted_at TEXT; "
        f"EXCEPTION WHEN duplicate_column THEN NULL; "
        f"END $$",
        # session_state_snapshots: per-revision workspace history.
        f"""
        CREATE TABLE IF NOT EXISTS {s}.session_state_snapshots (
            session_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            workspace_sha256 TEXT,
            workspace_json JSONB NOT NULL,
            captured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (session_id, revision)
        )
        """,
        f"CREATE INDEX IF NOT EXISTS session_state_snapshots_captured_idx "
        f"ON {s}.session_state_snapshots (session_id, captured_at DESC)",
        # agent_tool_calls: CAS-protected replay log.
        f"""
        CREATE TABLE IF NOT EXISTS {s}.agent_tool_calls (
            session_id TEXT NOT NULL REFERENCES {s}.session_state(session_id)
                ON DELETE CASCADE,
            idempotency_key TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            arguments_sha256 TEXT NOT NULL,
            expected_revision INTEGER NOT NULL,
            committed_revision INTEGER NOT NULL,
            result_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (session_id, idempotency_key)
        )
        """,
        f"CREATE INDEX IF NOT EXISTS agent_tool_calls_tool_idx "
        f"ON {s}.agent_tool_calls (tool_name, updated_at)",
        # chat_trail: conversation / event / snapshot-ref history.
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
        """,
        f"CREATE INDEX IF NOT EXISTS chat_trail_messages_idx ON {s}.chat_trail USING GIN (messages)",
        f"CREATE INDEX IF NOT EXISTS chat_trail_events_idx ON {s}.chat_trail USING GIN (events)",
        # async_tasks: generic background-job queue (embedding,
        # download, parse, table, storyline, compaction).
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
        """,
        f"CREATE UNIQUE INDEX IF NOT EXISTS async_tasks_idempotency_idx "
        f"ON {s}.async_tasks (session_id, artifact_id, kind, content_sha256)",
        f"CREATE INDEX IF NOT EXISTS async_tasks_pending_idx "
        f"ON {s}.async_tasks (status, next_attempt_at)",
        f"CREATE INDEX IF NOT EXISTS async_tasks_lease_idx "
        f"ON {s}.async_tasks (lease_expires_at)",
        f"CREATE INDEX IF NOT EXISTS async_tasks_session_idx "
        f"ON {s}.async_tasks (session_id, status)",
    ])
    return statements


def execute_statements(
    cursor: Any,
    schema_name: str,
    *,
    allow_schema_reset: bool = False,
) -> None:
    """Convenience: execute every DDL statement in order against ``cursor``.

    The caller owns the transaction; this function does not
    commit. The PostgresStateStore still wraps the call in
    one ``conn.commit()`` at the end so the whole schema
    upgrade is atomic.
    """
    for statement in schema_statements(
        schema_name, allow_schema_reset=allow_schema_reset
    ):
        cursor.execute(statement)