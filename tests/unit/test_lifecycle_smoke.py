"""Smoke test for littrace.lifecycle (enqueue_embedding_outbox, record_lifecycle_event,
dispatch_embedding_outbox). No mocks — these functions only do Postgres writes."""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.unit


_REAL_DSN = "postgresql://littrace:littrace@localhost:5433/littrace"


def _unique_schema(prefix: str) -> str:
    return f"littrace_test_{prefix}_{uuid.uuid4().hex[:8]}"


def test_record_lifecycle_event_and_enqueue_outbox():
    """record_lifecycle_event + enqueue_embedding_outbox must insert rows
    into the new 3-table layout (chat_trail.events + async_tasks)."""
    from littrace.config import LitTraceConfig, StorageConfig
    from littrace.lifecycle import (
        enqueue_embedding_outbox,
        record_lifecycle_event,
    )
    from littrace.state_db import (
        AsyncTaskRecord,
        state_store_from_config,
    )

    schema = _unique_schema("lifecycle")
    config = LitTraceConfig(
        metadata_store__schema_name=schema,  # type: ignore[call-arg]
        storage=StorageConfig(),
    )
    # Set the schema after construction (LitTraceConfig has a typed schema).
    config.metadata_store.schema_name = schema
    config.metadata_store.allow_schema_reset = True

    store = state_store_from_config(config)
    session_id = f"smoke-{uuid.uuid4().hex[:6]}"

    # chat_trail has a FK to session_state; create a stub row first.
    from littrace.state_db import SessionStateRecord

    store.upsert_session_state(SessionStateRecord(session_id=session_id))

    record_lifecycle_event(
        config,
        session_id=session_id,
        paper_id="p1",
        event_type="smoke_event",
    )
    outbox = enqueue_embedding_outbox(
        config,
        session_id=session_id,
        artifact_id="paper_pdf:p1",
        content_sha256="abc123",
    )
    assert isinstance(outbox, AsyncTaskRecord)
    assert outbox.kind == "artifact_outbox"

    # Verify the rows landed where the new schema expects them
    events = store.list_chat_events(session_id)
    assert any(e.get("event_type") == "smoke_event" for e in events)

    tasks = store.list_async_tasks(
        session_id=session_id, kind="artifact_outbox"
    )
    assert any(t.artifact_id == "paper_pdf:p1" for t in tasks)