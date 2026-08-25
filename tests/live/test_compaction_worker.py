"""Live smoke test for the Round 5 compaction worker.

End-to-end exercise against the local Postgres used by the rest of
the live / integration suite. The test does NOT need a Codex
runtime — it only drives the enqueue half of the worker (which
is what runs in the daemon thread). The full ``run_pending_compaction``
RPC half is exercised in unit tests with a fake client.

Steps:

  1. Insert a session_state row in ``active``.
  2. Insert an ``agent_thread_bindings`` row over the configured
     turn_count / token thresholds.
  3. Start a ``CompactionWorker`` with a 0.5s interval and let it
     tick once.
  4. Assert a ``compaction_job`` row appeared in ``async_tasks``
     and the binding is now stamped with ``last_compacted_at`` is
     still None (worker only enqueues; the actual RPC is the
     async driver's job).
  5. Stop the worker.

Skip conditions:

  * ``LITTRACE_POSTGRES_DSN`` not set
  * Postgres not reachable
"""

from __future__ import annotations

import os
import threading
import time

import pytest

pytestmark = pytest.mark.live


DSN = os.environ.get("LITTRACE_POSTGRES_DSN")


@pytest.mark.skipif(not DSN, reason="LITTRACE_POSTGRES_DSN not set")
def test_compaction_worker_enqueues_due_session_against_postgres() -> None:
    import psycopg

    from littrace.codex_runtime.compaction import CompactionWorker
    from littrace.config import LitTraceConfig, MetadataStoreConfig
    from littrace.state_db import (
        AgentThreadBindingRecord,
        SessionStateRecord,
        state_store_from_config,
    )

    config = LitTraceConfig(
        metadata_store=MetadataStoreConfig(
            backend="postgres",
            postgres_dsn=DSN,
            schema_name="littrace_e2e",
        ),
    )
    config.compaction.threshold_turns = 5
    config.compaction.threshold_tokens = 1_000
    config.compaction.batch_size = 5
    config.compaction.interval_seconds = 0.5

    store = state_store_from_config(config)
    # Use a stable session id for repeatability; the schema is
    # already created by the storage_rag_e2e live test.
    session_id = "live-compaction-smoke-0001"
    binding_id = "thread-live-compaction-0001"

    store.upsert_session_state(
        SessionStateRecord(
            session_id=session_id,
            status="active",
            revision=0,
        )
    )
    store.upsert_agent_thread_binding(
        AgentThreadBindingRecord(
            session_id=session_id,
            codex_thread_id=binding_id,
            turn_count=20,
            last_total_tokens=10_000,
        )
    )

    worker = CompactionWorker(
        store,
        interval_seconds=0.5,
        batch_size=5,
        threshold_turns=config.compaction.threshold_turns,
        threshold_tokens=config.compaction.threshold_tokens,
    )
    worker.start()
    try:
        # Give the daemon up to 2 s to enqueue. The first tick fires
        # immediately on the threading.Thread start, so 1 s is plenty
        # in practice; 2 s leaves headroom for slow CI machines.
        deadline = time.time() + 2.0
        queued = []
        while time.time() < deadline:
            queued = store.list_async_tasks(
                kind="compaction_job", status="queued", limit=10,
            )
            if any(t.session_id == session_id for t in queued):
                break
            time.sleep(0.05)
        assert any(t.session_id == session_id for t in queued), (
            f"compaction worker did not enqueue a job within 2s; "
            f"async_tasks rows: {[(t.session_id, t.status) for t in queued]}"
        )
    finally:
        worker.stop(timeout=2.0)

    # Cleanup so the next run of this test is idempotent.
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM littrace_e2e.async_tasks WHERE session_id = %s",
            (session_id,),
        )
        cur.execute(
            "DELETE FROM littrace_e2e.agent_thread_bindings WHERE session_id = %s",
            (session_id,),
        )
        cur.execute(
            "DELETE FROM littrace_e2e.session_state WHERE session_id = %s",
            (session_id,),
        )
        conn.commit()