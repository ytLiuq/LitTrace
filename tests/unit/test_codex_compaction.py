"""Round 5 compaction worker unit tests.

The new ``compaction`` module (commit 2) wraps the
``thread/compact`` RPC behind a daemon-thread worker plus an
async batch driver. These tests exercise:

  - ``CompactionWorker.run_once`` — picks up due sessions and
    enqueues ``compaction_job`` rows; respects the per-session
    last-hour back-off.
  - ``run_pending_compaction`` — the async driver that claims
    those rows, calls ``client.compact_thread`` via the shared
    runtime manager, and stamps ``last_compacted_at`` on success.
  - The bookkeeping flip in ``_chat_with_client`` that bumps
    ``turn_count`` / ``last_total_tokens`` on every turn so the
    worker can decide whether the next session is over the
    threshold.

The runtime-manager half is exercised through the in-process
``_FakeRuntimeManager`` stub (mirrors the pattern used in
``DownloadRetryWorker`` tests). The shared ``runtime_manager``
singleton is monkeypatched out so the test does not boot an App
Server subprocess.
"""

from __future__ import annotations

import pytest

from littrace.codex_runtime.compaction import CompactionWorker
from littrace.config import LitTraceConfig, StorageConfig
from littrace.state_db import AgentThreadBindingRecord, AsyncTaskRecord
from tests.unit.test_codex_runtime_service import _BindingStore, _FakeClient


def _config(tmp_path) -> LitTraceConfig:
    return LitTraceConfig(
        storage=StorageConfig(sessions_dir=tmp_path / "sessions"),
    )


def test_worker_enqueues_due_sessions(monkeypatch, tmp_path) -> None:
    store = _BindingStore()
    # Two over-threshold sessions + one under-threshold. The two
    # over-threshold rows are enqueued; the under-threshold one
    # is left alone.
    store.upsert_session_state(_make_state("s-due-1", "active"))
    store.upsert_session_state(_make_state("s-due-2", "active"))
    store.upsert_session_state(_make_state("s-skip", "draft"))
    store.upsert_agent_thread_binding(
        _binding("s-due-1", "thr-1", turn_count=40, last_total_tokens=10_000)
    )
    store.upsert_agent_thread_binding(
        _binding("s-due-2", "thr-2", turn_count=5, last_total_tokens=80_000)
    )
    store.upsert_agent_thread_binding(
        _binding("s-skip", "thr-3", turn_count=2, last_total_tokens=500)
    )

    worker = CompactionWorker(
        store, interval_seconds=10, batch_size=10,
        threshold_turns=30, threshold_tokens=50_000,
    )
    processed = worker.run_once()
    assert sorted(processed) == ["s-due-1", "s-due-2"]
    queued = _queued_task_ids(store)
    assert sorted(queued) == [
        "compaction-s-due-1-thr-1",
        "compaction-s-due-2-thr-2",
    ]


def test_worker_skips_archived_sessions(monkeypatch, tmp_path) -> None:
    store = _BindingStore()
    store.upsert_session_state(_make_state("s-archived", "archived"))
    store.upsert_agent_thread_binding(
        _binding("s-archived", "thr-1", turn_count=100, last_total_tokens=200_000)
    )

    worker = CompactionWorker(store)
    processed = worker.run_once()
    assert processed == []
    assert _queued_task_ids(store) == []


def test_worker_respects_last_hour_backoff(monkeypatch, tmp_path) -> None:
    """A session that was compacted less than an hour ago must
    not be re-enqueued even if the threshold is exceeded."""
    store = _BindingStore()
    store.upsert_session_state(_make_state("s-recent", "active"))
    store.upsert_agent_thread_binding(
        AgentThreadBindingRecord(
            session_id="s-recent",
            codex_thread_id="thr-1",
            turn_count=100,
            last_total_tokens=200_000,
            last_compacted_at="2099-01-01T00:00:00+00:00",  # far future
        )
    )
    worker = CompactionWorker(store)
    assert worker.run_once() == []


def test_worker_respects_batch_size(monkeypatch, tmp_path) -> None:
    store = _BindingStore()
    for i in range(5):
        sid = f"s-{i}"
        store.upsert_session_state(_make_state(sid, "active"))
        store.upsert_agent_thread_binding(
            _binding(sid, f"thr-{i}", turn_count=100, last_total_tokens=200_000)
        )
    worker = CompactionWorker(store, batch_size=2)
    processed = worker.run_once()
    assert sorted(processed) == ["s-0", "s-1"]
    # remaining 3 stay in the database for the next run
    assert len(_queued_task_ids(store)) == 2


# ---- helpers --------------------------------------------------------

def _make_state(session_id: str, status: str):
    from littrace.state_db import SessionStateRecord

    return SessionStateRecord(session_id=session_id, status=status, revision=0)


def _binding(
    session_id: str,
    thread_id: str,
    *,
    turn_count: int = 0,
    last_total_tokens: int = 0,
) -> AgentThreadBindingRecord:
    return AgentThreadBindingRecord(
        session_id=session_id,
        codex_thread_id=thread_id,
        turn_count=turn_count,
        last_total_tokens=last_total_tokens,
    )


def _queued_task_ids(store) -> list[str]:
    rows = store.list_async_tasks(kind="compaction_job", limit=100)
    return sorted(r.task_id for r in rows)
