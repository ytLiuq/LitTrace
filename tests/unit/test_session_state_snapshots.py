"""Lock down the session_state_snapshots side table.

Round 3 topic B moved per-revision workspace history from
``<session.root>/workspace/snapshots/workspace-*.json`` to a
Postgres side table. These tests assert the contract the rest of
the codebase now relies on:

  - a successful save_workspace captures exactly one snapshot
  - snapshots are append-only on (session_id, revision)
  - snapshots are time-ordered newest-first
  - cross-session lookup returns only the requested session
"""

from __future__ import annotations

import pytest

from littrace.config import LitTraceConfig, MetadataStoreConfig, StorageConfig
from littrace.session import create_chat_session, save_workspace
from littrace.state_db import SessionStateSnapshotRecord, state_store_from_config


_REAL_DSN = "postgresql://littrace:littrace@localhost:5433/littrace"


def _store():
    """In-memory alias of the real PostgresStateStore used by the chat path."""
    return state_store_from_config(
        LitTraceConfig(
            storage=StorageConfig(),
            metadata_store=MetadataStoreConfig(
                backend="postgres",
                postgres_dsn=_REAL_DSN,
                schema_name=f"littrace_test_snap_{__import__('uuid').uuid4().hex[:8]}",
                allow_schema_reset=True,
            ),
        )
    )


def test_snapshot_captured_on_save(monkeypatch) -> None:
    store = _store()
    monkeypatch.setattr("littrace.session._session_state_store", lambda *_a, **_k: store)
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=__import__("pathlib").Path("/tmp")))
    session = create_chat_session(config)
    save_workspace(session, __import__("littrace.models", fromlist=["LiteratureWorkspace"]).LiteratureWorkspace())
    snapshots = store.list_session_snapshots(session.session_id)
    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.session_id == session.session_id
    assert snap.revision == 1
    assert snap.workspace_json  # non-empty dict


def test_snapshot_revision_uniqueness(monkeypatch) -> None:
    """upsert_session_snapshot is INSERT ON CONFLICT DO NOTHING, so
    re-saving the same revision does not create a duplicate.
    """
    store = _store()
    snap = SessionStateSnapshotRecord(
        session_id="s-x",
        revision=3,
        workspace_json={"a": 1},
    )
    store.upsert_session_snapshot(snap)
    store.upsert_session_snapshot(snap.model_copy(update={"workspace_json": {"a": 999}}))
    rows = store.list_session_snapshots("s-x")
    assert len(rows) == 1
    assert rows[0].workspace_json == {"a": 1}


def test_snapshot_newest_first(monkeypatch) -> None:
    store = _store()
    for rev in (1, 2, 3):
        store.upsert_session_snapshot(
            SessionStateSnapshotRecord(
                session_id="s-time",
                revision=rev,
                workspace_json={"r": rev},
            )
        )
    rows = store.list_session_snapshots("s-time")
    revisions = [r.revision for r in rows]
    # captured_at is monotonic so newest revision should be first.
    assert revisions[0] == 3
    assert set(revisions) == {1, 2, 3}


def test_snapshot_cross_session_isolation(monkeypatch) -> None:
    store = _store()
    store.upsert_session_snapshot(
        SessionStateSnapshotRecord(session_id="s-a", revision=1, workspace_json={})
    )
    store.upsert_session_snapshot(
        SessionStateSnapshotRecord(session_id="s-b", revision=1, workspace_json={})
    )
    a_rows = store.list_session_snapshots("s-a")
    b_rows = store.list_session_snapshots("s-b")
    assert all(r.session_id == "s-a" for r in a_rows)
    assert all(r.session_id == "s-b" for r in b_rows)
    assert len(a_rows) == 1
    assert len(b_rows) == 1


def test_snapshot_limit_truncates(monkeypatch) -> None:
    store = _store()
    for rev in range(1, 6):
        store.upsert_session_snapshot(
            SessionStateSnapshotRecord(session_id="s-limit", revision=rev, workspace_json={})
        )
    rows = store.list_session_snapshots("s-limit", limit=2)
    assert len(rows) == 2
    assert {r.revision for r in rows} == {4, 5}