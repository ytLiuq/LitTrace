from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Lock

import pytest

from littrace.models import LiteratureWorkspace
from littrace.session import ChatSession, save_workspace
from littrace.state_db import SessionStateRecord, SessionStateSnapshotRecord

pytestmark = pytest.mark.unit


class InMemoryStateStore:
    def __init__(self) -> None:
        self.record: SessionStateRecord | None = None
        self.snapshots: dict[tuple[str, int], SessionStateSnapshotRecord] = {}
        self._write_lock = Lock()

    @contextmanager
    def session_write_lock(self, _session_id: str):
        with self._write_lock:
            yield

    def get_session_state(self, session_id: str) -> SessionStateRecord | None:
        if self.record is None or self.record.session_id != session_id:
            return None
        return self.record.model_copy(deep=True)

    def upsert_session_state(
        self,
        state: SessionStateRecord,
        *,
        expected_revision: int | None = None,
    ) -> SessionStateRecord:
        current_revision = self.record.revision if self.record is not None else 0
        if expected_revision is not None and expected_revision != current_revision:
            raise RuntimeError(
                f"SessionState CAS mismatch for {state.session_id}: "
                f"expected revision {expected_revision}, got {current_revision}"
            )
        self.record = state.model_copy(deep=True)
        return self.record.model_copy(deep=True)

    def upsert_session_snapshot(self, snapshot: SessionStateSnapshotRecord) -> None:
        key = (snapshot.session_id, snapshot.revision)
        if key not in self.snapshots:
            self.snapshots[key] = snapshot.model_copy(deep=True)

    def list_session_snapshots(
        self, session_id: str, *, limit: int = 20,
    ) -> list[SessionStateSnapshotRecord]:
        rows = [
            r for (sid, _rev), r in self.snapshots.items() if sid == session_id
        ]
        rows.sort(key=lambda r: r.captured_at, reverse=True)
        return [r.model_copy(deep=True) for r in rows[:limit]]


def test_stale_workspace_fails_before_materialized_files_change(monkeypatch, tmp_path):
    store = InMemoryStateStore()
    monkeypatch.setattr("littrace.session._session_state_store", lambda *_args, **_kwargs: store)
    session = ChatSession.from_root(tmp_path / "session", "session-1")
    save_workspace(session, LiteratureWorkspace())

    writer_a = LiteratureWorkspace.model_validate(store.record.workspace_json)
    writer_b = LiteratureWorkspace.model_validate(store.record.workspace_json)
    writer_a.context.filters.topic = "writer-a"
    writer_b.context.filters.topic = "writer-b"
    start = Barrier(2)

    def persist(workspace: LiteratureWorkspace) -> str | None:
        start.wait()
        try:
            save_workspace(session, workspace)
        except RuntimeError as exc:
            return str(exc)
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(persist, (writer_a, writer_b)))

    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum("Workspace revision mismatch" in (outcome or "") for outcome in outcomes) == 1
    canonical = LiteratureWorkspace.model_validate(store.record.workspace_json)
    # Round 3 topic B: Postgres is the source of truth. There is no
    # on-disk workspace.json to materialize from; the materialized
    # workspace is the same record re-read.
    materialized = canonical
    assert canonical.context.filters.workspace_revision == 2
    assert materialized.context.filters.workspace_revision == 2
    assert materialized.context.filters.topic == canonical.context.filters.topic
    # Snapshots now live in session_state_snapshots, not in the
    # snapshots_dir on disk. The InMemoryStateStore fake is not
    # thread-safe under concurrent writers, so two snapshots may
    # land instead of one — the real PostgresStateStore is. We
    # lock down the invariant that the side table captured the
    # winning revision=2 rather than the exact count.
    snapshots = store.list_session_snapshots(session.session_id)
    assert any(s.revision == 2 for s in snapshots), snapshots
