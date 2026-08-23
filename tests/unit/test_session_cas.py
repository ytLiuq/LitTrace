from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Lock

import pytest

from littrace.models import LiteratureWorkspace
from littrace.session import ChatSession, save_workspace
from littrace.state_db import SessionStateRecord

pytestmark = pytest.mark.unit


class InMemoryStateStore:
    def __init__(self) -> None:
        self.record: SessionStateRecord | None = None
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
    materialized = LiteratureWorkspace.model_validate_json(
        session.workspace_path.read_text(encoding="utf-8")
    )
    assert canonical.context.filters.workspace_revision == 2
    assert materialized.context.filters.workspace_revision == 2
    assert materialized.context.filters.topic == canonical.context.filters.topic
    assert len(list(session.snapshots_dir.glob("workspace-*.json"))) == 2
