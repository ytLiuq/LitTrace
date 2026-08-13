from littrace.state_db import (
    ArtifactOutboxRecord,
    EmbeddingJobRecord,
    PaperLifecycleEventRecord,
    PostgresStateStore,
)


def test_postgres_delete_session_removes_session_records(monkeypatch):
    store = PostgresStateStore("postgresql://example/littrace", schema_name="littrace_test")
    executed: list[tuple[str, dict[str, object] | None]] = []

    class FakeResult:
        rowcount = 1

    class FakeConnection:
        def execute(self, sql, params=None):
            executed.append((sql, params))
            return FakeResult()

        def commit(self):
            executed.append(("COMMIT", None))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(store, "_ensure_schema", lambda: None)
    monkeypatch.setattr(store, "_connect", lambda: FakeConnection())

    deleted = store.delete_session("s1")

    assert deleted == 7
    assert all("session_permissions" not in sql for sql, _ in executed if isinstance(sql, str))


def test_postgres_claim_pending_embedding_jobs_sets_lease(monkeypatch):
    store = PostgresStateStore("postgresql://example/littrace", schema_name="littrace_test")
    job = EmbeddingJobRecord(
        job_id="job1",
        profile_id="profile1",
        session_id="s1",
        artifact_id="paper_pdf:p1",
    )
    executed: list[tuple[str, dict[str, object] | None]] = []

    class FakeResult:
        def __init__(self, rows=None):
            self._rows = rows or []
            self.rowcount = len(self._rows)

        def fetchall(self):
            return self._rows

    class FakeConnection:
        def execute(self, sql, params=None):
            executed.append((sql, params))
            if sql.strip().startswith("SELECT payload"):
                return FakeResult([[job.model_dump(mode="json")]])
            return FakeResult([[job.model_dump(mode="json")]])

        def commit(self):
            executed.append(("COMMIT", None))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(store, "_ensure_schema", lambda: None)
    monkeypatch.setattr(store, "_connect", lambda: FakeConnection())

    claimed = store.claim_pending_embedding_jobs(worker_id="worker-1", limit=1, lease_seconds=30)

    assert claimed[0].status == "running"
    assert claimed[0].lease_owner == "worker-1"
    assert claimed[0].lease_expires_at is not None
    assert claimed[0].attempt_count == 1
    assert any("FOR UPDATE SKIP LOCKED" in sql for sql, _ in executed if isinstance(sql, str))
    claim_sql = next(sql for sql, _ in executed if isinstance(sql, str) and "FOR UPDATE SKIP LOCKED" in sql)
    assert claim_sql.index("LIMIT %(limit)s") < claim_sql.index("FOR UPDATE SKIP LOCKED")


def test_postgres_embedding_job_queue_report(monkeypatch):
    store = PostgresStateStore("postgresql://example/littrace", schema_name="littrace_test")

    class FakeResult:
        def __init__(self, rows=None):
            self._rows = rows or []

        def fetchall(self):
            return self._rows

        def fetchone(self):
            return self._rows[0] if self._rows else None

    class FakeConnection:
        def execute(self, sql, params=None):
            if "GROUP BY status" in sql:
                return FakeResult([("queued", 2), ("running", 1), ("dead", 1)])
            if "MIN(updated_at)" in sql:
                return FakeResult([(2, "2026-07-25T00:00:00+00:00")])
            if "lease_expires_at" in sql:
                return FakeResult([(1,)])
            if "last_error" in sql:
                return FakeResult([("ValueError: bad pdf",)])
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(store, "_ensure_schema", lambda: None)
    monkeypatch.setattr(store, "_connect", lambda: FakeConnection())

    report = store.embedding_job_queue_report()

    assert report.total == 4
    assert report.queued == 2
    assert report.running == 1
    assert report.dead == 1
    assert report.ready_to_claim == 2
    assert report.reclaimable_running == 1
    assert report.latest_error == "ValueError: bad pdf"


def test_postgres_requeue_dead_embedding_jobs(monkeypatch):
    store = PostgresStateStore("postgresql://example/littrace", schema_name="littrace_test")
    job = EmbeddingJobRecord(
        job_id="job1",
        profile_id="profile1",
        session_id="s1",
        artifact_id="paper_pdf:p1",
        status="dead",
        attempt_count=3,
        last_error="ValueError: bad pdf",
        completed_at="2026-07-25T00:00:00+00:00",
    )
    executed: list[tuple[str, dict[str, object] | None]] = []

    class FakeResult:
        def __init__(self, rows=None):
            self._rows = rows or []

        def fetchall(self):
            return self._rows

    class FakeConnection:
        def execute(self, sql, params=None):
            executed.append((sql, params))
            if sql.strip().startswith("SELECT payload"):
                return FakeResult([[job.model_dump(mode="json")]])
            return FakeResult()

        def commit(self):
            executed.append(("COMMIT", None))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(store, "_ensure_schema", lambda: None)
    monkeypatch.setattr(store, "_connect", lambda: FakeConnection())

    count = store.requeue_dead_embedding_jobs(session_id="s1", limit=5)

    assert count == 1
    update_params = next(params for sql, params in executed if sql.strip().startswith("UPDATE"))
    assert update_params["status"] == "queued"
    assert update_params["last_error"] is None
    assert update_params["completed_at"] is None


def test_postgres_appends_lifecycle_event(monkeypatch):
    store = PostgresStateStore("postgresql://example/littrace", schema_name="littrace_test")
    executed = []

    class Connection:
        def execute(self, sql, params=None):
            executed.append((sql, params))
            return object()
        def commit(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(store, "_ensure_schema", lambda: None)
    monkeypatch.setattr(store, "_connect", lambda: Connection())
    event = store.append_paper_lifecycle_event(PaperLifecycleEventRecord(
        event_id="event-1", session_id="s1", paper_id="p1", event_type="acquisition_verified",
    ))
    assert event.event_id == "event-1"
    sql, params = executed[0]
    assert "paper_lifecycle_events" in sql
    assert params["event_type"] == "acquisition_verified"


def test_postgres_claims_outbox_with_lease(monkeypatch):
    store = PostgresStateStore("postgresql://example/littrace", schema_name="littrace_test")
    record = ArtifactOutboxRecord(outbox_id="outbox-1", session_id="s1", artifact_id="paper_pdf:p1", content_sha256="abc")
    executed = []

    class Result:
        def __init__(self, rows=()): self.rows = rows
        def fetchall(self): return self.rows
    class Connection:
        def execute(self, sql, params=None):
            executed.append((sql, params))
            return Result([[
                record.outbox_id, record.session_id, record.artifact_id, record.event_type,
                record.content_sha256, record.status, record.attempt_count,
                record.next_attempt_at, record.lease_owner, record.lease_expires_at,
                record.last_error, record.payload, record.created_at, record.updated_at,
                record.completed_at,
            ]]) if "FROM littrace_test.artifact_outbox" in sql else Result()
        def commit(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(store, "_ensure_schema", lambda: None)
    monkeypatch.setattr(store, "_connect", lambda: Connection())
    claimed = store.claim_artifact_outbox(worker_id="worker", limit=1, lease_seconds=10)
    assert claimed[0].status == "running"
    assert claimed[0].attempt_count == 1
    assert claimed[0].lease_owner == "worker"
    assert any("FOR UPDATE SKIP LOCKED" in sql for sql, _ in executed)


def test_embedding_job_update_allows_running_job_to_complete(monkeypatch):
    store = PostgresStateStore("postgresql://example/littrace", schema_name="littrace_test")
    executed = []

    class Connection:
        def execute(self, sql, params=None):
            executed.append((sql, params))
            return object()
        def commit(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(store, "_ensure_schema", lambda: None)
    monkeypatch.setattr(store, "_connect", lambda: Connection())
    store.update_embedding_job(EmbeddingJobRecord(
        job_id="job-1", profile_id="p", session_id="s", artifact_id="paper_pdf:x",
        status="completed", completed_at="2026-01-01T00:00:00+00:00",
    ))
    sql = executed[0][0]
    assert "status = 'completed'" in sql
    assert "IN ('completed', 'running')" not in sql
    assert "lease_owner = EXCLUDED.lease_owner" in sql
