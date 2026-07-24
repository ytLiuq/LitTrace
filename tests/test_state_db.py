from littrace.state_db import PostgresStateStore


def test_postgres_delete_session_removes_artifact_permission_scope(monkeypatch):
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

    deleted = store.delete_session("s1", user_id="u1")

    permission_sql, permission_params = executed[-2]
    assert deleted == 5
    assert "session_permissions" in permission_sql
    assert "scope = 'artifact'" in permission_sql
    assert "resource_key LIKE %(artifact_prefix)s" in permission_sql
    assert "AND user_id = %(user_id)s" in permission_sql
    assert permission_params == {
        "session_id": "s1",
        "user_id": "u1",
        "artifact_prefix": "s1:%",
    }
