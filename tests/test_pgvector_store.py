from littrace.retrieval.pgvector_store import PgvectorRagStore, RagChunkRecord, pgvector_setup_sql
from littrace.retrieval.rag_profile import RagProfile
from littrace.retrieval.rag_search import rag_hits_to_evidence_spans


def test_pgvector_setup_sql_binds_profile_and_session():
    profile = RagProfile(
        profile_id="rag:123",
        user_id="u1",
        session_id="s1",
        namespace="u1.s1",
        topic="MXene pressure sensor",
        query_variants=["MXene pressure sensor"],
        backend="pgvector",
        postgres_schema="littrace_rag",
        collection_name="littrace_u1_s1",
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
        chunk_target_tokens=700,
        chunk_overlap_tokens=120,
        top_k=12,
        refresh_frequency="daily",
        auto_refresh_enabled=False,
        auto_download_open_access=True,
        login_required_policy="queue_only",
    )

    statements = pgvector_setup_sql(profile)

    assert any("CREATE EXTENSION IF NOT EXISTS vector" in stmt for stmt in statements)
    assert any("CREATE SCHEMA IF NOT EXISTS \"littrace_rag\"" in stmt for stmt in statements)
    assert any("vector(1536)" in stmt for stmt in statements)
    assert any("CHECK (user_id = 'u1')" in stmt for stmt in statements)
    assert any("CHECK (session_id = 's1')" in stmt for stmt in statements)
    assert any("USING hnsw (embedding vector_cosine_ops)" in stmt for stmt in statements)


def test_pgvector_ensure_schema_creates_extension_before_registering_type(monkeypatch):
    profile = RagProfile(
        profile_id="rag:123",
        user_id="u1",
        session_id="s1",
        namespace="u1.s1",
        topic="MXene pressure sensor",
        query_variants=["MXene pressure sensor"],
        backend="pgvector",
        postgres_schema="littrace_rag",
        collection_name="littrace_u1_s1",
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
        chunk_target_tokens=700,
        chunk_overlap_tokens=120,
        top_k=12,
        refresh_frequency="daily",
        auto_refresh_enabled=False,
        auto_download_open_access=True,
        login_required_policy="queue_only",
    )
    config = type("Config", (), {})()
    config.rag = type("Rag", (), {})()
    config.rag.backend = "pgvector"
    config.rag.postgres_dsn = "postgresql://u:p@localhost:5432/db"
    calls = []

    class FakeCursor:
        def execute(self, sql, params=None):
            calls.append(("execute", sql.strip(), params))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            calls.append(("commit",))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("psycopg.connect", lambda dsn: FakeConnection())
    monkeypatch.setattr(
        "pgvector.psycopg.register_vector",
        lambda connection: calls.append(("register_vector",)),
    )

    PgvectorRagStore(config, profile).ensure_schema()

    extension_index = next(
        index
        for index, call in enumerate(calls)
        if call[0] == "execute" and "CREATE EXTENSION IF NOT EXISTS vector" in call[1]
    )
    register_index = calls.index(("register_vector",))
    schema_index = next(
        index
        for index, call in enumerate(calls)
        if call[0] == "execute" and 'CREATE SCHEMA IF NOT EXISTS "littrace_rag"' in call[1]
    )
    assert extension_index < register_index < schema_index
    assert calls[extension_index + 1] == ("commit",)


def test_rag_chunk_record_keeps_chunk_payload():
    record = RagChunkRecord(
        chunk_id="chunk:1",
        paper_id="paper:1",
        text="The sensitivity reached 12.5 kPa-1.",
        embedding=[0.1, 0.2, 0.3],
        chunk_hash="hash:1",
        source_record_id="source:1",
        section="Results",
        page=4,
        metadata={"source": "docling"},
    )

    assert record.paper_id == "paper:1"
    assert record.embedding == [0.1, 0.2, 0.3]
    assert record.metadata["source"] == "docling"


def test_pgvector_query_chunks_returns_ranked_hits(monkeypatch):
    profile = RagProfile(
        profile_id="rag:123",
        user_id="u1",
        session_id="s1",
        namespace="u1.s1",
        topic="MXene pressure sensor",
        query_variants=["MXene pressure sensor"],
        backend="pgvector",
        postgres_schema="littrace_rag",
        collection_name="littrace_u1_s1",
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
        chunk_target_tokens=700,
        chunk_overlap_tokens=120,
        top_k=12,
        refresh_frequency="daily",
        auto_refresh_enabled=False,
        auto_download_open_access=True,
        login_required_policy="queue_only",
    )
    config = type("Config", (), {})()
    config.rag = type("Rag", (), {})()
    config.rag.backend = "pgvector"
    config.rag.postgres_dsn = "postgresql://u:p@localhost:5432/db"

    rows = [
        ("chunk:1", "paper:1", "src:1", "Results", 4, None, "hash:1", "first chunk", {"source": "docling"}, 0.94),
        ("chunk:2", "paper:2", None, None, None, None, "hash:2", "second chunk", {}, 0.87),
    ]
    calls = {}

    class FakeCursor:
        rowcount = 0

        def execute(self, sql, params=None):
            calls["sql"] = sql
            calls["params"] = params

        def fetchall(self):
            return rows

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            calls["committed"] = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("psycopg.connect", lambda dsn: FakeConnection())
    monkeypatch.setattr("pgvector.psycopg.register_vector", lambda connection: None)

    store = PgvectorRagStore(config, profile)
    hits = store.query_chunks([0.1, 0.2, 0.3], top_k=2)

    assert len(hits) == 2
    assert hits[0].chunk_id == "chunk:1"
    assert hits[0].score == 0.94
    assert "ORDER BY embedding <=> %(embedding)s::vector" in calls["sql"]
    assert calls["params"]["limit"] == 2


def test_pgvector_delete_missing_chunks_scopes_to_profile(monkeypatch):
    profile = RagProfile(
        profile_id="rag:123",
        user_id="u1",
        session_id="s1",
        namespace="u1.s1",
        topic="MXene pressure sensor",
        query_variants=["MXene pressure sensor"],
        backend="pgvector",
        postgres_schema="littrace_rag",
        collection_name="littrace_u1_s1",
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
        chunk_target_tokens=700,
        chunk_overlap_tokens=120,
        top_k=12,
        refresh_frequency="daily",
        auto_refresh_enabled=False,
        auto_download_open_access=True,
        login_required_policy="queue_only",
    )
    config = type("Config", (), {})()
    config.rag = type("Rag", (), {})()
    config.rag.backend = "pgvector"
    config.rag.postgres_dsn = "postgresql://u:p@localhost:5432/db"
    calls = {}

    class FakeCursor:
        rowcount = 7

        def execute(self, sql, params=None):
            calls["sql"] = sql
            calls["params"] = params

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            calls["committed"] = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("psycopg.connect", lambda dsn: FakeConnection())
    monkeypatch.setattr("pgvector.psycopg.register_vector", lambda connection: None)

    store = PgvectorRagStore(config, profile)
    deleted = store.delete_missing_chunks(["chunk:1", "chunk:2"])

    assert deleted == 7
    assert "NOT (chunk_id = ANY(%(chunk_ids)s))" in calls["sql"]
    assert calls["params"]["chunk_ids"] == ["chunk:1", "chunk:2"]


def test_rag_hits_convert_to_evidence_spans():
    profile = RagProfile(
        profile_id="rag:123",
        user_id="u1",
        session_id="s1",
        namespace="u1.s1",
        collection_name="littrace_u1_s1",
        topic="MXene pressure sensor",
        query_variants=["MXene pressure sensor"],
        source_routes=["crossref", "openalex"],
        backend="pgvector",
        postgres_schema="littrace_rag",
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
        chunk_target_tokens=700,
        chunk_overlap_tokens=120,
        top_k=12,
        refresh_frequency="daily",
        auto_refresh_enabled=True,
        auto_download_open_access=True,
        login_required_policy="queue_only",
    )
    hit = type(
        "Hit",
        (),
        {
            "chunk_id": "chunk:1",
            "paper_id": "paper:1",
            "text": "The sensor reached high sensitivity.",
            "score": 0.91,
            "chunk_hash": "hash:1",
            "source_record_id": None,
            "section": "Results",
            "page": 4,
            "table_id": None,
            "metadata": {},
        },
    )()

    spans = rag_hits_to_evidence_spans(profile, [hit], query="high sensitivity")

    assert spans[0].evidence_id == "rag:rag:123:chunk:1"
    assert spans[0].parser == "rag"
    assert spans[0].snippet == "The sensor reached high sensitivity."
    assert spans[0].confidence == 0.91
