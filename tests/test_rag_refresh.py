import asyncio

from littrace.config import LitTraceConfig, StorageConfig
from littrace.models import LiteratureWorkspace, ParsedPaper
from littrace.retrieval.rag_profile import RagProfile
from littrace.retrieval.rag_refresh import build_rag_chunk_drafts, refresh_session_rag_index
from littrace.session import create_chat_session


def _profile() -> RagProfile:
    return RagProfile(
        profile_id="rag:123",
        session_id="s1",
        namespace="s1",
        topic="MXene pressure sensor",
        query_variants=["MXene pressure sensor"],
        backend="pgvector",
        postgres_schema="littrace_rag",
        collection_name="littrace_s1",
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_dimension=16,
        chunk_target_tokens=20,
        chunk_overlap_tokens=4,
        top_k=12,
        refresh_frequency="daily",
        auto_refresh_enabled=False,
        auto_download_open_access=True,
        login_required_policy="queue_only",
    )


def test_build_rag_chunk_drafts_preserves_session_payload():
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": ParsedPaper(
                parsed=True,
                sections=[
                    {
                        "name": "Results",
                        "text": "The pressure sensor reached high sensitivity and stable cycling.",
                        "evidence": {"page": 4, "parser": "docling"},
                    }
                ],
            )
        }
    )

    chunks = build_rag_chunk_drafts(workspace, _profile())

    assert len(chunks) == 1
    assert chunks[0].paper_id == "p1"
    assert chunks[0].section == "Results"
    assert chunks[0].page == 4
    assert chunks[0].metadata["session_id"] == "s1"


def test_refresh_skips_cleanly_when_rag_disabled(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    config.rag.enabled = False
    session = create_chat_session(config)
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": ParsedPaper(
                parsed=True,
                sections=[{"name": "Intro", "text": "MXene sensors are flexible."}],
            )
        }
    )

    _, report = asyncio.run(refresh_session_rag_index(config, session, workspace))

    assert report.skipped is True
    assert report.skip_reason == "rag_disabled_or_unconfigured"
    assert report.warnings == []
    assert workspace.context.filters.rag_refresh_report["skip_reason"] == report.skip_reason


def test_refresh_uses_real_embedding_boundary_and_pgvector_store(monkeypatch, tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    config.rag.enabled = True
    config.rag.postgres_dsn = "postgresql://littrace:littrace@localhost:5433/littrace"
    config.rag.embedding_base_url = "https://embeddings.example.com/v1"
    config.rag.embedding_api_key = "test-key"
    config.rag.embedding_dimension = 3
    session = create_chat_session(config)
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": ParsedPaper(
                parsed=True,
                sections=[{"name": "Intro", "text": "MXene sensors are flexible."}],
            )
        }
    )
    calls: dict[str, object] = {}

    class FakeEmbeddingClient:
        async def embed_texts(self, texts):
            calls["embedded_texts"] = texts
            return [[0.1, 0.2, 0.3] for _ in texts]

    class FakePgvectorStore:
        def __init__(self, config, profile):
            calls["profile_id"] = profile.profile_id

        def upsert_chunks(self, chunks):
            records = list(chunks)
            calls["chunks"] = records
            return len(records)

        def delete_missing_chunks(self, chunk_ids):
            calls["kept_chunk_ids"] = list(chunk_ids)
            return 0

    monkeypatch.setattr(
        "littrace.retrieval.rag_refresh.embedding_client_from_config",
        lambda config, profile: FakeEmbeddingClient(),
    )
    monkeypatch.setattr("littrace.retrieval.rag_refresh.PgvectorRagStore", FakePgvectorStore)

    _, report = asyncio.run(refresh_session_rag_index(config, session, workspace))

    assert report.skipped is False
    assert report.upserted_count == 1
    assert report.stale_chunk_count == 0
    assert calls["embedded_texts"] == ["MXene sensors are flexible."]
    assert calls["chunks"][0].embedding == [0.1, 0.2, 0.3]
    assert calls["kept_chunk_ids"] == [calls["chunks"][0].chunk_id]
