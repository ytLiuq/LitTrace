from __future__ import annotations

import os

import pytest

from littrace.artifact_registry import artifact_registry_from_config
from littrace.artifact_store import BlobRef, artifact_store_from_config
from littrace.config import ArtifactStorageConfig, LitTraceConfig, MetadataStoreConfig, StorageConfig
from littrace.downloads import execute_downloads
from littrace.models import (
    AccessType,
    DownloadExecutionRequest,
    LiteratureWorkspace,
    PaperMetadata,
    ParsedPaper,
)
from littrace.rag_jobs import run_pending_embedding_jobs
from littrace.session_metrics import build_session_knowledge_metrics
from littrace.retrieval.pgvector_store import PgvectorRagStore
from littrace.retrieval.embeddings import embedding_client_from_config
from littrace.retrieval.rag_profile import load_session_rag_profile
from littrace.session import create_chat_session, save_workspace


SKIP = (
    os.environ.get("RUN_STORAGE_RAG_E2E") != "1"
    or not os.environ.get("LITTRACE_E2E_EMBEDDING_BASE_URL")
    or not os.environ.get("LITTRACE_E2E_EMBEDDING_API_KEY")
)
skip_reason = (
    "Set RUN_STORAGE_RAG_E2E=1 and configure a real LITTRACE_E2E_EMBEDDING_BASE_URL "
    "and LITTRACE_E2E_EMBEDDING_API_KEY to run this integration test."
)


@pytest.mark.anyio
@pytest.mark.skipif(SKIP, reason=skip_reason)
async def test_download_to_object_storage_and_refresh_rag_embeddings(
    tmp_path,
):
    pytest.importorskip("boto3")

    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
            sessions_dir=tmp_path / "sessions",
        ),
        artifact_storage=ArtifactStorageConfig(
            backend="s3",
            bucket="littrace-e2e",
            endpoint_url="http://127.0.0.1:9000",
            region="us-east-1",
            path_prefix="integration",
        ),
        metadata_store=MetadataStoreConfig(
            backend="postgres",
            postgres_dsn="postgresql://littrace:littrace@localhost:5433/littrace",
            schema_name="littrace_e2e",
        ),
    )
    config.rag.enabled = True
    config.rag.postgres_dsn = "postgresql://littrace:littrace@localhost:5433/littrace"
    config.rag.schema_name = "littrace_rag_e2e"
    config.rag.collection_prefix = "littrace_e2e"
    config.rag.embedding_base_url = os.environ["LITTRACE_E2E_EMBEDDING_BASE_URL"]
    config.rag.embedding_api_key = os.environ["LITTRACE_E2E_EMBEDDING_API_KEY"]
    config.rag.embedding_model = os.environ.get("LITTRACE_E2E_EMBEDDING_MODEL", "text-embedding-3-small")
    config.rag.embedding_dimension = int(os.environ.get("LITTRACE_E2E_EMBEDDING_DIMENSION", "1536"))
    config.rag.auto_refresh_enabled = False
    config.download_retry.max_attempts = 1

    papers = [
        PaperMetadata(paper_id="arxiv1706", title="Attention Is All You Need", year=2017, pdf_url="https://arxiv.org/pdf/1706.03762.pdf", access_type=AccessType.OPEN_ACCESS),
        PaperMetadata(paper_id="arxiv1810", title="BERT", year=2018, pdf_url="https://arxiv.org/pdf/1810.04805.pdf", access_type=AccessType.OPEN_ACCESS),
        PaperMetadata(paper_id="arxiv2005", title="Language Models are Few-Shot Learners", year=2020, pdf_url="https://arxiv.org/pdf/2005.14165.pdf", access_type=AccessType.OPEN_ACCESS),
    ]
    papers.extend([
        PaperMetadata(paper_id="arxiv-missing-1", title="Missing arXiv Record One", year=2024, pdf_url="https://arxiv.org/pdf/9999.99991.pdf", access_type=AccessType.OPEN_ACCESS),
        PaperMetadata(paper_id="arxiv-missing-2", title="Missing arXiv Record Two", year=2024, pdf_url="https://arxiv.org/pdf/9999.99992.pdf", access_type=AccessType.OPEN_ACCESS),
    ])

    _configure_minio_env()
    _ensure_bucket(config)

    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    workspace.papers = {paper.paper_id: paper for paper in papers}
    workspace.context.active_papers = [paper.paper_id for paper in papers]

    download_result = await execute_downloads(
        config,
        papers,
        DownloadExecutionRequest(
            paper_ids=[paper.paper_id for paper in papers],
            session_id=session.session_id,
            dry_run=False,
        ),
    )
    assert download_result.downloaded_count == 3
    assert download_result.requires_login_count == 0
    assert sum(item.status == "failed" for item in download_result.items) == 2

    # Parsing has its own integration coverage.  The durable queue needs parsed
    # content to verify the artifact can be embedded after outbox dispatch.
    for paper in papers[:3]:
        workspace.parsed_papers[paper.paper_id] = ParsedPaper(
            title=paper.title,
            parsed=True,
            sections=[{"name": "abstract", "text": f"Verified text for {paper.paper_id}."}],
        )
    workspace.context.filters.topic = "mixed acquisition batch"
    save_workspace(session, workspace, config=config)

    state_store = config_state_store(config)
    lifecycle_events = state_store.list_paper_lifecycle_events(
        session.session_id,
    )
    assert {event.event_type for event in lifecycle_events} >= {
        "discovered_relevant",
        "artifact_stored",
        "acquisition_verified",
    }

    processed = 0
    for _ in range(4):
        embedding_report = await run_pending_embedding_jobs(config)
        assert embedding_report.failed == 0
        processed += embedding_report.processed
        pending_after = state_store.list_pending_embedding_jobs(limit=20)
        if not pending_after:
            break
    else:
        assert pending_after == []
    assert processed >= 1
    session_jobs = state_store.list_embedding_jobs(
        session_id=session.session_id,
        limit=20,
    )
    assert session_jobs
    assert all(job.status == "completed" for job in session_jobs)

    loaded_profile = load_session_rag_profile(session)
    assert loaded_profile is not None
    store = PgvectorRagStore(config, loaded_profile)
    query_embedding = (await embedding_client_from_config(config, loaded_profile).embed_texts(["attention mechanisms"]))[0]
    hits = store.query_chunks(query_embedding, top_k=5)
    assert hits

    metrics = build_session_knowledge_metrics(config, session.session_id)
    assert metrics.discovery.value == 5
    assert metrics.acquisition.value == 0.6
    assert metrics.acquisition.numerator == 3
    assert metrics.acquisition.denominator == 5
    assert metrics.rag.value == 1.0
    assert metrics.consistency.value == 1.0

    registry = artifact_registry_from_config(config)
    record = registry.get(
        "paper_pdf:arxiv1706",
        session_id=session.session_id,
    )
    assert record is not None
    assert record.kind == "paper_pdf"

    object_store = artifact_store_from_config(config)
    storage_ref = download_result.items[0].storage_ref
    assert storage_ref is not None
    assert object_store.exists(
        BlobRef(
            backend=storage_ref["backend"],
            bucket=storage_ref.get("bucket"),
            object_key=storage_ref["object_key"],
            uri=storage_ref.get("uri"),
            sha256=storage_ref.get("sha256"),
            size_bytes=storage_ref.get("size_bytes"),
            content_type=storage_ref.get("content_type"),
            metadata=storage_ref.get("metadata", {}),
        )
    )


def config_state_store(config: LitTraceConfig):
    from littrace.state_db import state_store_from_config

    store = state_store_from_config(config)
    assert store is not None
    return store


def _configure_minio_env() -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "littrace")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "littrace123")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")


def _ensure_bucket(config: LitTraceConfig) -> None:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=config.artifact_storage.endpoint_url,
        region_name=config.artifact_storage.region,
    )
    try:
        client.head_bucket(Bucket=config.artifact_storage.bucket)
    except Exception:
        client.create_bucket(Bucket=config.artifact_storage.bucket)
