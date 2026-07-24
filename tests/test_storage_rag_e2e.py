from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from littrace.artifact_registry import artifact_registry_from_config
from littrace.artifact_store import BlobRef, artifact_store_from_config
from littrace.config import ArtifactStorageConfig, LitTraceConfig, MetadataStoreConfig, StorageConfig
from littrace.downloads import execute_downloads
from littrace.models import AccessType, DownloadExecutionRequest, LiteratureWorkspace, PaperMetadata
from littrace.rag_jobs import run_pending_embedding_jobs
from littrace.retrieval.pgvector_store import PgvectorRagStore
from littrace.retrieval.rag_profile import load_session_rag_profile
from littrace.session import create_chat_session, save_workspace
from littrace.skill_runner import parse_workspace_skill


SKIP = os.environ.get("RUN_STORAGE_RAG_E2E") != "1"
skip_reason = "Set RUN_STORAGE_RAG_E2E=1 to run the storage/RAG integration test."


class _EmbeddingHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path != "/v1/embeddings":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        inputs = payload.get("input") or []
        if not isinstance(inputs, list):
            inputs = [inputs]
        data = [
            {"index": index, "embedding": [0.1 + index * 0.01, 0.2 + index * 0.01, 0.3 + index * 0.01]}
            for index, _ in enumerate(inputs)
        ]
        body = json.dumps({"data": data}).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):  # noqa: D401
        return


@pytest.fixture
def embedding_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=2.0)


@pytest.mark.anyio
@pytest.mark.skipif(SKIP, reason=skip_reason)
async def test_download_to_object_storage_and_refresh_rag_embeddings(
    tmp_path,
    embedding_server,
):
    pytest.importorskip("boto3")

    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
            sessions_dir=tmp_path / "sessions",
            default_user_id="e2e-user",
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
    config.rag.embedding_base_url = embedding_server
    config.rag.embedding_api_key = "test-key"
    config.rag.embedding_dimension = 3
    config.rag.auto_refresh_enabled = False

    paper = PaperMetadata(
        paper_id="arxiv1706",
        title="Attention Is All You Need",
        year=2017,
        doi=None,
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        access_type=AccessType.OPEN_ACCESS,
    )

    _configure_minio_env()
    _ensure_bucket(config)

    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    workspace.papers[paper.paper_id] = paper
    workspace.context.active_papers = [paper.paper_id]

    download_result = await execute_downloads(
        config,
        [paper],
        DownloadExecutionRequest(
            paper_ids=[paper.paper_id],
            user_id=config.storage.default_user_id,
            session_id=session.session_id,
            dry_run=False,
        ),
    )
    assert download_result.downloaded_count == 1
    assert download_result.items[0].storage_ref is not None

    workspace, parse_report = await parse_workspace_skill(workspace, config)
    assert parse_report["parsed_count"] >= 1
    workspace.context.filters.topic = "Attention Is All You Need"
    save_workspace(session, workspace, config=config)

    state_store = config_state_store(config)
    pending_before = state_store.list_pending_embedding_jobs(limit=20)
    assert pending_before

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

    loaded_profile = load_session_rag_profile(session)
    assert loaded_profile is not None
    store = PgvectorRagStore(config, loaded_profile)
    hits = store.query_chunks([0.1, 0.2, 0.3], top_k=5)
    assert hits

    registry = artifact_registry_from_config(config)
    record = registry.get(
        "paper_pdf:arxiv1706",
        user_id=config.storage.default_user_id,
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
