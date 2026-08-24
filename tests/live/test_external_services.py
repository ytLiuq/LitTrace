"""Live external-service tests.

Consolidates ``test_live_download.py``, ``test_live_search.py``,
``test_live_daily_update.py`` and ``test_live_validation_matrix.py``.
All require real network/local Chrome + CDP — gated by the ``live``
marker so they only run when explicitly opted in (``pytest -m live``).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


# ---- test_live_download.py ----



import os

import pytest

from littrace.config import ArtifactStorageConfig, StorageConfig, load_config
from littrace.downloads import execute_downloads
from littrace.models import AccessType, DownloadExecutionRequest, PaperMetadata


@pytest.mark.live
@pytest.mark.anyio
async def test_live_mdpi_pdf_download(tmp_path):
    if os.environ.get("LITTRACE_LIVE_DOWNLOAD_TESTS") != "1":
        pytest.skip("set LITTRACE_LIVE_DOWNLOAD_TESTS=1 to run live publisher checks")

    config = load_config()
    config.storage = StorageConfig(
        paper_library_dir=tmp_path / "papers",
        metadata_dir=tmp_path / "metadata",
        cache_dir=tmp_path / "cache",
        sessions_dir=tmp_path / "sessions",
    )
    config.artifact_storage = ArtifactStorageConfig(local_root=tmp_path / "artifacts")
    config.cdp_downloader.auto_launch_chrome = True

    paper = PaperMetadata(
        paper_id="10.3390_s26051559",
        title="A Hybrid-Frequency Sampling Tactile Sensing System",
        doi="10.3390/s26051559",
        pdf_url="https://www.mdpi.com/1424-8220/26/5/1559/pdf",
        access_type=AccessType.OPEN_ACCESS,
    )
    result = await execute_downloads(
        config,
        [paper],
        DownloadExecutionRequest(paper_ids=[paper.paper_id], session_id="live-smoke"),
    )

    assert result.downloaded_count == 1, result.model_dump()
    target = config.storage.paper_library_dir / "unknown-year" / paper.paper_id / "paper.pdf"
    assert target.exists()
    assert target.stat().st_size > 1_000
    assert target.read_bytes()[:5] == b"%PDF-"

# ---- test_live_search.py ----

import httpx
import pytest


@pytest.mark.anyio



import os
from pathlib import Path
from uuid import uuid4

import pytest

from littrace.artifact_registry import artifact_registry_from_config
from littrace.artifact_store import BlobRef, artifact_store_from_config
from littrace.config import (
    ArtifactStorageConfig,
    MetadataStoreConfig,
    StorageConfig,
    load_config,
)
from littrace.rag_jobs import run_daily_rag_maintenance
from littrace.research_background import assess_research_background, set_workspace_research_background
from littrace.retrieval.rag_search import search_session_rag
from littrace.retrieval.search import filter_papers_by_retrieval_policy
from littrace.session import create_chat_session, load_workspace, save_workspace
from littrace.sentinel.agent import LiteratureSentinel
from littrace.sentinel.state import Watchlist


@pytest.mark.live
@pytest.mark.anyio
async def test_live_daily_update_runs_discovery_and_downloads(tmp_path: Path):
    if os.environ.get("LITTRACE_LIVE_DAILY_TESTS") != "1":
        pytest.skip("set LITTRACE_LIVE_DAILY_TESTS=1 to run the live daily update")

    config = load_config()
    config.storage = StorageConfig(
        paper_library_dir=tmp_path / "papers",
        metadata_dir=tmp_path / "metadata",
        cache_dir=tmp_path / "cache",
        sessions_dir=tmp_path / "sessions",
    )
    config.artifact_storage = ArtifactStorageConfig(local_root=tmp_path / "artifacts")
    config.api.enable_live_search = True
    config.cdp_downloader.auto_launch_chrome = True
    config.cdp_downloader.command_timeout_seconds = 120
    config.cdp_downloader.repository_download_timeout_seconds = 120
    # Keep this smoke test focused on the discovery/download contract. OCR has
    # its own parser tests and can be run as a follow-up workflow.
    config.sentinel.parse_on_daily = False

    watchlist = Watchlist(
        watchlist_id="live_daily_smoke",
        topic="Flexible piezoresistive sensor arrays",
        objective="Verify the daily literature discovery and PDF download path",
        query_variants=["flexible piezoresistive sensor array"],
        year_min=2024,
    )
    result = await LiteratureSentinel(config, watchlist).run()

    assert result.summary.new_candidates_count > 0, result.summary.model_dump()
    assert result.summary.downloaded_count > 0, result.summary.model_dump()
    assert result.digest_path and Path(result.digest_path).exists()
    assert result.run_dir and (Path(result.run_dir) / "run_summary.json").exists()
    downloaded_pdfs = list((tmp_path / "papers").rglob("paper.pdf"))
    assert downloaded_pdfs, "daily update reported downloads but wrote no local PDFs"
    assert all(pdf.stat().st_size > 1_000 for pdf in downloaded_pdfs)
    assert all(pdf.read_bytes()[:5] == b"%PDF-" for pdf in downloaded_pdfs)


@pytest.mark.live
@pytest.mark.anyio
async def test_live_daily_update_syncs_session_background_to_storage_and_rag(tmp_path: Path):
    """Exercise the scheduled per-session research-background update end to end."""
    if os.environ.get("LITTRACE_LIVE_DAILY_TESTS") != "1":
        pytest.skip("set LITTRACE_LIVE_DAILY_TESTS=1 to run the live daily update")

    config = load_config()
    if not config.rag.embedding_base_url or not config.rag.embedding_api_key:
        pytest.skip("real embedding endpoint is not configured")

    run_id = uuid4().hex[:12]
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "littrace")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "littrace123")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    config.storage = StorageConfig(
        paper_library_dir=tmp_path / "papers",
        metadata_dir=tmp_path / "metadata",
        cache_dir=tmp_path / "cache",
        sessions_dir=tmp_path / "sessions",
    )
    config.artifact_storage = ArtifactStorageConfig(
        backend="s3",
        bucket="littrace-e2e",
        endpoint_url="http://127.0.0.1:9000",
        region="us-east-1",
        path_prefix=f"daily-update/{run_id}",
    )
    config.metadata_store = MetadataStoreConfig(
        backend="postgres",
        postgres_dsn="postgresql://littrace:littrace@localhost:5433/littrace",
        schema_name=f"littrace_daily_{run_id}",
    )
    config.rag.enabled = True
    config.rag.backend = "pgvector"
    config.rag.postgres_dsn = "postgresql://littrace:littrace@localhost:5433/littrace"
    config.rag.schema_name = f"littrace_daily_rag_{run_id}"
    config.rag.collection_prefix = f"daily_{run_id}"
    config.rag.auto_refresh_enabled = True
    config.rag.auto_download_open_access = True
    config.api.enable_live_search = True
    config.cdp_downloader.auto_launch_chrome = True
    config.cdp_downloader.command_timeout_seconds = 120
    config.cdp_downloader.repository_download_timeout_seconds = 120
    config.parsing.default_parser = "docling"
    config.parsing.docling_workers = 2

    import boto3

    client = boto3.client("s3", endpoint_url=config.artifact_storage.endpoint_url)
    try:
        client.head_bucket(Bucket=config.artifact_storage.bucket)
    except Exception:
        client.create_bucket(Bucket=config.artifact_storage.bucket)

    session = create_chat_session(config)
    workspace = load_workspace(session)
    background = "近五年柔性薄膜压阻压力传感器的材料、微结构与长期稳定性"
    assessment = await assess_research_background(background, config)
    if not assessment.accepted or assessment.retrieval_policy is None:
        pytest.skip("real LLM research-background policy is not configured")
    set_workspace_research_background(
        workspace,
        assessment.background or background,
        topic=assessment.topic,
        retrieval_policy=assessment.retrieval_policy,
    )
    workspace.context.filters.year_min = 2021
    save_workspace(session, workspace, config=config)

    report = await run_daily_rag_maintenance(config)
    refreshed = load_workspace(session)
    artifacts = artifact_registry_from_config(config).list_for_session(
        session_id=session.session_id
    )
    pdf_artifacts = [artifact for artifact in artifacts if artifact.kind == "paper_pdf"]
    store = artifact_store_from_config(config)
    rag_result = await search_session_rag(
        config,
        session,
        "MWCNT PDMS flexible piezoresistive pressure sensor",
        top_k=3,
    )

    assert report.sessions_failed == 0, report.model_dump()
    assert refreshed.context.filters.research_background_last_sync_at
    assert refreshed.context.filters.research_background_last_downloaded_count > 0
    assert refreshed.context.filters.research_background_last_parsed_count > 0
    assert pdf_artifacts
    assert all(
        store.exists(
            BlobRef(
                backend=artifact.backend,
                bucket=artifact.bucket,
                object_key=artifact.object_key,
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
            )
        )
        for artifact in pdf_artifacts
    )
    assert rag_result is not None and rag_result.hits
    downloaded_paper_ids = {
        artifact.paper_id for artifact in pdf_artifacts if artifact.paper_id
    }
    downloaded_papers = [
        refreshed.papers[paper_id]
        for paper_id in downloaded_paper_ids
        if paper_id in refreshed.papers
    ]
    # Validate the actual session policy emitted by the research-background
    # assessment rather than hard-coding a topic-specific vocabulary here.
    assert downloaded_papers
    accepted, rejected = filter_papers_by_retrieval_policy(
        downloaded_papers,
        assessment.retrieval_policy,
    )
    assert len(accepted) == len(downloaded_papers), rejected

# ---- test_live_validation_matrix.py ----



import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _require_live() -> None:
    if os.environ.get("LITTRACE_LIVE_TESTS") != "1":
        pytest.skip("set LITTRACE_LIVE_TESTS=1 to run real validation tests")


def _run(script: str, *args: str, timeout: int) -> str:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    if completed.returncode:
        pytest.fail(f"{script} failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    return completed.stdout


def _json_output(output: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    pytest.fail(f"no JSON object in live command output: {output[-2000:]}")


@pytest.mark.live
def test_live_dual_path_30_paper_metrics() -> None:
    _require_live()
    output = _run("run_dual_path_30_e2e.py", "--path", "both", timeout=3600)
    payload = _json_output(output)
    assert len(payload["results"]) == 2
    for result in payload["results"]:
        assert result["object_records"] > 0
        assert result["objects_present"] == result["object_records"]
        assert result["metrics"]


@pytest.mark.live
def test_live_authenticated_publisher_downloads() -> None:
    _require_live()
    _run("run_seven_publisher_download_e2e.py", "--timeout", "300", "--user-wait-seconds", "90", timeout=2400)


@pytest.mark.live
def test_live_multi_topic_retrieval_policies() -> None:
    _require_live()
    output = _run("run_multi_topic_policy_e2e.py", timeout=1800)
    payload = _json_output(output)
    assert len(payload["topics"]) == 3
    assert all(item["gate"] == "accepted" or item.get("reason") for item in payload["topics"])
    accepted = [item for item in payload["topics"] if item["gate"] == "accepted"]
    assert accepted
    assert len({item["canonical_topic"] for item in accepted}) == len(accepted)
    assert all(item["accepted"] > 0 for item in accepted)


@pytest.mark.live
def test_live_embedding_failure_recovery() -> None:
    _require_live()
    _run("run_failure_recovery_e2e.py", timeout=900)
