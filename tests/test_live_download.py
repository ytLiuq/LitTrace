"""Opt-in integration checks for the real publisher download path.

Run with ``LITTRACE_LIVE_DOWNLOAD_TESTS=1``. These tests intentionally require
network access and a usable Chrome/CDP environment; they are not part of the
fast unit-test suite.
"""

from __future__ import annotations

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
