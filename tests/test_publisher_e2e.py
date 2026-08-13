import os

import pytest

from littrace.access_layer.cdp_downloader import CDPDownloadResult
from littrace.config import LitTraceConfig, StorageConfig
from littrace.models import PaperMetadata
from littrace.publisher_e2e import (
    _case_dois,
    run_interactive_publisher_e2e,
    run_publisher_golden_e2e,
)


def test_case_dois_reads_expected_dois():
    assert _case_dois({"expected_dois": ["10.1021/example", "10.1039/example"]}) == [
        "10.1021/example",
        "10.1039/example",
    ]
    assert _case_dois({"expected_dois": "10.1002/example"}) == ["10.1002/example"]


@pytest.mark.anyio
async def test_real_publisher_golden_e2e_requires_full_text(tmp_path):
    if os.environ.get("LITTRACE_RUN_PUBLISHER_E2E") != "1":
        pytest.skip("Set LITTRACE_RUN_PUBLISHER_E2E=1 to run real publisher PDF E2E tests.")
    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
            cache_dir=tmp_path / "cache",
            sessions_dir=tmp_path / "sessions",
        )
    )

    report = await run_publisher_golden_e2e(config, timeout_seconds=180.0)

    assert report.paper_count > 0
    assert report.passed, report.model_dump_json(indent=2)


@pytest.mark.anyio
async def test_interactive_publisher_e2e_uses_cdp_downloader(monkeypatch, tmp_path):
    paper = PaperMetadata(
        paper_id="wiley",
        title="Wiley paper",
        doi="10.1002/adfm.202316712",
        publisher="Wiley",
        journal="Advanced Functional Materials",
        year=2024,
    )

    async def fake_fetch_crossref(_client, _doi):
        return paper

    async def fake_resolve(_client, _paper, _config):
        class Report:
            best_pdf_url = None
            warnings = []
            candidates = []
        return Report()

    def fake_download(_config, doi, target_path, email=None):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"%PDF-1.7\n")
        return CDPDownloadResult(
            doi=doi,
            publisher="wiley",
            target_path=str(target_path),
            downloaded=True,
            method="cdp_fetch_blob",
            source_url="https://advanced.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/adfm.202316712?download=true",
            file_size=9,
            steps=["cdp_open", "fetch_blob"],
        )

    monkeypatch.setattr("littrace.publisher_e2e.fetch_crossref_paper_by_doi", fake_fetch_crossref)
    monkeypatch.setattr("littrace.publisher_e2e.resolve_full_text_for_paper", fake_resolve)
    monkeypatch.setattr("littrace.publisher_e2e.download_paper_via_cdp", fake_download)
    async def fake_parse(workspace, _config):
        return workspace, {"parsed_count": 1}

    monkeypatch.setattr("littrace.publisher_e2e.parse_workspace_skill", fake_parse)

    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
            cache_dir=tmp_path / "cache",
            sessions_dir=tmp_path / "sessions",
        )
    )

    report = await run_interactive_publisher_e2e(config, "10.1002/adfm.202316712")

    assert report.completed
    assert report.downloaded_pdf
    assert report.parsed_full_text
    assert report.session_name == "cdp"
    assert report.method == "cdp_fetch_blob"
    assert "fetch_blob" in report.cdp_steps


@pytest.mark.anyio
async def test_interactive_publisher_e2e_reports_cdp_user_action(monkeypatch, tmp_path):
    paper = PaperMetadata(
        paper_id="wiley",
        title="Wiley paper",
        doi="10.1002/adfm.202316712",
        publisher="Wiley",
        journal="Advanced Functional Materials",
        year=2024,
    )

    async def fake_fetch_crossref(_client, _doi):
        return paper

    async def fake_resolve(_client, _paper, _config):
        class Report:
            best_pdf_url = None
            warnings = ["resolver warning"]
            candidates = []
        return Report()

    def fake_download(_config, doi, target_path, email=None):
        return CDPDownloadResult(
            doi=doi,
            publisher="wiley",
            target_path=str(target_path),
            downloaded=False,
            requires_user_action=True,
            user_action="请在已打开的本地 Chrome 窗口中完成 Cloudflare 人机验证。",
            error="Cloudflare verification did not complete in the local CDP browser.",
            warnings=["cdp warning"],
            steps=["cloudflare_wait"],
        )

    monkeypatch.setattr("littrace.publisher_e2e.fetch_crossref_paper_by_doi", fake_fetch_crossref)
    monkeypatch.setattr("littrace.publisher_e2e.resolve_full_text_for_paper", fake_resolve)
    monkeypatch.setattr("littrace.publisher_e2e.download_paper_via_cdp", fake_download)

    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
            cache_dir=tmp_path / "cache",
            sessions_dir=tmp_path / "sessions",
        )
    )

    report = await run_interactive_publisher_e2e(config, "10.1002/adfm.202316712")

    assert not report.completed
    assert report.needs_user_action
    assert "Cloudflare" in (report.user_action_message or "")
    assert report.last_access_state == "user_action_required"
    assert "resolver warning" in report.warnings
    assert "cdp warning" in report.warnings
