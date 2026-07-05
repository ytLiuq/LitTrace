from littrace.attachments import attach_pdf_to_paper
from littrace.auto_resume import (
    auto_archive_login_downloads,
    auto_resume_downloaded_pdfs,
    run_browser_session_download_handoff_test,
    watch_and_resume_downloads,
)
from littrace.access import target_pdf_path
from littrace.config import LitTraceConfig, StorageConfig
from littrace.context import add_papers
from littrace.models import AccessType, LiteratureWorkspace, PaperMetadata


def test_auto_resume_parses_ready_pdf_and_exports_artifacts(tmp_path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4")
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026)],
    )
    attach_pdf_to_paper(config, workspace, "p1", source)

    workspace, result = auto_resume_downloaded_pdfs(config, workspace)

    assert result.ready_to_parse_count == 1
    assert "p1" in workspace.parsed_papers


def test_auto_archive_login_downloads_moves_recent_pdf_to_target(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026)],
    )
    target = target_pdf_path(config, workspace.papers["p1"])
    target.parent.mkdir(parents=True)
    downloaded = target.parent / "publisher-download.pdf"
    downloaded.write_bytes(b"%PDF-1.4")

    count, warnings = auto_archive_login_downloads(config, workspace)

    assert count == 1
    assert target.exists()
    assert not downloaded.exists()
    assert "Auto-archived" in warnings[0]


def test_watch_and_resume_downloads_completes_when_pdf_appears(tmp_path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4")
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026)],
    )
    attach_pdf_to_paper(config, workspace, "p1", source)

    workspace, result = watch_and_resume_downloads(
        config,
        workspace,
        timeout_seconds=0.1,
        poll_interval_seconds=0.1,
    )

    assert result.completed
    assert result.resume_result.ready_to_parse_count == 1
    assert "p1" in workspace.parsed_papers


def test_browser_session_download_handoff_reports_timeout_for_gated_paper(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Gated Paper",
                year=2026,
                source_urls=["https://example.org/article"],
                access_type=AccessType.REQUIRES_LOGIN,
            )
        ],
    )

    workspace, result = run_browser_session_download_handoff_test(
        config,
        workspace,
        timeout_seconds=0.1,
        poll_interval_seconds=0.1,
    )

    assert result.planned_count == 1
    assert result.watch_result.completed is False
    assert result.target_paths
    assert "No authorized PDF appeared" in result.warnings[0]
