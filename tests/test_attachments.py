from pathlib import Path

from littrace.attachments import attach_pdf_to_paper, check_download_presence
from littrace.config import LitTraceConfig, StorageConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata


def test_attach_pdf_to_paper_copies_into_library(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4")
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026)],
    )

    result = attach_pdf_to_paper(config, workspace, "p1", source)

    assert result.attached
    assert Path(result.target_path).exists()
    assert Path(result.target_path).read_bytes() == b"%PDF-1.4"


def test_check_download_presence_reports_ready_pdf(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026)],
    )
    attach_pdf_to_paper(config, workspace, "p1", tmp_path / "missing.pdf")
    target = tmp_path / "papers" / "2026" / "p1" / "paper.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF-1.4")

    report = check_download_presence(config, workspace)

    assert report.ready_to_parse_count == 1
    assert report.missing_count == 0
