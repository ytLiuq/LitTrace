from littrace.attachments import attach_pdf_to_paper
from littrace.auto_resume import auto_resume_downloaded_pdfs
from littrace.config import LitTraceConfig, StorageConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata


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
