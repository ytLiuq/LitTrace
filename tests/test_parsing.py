from littrace.config import LitTraceConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.parsing import parse_workspace_papers


def test_parse_workspace_papers_fails_without_local_pdf():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="A materials paper",
                abstract="This method has limitations.",
                year=2026,
            )
        ],
    )

    workspace, report = parse_workspace_papers(workspace, LitTraceConfig())

    assert report["parser"] == "docling"
    assert report["missing_pdf_count"] == 1
    assert report["failed_count"] == 1
    assert report["parsed_count"] == 0
    assert workspace.parsed_papers["p1"].title == "A materials paper"
    assert "metadata/abstract fallback is disabled" in workspace.parsed_papers["p1"].error
