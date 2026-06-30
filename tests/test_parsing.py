from littrace.config import LitTraceConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.parsing import parse_workspace_papers


def test_parse_workspace_papers_uses_metadata_only_tool_without_local_pdf():
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

    assert report["parser"] == "metadata_only"
    assert report["missing_pdf_count"] == 1
    assert workspace.parsed_papers["p1"]["title"] == "A materials paper"
