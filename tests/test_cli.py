from littrace.cli import format_context_panel
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata


def test_format_context_panel_empty():
    assert format_context_panel(LiteratureWorkspace()) == "[上下文窗] 当前没有文献。"


def test_format_context_panel_lists_papers():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026, journal="ACS Nano")],
    )

    panel = format_context_panel(workspace)

    assert "[上下文窗] 当前文献 1 篇" in panel
    assert "Paper" in panel
