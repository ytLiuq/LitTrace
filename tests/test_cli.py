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

    assert "[上下文窗:显示] 当前文献 1 篇" in panel
    assert "Paper" in panel
    assert "选择第 1、3 篇下载" in panel


def test_format_context_panel_marks_selected_papers():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(paper_id="p1", title="Paper", year=2026, journal="ACS Nano"),
        ],
    )
    workspace.context.selected_for_download = ["p1"]

    panel = format_context_panel(workspace)

    assert "已选下载 1 篇" in panel
    assert "* 1. Paper" in panel
