from littrace.cli import (
    ShellState,
    _paper_id_for_index,
    _parse_attach_args,
    _parse_index_arg,
    _parse_publisher_retrieve_args,
    format_context_panel,
    format_dashboard,
)
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


def test_cli_index_helpers():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper")],
    )

    assert _parse_index_arg("/login 1") == 1
    assert _parse_attach_args("/attach 1 /tmp/a.pdf") == (1, "/tmp/a.pdf")
    assert _parse_publisher_retrieve_args("/publisher-retrieve acs MXene sensor") == (
        "acs",
        "MXene sensor",
    )
    assert _paper_id_for_index(workspace, 1) == "p1"
    assert _paper_id_for_index(workspace, 2) is None


def test_format_dashboard_summarizes_shell_state():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper")],
    )
    state = ShellState(workspace=workspace, session_id="s1", session_root="/tmp/s1")

    dashboard = format_dashboard(state)

    assert "[LitTrace Dashboard]" in dashboard
    assert "1 papers" in dashboard
    assert "/attach N path.pdf" in dashboard
