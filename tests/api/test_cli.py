"""CLI dashboard formatting test — pure rendering helper, no I/O."""

from __future__ import annotations

import pytest

from littrace.cli import ShellState, format_dashboard
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata


pytestmark = pytest.mark.api


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
