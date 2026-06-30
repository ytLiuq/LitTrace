from littrace.config import LitTraceConfig
from littrace.login_flow import launch_login_for_paper, login_action_for_paper
from littrace.models import AccessType, PaperMetadata


def test_login_action_for_gated_paper_has_target_and_instructions():
    paper = PaperMetadata(
        paper_id="p1",
        title="Gated",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://example.org/gated"],
    )

    action = login_action_for_paper(LitTraceConfig(), paper)

    assert action.action == "open_login_popup"
    assert str(action.login_url) == "https://example.org/gated"
    assert action.target_path.endswith("paper.pdf")
    assert "allowed" in action.login_instructions[1]


def test_launch_login_dry_run_does_not_open_browser():
    paper = PaperMetadata(
        paper_id="p1",
        title="Gated",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://example.org/gated"],
    )

    result = launch_login_for_paper(LitTraceConfig(), paper, dry_run=True)

    assert not result.opened
    assert str(result.login_url) == "https://example.org/gated"
    assert result.target_path
