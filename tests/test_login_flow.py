from littrace.config import LitTraceConfig
from littrace.login_flow import (
    browser_login_session_for_paper,
    launch_login_for_paper,
    login_action_for_paper,
)
from littrace.models import AccessType, FullTextCandidate, FullTextResolutionReport, PaperMetadata


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


def test_login_action_prefers_full_text_landing_candidate():
    paper = PaperMetadata(
        paper_id="p1",
        title="Gated",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://example.org/gated"],
    )
    report = FullTextResolutionReport(
        paper_id="p1",
        candidates=[
            FullTextCandidate(
                paper_id="p1",
                url="https://publisher.example.org/article",
                source="crossref.resource",
                requires_login=True,
            )
        ],
    )

    action = login_action_for_paper(LitTraceConfig(), paper, report)

    assert str(action.login_url) == "https://publisher.example.org/article"


def test_browser_login_session_plan_contains_download_handoff():
    paper = PaperMetadata(
        paper_id="p1",
        title="Gated",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://example.org/gated"],
    )

    plan = browser_login_session_for_paper(LitTraceConfig(), paper, browser_profile="test-profile")

    assert str(plan.login_url) == "https://example.org/gated"
    assert plan.browser_profile == "test-profile"
    assert plan.target_path.endswith("paper.pdf")
    assert "--download-dir" in plan.browser_act_command
