from littrace.config import BrowserAutomationConfig, LitTraceConfig
from littrace.access_layer.authorized_pdf_archiver import AuthorizedPdfArchiveResult
from littrace.access_layer.login_flow import (
    authorized_pdf_url_for_paper,
    browser_login_session_for_paper,
    BrowserAuthorizationWaitResult,
    detect_user_confirmation_required,
    discover_institutional_login_url_from_browser_session,
    discover_pdf_url_from_browser_session,
    fetch_authorized_pdf_after_user_auth,
    launch_login_for_paper,
    login_action_for_paper,
    open_browser_login_session,
    open_institutional_login_if_available,
    resume_browser_auth_after_user_close,
    wait_for_browser_authorization,
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
    assert "browser" in plan.browser_act_command
    assert "open" in plan.browser_act_command
    assert plan.session_name == "littrace-p1-auth"
    assert "--headed" in plan.browser_act_command
    assert plan.background_resume_command
    assert "--headed" not in plan.background_resume_command


def test_browser_login_session_plan_can_use_chat_scoped_publisher_window():
    paper = PaperMetadata(
        paper_id="p1",
        title="Gated",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://example.org/gated"],
    )

    plan = browser_login_session_for_paper(
        LitTraceConfig(),
        paper,
        browser_session_name="littrace-chat1-publisher",
    )

    assert plan.session_name == "littrace-chat1-publisher"
    assert "--session" in plan.browser_act_command
    assert "littrace-chat1-publisher" in plan.browser_act_command


def test_browser_login_session_plan_infers_acs_pdf_url():
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )

    plan = browser_login_session_for_paper(LitTraceConfig(), paper)

    assert plan.pdf_url == "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548"
    assert plan.background_pdf_command
    assert "--headed" not in plan.background_pdf_command


def test_open_browser_login_session_blocks_confirm_fallback_by_default(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    calls = []

    class FakeRunResult:
        def __init__(self, returncode, stderr="", recoverable=False, api_key_required=False):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = stderr
            self.recoverable_window_closed = False
            self.recoverable_browser_open_failed = recoverable
            self.api_key_required = api_key_required

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        if len(calls) == 1:
            return FakeRunResult(1, 'Error: Session "littrace-acs-auth" not found', True)
        return FakeRunResult(1, "Error 230404: Unknown error", True)

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.access_layer.login_flow.time.sleep", lambda *_args, **_kwargs: None)

    config = LitTraceConfig(
        browser=BrowserAutomationConfig(
            default_browser_id="direct_local_105121787802550357",
            default_browser_type="chrome-direct",
            confirm_before_use=True,
            allow_confirm_browser_fallback=False,
            chrome_direct_open_retries=1,
        )
    )

    result = open_browser_login_session(config, paper)

    assert not result.opened
    assert result.fallback_blocked
    assert "allow_confirm_browser_fallback is false" in (result.error or "")
    assert len(calls) == 3
    assert calls[0][2] == "navigate"
    assert calls[1][4] == calls[2][4]
    browser_ids = [call[4] for call in calls if "browser" in call and "open" in call]
    assert "littrace-publisher-auth" not in browser_ids


def test_open_browser_login_session_does_not_invent_fixed_browser_fallback_without_default_id(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    calls = []

    class FakeRunResult:
        def __init__(self, returncode, stderr="", recoverable=False):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = stderr
            self.recoverable_window_closed = recoverable
            self.recoverable_browser_open_failed = recoverable
            self.api_key_required = False

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        if len(calls) == 1:
            return FakeRunResult(1, 'Error: Session "littrace-acs-auth" not found', True)
        return FakeRunResult(1, "Error 230404: Unknown error", True)

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.access_layer.login_flow.time.sleep", lambda *_args, **_kwargs: None)

    config = LitTraceConfig(
        browser=BrowserAutomationConfig(
            allow_confirm_browser_fallback=False,
            chrome_direct_open_retries=1,
        )
    )

    result = open_browser_login_session(config, paper)

    assert not result.opened
    browser_ids = [call[4] for call in calls if "browser" in call and "open" in call]
    assert browser_ids == ["littrace-publisher-auth", "littrace-publisher-auth"]


def test_open_browser_login_session_can_fallback_when_explicitly_allowed(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    calls = []

    class FakeRunResult:
        def __init__(self, returncode, stderr="", recoverable=False):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = stderr
            self.recoverable_window_closed = False
            self.recoverable_browser_open_failed = recoverable
            self.api_key_required = False

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        if len(calls) == 1:
            return FakeRunResult(1, 'Error: Session "littrace-acs-auth" not found', True)
        if len(calls) in {2, 3}:
            return FakeRunResult(1, "Error 230404: Unknown error", True)
        return FakeRunResult(0)

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.access_layer.login_flow.time.sleep", lambda *_args, **_kwargs: None)
    config = LitTraceConfig(
        browser=BrowserAutomationConfig(
            default_browser_id="direct_local_105121787802550357",
            allow_confirm_browser_fallback=True,
            chrome_direct_open_retries=1,
        )
    )

    result = open_browser_login_session(config, paper)

    assert result.opened
    assert result.fallback_used
    assert len(calls) == 4
    assert calls[2][4] != calls[3][4]


def test_open_browser_login_session_falls_back_after_nonrecoverable_direct_failure(monkeypatch):
    paper = PaperMetadata(
        paper_id="wiley",
        title="Wiley paper",
        doi="10.1002/adfm.202316712",
        publisher="Wiley",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://advanced.onlinelibrary.wiley.com/doi/10.1002/adfm.202316712"],
    )
    calls = []

    class FakeRunResult:
        def __init__(self, returncode, stderr="", recoverable=False):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = stderr
            self.recoverable_window_closed = recoverable
            self.recoverable_browser_open_failed = recoverable
            self.api_key_required = False

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        if len(calls) == 1:
            return FakeRunResult(1, 'Error: Session "littrace-wiley-auth" not found', True)
        if len(calls) == 2:
            return FakeRunResult(1, "Error 210101: Connection health check failed.")
        return FakeRunResult(0)

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.access_layer.login_flow.time.sleep", lambda *_args, **_kwargs: None)
    config = LitTraceConfig(
        browser=BrowserAutomationConfig(
            default_browser_id="direct_local_105121787802550357",
            default_browser_type="chrome-direct",
            confirm_before_use=True,
            allow_confirm_browser_fallback=True,
            chrome_direct_open_retries=1,
        )
    )

    result = open_browser_login_session(config, paper)

    assert result.opened
    assert result.fallback_used
    browser_ids = [call[4] for call in calls if "browser" in call and "open" in call]
    assert browser_ids == [
        "direct_local_105121787802550357",
        "chrome_local_104956678805389514",
    ]


def test_open_browser_login_session_retries_same_browser_after_recoverable_open_failure(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    calls = []

    class FakeRunResult:
        def __init__(self, returncode, stderr="", recoverable=False):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = stderr
            self.recoverable_window_closed = recoverable
            self.recoverable_browser_open_failed = recoverable
            self.api_key_required = False

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        if len(calls) == 1:
            return FakeRunResult(1, 'Error: Session "littrace-acs-auth" not found', True)
        if len(calls) == 2:
            return FakeRunResult(1, "Browser window not found", True)
        return FakeRunResult(0)

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.access_layer.login_flow.time.sleep", lambda *_args, **_kwargs: None)

    config = LitTraceConfig(
        browser=BrowserAutomationConfig(chrome_direct_open_retries=1)
    )

    result = open_browser_login_session(config, paper)

    assert result.opened
    assert not result.fallback_used
    assert len(calls) == 3
    assert calls[1][4] == calls[2][4]


def test_open_browser_login_session_uses_configured_multi_retry(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    calls = []

    class FakeRunResult:
        def __init__(self, returncode, stderr="", recoverable=False):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = stderr
            self.recoverable_window_closed = recoverable
            self.recoverable_browser_open_failed = recoverable
            self.api_key_required = False

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        if len(calls) == 1:
            return FakeRunResult(1, 'Error: Session "littrace-acs-auth" not found', True)
        if len(calls) in {2, 3, 4}:
            return FakeRunResult(1, "Error 230404: Unknown error", True)
        return FakeRunResult(0)

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.access_layer.login_flow.time.sleep", lambda *_args, **_kwargs: None)
    config = LitTraceConfig(
        browser=BrowserAutomationConfig(chrome_direct_open_retries=3)
    )

    result = open_browser_login_session(config, paper)

    assert result.opened
    assert len(calls) == 5
    assert all(call[4] == calls[1][4] for call in calls[1:])


def test_open_browser_login_session_prewarms_chrome_direct_before_open(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    events = []

    class FakeRunResult:
        def __init__(self, returncode, stderr="", recoverable=False):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = stderr
            self.recoverable_window_closed = recoverable
            self.recoverable_browser_open_failed = recoverable
            self.api_key_required = False

    class FakePrewarm:
        attempted = True
        ok = True
        error = None

    def fake_run(_config, args, **_kwargs):
        events.append(args[2] if len(args) > 2 else args[0])
        if len(events) == 1:
            return FakeRunResult(1, 'Error: Session "littrace-acs-auth" not found', True)
        return FakeRunResult(0)

    def fake_prewarm(_config):
        events.append("prewarm")
        return FakePrewarm()

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.access_layer.login_flow.prewarm_chrome_direct", fake_prewarm)
    config = LitTraceConfig(
        browser=BrowserAutomationConfig(
            default_browser_id="direct_local_105121787802550357",
            default_browser_type="chrome-direct",
        )
    )

    result = open_browser_login_session(config, paper)

    assert result.opened
    assert events == ["navigate", "prewarm", "browser"]


def test_open_browser_login_session_reuses_existing_chat_window_with_navigate(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    calls = []

    class FakeRunResult:
        returncode = 0
        stdout = "navigated"
        stderr = ""
        recoverable_window_closed = False
        recoverable_browser_open_failed = False
        api_key_required = False

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        return FakeRunResult()

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)

    result = open_browser_login_session(
        LitTraceConfig(),
        paper,
        browser_session_name="littrace-chat1-publisher",
    )

    assert result.opened
    assert result.session_name == "littrace-chat1-publisher"
    assert calls == [
        [
            "--session",
            "littrace-chat1-publisher",
            "navigate",
            "https://pubs.acs.org/doi/10.1021/acsomega.2c06548",
        ]
    ]


def test_open_browser_login_session_stops_when_browser_act_api_key_required(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    calls = []

    class FakeRunResult:
        def __init__(self, stderr, recoverable=False, api_key_required=False):
            self.returncode = 1
            self.stdout = ""
            self.stderr = stderr
            self.recoverable_window_closed = recoverable
            self.recoverable_browser_open_failed = recoverable
            self.api_key_required = api_key_required

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        if len(calls) == 1:
            return FakeRunResult('Error: Session "littrace-acs-auth" not found', recoverable=True)
        return FakeRunResult(
            "Error 230103: API key required for browser open",
            api_key_required=True,
        )

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)

    result = open_browser_login_session(LitTraceConfig(), paper)

    assert not result.opened
    assert "API key required" in (result.error or "")
    assert len(calls) == 2


def test_authorized_pdf_url_prefers_full_text_pdf_candidate():
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
                url="https://publisher.example.org/paper.pdf",
                source="crossref.link",
                content_type="application/pdf",
                is_pdf=True,
                requires_login=True,
            )
        ],
    )

    assert authorized_pdf_url_for_paper(paper, report) == "https://publisher.example.org/paper.pdf"


def test_browser_login_session_plan_opens_mdpi_article_before_pdf_discovery():
    paper = PaperMetadata(
        paper_id="mdpi",
        title="MDPI paper",
        doi="10.3390/s24010001",
        publisher="MDPI",
        source_urls=["https://www.mdpi.com/1424-8220/24/1/1"],
        pdf_url="https://www.mdpi.com/1424-8220/24/1/1/pdf",
    )

    plan = browser_login_session_for_paper(LitTraceConfig(), paper)

    assert str(plan.login_url) == "https://www.mdpi.com/1424-8220/24/1/1"
    assert plan.pdf_url == "https://www.mdpi.com/1424-8220/24/1/1/pdf"


def test_browser_login_session_plan_opens_sciencedirect_article_before_pdf_discovery():
    paper = PaperMetadata(
        paper_id="elsevier",
        title="ScienceDirect paper",
        doi="10.1016/j.nanoen.2024.109999",
        publisher="Elsevier",
        source_urls=["https://www.sciencedirect.com/science/article/pii/S2211285524009999"],
        pdf_url="https://www.sciencedirect.com/science/article/pii/S2211285524009999/pdfft",
    )

    plan = browser_login_session_for_paper(LitTraceConfig(), paper)

    assert str(plan.login_url) == "https://www.sciencedirect.com/science/article/pii/S2211285524009999"
    assert plan.pdf_url == "https://www.sciencedirect.com/science/article/pii/S2211285524009999/pdfft"


def test_browser_login_session_plan_opens_rsc_article_before_pdf_discovery():
    paper = PaperMetadata(
        paper_id="rsc",
        title="RSC paper",
        doi="10.1039/D4TC00001A",
        publisher="Royal Society of Chemistry",
        source_urls=["https://pubs.rsc.org/en/content/articlelanding/2024/tc/d4tc00001a"],
        pdf_url="https://pubs.rsc.org/en/content/articlepdf/2024/tc/d4tc00001a",
    )

    plan = browser_login_session_for_paper(LitTraceConfig(), paper)

    assert str(plan.login_url) == "https://pubs.rsc.org/en/content/articlelanding/2024/tc/d4tc00001a"
    assert plan.pdf_url == "https://pubs.rsc.org/en/content/articlepdf/2024/tc/d4tc00001a"


def test_resume_browser_auth_after_user_close_reports_recoverable_window_loss(monkeypatch):
    paper = PaperMetadata(
        paper_id="p1",
        title="Gated",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://example.org/gated"],
    )

    class FakeRunResult:
        returncode = 1
        stdout = ""
        stderr = "Error: Browser window not found"
        recoverable_window_closed = True

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", lambda *_args, **_kwargs: FakeRunResult())

    result = resume_browser_auth_after_user_close(LitTraceConfig(), paper)

    assert result.attempted
    assert result.recoverable_window_closed
    assert result.error is None
    assert result.session_name == "littrace-p1-resume"


def test_fetch_authorized_pdf_after_user_auth_uses_background_pdf_command(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    calls = []

    class FakeRunResult:
        returncode = 0
        stdout = "opened"
        stderr = ""
        recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        return FakeRunResult()

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr(
        "littrace.access_layer.login_flow.archive_authorized_pdf_response",
        lambda *_args, **_kwargs: AuthorizedPdfArchiveResult(
            paper_id="acs",
            pdf_url="https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548",
            target_path="/tmp/paper.pdf",
            archived=True,
        ),
    )

    result = fetch_authorized_pdf_after_user_auth(LitTraceConfig(), paper)

    assert not result.opened_pdf
    assert result.pdf_url == "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548"
    assert not any("browser" in call and "open" in call for call in calls)


def test_discover_pdf_url_from_browser_session_extracts_frontend_pdf_link(monkeypatch):
    class FakeRunResult:
        returncode = 0
        stdout = '{"pdfUrl":"https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548","title":"Article","text":""}'
        stderr = ""
        recoverable_window_closed = False

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", lambda *_args, **_kwargs: FakeRunResult())

    result = discover_pdf_url_from_browser_session(LitTraceConfig(), "littrace-acs-auth")

    assert result.pdf_url == "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548"


def test_discover_pdf_url_marks_cloudflare_confirmation_required(monkeypatch):
    class FakeRunResult:
        returncode = 0
        stdout = '{"pdfUrl":null,"title":"Just a moment...","url":"https://pubs.acs.org/doi/10.1021/acsomega.2c06548","text":"请验证您是真人 Cloudflare"}'
        stderr = ""
        recoverable_window_closed = False

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", lambda *_args, **_kwargs: FakeRunResult())

    result = discover_pdf_url_from_browser_session(LitTraceConfig(), "littrace-acs-auth")

    assert result.requires_user_confirmation
    assert result.pdf_url is None


def test_discover_pdf_url_accepts_current_pdf_viewer_url(monkeypatch):
    class FakeRunResult:
        returncode = 0
        stdout = '{"pdfUrl":null,"title":"","url":"https://www.nature.com/articles/s41598-024-60080-z.pdf","text":""}'
        stderr = ""
        recoverable_window_closed = False

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", lambda *_args, **_kwargs: FakeRunResult())

    result = discover_pdf_url_from_browser_session(LitTraceConfig(), "littrace-nature-auth")

    assert result.pdf_url == "https://www.nature.com/articles/s41598-024-60080-z.pdf"
    assert not result.requires_user_confirmation


def test_discover_pdf_url_marks_cloudflare_query_confirmation_required(monkeypatch):
    class FakeRunResult:
        returncode = 0
        stdout = '{"pdfUrl":null,"title":"","url":"https://www.science.org/doi/pdf/10.1126/sciadv.ads2297?__cf_chl_rt_tk=token","text":""}'
        stderr = ""
        recoverable_window_closed = False

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", lambda *_args, **_kwargs: FakeRunResult())

    result = discover_pdf_url_from_browser_session(LitTraceConfig(), "littrace-science-auth")

    assert result.requires_user_confirmation


def test_discover_pdf_url_uses_state_fallback_for_cloudflare(monkeypatch):
    calls = []

    class FakeRunResult:
        returncode = 0
        stdout = ""
        stderr = ""
        recoverable_window_closed = False

    class FakeStateResult:
        returncode = 0
        stdout = "title=Just a moment...\nVerify you are human\nCloudflare"
        stderr = ""
        recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        return FakeStateResult() if args[-1] == "state" else FakeRunResult()

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)

    result = discover_pdf_url_from_browser_session(LitTraceConfig(), "littrace-wiley-auth")

    assert result.requires_user_confirmation
    assert result.access_state == "confirmation_required"
    assert len(calls) == 2


def test_discover_pdf_url_extracts_sciencedirect_pdfft_link(monkeypatch):
    class FakeRunResult:
        returncode = 0
        stdout = '{"pdfUrl":"https://www.sciencedirect.com/science/article/pii/S2211285524009999/pdfft?isDTMRedir=true","title":"Article","text":""}'
        stderr = ""
        recoverable_window_closed = False

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", lambda *_args, **_kwargs: FakeRunResult())

    result = discover_pdf_url_from_browser_session(LitTraceConfig(), "littrace-elsevier-auth")

    assert result.pdf_url == "https://www.sciencedirect.com/science/article/pii/S2211285524009999/pdfft?isDTMRedir=true"


def test_discover_pdf_url_extracts_rsc_articlepdf_link(monkeypatch):
    class FakeRunResult:
        returncode = 0
        stdout = '{"pdfUrl":"https://pubs.rsc.org/en/content/articlepdf/2024/tc/d4tc00001a","title":"Article","text":""}'
        stderr = ""
        recoverable_window_closed = False

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", lambda *_args, **_kwargs: FakeRunResult())

    result = discover_pdf_url_from_browser_session(LitTraceConfig(), "littrace-rsc-auth")

    assert result.pdf_url == "https://pubs.rsc.org/en/content/articlepdf/2024/tc/d4tc00001a"


def test_detect_user_confirmation_required_reads_page_text(monkeypatch):
    class FakeRunResult:
        returncode = 0
        stdout = "Just a moment...\n请验证您是真人"
        stderr = ""
        recoverable_window_closed = False

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", lambda *_args, **_kwargs: FakeRunResult())

    result = detect_user_confirmation_required(LitTraceConfig(), "littrace-acs-auth")

    assert result.requires_user_confirmation


def test_fetch_authorized_pdf_prefers_frontend_pdf_link_over_fallback(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    calls = []

    class FakeRunResult:
        returncode = 0
        stdout = ""
        stderr = ""
        recoverable_window_closed = False

    def fake_discover(*_args, **_kwargs):
        class Frontend:
            pdf_url = "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?download=true"
            access_state = "authorized"
            recoverable_window_closed = False
            requires_user_confirmation = False
            requires_login = False
            stdout = ""
            stderr = ""
        return Frontend()

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        return FakeRunResult()

    monkeypatch.setattr("littrace.access_layer.login_flow.discover_pdf_url_from_browser_session", fake_discover)
    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr(
        "littrace.access_layer.login_flow.archive_authorized_pdf_response",
        lambda *_args, **_kwargs: AuthorizedPdfArchiveResult(
            paper_id="acs",
            pdf_url="https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?download=true",
            target_path="/tmp/paper.pdf",
            archived=True,
        ),
    )

    result = fetch_authorized_pdf_after_user_auth(LitTraceConfig(), paper)

    assert result.pdf_url == "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?download=true"
    assert calls == []


def test_fetch_authorized_pdf_opens_pdf_session_after_auth_archive_failure(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    calls = []
    archive_calls = []

    class FakeRunResult:
        returncode = 0
        stdout = "opened"
        stderr = ""
        recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        return FakeRunResult()

    def fake_archive(_config, _paper, session_name, pdf_url, **_kwargs):
        archive_calls.append(session_name)
        return AuthorizedPdfArchiveResult(
            paper_id="acs",
            pdf_url=pdf_url,
            target_path="/tmp/paper.pdf",
            archived=len(archive_calls) == 2,
            error=None if len(archive_calls) == 2 else "auth session network logs not ready",
        )

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.access_layer.login_flow.archive_authorized_pdf_response", fake_archive)

    result = fetch_authorized_pdf_after_user_auth(
        LitTraceConfig(),
        paper,
        auth_wait_result=BrowserAuthorizationWaitResult(
            session_name="littrace-acs-auth",
            authorized=True,
            attempts=1,
            elapsed_seconds=0.1,
            pdf_url="https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548",
        ),
    )

    assert result.opened_pdf
    assert result.archive_result
    assert result.archive_result.archived
    assert archive_calls == ["littrace-acs-auth", "littrace-acs-pdf"]
    browser_open_calls = [call for call in calls if "browser" in call and "open" in call]
    assert browser_open_calls
    assert browser_open_calls[0][-1] == "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548"


def test_fetch_authorized_pdf_keeps_landing_first_publishers_in_auth_session(monkeypatch):
    paper = PaperMetadata(
        paper_id="mdpi",
        title="MDPI paper",
        doi="10.3390/s24010001",
        publisher="MDPI",
        source_urls=["https://www.mdpi.com/1424-8220/24/1/1"],
        pdf_url="https://www.mdpi.com/1424-8220/24/1/1/pdf?version=1",
    )
    calls = []

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        raise AssertionError("landing-first publishers should not open a second PDF session")

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr(
        "littrace.access_layer.login_flow.archive_authorized_pdf_response",
        lambda *_args, **_kwargs: AuthorizedPdfArchiveResult(
            paper_id="mdpi",
            pdf_url="https://www.mdpi.com/1424-8220/24/1/1/pdf?version=1",
            target_path="/tmp/paper.pdf",
            archived=False,
            error="browser click fallback failed",
        ),
    )

    result = fetch_authorized_pdf_after_user_auth(
        LitTraceConfig(),
        paper,
        auth_wait_result=BrowserAuthorizationWaitResult(
            session_name="littrace-mdpi-auth",
            authorized=True,
            attempts=1,
            elapsed_seconds=0.1,
            pdf_url="https://www.mdpi.com/1424-8220/24/1/1/pdf?version=1",
        ),
    )

    assert result.session_name == "littrace-mdpi-auth"
    assert not result.opened_pdf
    assert "browser click fallback failed" in (result.error or "")
    assert calls == []


def test_fetch_authorized_pdf_stops_when_auth_session_is_not_authorized(monkeypatch):
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
        publisher="American Chemical Society",
        access_type=AccessType.REQUIRES_LOGIN,
        source_urls=["https://pubs.acs.org/doi/10.1021/acsomega.2c06548"],
    )
    calls = []

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        raise AssertionError("should not open PDF browser session without authorization")

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)
    monkeypatch.setattr(
        "littrace.access_layer.login_flow.wait_for_browser_authorization",
        lambda *_args, **_kwargs: BrowserAuthorizationWaitResult(
            session_name="littrace-acs-auth",
            authorized=False,
            attempts=1,
            elapsed_seconds=0.1,
            error="No active session",
        ),
    )

    result = fetch_authorized_pdf_after_user_auth(LitTraceConfig(), paper)

    assert not result.attempted
    assert "No active session" in (result.error or "")
    assert calls == []


def test_wait_for_browser_authorization_polls_until_pdf_link(monkeypatch):
    calls = []

    def fake_discover(*_args, **_kwargs):
        calls.append(1)

        class Result:
            pdf_url = None if len(calls) == 1 else "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548"
            access_state = "confirmation_required" if len(calls) == 1 else "authorized"
            recoverable_window_closed = False
            requires_user_confirmation = len(calls) == 1
            requires_login = False
            stdout = "Just a moment..." if len(calls) == 1 else "article page"
            stderr = ""

        return Result()

    monkeypatch.setattr("littrace.access_layer.login_flow.discover_pdf_url_from_browser_session", fake_discover)

    result = wait_for_browser_authorization(
        LitTraceConfig(),
        "littrace-acs-auth",
        timeout_seconds=3,
        poll_interval_seconds=0.01,
    )

    assert result.authorized
    assert result.pdf_url == "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548"
    assert result.attempts == 2


def test_wait_for_browser_authorization_tolerates_stale_session_before_ready(monkeypatch):
    calls = []

    def fake_discover(*_args, **_kwargs):
        calls.append(1)

        class Result:
            pdf_url = None if len(calls) == 1 else "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548"
            access_state = "unknown" if len(calls) == 1 else "authorized"
            recoverable_window_closed = len(calls) == 1
            requires_user_confirmation = False
            requires_login = False
            stdout = ""
            stderr = 'Error: Session "littrace-acs-auth" not found' if len(calls) == 1 else ""

        return Result()

    monkeypatch.setattr("littrace.access_layer.login_flow.discover_pdf_url_from_browser_session", fake_discover)

    result = wait_for_browser_authorization(
        LitTraceConfig(),
        "littrace-acs-auth",
        timeout_seconds=3,
        poll_interval_seconds=0.01,
    )

    assert result.authorized
    assert result.attempts == 2


def test_wait_for_browser_authorization_times_out(monkeypatch):
    class Result:
        pdf_url = None
        access_state = "confirmation_required"
        recoverable_window_closed = False
        requires_user_confirmation = True
        requires_login = False
        stdout = "Just a moment..."
        stderr = ""

    monkeypatch.setattr(
        "littrace.access_layer.login_flow.discover_pdf_url_from_browser_session",
        lambda *_args, **_kwargs: Result(),
    )

    result = wait_for_browser_authorization(
        LitTraceConfig(),
        "littrace-acs-auth",
        timeout_seconds=0.01,
        poll_interval_seconds=0.01,
    )

    assert not result.authorized
    assert result.requires_user_confirmation
    assert "human verification" in (result.error or "")


def test_wait_for_browser_authorization_does_not_treat_wiley_login_page_as_authorized(
    monkeypatch,
):
    class Result:
        pdf_url = "https://advanced.onlinelibrary.wiley.com/doi/pdf/10.1002/adfm.202316712"
        access_state = "login_required"
        recoverable_window_closed = False
        requires_user_confirmation = False
        requires_login = True
        stdout = "Login / Register wol_publication_access=no PDF"
        stderr = ""

    monkeypatch.setattr(
        "littrace.access_layer.login_flow.discover_pdf_url_from_browser_session",
        lambda *_args, **_kwargs: Result(),
    )

    result = wait_for_browser_authorization(
        LitTraceConfig(),
        "littrace-wiley-auth",
        timeout_seconds=0.01,
        poll_interval_seconds=0.01,
    )

    assert not result.authorized
    assert result.pdf_url is None
    assert result.requires_login
    assert result.access_state == "login_required"
    assert "requires publisher or institutional login" in (result.error or "")


def test_discover_pdf_url_treats_wiley_full_access_as_authorized(monkeypatch):
    class Result:
        returncode = 0
        stdout = (
            '{"pdfUrl":"https://advanced.onlinelibrary.wiley.com/doi/pdf/10.1002/adfm.202316712",'
            '"accessState":"authorized","loginRequired":true,"accessDenied":true,'
            '"fullAccess":true,"confirmationRequired":false,'
            '"title":"Wiley","url":"https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adfm.202316712",'
            '"text":"Medical College Of Shanghai Login / Register Full Access PDF"}'
        )
        stderr = ""
        recoverable_window_closed = False

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", lambda *_args, **_kwargs: Result())

    result = discover_pdf_url_from_browser_session(
        LitTraceConfig(),
        "littrace-wiley-auth",
    )

    assert result.access_state == "authorized"
    assert not result.requires_login
    assert result.pdf_url.endswith("/doi/pdf/10.1002/adfm.202316712")


def test_discover_institutional_login_url_prefers_wiley_ssostart(monkeypatch):
    class Result:
        returncode = 0
        stdout = (
            '{"institutionalUrl":"https://advanced.onlinelibrary.wiley.com/action/ssostart?'
            'redirectUri=%2Fdoi%2Fabs%2F10.1002%2Fadfm.202316712",'
            '"candidates":[]}'
        )
        stderr = ""
        recoverable_window_closed = False

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", lambda *_args, **_kwargs: Result())

    result = discover_institutional_login_url_from_browser_session(
        LitTraceConfig(),
        "littrace-wiley-auth",
    )

    assert result.institutional_url
    assert "/action/ssostart" in result.institutional_url


def test_open_institutional_login_if_available_navigates_when_login_required(monkeypatch):
    calls = []

    class Discovery:
        requires_login = True
        stdout = "Login / Register wol_publication_access=no"
        stderr = ""

    class Link:
        institutional_url = "https://advanced.onlinelibrary.wiley.com/action/ssostart?redirectUri=%2Fdoi%2Fabs%2F10.1002%2Fadfm.202316712"
        stdout = "institutional"
        stderr = ""
        error = None

    class RunResult:
        returncode = 0
        stdout = "navigated"
        stderr = ""
        recoverable_window_closed = False

    monkeypatch.setattr(
        "littrace.access_layer.login_flow.discover_pdf_url_from_browser_session",
        lambda *_args, **_kwargs: Discovery(),
    )
    monkeypatch.setattr(
        "littrace.access_layer.login_flow.discover_institutional_login_url_from_browser_session",
        lambda *_args, **_kwargs: Link(),
    )

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        return RunResult()

    monkeypatch.setattr("littrace.access_layer.login_flow.run_browser_act", fake_run)

    result = open_institutional_login_if_available(
        LitTraceConfig(),
        "littrace-wiley-auth",
    )

    assert result.opened
    assert calls[-1][2] == "navigate"
    assert "/action/ssostart" in calls[-1][3]
