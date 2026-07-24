from __future__ import annotations

from fastapi import APIRouter

from littrace.attachments import (
    AttachmentResult,
    DownloadPresenceReport,
    attach_pdf_to_paper,
    check_download_presence,
)
from littrace.auto_resume import (
    AutoResumeResult,
    BrowserSessionDownloadTestResult,
    DownloadWatchResult,
    auto_resume_downloaded_pdfs_async,
    run_browser_session_download_handoff_test,
    watch_and_resume_downloads_async,
)
from littrace.access_layer import (
    BrowserLoginSessionPlan,
    LoginLaunchResult,
    browser_login_session_for_paper,
    launch_login_for_paper,
    publisher_window_session_name_for_chat,
)
from littrace.models import DownloadExecutionRequest, DownloadExecutionResult
from littrace.publisher_session import PublisherSessionE2EReport, build_publisher_session_e2e_report
from littrace.skill_runner import build_download_plan_skill, execute_downloads_skill
from littrace.session import load_or_create_session, save_workspace


class _AppProxy:
    def __getattr__(self, name: str):
        from littrace.api import app as api_app

        return getattr(api_app, name)


api_app = _AppProxy()

router = APIRouter()


@router.post("/downloads/plan", response_model=object)
async def download_plan():
    return await build_download_plan_skill(api_app.load_config(), api_app.WORKSPACE)


@router.post("/downloads/execute", response_model=DownloadExecutionResult)
async def downloads_execute(request: DownloadExecutionRequest) -> DownloadExecutionResult:
    return await execute_downloads_skill(api_app.load_config(), api_app.WORKSPACE, request)


@router.post("/downloads/login/{paper_id}", response_model=LoginLaunchResult)
def downloads_login(paper_id: str, dry_run: bool = False) -> LoginLaunchResult:
    config = api_app.load_config()
    paper = api_app.WORKSPACE.papers[paper_id]
    return launch_login_for_paper(
        config, paper, api_app.WORKSPACE.full_text_reports.get(paper_id), dry_run=dry_run
    )


@router.post("/downloads/browser-session/{paper_id}", response_model=BrowserLoginSessionPlan)
def downloads_browser_session(
    paper_id: str,
    browser_profile: str = "littrace-auth",
    session_id: str | None = None,
) -> BrowserLoginSessionPlan:
    config = api_app.load_config()
    paper = api_app.WORKSPACE.papers[paper_id]
    return browser_login_session_for_paper(
        config,
        paper,
        api_app.WORKSPACE.full_text_reports.get(paper_id),
        browser_profile=browser_profile,
        browser_session_name=publisher_window_session_name_for_chat(session_id),
    )


@router.post("/downloads/check", response_model=DownloadPresenceReport)
def downloads_check() -> DownloadPresenceReport:
    return check_download_presence(api_app.load_config(), api_app.WORKSPACE)


@router.post("/downloads/resume", response_model=AutoResumeResult)
async def downloads_resume(session_id: str | None = None) -> AutoResumeResult:
    config = api_app.load_config()
    session = load_or_create_session(config, session_id) if session_id else None
    workspace, result = await auto_resume_downloaded_pdfs_async(config, api_app.WORKSPACE, session)
    api_app._set_workspace(workspace)
    if session:
        save_workspace(session, api_app.WORKSPACE, config=config)
    return result


@router.post("/downloads/watch", response_model=DownloadWatchResult)
async def downloads_watch(
    session_id: str | None = None,
    timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 2.0,
) -> DownloadWatchResult:
    config = api_app.load_config()
    session = load_or_create_session(config, session_id) if session_id else None
    workspace, result = await watch_and_resume_downloads_async(
        config,
        api_app.WORKSPACE,
        session,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    api_app._set_workspace(workspace)
    if session:
        save_workspace(session, api_app.WORKSPACE, config=config)
    return result


@router.post("/papers/{paper_id}/attach-pdf", response_model=AttachmentResult)
def attach_pdf(paper_id: str, source_path: str) -> AttachmentResult:
    return attach_pdf_to_paper(api_app.load_config(), api_app.WORKSPACE, paper_id, source_path)


@router.post("/papers/{paper_id}/attach-si", response_model=object)
def attach_si(paper_id: str, source_path: str, session_id: str | None = None):
    from littrace.supplementary import attach_supplementary_file

    config = api_app.load_config()
    session = load_or_create_session(config, session_id)
    result = attach_supplementary_file(api_app.WORKSPACE, session, paper_id, source_path)
    save_workspace(session, api_app.WORKSPACE, config=config)
    return result


@router.post("/downloads/browser-session-test", response_model=BrowserSessionDownloadTestResult)
def downloads_browser_session_test(
    timeout_seconds: float = 5.0,
) -> BrowserSessionDownloadTestResult:
    config = api_app.load_config()
    workspace, result = run_browser_session_download_handoff_test(
        config, api_app.WORKSPACE, timeout_seconds=timeout_seconds
    )
    api_app._set_workspace(workspace)
    return result


@router.post("/downloads/publisher-session-test", response_model=PublisherSessionE2EReport)
def downloads_publisher_session_test(
    publisher_family: str | None = None,
    timeout_seconds: float = 5.0,
) -> PublisherSessionE2EReport:
    config = api_app.load_config()
    workspace, result = build_publisher_session_e2e_report(
        config,
        api_app.WORKSPACE,
        publisher_family=publisher_family,
        timeout_seconds=timeout_seconds,
    )
    api_app._set_workspace(workspace)
    return result
