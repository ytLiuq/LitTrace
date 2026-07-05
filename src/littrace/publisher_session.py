from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.auto_resume import BrowserSessionDownloadTestResult, run_browser_session_download_handoff_test
from littrace.config import LitTraceConfig
from littrace.login_flow import (
    AuthorizedPdfFetchResult,
    BrowserLoginSessionPlan,
    browser_login_session_plans_for_workspace,
    fetch_authorized_pdf_after_user_auth,
    publisher_window_session_name_for_chat,
)
from littrace.models import LiteratureWorkspace
from littrace.session import ChatSession


class PublisherSessionE2EReport(BaseModel):
    publisher_family: str | None = None
    planned_count: int = 0
    opened_count: int = 0
    completed: bool = False
    parsed_count: int = 0
    target_paths: list[str] = Field(default_factory=list)
    browser_plans: list[BrowserLoginSessionPlan] = Field(default_factory=list)
    authorized_pdf_fetches: list[AuthorizedPdfFetchResult] = Field(default_factory=list)
    download_test: BrowserSessionDownloadTestResult | None = None
    warnings: list[str] = Field(default_factory=list)


def build_publisher_session_e2e_report(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    session: ChatSession | None = None,
    publisher_family: str | None = None,
    timeout_seconds: float = 5.0,
) -> tuple[LiteratureWorkspace, PublisherSessionE2EReport]:
    browser_session_name = publisher_window_session_name_for_chat(session.session_id if session else None)
    plans = browser_login_session_plans_for_workspace(
        config,
        workspace,
        browser_session_name=browser_session_name,
    )
    if publisher_family:
        family_lowered = publisher_family.lower()
        plans = [
            plan
            for plan in plans
            if family_lowered in (workspace.papers[plan.paper_id].publisher or "").lower()
            or family_lowered in (workspace.papers[plan.paper_id].journal or "").lower()
        ]
    fetches: list[AuthorizedPdfFetchResult] = []
    for plan in plans:
        paper = workspace.papers[plan.paper_id]
        fetches.append(
            fetch_authorized_pdf_after_user_auth(
                config,
                paper,
                workspace.full_text_reports.get(plan.paper_id),
                timeout_seconds=min(max(timeout_seconds, 1.0), 60.0),
                browser_session_name=browser_session_name,
            )
        )
    workspace, download_test = run_browser_session_download_handoff_test(
        config,
        workspace,
        session=session,
        timeout_seconds=timeout_seconds,
    )
    warnings = list(download_test.warnings)
    for fetch in fetches:
        if fetch.error:
            warnings.append(f"authorized_pdf_fetch:{fetch.paper_id}: {fetch.error}")
        if fetch.archive_result and fetch.archive_result.warning:
            warnings.append(
                f"authorized_pdf_archive:{fetch.paper_id}: {fetch.archive_result.warning}"
            )
        elif fetch.recoverable_window_closed:
            warnings.append(
                f"authorized_pdf_fetch:{fetch.paper_id}: browser window closed after PDF handoff; continuing download watch."
            )
    if publisher_family and not plans:
        warnings.append(f"No active papers matched publisher family: {publisher_family}.")
    return workspace, PublisherSessionE2EReport(
        publisher_family=publisher_family,
        planned_count=len(plans),
        opened_count=0,
        completed=download_test.watch_result.completed,
        parsed_count=download_test.watch_result.resume_result.parsed_count,
        target_paths=[plan.target_path for plan in plans],
        browser_plans=plans,
        authorized_pdf_fetches=fetches,
        download_test=download_test,
        warnings=warnings,
    )
