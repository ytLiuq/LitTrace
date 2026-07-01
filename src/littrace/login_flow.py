from __future__ import annotations

import webbrowser
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl

from littrace.access import target_pdf_path
from littrace.config import LitTraceConfig
from littrace.models import DownloadExecutionItem, FullTextResolutionReport, PaperMetadata


class LoginLaunchRequest(BaseModel):
    paper_id: str
    dry_run: bool = False


class LoginLaunchResult(BaseModel):
    paper_id: str
    opened: bool
    login_url: HttpUrl | None = None
    target_path: str | None = None
    instructions: list[str] = Field(default_factory=list)
    error: str | None = None


class BrowserLoginSessionPlan(BaseModel):
    paper_id: str
    login_url: HttpUrl | None = None
    target_path: str
    download_dir: str
    browser_profile: str = "littrace-auth"
    automation_steps: list[str] = Field(default_factory=list)
    browser_act_command: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    requires_user_login: bool = True
    error: str | None = None


def login_action_for_paper(
    config: LitTraceConfig,
    paper: PaperMetadata,
    full_text_report: FullTextResolutionReport | None = None,
) -> DownloadExecutionItem:
    pdf_path = target_pdf_path(config, paper)
    login_url = _login_url_for_paper(paper, full_text_report)
    return DownloadExecutionItem(
        paper_id=paper.paper_id,
        action="open_login_popup",
        status="requires_login",
        target_path=str(pdf_path),
        login_url=str(login_url) if login_url else None,
        login_instructions=login_instructions(pdf_path),
        error=None if login_url else "No login or landing URL is available",
    )


def launch_login_for_paper(
    config: LitTraceConfig,
    paper: PaperMetadata,
    full_text_report: FullTextResolutionReport | None = None,
    dry_run: bool = False,
) -> LoginLaunchResult:
    action = login_action_for_paper(config, paper, full_text_report)
    if not action.login_url:
        return LoginLaunchResult(
            paper_id=paper.paper_id,
            opened=False,
            target_path=action.target_path,
            instructions=action.login_instructions,
            error=action.error or "No login URL available",
        )

    opened = False
    if not dry_run:
        opened = webbrowser.open(str(action.login_url), new=1, autoraise=True)

    return LoginLaunchResult(
        paper_id=paper.paper_id,
        opened=opened if not dry_run else False,
        login_url=action.login_url,
        target_path=action.target_path,
        instructions=action.login_instructions,
    )


def browser_login_session_for_paper(
    config: LitTraceConfig,
    paper: PaperMetadata,
    full_text_report: FullTextResolutionReport | None = None,
    browser_profile: str = "littrace-auth",
) -> BrowserLoginSessionPlan:
    action = login_action_for_paper(config, paper, full_text_report)
    target = Path(action.target_path or target_pdf_path(config, paper))
    login_url = action.login_url
    steps = [
        "Open the publisher landing or PDF page in a persistent browser session.",
        "Let the user complete institutional, society, or publisher login.",
        "Wait for the user-authorized PDF response or browser download.",
        f"Save or move the resulting PDF to {target}.",
        "Return control to LitTrace for /check-downloads and parsing.",
    ]
    command = [
        "browser-act",
        "open",
        "--profile",
        browser_profile,
        str(login_url or ""),
        "--download-dir",
        str(target.parent),
    ]
    return BrowserLoginSessionPlan(
        paper_id=paper.paper_id,
        login_url=login_url,
        target_path=str(target),
        download_dir=str(target.parent),
        browser_profile=browser_profile,
        automation_steps=steps,
        browser_act_command=command if login_url else [],
        instructions=login_instructions(target),
        error=action.error,
    )


def login_instructions(target_path: Path) -> list[str]:
    return [
        "Open the authorized publisher, institution, or society login page.",
        "Sign in using an account or institutional route that you are allowed to use.",
        f"Download the PDF manually to: {target_path}",
        "Return to LitTrace and run parsing after the PDF is present.",
    ]


def _login_url_for_paper(
    paper: PaperMetadata,
    full_text_report: FullTextResolutionReport | None,
) -> str | None:
    if full_text_report is not None:
        login_candidates = [
            candidate
            for candidate in full_text_report.candidates
            if candidate.requires_login and not candidate.is_pdf
        ]
        if login_candidates:
            return str(login_candidates[0].url)
        if full_text_report.best_landing_url:
            return str(full_text_report.best_landing_url)
    if paper.pdf_url:
        return str(paper.pdf_url)
    return str(paper.source_urls[0]) if paper.source_urls else None
