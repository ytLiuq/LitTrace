from __future__ import annotations

import webbrowser
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field, HttpUrl

from littrace.access import target_pdf_path
from littrace.authorized_pdf_archiver import AuthorizedPdfArchiveResult, archive_authorized_pdf_response
from littrace.browser import (
    browser_act_command,
    browser_open_args,
    browser_session_name_for_paper,
    prewarm_chrome_direct,
    publisher_window_session_name,
    run_browser_act,
)
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
    background_resume_command: list[str] = Field(default_factory=list)
    background_pdf_command: list[str] = Field(default_factory=list)
    session_name: str | None = None
    pdf_url: str | None = None
    instructions: list[str] = Field(default_factory=list)
    requires_user_login: bool = True
    error: str | None = None


class BrowserLoginOpenResult(BaseModel):
    opened: bool
    session_name: str
    browser_id: str | None = None
    fallback_used: bool = False
    fallback_blocked: bool = False
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class BrowserAuthResumeResult(BaseModel):
    paper_id: str
    attempted: bool
    reopened: bool = False
    recoverable_window_closed: bool = False
    session_name: str | None = None
    target_path: str | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class AuthorizedPdfFetchResult(BaseModel):
    paper_id: str
    attempted: bool
    opened_pdf: bool = False
    recoverable_window_closed: bool = False
    session_name: str | None = None
    pdf_url: str | None = None
    target_path: str | None = None
    stdout: str = ""
    stderr: str = ""
    archive_result: AuthorizedPdfArchiveResult | None = None
    error: str | None = None


class BrowserPdfLinkDiscoveryResult(BaseModel):
    session_name: str
    attempted: bool
    pdf_url: str | None = None
    source: str = "browser_dom"
    requires_user_confirmation: bool = False
    recoverable_window_closed: bool = False
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class BrowserAuthorizationWaitResult(BaseModel):
    session_name: str
    authorized: bool
    attempts: int
    elapsed_seconds: float
    pdf_url: str | None = None
    recoverable_window_closed: bool = False
    requires_user_confirmation: bool = False
    last_stdout: str = ""
    last_stderr: str = ""
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
    browser_session_name: str | None = None,
) -> BrowserLoginSessionPlan:
    action = login_action_for_paper(config, paper, full_text_report)
    target = Path(action.target_path or target_pdf_path(config, paper))
    login_url = action.login_url
    steps = [
        "Open the publisher landing or PDF page in a persistent browser session.",
        "Let the user complete institutional, society, or publisher login.",
        "If the authorization window is closed, reopen a background session with the persisted browser state.",
        "Wait for the user-authorized PDF response or browser download.",
        f"Save or move the resulting PDF to {target}.",
        "Return control to LitTrace for /check-downloads and parsing.",
    ]
    browser_id = config.browser.default_browser_id or config.browser.default_browser_name
    session_name = browser_session_name or browser_session_name_for_paper(paper.paper_id, "auth")
    pdf_url = authorized_pdf_url_for_paper(paper, full_text_report)
    command = browser_act_command(
        config,
        browser_open_args(
            config,
            session_name,
            browser_id,
            str(login_url or ""),
            headed=True,
        ),
    )
    resume_command = browser_act_command(
        config,
        browser_open_args(
            config,
            browser_session_name_for_paper(paper.paper_id, "resume"),
            browser_id,
            str(login_url or ""),
        ),
    )
    pdf_command = browser_act_command(
        config,
        browser_open_args(
            config,
            browser_session_name_for_paper(paper.paper_id, "pdf"),
            browser_id,
            pdf_url or str(login_url or ""),
        ),
    )
    return BrowserLoginSessionPlan(
        paper_id=paper.paper_id,
        login_url=login_url,
        target_path=str(target),
        download_dir=str(target.parent),
        browser_profile=browser_profile,
        automation_steps=steps,
        browser_act_command=command if login_url else [],
        background_resume_command=resume_command if login_url else [],
        background_pdf_command=pdf_command if pdf_url else [],
        session_name=session_name,
        pdf_url=pdf_url,
        instructions=login_instructions(target),
        error=action.error,
    )


def resume_browser_auth_after_user_close(
    config: LitTraceConfig,
    paper: PaperMetadata,
    full_text_report: FullTextResolutionReport | None = None,
    timeout_seconds: float = 30.0,
    browser_session_name: str | None = None,
) -> BrowserAuthResumeResult:
    plan = browser_login_session_for_paper(
        config,
        paper,
        full_text_report,
        browser_session_name=browser_session_name,
    )
    if not plan.login_url or not plan.background_resume_command:
        return BrowserAuthResumeResult(
            paper_id=paper.paper_id,
            attempted=False,
            target_path=plan.target_path,
            error=plan.error or "No login URL available for browser resume.",
        )
    args = plan.background_resume_command[1:]
    result = run_browser_act(config, args, timeout_seconds=timeout_seconds)
    return BrowserAuthResumeResult(
        paper_id=paper.paper_id,
        attempted=True,
        reopened=result.returncode == 0,
        recoverable_window_closed=result.recoverable_window_closed,
        session_name=browser_session_name_for_paper(paper.paper_id, "resume"),
        target_path=plan.target_path,
        stdout=result.stdout,
        stderr=result.stderr,
        error=None if result.returncode == 0 or result.recoverable_window_closed else result.stderr or result.stdout,
    )


def open_browser_login_session(
    config: LitTraceConfig,
    paper: PaperMetadata,
    full_text_report: FullTextResolutionReport | None = None,
    timeout_seconds: float = 60.0,
    browser_session_name: str | None = None,
) -> BrowserLoginOpenResult:
    plan = browser_login_session_for_paper(
        config,
        paper,
        full_text_report,
        browser_session_name=browser_session_name,
    )
    if plan.error or not plan.login_url:
        return BrowserLoginOpenResult(
            opened=False,
            session_name=plan.session_name or browser_session_name_for_paper(paper.paper_id, "auth"),
            error=plan.error or "No login URL available.",
        )
    browser_ids = _candidate_browser_ids(config)
    last_stdout = ""
    last_stderr = ""
    last_browser_id = browser_ids[0] if browser_ids else None
    session_name = plan.session_name or browser_session_name_for_paper(paper.paper_id, "auth")
    navigate = _navigate_existing_browser_session(
        config,
        session_name,
        str(plan.login_url),
        timeout_seconds=min(timeout_seconds, 15.0),
    )
    if navigate.returncode == 0:
        return BrowserLoginOpenResult(
            opened=True,
            session_name=session_name,
            browser_id=last_browser_id,
            stdout=navigate.stdout,
            stderr=navigate.stderr,
        )
    if navigate.returncode != 0 and not navigate.recoverable_window_closed:
        last_stdout = navigate.stdout
        last_stderr = navigate.stderr
    prewarm = prewarm_chrome_direct(config)
    if prewarm.attempted and not prewarm.ok:
        last_stderr = "\n".join(
            part for part in [last_stderr, f"Chrome direct prewarm failed: {prewarm.error}"] if part
        )
    for index, browser_id in enumerate(browser_ids):
        last_browser_id = browser_id
        args = browser_open_args(
            config,
            session_name,
            browser_id,
            str(plan.login_url),
            headed=True,
        )
        result = run_browser_act(config, args, timeout_seconds=timeout_seconds)
        last_stdout = result.stdout
        last_stderr = result.stderr
        if result.returncode != 0 and result.recoverable_browser_open_failed:
            for _attempt in range(max(config.browser.chrome_direct_open_retries, 1)):
                time.sleep(max(config.browser.chrome_direct_retry_delay_seconds, 0.0))
                retry = run_browser_act(config, args, timeout_seconds=timeout_seconds)
                last_stdout = retry.stdout
                last_stderr = retry.stderr
                if retry.returncode == 0:
                    return BrowserLoginOpenResult(
                        opened=True,
                        session_name=session_name,
                        browser_id=browser_id,
                        fallback_used=index > 0,
                        stdout=retry.stdout,
                        stderr=retry.stderr,
                    )
                result = retry
                if not retry.recoverable_browser_open_failed:
                    break
        if result.returncode == 0:
            return BrowserLoginOpenResult(
                opened=True,
                session_name=session_name,
                browser_id=browser_id,
                fallback_used=index > 0,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        if result.api_key_required:
            break
        if not result.recoverable_browser_open_failed:
            break
    final_error = last_stderr or last_stdout or "Could not open browser login session."
    fallback_message = _blocked_browser_fallback_message(config, final_error)
    return BrowserLoginOpenResult(
        opened=False,
        session_name=session_name,
        browser_id=last_browser_id,
        fallback_blocked=fallback_message is not None,
        stdout=last_stdout,
        stderr=last_stderr,
        error=fallback_message or final_error,
    )


def fetch_authorized_pdf_after_user_auth(
    config: LitTraceConfig,
    paper: PaperMetadata,
    full_text_report: FullTextResolutionReport | None = None,
    timeout_seconds: float = 60.0,
    auth_wait_result: BrowserAuthorizationWaitResult | None = None,
    browser_session_name: str | None = None,
) -> AuthorizedPdfFetchResult:
    plan = browser_login_session_for_paper(
        config,
        paper,
        full_text_report,
        browser_session_name=browser_session_name,
    )
    auth_wait = auth_wait_result or wait_for_browser_authorization(
        config,
        plan.session_name or browser_session_name_for_paper(paper.paper_id, "auth"),
        timeout_seconds=min(timeout_seconds, 20.0),
    )
    pdf_url = auth_wait.pdf_url or plan.pdf_url
    if not pdf_url:
        return AuthorizedPdfFetchResult(
            paper_id=paper.paper_id,
            attempted=False,
            target_path=plan.target_path,
            error=auth_wait.error or "No publisher PDF URL could be inferred after authorization.",
        )
    auth_session_name = plan.session_name or browser_session_name_for_paper(
        paper.paper_id, "auth"
    )
    if not auth_wait.authorized:
        return AuthorizedPdfFetchResult(
            paper_id=paper.paper_id,
            attempted=False,
            recoverable_window_closed=auth_wait.recoverable_window_closed,
            session_name=auth_session_name,
            pdf_url=pdf_url,
            target_path=plan.target_path,
            stdout=auth_wait.last_stdout,
            stderr=auth_wait.last_stderr,
            error=auth_wait.error
            or "No live authorized browser session is available for PDF retrieval.",
        )
    archive_result = archive_authorized_pdf_response(
        config,
        paper,
        auth_session_name,
        pdf_url,
        timeout_seconds=min(timeout_seconds, 20.0),
    )
    if archive_result.archived:
        return AuthorizedPdfFetchResult(
            paper_id=paper.paper_id,
            attempted=True,
            opened_pdf=False,
            recoverable_window_closed=auth_wait.recoverable_window_closed,
            session_name=auth_session_name,
            pdf_url=pdf_url,
            target_path=plan.target_path,
            stdout=auth_wait.last_stdout,
            stderr=auth_wait.last_stderr,
            archive_result=archive_result,
        )
    if _is_landing_first_publisher(paper):
        return AuthorizedPdfFetchResult(
            paper_id=paper.paper_id,
            attempted=True,
            opened_pdf=False,
            recoverable_window_closed=auth_wait.recoverable_window_closed,
            session_name=auth_session_name,
            pdf_url=pdf_url,
            target_path=plan.target_path,
            stdout=auth_wait.last_stdout,
            stderr=auth_wait.last_stderr,
            archive_result=archive_result,
            error=archive_result.error,
        )
    browser_id = config.browser.default_browser_id or config.browser.default_browser_name
    pdf_session_name = browser_session_name_for_paper(paper.paper_id, "pdf")
    args = browser_open_args(config, pdf_session_name, browser_id, pdf_url)
    result = run_browser_act(config, args, timeout_seconds=timeout_seconds)
    if result.returncode == 0:
        archive_result = archive_authorized_pdf_response(
            config,
            paper,
            pdf_session_name,
            pdf_url,
            timeout_seconds=min(timeout_seconds, 20.0),
        )
    return AuthorizedPdfFetchResult(
        paper_id=paper.paper_id,
        attempted=True,
        opened_pdf=result.returncode == 0,
        recoverable_window_closed=result.recoverable_window_closed or auth_wait.recoverable_window_closed,
        session_name=pdf_session_name,
        pdf_url=pdf_url,
        target_path=plan.target_path,
        stdout=result.stdout,
        stderr=result.stderr,
        archive_result=archive_result,
        error=None
        if (result.returncode == 0 or result.recoverable_window_closed)
        and not (archive_result and archive_result.error and archive_result.warning != "needs_binary_body_export")
        else result.stderr or result.stdout or (archive_result.error if archive_result else None),
    )


def wait_for_browser_authorization(
    config: LitTraceConfig,
    session_name: str,
    timeout_seconds: float = 180.0,
    poll_interval_seconds: float = 2.0,
) -> BrowserAuthorizationWaitResult:
    start = time.monotonic()
    attempts = 0
    last: BrowserPdfLinkDiscoveryResult | None = None
    while True:
        attempts += 1
        last = discover_pdf_url_from_browser_session(
            config,
            session_name,
            timeout_seconds=min(max(poll_interval_seconds, 1.0), 10.0),
        )
        if last.pdf_url:
            return BrowserAuthorizationWaitResult(
                session_name=session_name,
                authorized=True,
                attempts=attempts,
                elapsed_seconds=round(time.monotonic() - start, 3),
                pdf_url=last.pdf_url,
                last_stdout=last.stdout,
                last_stderr=last.stderr,
            )
        elapsed = time.monotonic() - start
        if last.recoverable_window_closed and elapsed >= timeout_seconds:
            return BrowserAuthorizationWaitResult(
                session_name=session_name,
                authorized=False,
                attempts=attempts,
                elapsed_seconds=round(elapsed, 3),
                recoverable_window_closed=True,
                requires_user_confirmation=last.requires_user_confirmation,
                last_stdout=last.stdout,
                last_stderr=last.stderr,
                error="Authorization browser session closed before a PDF link was visible.",
            )
        if elapsed >= timeout_seconds:
            return BrowserAuthorizationWaitResult(
                session_name=session_name,
                authorized=False,
                attempts=attempts,
                elapsed_seconds=round(elapsed, 3),
                last_stdout=last.stdout if last else "",
                last_stderr=last.stderr if last else "",
                requires_user_confirmation=last.requires_user_confirmation if last else False,
                error="Timed out waiting for user authorization to complete.",
            )
        time.sleep(max(poll_interval_seconds, 0.2))


def discover_pdf_url_from_browser_session(
    config: LitTraceConfig,
    session_name: str,
    timeout_seconds: float = 20.0,
) -> BrowserPdfLinkDiscoveryResult:
    script = r"""
(() => {
  const normalize = (href) => {
    try { return new URL(href, location.href).href; } catch (_) { return null; }
  };
  const anchors = Array.from(document.querySelectorAll('a[href]'));
  const currentUrl = location.href || '';
  const currentLooksPdf = /\.pdf(?:$|[?#])|\/pdf(?:$|[/?#])|\/doi\/pdf\//i.test(currentUrl);
  const scored = anchors.map((a) => {
    const text = [
      a.textContent || '',
      a.getAttribute('title') || '',
      a.getAttribute('aria-label') || '',
      a.getAttribute('href') || ''
    ].join(' ').toLowerCase();
    const href = normalize(a.getAttribute('href'));
    let score = 0;
    if (href && href.includes('/doi/pdf/')) score += 10;
    if (href && href.includes('/content/articlepdf/')) score += 10;
    if (href && href.includes('/pdfft')) score += 10;
    if (href && href.includes('/pdf?')) score += 8;
    if (href && /\/pdf(?:$|[?#/])/.test(href)) score += 6;
    if (text.includes('open pdf') || text === 'pdf' || text.includes(' pdf')) score += 5;
    if (text.includes('download pdf')) score += 4;
    if (text.includes('view pdf')) score += 3;
    return { href, score };
  }).filter((item) => item.href && item.score > 0);
  scored.sort((a, b) => b.score - a.score);
  return JSON.stringify({
    pdfUrl: currentLooksPdf ? currentUrl : (scored.length ? scored[0].href : null),
    title: document.title || '',
    url: currentUrl,
    text: document.body ? document.body.innerText.slice(0, 3000) : ''
  });
})()
""".strip()
    result = run_browser_act(
        config,
        ["--session", session_name, "eval", script],
        timeout_seconds=timeout_seconds,
    )
    payload_text = _extract_browser_eval_payload(result.stdout)
    pdf_url = _extract_pdf_url_from_payload(payload_text)
    if not _is_json_payload(payload_text):
        pdf_url = pdf_url or _extract_url_from_browser_eval(result.stdout)
    requires_confirmation = _looks_like_user_confirmation_required(
        f"{payload_text}\n{result.stdout}\n{result.stderr}"
    )
    return BrowserPdfLinkDiscoveryResult(
        session_name=session_name,
        attempted=True,
        pdf_url=pdf_url,
        requires_user_confirmation=requires_confirmation,
        recoverable_window_closed=result.recoverable_window_closed,
        stdout=result.stdout,
        stderr=result.stderr,
        error=None
        if pdf_url or requires_confirmation or result.recoverable_window_closed
        else result.stderr or result.stdout,
    )


def detect_user_confirmation_required(
    config: LitTraceConfig,
    session_name: str,
    timeout_seconds: float = 10.0,
) -> BrowserPdfLinkDiscoveryResult:
    script = r"""
(() => [
  document.title || '',
  location.href || '',
  document.body ? document.body.innerText.slice(0, 3000) : ''
].join('\n'))()
""".strip()
    result = run_browser_act(
        config,
        ["--session", session_name, "eval", script],
        timeout_seconds=timeout_seconds,
    )
    return BrowserPdfLinkDiscoveryResult(
        session_name=session_name,
        attempted=True,
        requires_user_confirmation=_looks_like_user_confirmation_required(
            f"{result.stdout}\n{result.stderr}"
        ),
        recoverable_window_closed=result.recoverable_window_closed,
        stdout=result.stdout,
        stderr=result.stderr,
        error=None if result.returncode == 0 or result.recoverable_window_closed else result.stderr or result.stdout,
    )


def browser_login_session_plans_for_workspace(
    config: LitTraceConfig,
    workspace,
    browser_profile: str = "littrace-auth",
    browser_session_name: str | None = None,
) -> list[BrowserLoginSessionPlan]:
    plans: list[BrowserLoginSessionPlan] = []
    for paper_id in workspace.context.active_papers:
        paper = workspace.papers[paper_id]
        if paper.access_type.value not in {"requires_login", "metadata_only"}:
            continue
        plans.append(
            browser_login_session_for_paper(
                config,
                paper,
                workspace.full_text_reports.get(paper_id),
                browser_profile=browser_profile,
                browser_session_name=browser_session_name,
            )
        )
    return plans


def publisher_window_session_name_for_chat(session_id: str | None = None) -> str:
    return publisher_window_session_name(session_id)


def login_instructions(target_path: Path) -> list[str]:
    return [
        "Open the authorized publisher, institution, or society login page.",
        "Sign in using an account or institutional route that you are allowed to use.",
        "Keep the authorization browser window open until LitTrace reports that the PDF has been archived.",
        f"LitTrace will archive the authorized PDF to: {target_path}",
        "Return to LitTrace; parsing and evidence extraction continue automatically once the PDF is present.",
    ]


def _navigate_existing_browser_session(
    config: LitTraceConfig,
    session_name: str,
    url: str,
    timeout_seconds: float,
):
    return run_browser_act(
        config,
        ["--session", session_name, "navigate", url],
        timeout_seconds=timeout_seconds,
    )


def _candidate_browser_ids(config: LitTraceConfig) -> list[str]:
    candidates = [config.browser.default_browser_id]
    if config.browser.allow_confirm_browser_fallback:
        if config.browser.default_browser_id != "chrome_local_104956678805389514":
            candidates.append("chrome_local_104956678805389514")
        if config.browser.default_browser_id != "direct_local_105121787802550357":
            candidates.append("direct_local_105121787802550357")
        if not config.browser.default_browser_id:
            candidates.append(config.browser.default_browser_name)
    elif not config.browser.default_browser_id:
        candidates.append(config.browser.default_browser_name)
    result: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in result:
            result.append(candidate)
    return result


def _blocked_browser_fallback_message(config: LitTraceConfig, last_error: str) -> str | None:
    if config.browser.allow_confirm_browser_fallback:
        return None
    if not config.browser.default_browser_id:
        return None
    if not config.browser.confirm_before_use:
        return None
    detail = last_error.strip() or "Configured browser could not be opened."
    return (
        f"{detail}\n"
        "LitTrace did not fall back to BrowserAct's confirm/sign-in browser because "
        "browser.allow_confirm_browser_fallback is false. The BrowserAct API key only "
        "authenticates the CLI/API; it does not bypass BrowserAct web confirmation pages. "
        "Fix the configured local chrome-direct browser, or explicitly enable fallback "
        "only when you want to complete BrowserAct confirmation in a browser window."
    )


def authorized_pdf_url_for_paper(
    paper: PaperMetadata,
    full_text_report: FullTextResolutionReport | None = None,
) -> str | None:
    if full_text_report is not None:
        pdf_candidates = [
            candidate
            for candidate in full_text_report.candidates
            if candidate.is_pdf or "pdf" in candidate.content_type.lower()
        ]
        if pdf_candidates:
            return str(pdf_candidates[0].url)
        if full_text_report.best_pdf_url:
            return str(full_text_report.best_pdf_url)
    if paper.pdf_url:
        return str(paper.pdf_url)
    if paper.doi and _is_acs_paper(paper):
        return f"https://pubs.acs.org/doi/pdf/{paper.doi}"
    for url in paper.source_urls:
        url_text = str(url)
        if url_text.lower().endswith(".pdf") or "/pdf" in url_text.lower():
            return url_text
    return None


def _extract_url_from_browser_eval(output: str) -> str | None:
    for raw in reversed(output.splitlines()):
        line = raw.strip().strip('"').strip("'")
        if not line:
            continue
        parsed = urlparse(line)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return line
        marker = "http"
        if marker in line:
            candidate = line[line.find(marker) :].strip().strip('"').strip("'")
            parsed = urlparse(candidate)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return candidate
    return None


def _extract_browser_eval_payload(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _extract_pdf_url_from_payload(payload_text: str) -> str | None:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    pdf_url = payload.get("pdfUrl")
    if not isinstance(pdf_url, str):
        current_url = payload.get("url")
        pdf_url = current_url if isinstance(current_url, str) and _looks_like_pdf_url(current_url) else None
    if not pdf_url:
        return None
    parsed = urlparse(pdf_url)
    return pdf_url if parsed.scheme in {"http", "https"} and parsed.netloc else None


def _is_json_payload(payload_text: str) -> bool:
    try:
        json.loads(payload_text)
    except json.JSONDecodeError:
        return False
    return True


def _looks_like_user_confirmation_required(output: str) -> bool:
    lowered = output.lower()
    markers = [
        "cloudflare",
        "__cf_chl",
        "cf_chl",
        "just a moment",
        "verify you are human",
        "请验证您是真人",
        "正在进行安全验证",
        "captcha",
        "human verification",
    ]
    return any(marker in lowered for marker in markers)


def _looks_like_pdf_url(url: str) -> bool:
    lowered = url.lower()
    return (
        lowered.endswith(".pdf")
        or ".pdf?" in lowered
        or ".pdf#" in lowered
        or "/doi/pdf/" in lowered
        or "/content/articlepdf/" in lowered
        or "/pdfft" in lowered
        or lowered.endswith("/pdf")
        or "/pdf?" in lowered
        or "/pdf#" in lowered
    )


def _is_acs_paper(paper: PaperMetadata) -> bool:
    haystack = " ".join(
        value or ""
        for value in [
            paper.publisher,
            paper.journal,
            str(paper.source_urls[0]) if paper.source_urls else "",
        ]
    ).lower()
    return "american chemical society" in haystack or "acs" in haystack or "pubs.acs.org" in haystack


def _is_landing_first_publisher(paper: PaperMetadata) -> bool:
    haystack = " ".join(
        value or ""
        for value in [
            paper.publisher,
            paper.journal,
            str(paper.source_urls[0]) if paper.source_urls else "",
            str(paper.pdf_url) if paper.pdf_url else "",
        ]
    ).lower()
    markers = [
        "mdpi",
        "elsevier",
        "sciencedirect",
        "science direct",
        "royal society of chemistry",
        "rsc",
        "pubs.rsc.org",
    ]
    return any(marker in haystack for marker in markers)


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
    if _is_landing_first_publisher(paper) and paper.source_urls:
        return str(paper.source_urls[0])
    if paper.pdf_url:
        return str(paper.pdf_url)
    return str(paper.source_urls[0]) if paper.source_urls else None
