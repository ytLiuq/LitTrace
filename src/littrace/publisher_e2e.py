from __future__ import annotations

from pathlib import Path
import time

import httpx
from pydantic import BaseModel, Field

from littrace.access_layer.cdp import CDPDownloadResult, download_paper_via_cdp
from littrace.config import LitTraceConfig
from littrace.context import add_papers
from littrace.retrieval.full_text import fetch_crossref_paper_by_doi, resolve_full_text_for_paper
from littrace.evaluation.golden_eval import _load_cases
from littrace.models import AccessType, LiteratureWorkspace
from littrace.evidence.parsing import local_pdf_path
from littrace.skill_runner import parse_workspace_skill


class PublisherE2EPaperResult(BaseModel):
    case_id: str
    doi: str
    paper_id: str | None = None
    publisher: str | None = None
    access_type: AccessType | None = None
    resolved_pdf_url: str | None = None
    downloaded_pdf: bool = False
    parsed_full_text: bool = False
    target_path: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PublisherE2EReport(BaseModel):
    case_count: int = 0
    paper_count: int = 0
    downloaded_count: int = 0
    parsed_count: int = 0
    passed: bool = False
    results: list[PublisherE2EPaperResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class InteractivePublisherE2EReport(BaseModel):
    doi: str
    session_name: str
    target_path: str | None = None
    opened: bool = False
    institutional_login_opened: bool = False
    downloaded_pdf: bool = False
    parsed_full_text: bool = False
    completed: bool = False
    attempts: int = 0
    elapsed_seconds: float = 0.0
    last_access_state: str | None = None
    last_error: str | None = None
    needs_user_action: bool = False
    user_action_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    method: str | None = None
    source_url: str | None = None
    cdp_steps: list[str] = Field(default_factory=list)


async def run_publisher_golden_e2e(
    config: LitTraceConfig,
    cases_dir: Path | None = None,
    timeout_seconds: float = 180.0,
) -> PublisherE2EReport:
    cases = _load_cases(cases_dir or config.eval_golden_set_dir)
    results: list[PublisherE2EPaperResult] = []
    warnings: list[str] = []
    if not cases:
        warnings.append(f"No golden cases found under {cases_dir or config.eval_golden_set_dir}.")

    timeout = httpx.Timeout(config.api.request_timeout_seconds)
    headers = {"User-Agent": config.api.user_agent}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for case in cases:
            case_id = str(case.get("case_id") or case.get("topic") or "case")
            dois = _case_dois(case)
            for doi in dois:
                result = await _run_one_publisher_e2e(
                    client,
                    config,
                    case_id,
                    doi,
                    timeout_seconds=timeout_seconds,
                )
                results.append(result)

    downloaded_count = sum(result.downloaded_pdf for result in results)
    parsed_count = sum(result.parsed_full_text for result in results)
    return PublisherE2EReport(
        case_count=len(cases),
        paper_count=len(results),
        downloaded_count=downloaded_count,
        parsed_count=parsed_count,
        passed=bool(results) and downloaded_count == len(results) and parsed_count == len(results),
        results=results,
        warnings=warnings,
    )


async def run_interactive_publisher_e2e(
    config: LitTraceConfig,
    doi: str,
    session_name: str | None = None,
    timeout_seconds: float = 900.0,
    poll_interval_seconds: float = 5.0,
    wait_for_user_action: bool = True,
    user_action_timeout_seconds: float | None = None,
    max_browser_reopens: int = 2,
    close_session: bool = False,
) -> InteractivePublisherE2EReport:
    timeout = httpx.Timeout(config.api.request_timeout_seconds)
    headers = {"User-Agent": config.api.user_agent}
    start = _monotonic()
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        paper = await fetch_crossref_paper_by_doi(client, doi)
        if paper is None:
            return InteractivePublisherE2EReport(
                doi=doi,
                session_name=session_name or "",
                completed=False,
                last_error="Crossref DOI lookup failed.",
            )
        report = await resolve_full_text_for_paper(client, paper, config)
        paper.pdf_url = report.best_pdf_url or paper.pdf_url
        target_path = local_pdf_path(config, paper)
        download = download_paper_via_cdp(
            config,
            paper.doi or doi,
            target_path,
        )
        if download.downloaded:
            workspace = add_papers(LiteratureWorkspace(), [paper])
            workspace.full_text_reports[paper.paper_id] = report
            workspace, parse_report = await parse_workspace_skill(workspace, config)
            return _interactive_report(
                doi=doi,
                download=download,
                parsed=bool(parse_report.get("parsed_count")),
                start=start,
                warnings=report.warnings,
            )
        return InteractivePublisherE2EReport(
            doi=doi,
            session_name="cdp",
            target_path=str(target_path),
            opened=True,
            downloaded_pdf=target_path.exists(),
            completed=False,
            elapsed_seconds=round(_monotonic() - start, 3),
            last_access_state="user_action_required" if download.requires_user_action else "failed",
            last_error=download.error,
            needs_user_action=download.requires_user_action,
            user_action_message=download.user_action,
            warnings=[*report.warnings, *download.warnings],
            method=download.method,
            source_url=download.source_url,
            cdp_steps=download.steps,
        )


async def _run_one_publisher_e2e(
    client: httpx.AsyncClient,
    config: LitTraceConfig,
    case_id: str,
    doi: str,
    timeout_seconds: float,
) -> PublisherE2EPaperResult:
    paper = await fetch_crossref_paper_by_doi(client, doi)
    if paper is None:
        return PublisherE2EPaperResult(
            case_id=case_id, doi=doi, error="Crossref DOI lookup failed."
        )
    report = await resolve_full_text_for_paper(client, paper, config)
    paper.pdf_url = report.best_pdf_url or paper.pdf_url

    workspace = add_papers(LiteratureWorkspace(), [paper])
    workspace.full_text_reports[paper.paper_id] = report
    target_path = local_pdf_path(config, paper)
    download = download_paper_via_cdp(config, paper.doi or doi, target_path)
    if not download.downloaded:
        return PublisherE2EPaperResult(
            case_id=case_id,
            doi=doi,
            paper_id=paper.paper_id,
            publisher=paper.publisher,
            access_type=paper.access_type,
            resolved_pdf_url=download.source_url or str(report.best_pdf_url or ""),
            target_path=str(target_path),
            error=download.error or "CDP publisher download failed.",
            warnings=[*report.warnings, *download.warnings],
        )

    workspace, parse_report = await parse_workspace_skill(workspace, config)
    parsed = bool(parse_report.get("parsed_count"))
    return PublisherE2EPaperResult(
        case_id=case_id,
        doi=doi,
        paper_id=paper.paper_id,
        publisher=paper.publisher,
        access_type=paper.access_type,
        resolved_pdf_url=str(paper.pdf_url) if paper.pdf_url else None,
        downloaded_pdf=target_path.exists(),
        parsed_full_text=parsed,
        target_path=str(target_path),
        error=None if parsed else "PDF downloaded but full-text parsing failed.",
        warnings=report.warnings,
    )


def _case_dois(case: dict[str, object]) -> list[str]:
    raw = case.get("expected_dois") or []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


def _interactive_report(
    doi: str,
    download: CDPDownloadResult,
    parsed: bool,
    start: float,
    warnings: list[str],
) -> InteractivePublisherE2EReport:
    return InteractivePublisherE2EReport(
        doi=doi,
        session_name="cdp",
        target_path=download.target_path,
        opened=True,
        downloaded_pdf=download.downloaded,
        parsed_full_text=parsed,
        completed=download.downloaded,
        attempts=1,
        elapsed_seconds=round(_monotonic() - start, 3),
        last_access_state="downloaded" if download.downloaded else "failed",
        last_error=None if download.downloaded else download.error,
        needs_user_action=download.requires_user_action,
        user_action_message=download.user_action,
        warnings=[*warnings, *download.warnings],
        method=download.method,
        source_url=download.source_url,
        cdp_steps=download.steps,
    )


def _user_action_message(access_state: str | None) -> str | None:
    if access_state == "confirmation_required":
        return "请在已打开的浏览器授权窗口中完成人机验证或 Cloudflare 确认。"
    if access_state == "login_required":
        return "请在已打开的浏览器授权窗口中完成出版商或机构登录。"
    if access_state == "recoverable_window_closed":
        return "授权浏览器窗口已失效，LitTrace 会重新打开本地出版商窗口。"
    return None


def _monotonic() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(max(seconds, 0.0))
