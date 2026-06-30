from __future__ import annotations

from pathlib import Path

import httpx

from littrace.access import build_download_plan, paper_storage_dir
from littrace.config import LitTraceConfig
from littrace.models import (
    AccessType,
    DownloadExecutionItem,
    DownloadExecutionRequest,
    DownloadExecutionResult,
    PaperMetadata,
)


async def execute_downloads(
    config: LitTraceConfig,
    papers: list[PaperMetadata],
    request: DownloadExecutionRequest,
) -> DownloadExecutionResult:
    selected_ids = set(request.paper_ids)
    target_papers = [paper for paper in papers if not selected_ids or paper.paper_id in selected_ids]
    plan = build_download_plan(config, target_papers, selected_ids)
    items: list[DownloadExecutionItem] = []

    timeout = httpx.Timeout(config.api.request_timeout_seconds)
    headers = {"User-Agent": config.api.user_agent}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for plan_item in plan.items:
            paper = next(paper for paper in target_papers if paper.paper_id == plan_item.paper_id)
            items.append(await _execute_one(client, config, paper, request.dry_run))

    return DownloadExecutionResult(
        items=items,
        downloaded_count=sum(item.status == "downloaded" for item in items),
        requires_login_count=sum(item.status == "requires_login" for item in items),
        skipped_count=sum(item.status == "skipped" for item in items),
    )


async def _execute_one(
    client: httpx.AsyncClient,
    config: LitTraceConfig,
    paper: PaperMetadata,
    dry_run: bool,
) -> DownloadExecutionItem:
    if paper.access_type == AccessType.REQUIRES_LOGIN:
        target_path = _target_pdf_path(config, paper)
        return DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="open_login_popup",
            status="requires_login",
            target_path=str(target_path),
            login_url=str(paper.pdf_url or (paper.source_urls[0] if paper.source_urls else None)),
            login_instructions=[
                "Open the login URL in an authenticated browser session.",
                "Sign in through the institution, publisher account, or other authorized route.",
                f"Download the PDF manually to: {target_path}",
                "Return to LitTrace and run parsing after the PDF is present.",
            ],
        )
    if paper.access_type != AccessType.OPEN_ACCESS or not paper.pdf_url:
        return DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="skip",
            status="skipped",
            error="No open-access PDF URL is available",
        )

    target_path = _target_pdf_path(config, paper)
    if dry_run:
        return DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="download",
            status="planned",
            target_path=str(target_path),
        )

    try:
        response = await client.get(str(paper.pdf_url))
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not str(paper.pdf_url).lower().endswith(".pdf"):
            return DownloadExecutionItem(
                paper_id=paper.paper_id,
                action="download",
                status="skipped",
                target_path=str(target_path),
                error=f"Response does not look like a PDF: {content_type}",
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(response.content)
        return DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="download",
            status="downloaded",
            target_path=str(target_path),
        )
    except httpx.HTTPError as exc:
        return DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="download",
            status="failed",
            target_path=str(target_path),
            error=f"{exc.__class__.__name__}: {exc}",
        )


def _target_pdf_path(config: LitTraceConfig, paper: PaperMetadata) -> Path:
    return paper_storage_dir(config, paper) / "paper.pdf"
