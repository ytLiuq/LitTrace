from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.access_layer.browser_sessions import browser_login_session_plans_for_workspace
from littrace.access_layer.download_planning import check_download_presence, target_pdf_path
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.session import ChatSession
from littrace.skill_runner import (
    build_comparison_matrix_skill,
    export_session_bundle_skill,
    extract_tables_skill,
    parse_workspace_skill,
)


class AutoResumeResult(BaseModel):
    ready_to_parse_count: int
    auto_archived_count: int = 0
    parsed_count: int
    performance_cell_count: int
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DownloadWatchResult(BaseModel):
    completed: bool
    attempts: int
    elapsed_seconds: float
    resume_result: AutoResumeResult


class BrowserSessionDownloadTestResult(BaseModel):
    planned_count: int
    target_paths: list[str] = Field(default_factory=list)
    download_dirs: list[str] = Field(default_factory=list)
    watch_result: DownloadWatchResult
    warnings: list[str] = Field(default_factory=list)


async def auto_resume_downloaded_pdfs_async(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    session: ChatSession | None = None,
) -> tuple[LiteratureWorkspace, AutoResumeResult]:
    archived_count, archive_warnings = auto_archive_login_downloads(config, workspace)
    presence = check_download_presence(config, workspace)
    warnings = [*archive_warnings, *presence.warnings]
    if presence.ready_to_parse_count:
        workspace, parse_report = await parse_workspace_skill(workspace, config)
        workspace, table_harness = await extract_tables_skill(workspace, config)
        matrix = build_comparison_matrix_skill(workspace)
        warnings.extend(parse_report.get("warnings", []))
        warnings.extend(getattr(table_harness, "warnings", []))
        warnings.extend(getattr(matrix, "warnings", []))
    else:
        parse_report = {"parsed_count": 0}

    artifacts = await export_session_bundle_skill(session, workspace, config) if session else {}
    result = AutoResumeResult(
        ready_to_parse_count=presence.ready_to_parse_count,
        auto_archived_count=archived_count,
        parsed_count=int(parse_report.get("parsed_count") or 0),
        performance_cell_count=len(workspace.performance_cells),
        artifact_paths=artifacts,
        warnings=warnings,
    )
    return workspace, result


def auto_resume_downloaded_pdfs(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    session: ChatSession | None = None,
) -> tuple[LiteratureWorkspace, AutoResumeResult]:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(auto_resume_downloaded_pdfs_async(config, workspace, session))
    raise RuntimeError(
        "auto_resume_downloaded_pdfs() is synchronous-only; use auto_resume_downloaded_pdfs_async() "
        "inside async contexts."
    )


def watch_and_resume_downloads(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    session: ChatSession | None = None,
    timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 2.0,
) -> tuple[LiteratureWorkspace, DownloadWatchResult]:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            watch_and_resume_downloads_async(
                config,
                workspace,
                session=session,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
        )
    raise RuntimeError(
        "watch_and_resume_downloads() is synchronous-only; use watch_and_resume_downloads_async() "
        "inside async contexts."
    )


async def watch_and_resume_downloads_async(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    session: ChatSession | None = None,
    timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 2.0,
) -> tuple[LiteratureWorkspace, DownloadWatchResult]:
    import asyncio

    start = time.monotonic()
    attempts = 0
    last_result: AutoResumeResult | None = None
    while True:
        attempts += 1
        workspace, last_result = await auto_resume_downloaded_pdfs_async(config, workspace, session)
        if (
            last_result.ready_to_parse_count
            or last_result.auto_archived_count
            or last_result.parsed_count
        ):
            return workspace, DownloadWatchResult(
                completed=True,
                attempts=attempts,
                elapsed_seconds=round(time.monotonic() - start, 3),
                resume_result=last_result,
            )
        elapsed = time.monotonic() - start
        if elapsed >= timeout_seconds:
            return workspace, DownloadWatchResult(
                completed=False,
                attempts=attempts,
                elapsed_seconds=round(elapsed, 3),
                resume_result=last_result,
            )
        await asyncio.sleep(max(poll_interval_seconds, 0.1))


def run_browser_session_download_handoff_test(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    session: ChatSession | None = None,
    timeout_seconds: float = 5.0,
    poll_interval_seconds: float = 1.0,
) -> tuple[LiteratureWorkspace, BrowserSessionDownloadTestResult]:
    plans = browser_login_session_plans_for_workspace(config, workspace)
    workspace, watch = watch_and_resume_downloads(
        config,
        workspace,
        session=session,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    warnings: list[str] = []
    if not plans:
        warnings.append("No login-capable papers in current active context.")
    if plans and not watch.completed:
        warnings.append(
            "No authorized PDF appeared before timeout. Keep the browser session open, finish login, and retry."
        )
    return workspace, BrowserSessionDownloadTestResult(
        planned_count=len(plans),
        target_paths=[plan.target_path for plan in plans],
        download_dirs=sorted({plan.download_dir for plan in plans}),
        watch_result=watch,
        warnings=warnings,
    )


def auto_archive_login_downloads(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
) -> tuple[int, list[str]]:
    archived = 0
    warnings: list[str] = []
    for paper_id in workspace.context.active_papers:
        paper = workspace.papers[paper_id]
        target = target_pdf_path(config, paper)
        if target.exists():
            continue
        candidates = _download_candidates(target.parent)
        if not candidates:
            continue
        source = candidates[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        archived += 1
        warnings.append(
            f"Auto-archived browser download for {paper_id}: {source.name} -> paper.pdf"
        )
    return archived, warnings


def _download_candidates(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    candidates = [
        path
        for path in folder.glob("*.pdf")
        if path.name != "paper.pdf" and path.is_file() and path.stat().st_size > 0
    ]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
