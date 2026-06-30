from __future__ import annotations

from pathlib import Path

from littrace.config import DownloadMode, LitTraceConfig
from littrace.models import AccessType, DownloadPlan, DownloadPlanItem, PaperMetadata


class DownloadDecision(str):
    """Human-readable download decision for UI/API handoff."""


def paper_storage_dir(config: LitTraceConfig, paper: PaperMetadata) -> Path:
    year = str(paper.year or "unknown-year")
    doi_slug = (paper.doi or paper.paper_id).replace("/", "_").replace(":", "_")
    return config.storage.paper_library_dir / year / doi_slug


def plan_download(
    config: LitTraceConfig,
    paper: PaperMetadata,
    selected_paper_ids: set[str] | None = None,
) -> DownloadDecision:
    mode = config.paper_download.mode
    selected_paper_ids = selected_paper_ids or set()

    if mode == DownloadMode.METADATA_ONLY:
        return DownloadDecision("save_metadata_only")
    if mode == DownloadMode.ASK_EACH_TIME:
        return DownloadDecision("ask_user")
    if mode == DownloadMode.DOWNLOAD_SELECTED:
        return DownloadDecision(
            "download" if paper.paper_id in selected_paper_ids else "skip_unselected"
        )
    if mode == DownloadMode.DOWNLOAD_OPEN_ACCESS:
        return DownloadDecision(
            "download" if paper.access_type == AccessType.OPEN_ACCESS else "skip_not_open_access"
        )
    if mode == DownloadMode.DOWNLOAD_ALL_ALLOWED:
        if paper.access_type == AccessType.OPEN_ACCESS:
            return DownloadDecision("download")
        if (
            paper.access_type == AccessType.REQUIRES_LOGIN
            and config.paper_download.allow_requires_login_download
        ):
            return DownloadDecision("open_login_popup")
    return DownloadDecision("skip_unavailable")


def build_download_plan(
    config: LitTraceConfig,
    papers: list[PaperMetadata],
    selected_paper_ids: set[str] | None = None,
) -> DownloadPlan:
    items: list[DownloadPlanItem] = []
    requires_login_count = 0
    downloadable_count = 0

    for paper in papers:
        decision = str(plan_download(config, paper, selected_paper_ids))
        can_download = paper.access_type in {
            AccessType.OPEN_ACCESS,
            AccessType.REQUIRES_LOGIN,
        }
        requires_login = paper.access_type == AccessType.REQUIRES_LOGIN
        if decision == "open_login_popup" or (
            decision == "ask_user" and requires_login
        ):
            requires_login_count += 1
        if decision in {"download", "open_login_popup"} or (
            decision == "ask_user" and can_download
        ):
            downloadable_count += 1

        items.append(
            DownloadPlanItem(
                paper_id=paper.paper_id,
                title=paper.title,
                access_type=paper.access_type,
                decision=decision,
                can_download=can_download,
                requires_login=requires_login,
                target_dir=str(paper_storage_dir(config, paper)),
            )
        )

    return DownloadPlan(
        items=items,
        target_root=str(config.storage.paper_library_dir),
        requires_login_count=requires_login_count,
        downloadable_count=downloadable_count,
    )
