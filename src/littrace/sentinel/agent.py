from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from littrace.config import LitTraceConfig
from littrace.models import AccessType, DownloadExecutionRequest, LiteratureWorkspace, PaperSearchRequest
from littrace.sentinel.digest import build_digest_markdown, save_digest
from littrace.sentinel.resource_pack import ResourcePack, build_resource_pack, save_resource_pack
from littrace.sentinel.state import AccessTask, SentinelRunSummary, SentinelState, Watchlist
from littrace.sentinel.storage import (
    SentinelStore,
    ensure_sentinel_store,
    load_sentinel_state,
    load_sentinel_workspace,
    save_evidence_base,
    save_sentinel_state,
    save_sentinel_workspace,
    save_watchlist,
    timestamp_run_id,
    touch_run_dir,
)
from littrace.skill_runner import (
    build_quality_report_skill,
    execute_downloads_skill,
    extract_tables_skill,
    parse_workspace_skill,
    resolve_workspace_full_text_skill,
    search_papers_skill,
)
from littrace.context import add_ranked_candidate_papers


@dataclass
class SentinelRunResult:
    store: SentinelStore
    state: SentinelState
    workspace: LiteratureWorkspace
    resource_pack: ResourcePack
    summary: SentinelRunSummary
    digest_path: str | None = None
    run_dir: str | None = None


class LiteratureSentinelAgent:
    def __init__(self, config: LitTraceConfig, watchlist: Watchlist):
        self.config = config
        self.watchlist = watchlist
        self.store = ensure_sentinel_store(config, watchlist)
        save_watchlist(self.store, watchlist)

    def load_state(self) -> SentinelState:
        return load_sentinel_state(self.store)

    def load_workspace(self) -> LiteratureWorkspace:
        return load_sentinel_workspace(self.store)

    async def run(self) -> SentinelRunResult:
        run_id = timestamp_run_id()
        started_at = datetime.now().isoformat(timespec="seconds")
        state = self.load_state()
        workspace = self.load_workspace()
        state.watchlist = self.watchlist
        run_dir = touch_run_dir(self.store, run_id)

        request = PaperSearchRequest(
            topic=self.watchlist.topic,
            discipline="materials chemistry",
            year_min=self.watchlist.year_min,
            live=self.config.api.enable_live_search,
            query_variants=list(self.watchlist.query_variants),
        )

        search = await search_papers_skill(request, self.config)
        candidate_papers = search.result.papers
        if candidate_papers:
            workspace = add_ranked_candidate_papers(
                workspace,
                candidate_papers,
                request,
                active_limit=self.config.literature_context.active_context_limit,
            )

        seen = set(state.seen_paper_ids)
        new_candidates = [paper for paper in candidate_papers if paper.paper_id not in seen]
        for paper in new_candidates:
            state.seen_paper_ids.append(paper.paper_id)

        workspace = await resolve_workspace_full_text_skill(workspace, self.config)

        access_tasks = _build_access_tasks(workspace, state)
        state.access_queue = _merge_access_tasks(state.access_queue, access_tasks)

        downloadable_ids = [
            paper.paper_id
            for paper in workspace.papers.values()
            if paper.access_type == AccessType.OPEN_ACCESS and paper.paper_id in workspace.context.active_papers
        ]
        downloaded_count = 0
        if downloadable_ids:
            download_result = await execute_downloads_skill(
                self.config,
                workspace,
                DownloadExecutionRequest(paper_ids=downloadable_ids, dry_run=False),
            )
            downloaded_count = download_result.downloaded_count

        workspace, parse_report = await parse_workspace_skill(workspace, self.config)
        workspace, table_harness = await extract_tables_skill(workspace, self.config)
        quality_report = build_quality_report_skill(self.config, workspace)

        quality_warnings = [
            *quality_report.warnings,
            *table_harness.warnings,
            *parse_report.get("warnings", []),
        ]
        resource_pack = build_resource_pack(
            workspace,
            state,
            quality_warnings=quality_warnings,
        )
        resource_pack_file = save_resource_pack(self.store.root, resource_pack, run_id=run_id)
        save_evidence_base(self.store, run_id, workspace, resource_pack, quality_report)
        digest_markdown = build_digest_markdown(resource_pack, state)
        digest_path, digest_record = save_digest(self.store.root, run_id, digest_markdown, resource_pack, state)
        finished_at = datetime.now().isoformat(timespec="seconds")
        state.last_run_at = finished_at
        state.warnings = quality_warnings
        save_sentinel_workspace(self.store, workspace)
        save_sentinel_state(self.store, state)

        summary = SentinelRunSummary(
            run_id=run_id,
            watchlist_id=self.watchlist.watchlist_id,
            topic=self.watchlist.topic,
            started_at=started_at,
            finished_at=finished_at,
            new_candidates_count=len(new_candidates),
            downloaded_count=downloaded_count,
            parsed_count=int(parse_report.get("parsed_count", 0)),
            access_task_count=len(state.access_queue),
            digest_path=str(digest_path),
            resource_pack_path=str(resource_pack_file),
            quality_score=quality_report.metrics.get("overall_score"),
            warnings=quality_warnings,
        )
        (run_dir / "run_summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        (run_dir / "digest.md").write_text(digest_markdown, encoding="utf-8")
        return SentinelRunResult(
            store=self.store,
            state=state,
            workspace=workspace,
            resource_pack=resource_pack,
            summary=summary,
            digest_path=str(digest_path),
            run_dir=str(run_dir),
        )

    def access_review(self) -> list[AccessTask]:
        return list(self.load_state().access_queue)

    async def resume_after_login(self) -> SentinelRunResult:
        run_id = timestamp_run_id()
        started_at = datetime.now().isoformat(timespec="seconds")
        state = self.load_state()
        workspace = self.load_workspace()
        queued_tasks = [task for task in state.access_queue if task.retry_after_login]
        login_papers = [task.paper_id for task in queued_tasks if task.paper_id in workspace.papers]
        downloaded_count = 0
        if login_papers:
            download_result = await execute_downloads_skill(
                self.config,
                workspace,
                DownloadExecutionRequest(paper_ids=login_papers, dry_run=False),
            )
            downloaded_count = download_result.downloaded_count
            workspace, parse_report = await parse_workspace_skill(workspace, self.config)
        else:
            parse_report = {"parsed_count": 0, "warnings": []}
        completed_ids = {
            paper_id
            for paper_id in login_papers
            if paper_id in workspace.parsed_papers and workspace.parsed_papers[paper_id].parsed
        }
        if completed_ids:
            state.access_queue = [task for task in state.access_queue if task.paper_id not in completed_ids]
        state.warnings = [*state.warnings, *parse_report.get("warnings", [])]
        finished_at = datetime.now().isoformat(timespec="seconds")
        state.last_run_at = finished_at
        save_sentinel_workspace(self.store, workspace)
        save_sentinel_state(self.store, state)
        resource_pack = build_resource_pack(workspace, state, quality_warnings=list(state.warnings))
        resource_pack_file = save_resource_pack(self.store.root, resource_pack, run_id=run_id)
        save_evidence_base(self.store, run_id, workspace, resource_pack)
        summary = SentinelRunSummary(
            run_id=run_id,
            watchlist_id=self.watchlist.watchlist_id,
            topic=self.watchlist.topic,
            started_at=started_at,
            finished_at=finished_at,
            new_candidates_count=0,
            downloaded_count=downloaded_count,
            parsed_count=int(parse_report.get("parsed_count", 0)),
            access_task_count=len(state.access_queue),
            resource_pack_path=str(resource_pack_file),
            quality_score=None,
            warnings=list(state.warnings),
        )
        return SentinelRunResult(
            store=self.store,
            state=state,
            workspace=workspace,
            resource_pack=resource_pack,
            summary=summary,
        )


def _build_access_tasks(workspace: LiteratureWorkspace, state: SentinelState) -> list[AccessTask]:
    tasks: list[AccessTask] = []
    for paper_id in workspace.context.active_papers:
        paper = workspace.papers[paper_id]
        report = workspace.full_text_reports.get(paper_id)
        if paper.access_type == AccessType.REQUIRES_LOGIN or (
            report and report.login_required_candidate_count > 0 and not report.best_pdf_url
        ):
            tasks.append(
                AccessTask(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    doi=paper.doi,
                    publisher=paper.publisher,
                    landing_url=str(report.best_landing_url) if report and report.best_landing_url else None,
                    reason="requires_institution_login",
                )
            )
    return tasks


def _merge_access_tasks(existing: list[AccessTask], new_tasks: list[AccessTask]) -> list[AccessTask]:
    by_paper_id = {task.paper_id: task for task in existing}
    for task in new_tasks:
        by_paper_id[task.paper_id] = task
    return list(by_paper_id.values())
