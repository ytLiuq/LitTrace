"""User-facing topic search pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from littrace.config import DownloadMode, LitTraceConfig
from littrace.artifact_registry import artifact_registry_from_config
from littrace.artifact_store import BlobRef, artifact_store_from_config
from littrace.context import add_ranked_candidate_papers
from littrace.downloads import execute_downloads
from littrace.models import DownloadExecutionRequest, LiteratureWorkspace, PaperSearchRequest
from littrace.parse_jobs import enqueue_parse_job, run_pending_parse_jobs
from littrace.rag_jobs import run_pending_embedding_jobs
from littrace.session import ChatSession, load_workspace, save_workspace
from littrace.skills.search_papers import run as search_papers_skill


@dataclass
class TopicSearchRunResult:
    workspace: LiteratureWorkspace
    requested_rag_ready: int
    candidate_count: int = 0
    downloaded_count: int = 0
    parsed_count: int = 0
    embedded_count: int = 0
    requires_login_count: int = 0
    failed_download_count: int = 0
    status: str = "running"
    warnings: list[str] = field(default_factory=list)

    @property
    def rag_ready_count(self) -> int:
        return int(getattr(self.workspace.context.filters, "rag_ready_count", 0) or 0)

    @property
    def target_met(self) -> bool:
        return self.rag_ready_count >= self.requested_rag_ready


def _persist_topic_workspace(
    session: ChatSession,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> LiteratureWorkspace:
    """Persist topic updates while tolerating concurrent worker commits."""
    for _attempt in range(3):
        try:
            save_workspace(session, workspace, config=config)
            return workspace
        except RuntimeError as exc:
            if "Workspace revision mismatch" not in str(exc):
                raise
            latest = load_workspace(session)
            latest.papers.update(workspace.papers)
            latest.parsed_papers.update(workspace.parsed_papers)
            latest.full_text_reports.update(workspace.full_text_reports)
            latest.context.active_papers = list(workspace.context.active_papers)
            latest.context.selected_for_download = list(
                dict.fromkeys(
                    latest.context.selected_for_download
                    + workspace.context.selected_for_download
                )
            )
            latest.context.filters = workspace.context.filters
            latest.context.filters.workspace_revision = (
                latest.context.filters.workspace_revision
            )
            workspace = latest
    raise RuntimeError("Topic workspace could not be persisted after concurrent updates")


async def run_topic_search(
    config: LitTraceConfig,
    session: ChatSession,
    request: PaperSearchRequest,
    *,
    requested_rag_ready: int,
    canonical_topic: str | None = None,
    keywords: str = "",
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> TopicSearchRunResult:
    def progress(stage: str, **payload: object) -> None:
        if progress_callback is not None:
            progress_callback({"stage": stage, **payload})

    progress("search_started", query=request.topic)
    if progress_callback is None:
        # Keep the call signature compatible with lightweight test/mocking
        # adapters that implement the historical two-argument contract.
        search = await search_papers_skill(request, config)
    else:
        search = await search_papers_skill(
            request, config, progress_callback=progress_callback
        )
    progress("search_finished", count=len(search.result.papers))
    # Expand the candidate pool before acquisition when the first response is
    # smaller than the requested RAG target. Source adapters cap each request
    # at 100; merging by paper_id keeps retries idempotent and avoids replacing
    # good candidates with a later, narrower response.
    expansion_limit = min(100, max(request.limit * 2, requested_rag_ready * 5))
    expansion_rounds = 0
    while len(search.result.papers) < requested_rag_ready and expansion_limit > request.limit:
        expanded_request = request.model_copy(update={"limit": expansion_limit})
        if progress_callback is None:
            expanded = await search_papers_skill(expanded_request, config)
        else:
            expanded = await search_papers_skill(
                expanded_request, config, progress_callback=progress_callback
            )
        progress("candidate_expansion", count=len(expanded.result.papers), limit=expansion_limit)
        by_id = {paper.paper_id: paper for paper in search.result.papers}
        by_id.update({paper.paper_id: paper for paper in expanded.result.papers})
        if len(by_id) == len(search.result.papers):
            break
        search.result.papers = list(by_id.values())
        if search.diagnostics and expanded.diagnostics:
            search.diagnostics.errors.extend(expanded.diagnostics.errors)
            search.diagnostics.source_health.update(expanded.diagnostics.source_health)
            search.diagnostics.source_counts.update(expanded.diagnostics.source_counts)
        expansion_rounds += 1
        if expansion_limit >= 100:
            break
        expansion_limit = min(100, expansion_limit * 2)
    # Search results are an incremental update. Preserve previously parsed
    # papers, pipeline statuses, and RAG metadata when the user repeats a
    # topic search; replacing the workspace with a blank model would make
    # idempotent parse jobs look like they had never run.
    previous_workspace = load_workspace(session)
    previous_active = list(previous_workspace.context.active_papers)
    workspace = add_ranked_candidate_papers(
        previous_workspace,
        search.result.papers,
        request,
        active_limit=max(request.limit, config.literature_context.active_context_limit),
    )
    active_limit = max(request.limit, config.literature_context.active_context_limit)
    priority_preserved = [
        paper_id for paper_id in previous_workspace.context.pinned_papers
        if paper_id in previous_active
    ] + [
        paper_id for paper_id in previous_workspace.parsed_papers
        if paper_id in previous_active
    ]
    preserved_active = [
        paper_id
        for paper_id in dict.fromkeys(priority_preserved + previous_active)
        if paper_id in workspace.papers
        and paper_id not in workspace.context.excluded_papers
        and paper_id not in workspace.context.active_papers
    ]
    workspace.context.active_papers = (
        workspace.context.active_papers + preserved_active
    )[:active_limit]
    filters = workspace.context.filters
    filters.topic = canonical_topic or request.topic
    filters.search_query = request.topic
    filters.year_min = request.year_min
    filters.search_mode = "live" if search.use_live else "mock"
    filters.requested_rag_ready_count = requested_rag_ready
    filters.paper_pipeline_status = {
        paper.paper_id: "candidate" for paper in search.result.papers
    }
    if search.diagnostics:
        filters.search_diagnostics = {
            **search.diagnostics.__dict__,
            "source_health": {
                name: health.model_dump(mode="json")
                for name, health in search.diagnostics.source_health.items()
            },
        }
    workspace = _persist_topic_workspace(session, workspace, config)
    filters = workspace.context.filters
    result = TopicSearchRunResult(
        workspace=workspace,
        requested_rag_ready=requested_rag_ready,
        candidate_count=len(search.result.papers),
        warnings=list(search.diagnostics.errors[:5]) if search.diagnostics else [],
    )
    if expansion_rounds:
        result.warnings.append(f"候选池已扩展 {expansion_rounds} 轮，合并后 {len(search.result.papers)} 篇。")
    if not search.result.papers:
        result.warnings.append("没有检索到候选文献。")
        result.status = "exhausted"
        return result

    # A topic search owns the whole candidate set. Override the user's
    # general download preference for this run so every candidate enters the
    # downloader; unavailable papers will receive an explicit failed status,
    # while gated papers enter the existing CDP/auth path.
    download_config = config.model_copy(deep=True)
    download_config.paper_download.mode = DownloadMode.DOWNLOAD_SELECTED
    download_config.cdp_downloader.auto_launch_chrome = True
    download_config.cdp_downloader.headless = False
    download_result = await execute_downloads(
        download_config,
        list(workspace.papers.values()),
        DownloadExecutionRequest(
            paper_ids=list(workspace.context.active_papers),
            session_id=session.session_id,
            target="storage_only",
        ),
    )
    progress(
        "download_finished",
        downloaded=download_result.downloaded_count,
        requires_login=download_result.requires_login_count,
        failed=sum(1 for item in download_result.items if item.status == "failed"),
    )
    result.downloaded_count = download_result.downloaded_count
    result.requires_login_count = download_result.requires_login_count
    failed_ids = {item.paper_id for item in download_result.items if item.status == "failed"}
    result.failed_download_count = len(failed_ids)
    for item in download_result.items:
        filters.paper_pipeline_status[item.paper_id] = item.status
        if item.error:
            result.warnings.append(f"{item.paper_id}: {item.error}")
    filters.downloaded_full_text_count = result.downloaded_count
    workspace = _persist_topic_workspace(session, workspace, config)
    filters = workspace.context.filters
    downloaded_ids = [item.paper_id for item in download_result.items if item.status == "downloaded"]
    # A downloader result is not sufficient proof of object-storage success;
    # validate each registered artifact before creating parse jobs.
    registry = artifact_registry_from_config(download_config)
    artifact_store = artifact_store_from_config(download_config)
    stored_ids: list[str] = []
    for paper_id in downloaded_ids:
        record = registry.find_in_session(
            f"paper_pdf:{paper_id}", session_id=session.session_id
        )
        if record is None:
            filters.paper_pipeline_status[paper_id] = "storage_failed"
            result.warnings.append(f"{paper_id}: 下载器返回成功但未登记对象存储 artifact")
            continue
        ref = BlobRef(
            backend=record.backend,
            bucket=record.bucket,
            object_key=record.object_key,
            sha256=record.sha256,
            size_bytes=record.size_bytes,
            content_type=record.content_type,
        )
        if not artifact_store.exists(ref):
            filters.paper_pipeline_status[paper_id] = "storage_failed"
            result.warnings.append(f"{paper_id}: 对象存储 artifact 不存在")
            continue
        stored_ids.append(paper_id)
    result.downloaded_count = len(stored_ids)
    filters.downloaded_full_text_count = result.downloaded_count
    workspace = _persist_topic_workspace(session, workspace, config)
    filters = workspace.context.filters
    downloaded_ids = stored_ids
    if not downloaded_ids:
        result.workspace = load_workspace(session)
        result.status = "waiting_for_auth" if result.requires_login_count else "exhausted"
        return result

    current = load_workspace(session)
    enqueue_parse_job(config, session, current, downloaded_ids)
    parse_report = await run_pending_parse_jobs(
        config,
        limit=len(downloaded_ids),
        session_id=session.session_id,
    )
    progress("parse_finished", parsed=parse_report.parsed, total=len(downloaded_ids))
    result.parsed_count = parse_report.parsed
    result.warnings.extend(parse_report.warnings)
    embedding_report = await run_pending_embedding_jobs(
        config,
        limit=len(downloaded_ids),
        session_id=session.session_id,
    )
    progress("rag_finished", ready=len(embedding_report.ready_paper_ids), total=len(downloaded_ids))
    result.embedded_count = embedding_report.processed
    result.warnings.extend(embedding_report.warnings)
    result.workspace = load_workspace(session)
    result.workspace.context.filters.requested_rag_ready_count = requested_rag_ready
    ready_ids = set(embedding_report.ready_paper_ids)
    all_downloaded_ids = set(downloaded_ids)
    result.workspace.context.filters.rag_ready_count = len(ready_ids)
    for paper_id in downloaded_ids:
        parsed = result.workspace.parsed_papers.get(paper_id)
        if paper_id in ready_ids:
            result.workspace.context.filters.paper_pipeline_status[paper_id] = "rag_ready"
        elif parsed is not None and parsed.parsed:
            result.workspace.context.filters.paper_pipeline_status[paper_id] = "parsed"

    # Retry transient acquisition/embedding failures without re-downloading
    # papers that are already RAG-ready. This is bounded so a gated publisher
    # or a permanently bad source cannot block the UI forever.
    for retry_index in range(2):
        if len(ready_ids) >= requested_rag_ready:
            break
        current = load_workspace(session)
        retry_ids = [
            paper_id
            for paper_id in current.context.active_papers
            if paper_id not in ready_ids
            and current.context.filters.paper_pipeline_status.get(paper_id)
            not in {"requires_login", "auth_required"}
        ]
        if not retry_ids:
            break
        retry_result = await execute_downloads(
            download_config,
            [current.papers[paper_id] for paper_id in retry_ids if paper_id in current.papers],
            DownloadExecutionRequest(
                paper_ids=retry_ids,
                session_id=session.session_id,
                target="storage_only",
            ),
        )
        progress(
            "download_retry_finished", retry=retry_index + 1,
            downloaded=retry_result.downloaded_count,
            failed=sum(1 for item in retry_result.items if item.status == "failed"),
        )
        retry_downloaded = [
            item.paper_id for item in retry_result.items if item.status == "downloaded"
        ]
        new_downloaded = set(retry_downloaded) - all_downloaded_ids
        all_downloaded_ids.update(retry_downloaded)
        result.downloaded_count += len(new_downloaded)
        failed_ids.update(item.paper_id for item in retry_result.items if item.status == "failed")
        result.failed_download_count = len(failed_ids)
        for item in retry_result.items:
            current.context.filters.paper_pipeline_status[item.paper_id] = item.status
        if not retry_downloaded:
            result.warnings.append(f"第 {retry_index + 2} 轮下载未产生新增成功文献。")
            break
        current = _persist_topic_workspace(session, current, config)
        retry_workspace = load_workspace(session)
        enqueue_parse_job(config, session, retry_workspace, retry_downloaded)
        retry_parse = await run_pending_parse_jobs(
            config, limit=len(retry_downloaded), session_id=session.session_id,
        )
        progress("parse_retry_finished", retry=retry_index + 1, parsed=retry_parse.parsed)
        result.parsed_count += retry_parse.parsed
        result.warnings.extend(retry_parse.warnings)
        retry_embedding = await run_pending_embedding_jobs(
            config, limit=len(retry_downloaded), session_id=session.session_id,
        )
        progress(
            "rag_retry_finished", retry=retry_index + 1,
            ready=len(retry_embedding.ready_paper_ids),
        )
        result.embedded_count += retry_embedding.processed
        result.warnings.extend(retry_embedding.warnings)
        ready_ids.update(retry_embedding.ready_paper_ids)
        result.workspace = load_workspace(session)
        result.workspace.context.filters.rag_ready_count = len(ready_ids)

    if len(ready_ids) < requested_rag_ready:
        result.warnings.append(
            f"RAG ready 未达到目标：{len(ready_ids)}/{requested_rag_ready}。"
        )
        result.status = "waiting_for_auth" if result.requires_login_count else "exhausted"
    else:
        result.status = "completed"
    result.workspace = _persist_topic_workspace(session, result.workspace, config)
    return result
