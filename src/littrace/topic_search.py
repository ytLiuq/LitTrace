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
from littrace.retrieval.rag_refresh import refresh_session_rag_index
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

    def source_progress(payload: dict[str, object]) -> None:
        # The source skill emits its own pre-truncation ``search_finished``
        # count. Suppress that duplicate so the UI reports the final bounded
        # candidate pool exactly once.
        if progress_callback is not None and payload.get("stage") != "search_finished":
            progress_callback(payload)

    progress("search_started", query=request.topic)
    if progress_callback is None:
        # Keep the call signature compatible with lightweight test/mocking
        # adapters that implement the historical two-argument contract.
        search = await search_papers_skill(request, config)
    else:
        search = await search_papers_skill(
            request, config, progress_callback=source_progress
        )
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
                expanded_request, config, progress_callback=source_progress
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
    # Source adapters may return a merged list larger than the requested
    # client-facing reserve. Keep only the ranked head for acquisition and
    # context accounting; the adapters already performed the broad recall.
    if len(search.result.papers) > request.limit:
        search.result.papers = search.result.papers[: request.limit]
    progress("search_finished", count=len(search.result.papers))
    # Search results are an incremental update. Preserve previously parsed
    # papers, pipeline statuses, and RAG metadata when the user repeats a
    # topic search; replacing the workspace with a blank model would make
    # idempotent parse jobs look like they had never run.
    previous_workspace = load_workspace(session)
    workspace = add_ranked_candidate_papers(
        previous_workspace,
        search.result.papers,
        request,
        # The candidate pool is deliberately wider than the active context.
        # Only papers that complete RAG processing may become active below.
        active_limit=requested_rag_ready,
    )
    # Do not expose raw candidates in the literature context while acquisition
    # is still running. Parse jobs temporarily add only verified stored IDs and
    # the final context is pruned to RAG-ready IDs.
    workspace.context.active_papers = []
    filters = workspace.context.filters
    filters.topic = canonical_topic or request.topic
    filters.search_query = request.topic
    filters.year_min = request.year_min
    filters.year_max = request.year_max
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

    # A topic search owns a staged acquisition queue. The first batch is no
    # larger than the requested RAG target; failed papers are replaced from
    # the ranked reserve only when needed.
    download_config = config.model_copy(deep=True)
    download_config.paper_download.mode = DownloadMode.DOWNLOAD_SELECTED
    download_config.cdp_downloader.auto_launch_chrome = True
    download_config.cdp_downloader.headless = False
    candidate_ids = [paper.paper_id for paper in search.result.papers]
    attempted_ids: set[str] = set()
    ready_ids: list[str] = []
    stored_ids_seen: set[str] = set()
    failed_ids: set[str] = set()
    login_ids: set[str] = set()

    async def process_batch(batch_ids: list[str], *, retry_index: int = 0) -> None:
        nonlocal workspace
        if not batch_ids:
            return
        batch_papers = [workspace.papers[paper_id] for paper_id in batch_ids if paper_id in workspace.papers]
        download_result = await execute_downloads(
            download_config,
            batch_papers,
            DownloadExecutionRequest(
                paper_ids=batch_ids,
                session_id=session.session_id,
                target="storage_only",
            ),
        )
        progress(
            "download_finished" if retry_index == 0 else "download_retry_finished",
            retry=retry_index or None,
            downloaded=download_result.downloaded_count,
            requires_login=download_result.requires_login_count,
            failed=sum(1 for item in download_result.items if item.status == "failed"),
        )
        for item in download_result.items:
            if item.status in {"requires_login", "auth_required"}:
                login_ids.add(item.paper_id)
            if item.status == "failed":
                failed_ids.add(item.paper_id)
            if item.error:
                result.warnings.append(f"{item.paper_id}: {item.error}")
            workspace.context.filters.paper_pipeline_status[item.paper_id] = item.status

        downloaded_ids = [item.paper_id for item in download_result.items if item.status == "downloaded"]
        registry = artifact_registry_from_config(download_config)
        artifact_store = artifact_store_from_config(download_config)
        stored_ids: list[str] = []
        for paper_id in downloaded_ids:
            record = registry.find_in_session(f"paper_pdf:{paper_id}", session_id=session.session_id)
            if record is None:
                failed_ids.add(paper_id)
                workspace.context.filters.paper_pipeline_status[paper_id] = "storage_failed"
                result.warnings.append(f"{paper_id}: 下载成功但对象存储 artifact 未登记")
                continue
            ref = BlobRef(
                backend=record.backend, bucket=record.bucket,
                object_key=record.object_key, sha256=record.sha256,
                size_bytes=record.size_bytes, content_type=record.content_type,
            )
            if not artifact_store.exists(ref):
                failed_ids.add(paper_id)
                workspace.context.filters.paper_pipeline_status[paper_id] = "storage_failed"
                result.warnings.append(f"{paper_id}: 对象存储 artifact 不存在")
                continue
            stored_ids.append(paper_id)
            stored_ids_seen.add(paper_id)

        result.downloaded_count = len(stored_ids_seen)
        result.requires_login_count = len(login_ids)
        result.failed_download_count = len(failed_ids)
        workspace.context.filters.downloaded_full_text_count = result.downloaded_count
        if not stored_ids:
            workspace = _persist_topic_workspace(session, workspace, config)
            return

        # Parse commit validation requires the papers to be active temporarily;
        # failed/unverified candidates never enter this list.
        workspace.context.active_papers = list(dict.fromkeys(ready_ids + stored_ids))
        workspace = _persist_topic_workspace(session, workspace, config)
        current = load_workspace(session)
        parse_job = enqueue_parse_job(config, session, current, stored_ids)
        parse_report = await run_pending_parse_jobs(
            config, limit=len(stored_ids), session_id=session.session_id,
            task_ids={parse_job.task_id} if parse_job is not None else None,
        )
        result.parsed_count += parse_report.parsed
        result.warnings.extend(parse_report.warnings)
        progress(
            "parse_finished" if retry_index == 0 else "parse_retry_finished",
            retry=retry_index or None, parsed=parse_report.parsed, total=len(stored_ids),
        )
        embedding_report = await run_pending_embedding_jobs(
            config, limit=len(stored_ids), session_id=session.session_id,
            artifact_ids={f"paper_pdf:{paper_id}" for paper_id in stored_ids},
        )
        result.embedded_count += embedding_report.processed
        result.warnings.extend(embedding_report.warnings)
        batch_ready = list(embedding_report.ready_paper_ids)
        if parse_report.parsed and not batch_ready:
            current = load_workspace(session)
            _, direct_rag = await refresh_session_rag_index(
                config, session, current, artifact_ids=set(stored_ids),
            )
            if not direct_rag.skipped:
                save_workspace(session, current, config=config)
                batch_ready = list(direct_rag.paper_ids)
        ready_ids.extend(paper_id for paper_id in batch_ready if paper_id not in ready_ids)
        workspace = load_workspace(session)
        for paper_id in stored_ids:
            workspace.context.filters.paper_pipeline_status[paper_id] = (
                "rag_ready" if paper_id in ready_ids else "parsed"
            )
        workspace.context.filters.rag_ready_count = len(ready_ids)
        workspace.context.active_papers = list(dict.fromkeys(ready_ids))
        workspace = _persist_topic_workspace(session, workspace, config)
        progress(
            "rag_finished" if retry_index == 0 else "rag_retry_finished",
            retry=retry_index or None, ready=len(ready_ids), total=len(stored_ids),
        )

    max_attempts = min(len(candidate_ids), max(requested_rag_ready * 3, requested_rag_ready))
    wave = 0
    while len(ready_ids) < requested_rag_ready and len(attempted_ids) < max_attempts:
        remaining = requested_rag_ready - len(ready_ids)
        batch = [paper_id for paper_id in candidate_ids if paper_id not in attempted_ids][:remaining]
        if not batch:
            break
        attempted_ids.update(batch)
        await process_batch(batch, retry_index=wave)
        wave += 1
        if wave >= 3:
            break
    if len(ready_ids) < requested_rag_ready:
        result.warnings.append(
            f"RAG ready 未达到目标：{len(ready_ids)}/{requested_rag_ready}。"
        )
        result.status = "waiting_for_auth" if result.requires_login_count else "exhausted"
    else:
        result.status = "completed"
    workspace.context.active_papers = ready_ids[:requested_rag_ready]
    workspace.context.filters.rag_ready_count = len(workspace.context.active_papers)
    result.workspace = _persist_topic_workspace(session, workspace, config)
    return result
