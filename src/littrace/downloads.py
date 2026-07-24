from __future__ import annotations

from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import httpx

from littrace.artifact_store import (
    ArtifactKeyContext,
    artifact_store_from_config,
    build_artifact_object_key,
)
from littrace.artifact_registry import ArtifactRecord, artifact_registry_from_config
from littrace.access_layer.cdp import download_paper_via_cdp, identify_publisher
from littrace.access_layer.paths import build_download_plan, target_pdf_path
from littrace.config import LitTraceConfig
from littrace.download_tasks import (
    DownloadTask,
    DownloadTaskStatus,
    download_task_store_from_config,
)
from littrace.state_db import EmbeddingJobRecord, SessionPermissionGrant, state_store_from_config
from littrace.retrieval.full_text import resolve_full_text_for_paper
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
    target_papers = [
        paper for paper in papers if not selected_ids or paper.paper_id in selected_ids
    ]
    plan = build_download_plan(config, target_papers, selected_ids)
    items: list[DownloadExecutionItem] = []

    timeout = httpx.Timeout(config.api.request_timeout_seconds)
    headers = {"User-Agent": config.api.user_agent}
    task_store = download_task_store_from_config(config)
    user_id = request.user_id or config.storage.default_user_id
    session_id = request.session_id or "adhoc"
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for plan_item in plan.items:
            paper = next(paper for paper in target_papers if paper.paper_id == plan_item.paper_id)
            task = DownloadTask.from_paper(
                config,
                paper,
                session_id=session_id,
                user_id=user_id,
            )
            if not request.dry_run and config.download_retry.enabled:
                task_store.upsert(task)
            item, task = await _execute_one(client, config, paper, request.dry_run, task)
            if not request.dry_run and config.download_retry.enabled:
                task_store.upsert(task)
            items.append(item)

    return DownloadExecutionResult(
        items=items,
        downloaded_count=sum(item.status == "downloaded" for item in items),
        requires_login_count=sum(
            item.action == "cdp_publisher_download" or item.status == "requires_login"
            for item in items
        ),
        skipped_count=sum(item.status == "skipped" for item in items),
    )


async def _execute_one(
    client: httpx.AsyncClient,
    config: LitTraceConfig,
    paper: PaperMetadata,
    dry_run: bool,
    task: DownloadTask,
) -> tuple[DownloadExecutionItem, DownloadTask]:
    if paper.access_type == AccessType.REQUIRES_LOGIN and paper.doi:
        return _execute_cdp_download(config, paper, dry_run, task)
    pdf_url = paper.pdf_url
    if paper.access_type == AccessType.OPEN_ACCESS and not pdf_url:
        report = await resolve_full_text_for_paper(client, paper, config)
        pdf_url = report.best_pdf_url
    if paper.access_type != AccessType.OPEN_ACCESS or not pdf_url:
        error = "Full text PDF is required, but no verified PDF URL is available."
        task.mark(DownloadTaskStatus.FAILED, error=error)
        task.schedule_retry(config.download_retry.base_delay_seconds)
        return DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="download",
            status="failed",
            error=error,
            task_id=task.task_id,
        ), task

    target_path = _target_pdf_path(config, paper)
    if dry_run:
        return DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="download",
            status="planned",
            target_path=str(target_path),
        ), task

    try:
        task.attempt_count += 1
        task.mark(DownloadTaskStatus.DOWNLOADING)
        response = await client.get(str(pdf_url))
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if not _looks_like_pdf_bytes(response.content):
            if paper.access_type == AccessType.OPEN_ACCESS:
                for candidate_url in _extract_pdf_links_from_html(str(response.url), response.text):
                    try:
                        candidate_response = await client.get(candidate_url)
                        candidate_response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        if paper.doi and exc.response.status_code in {401, 403, 429}:
                            return _execute_cdp_download(
                                config,
                                paper,
                                dry_run,
                                task,
                                prior_error=f"{exc.__class__.__name__}: {exc}",
                            )
                        continue
                    except httpx.HTTPError:
                        continue
                    if not _looks_like_pdf_bytes(candidate_response.content):
                        continue
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(candidate_response.content)
                    try:
                        storage_ref = _store_pdf_artifact(
                            config,
                            task,
                            paper,
                            candidate_response.content,
                            content_type=candidate_response.headers.get(
                                "content-type", "application/pdf"
                            ),
                        )
                    except Exception as exc:
                        error = f"{exc.__class__.__name__}: {exc}"
                        task.mark(DownloadTaskStatus.FAILED, error=error)
                        task.schedule_retry(config.download_retry.base_delay_seconds)
                        return DownloadExecutionItem(
                            paper_id=paper.paper_id,
                            action="download",
                            status="failed",
                            target_path=str(target_path),
                            error=error,
                            task_id=task.task_id,
                        ), task
                    task.mark(DownloadTaskStatus.VERIFIED)
                    return DownloadExecutionItem(
                        paper_id=paper.paper_id,
                        action="download",
                        status="downloaded",
                        target_path=str(target_path),
                        task_id=task.task_id,
                        storage_ref=storage_ref,
                    ), task
            error = f"Response does not look like a PDF: {content_type}"
            if paper.doi and _should_try_cdp_fallback(response):
                return _execute_cdp_download(
                    config,
                    paper,
                    dry_run,
                    task,
                    prior_error=error,
                )
            if _looks_like_human_verification_response(response):
                error = (
                    "Source returned a human-verification or access-block page instead of "
                    "PDF bytes."
                )
                task.requires_login = True
                task.mark(DownloadTaskStatus.AUTH_REQUIRED, error=error)
                return DownloadExecutionItem(
                    paper_id=paper.paper_id,
                    action="download",
                    status="requires_login",
                    target_path=str(target_path),
                    error=error,
                    task_id=task.task_id,
                    login_instructions=[
                        "请在浏览器中完成该来源的人机验证/访问确认后重试下载。"
                    ],
                ), task
            task.mark(DownloadTaskStatus.FAILED, error=error)
            task.schedule_retry(config.download_retry.base_delay_seconds)
            return DownloadExecutionItem(
                paper_id=paper.paper_id,
                action="download",
                status="failed",
                target_path=str(target_path),
                error=error,
                task_id=task.task_id,
            ), task
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(response.content)
        try:
            storage_ref = _store_pdf_artifact(
                config,
                task,
                paper,
                response.content,
                content_type=content_type or "application/pdf",
            )
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            task.mark(DownloadTaskStatus.FAILED, error=error)
            task.schedule_retry(config.download_retry.base_delay_seconds)
            return DownloadExecutionItem(
                paper_id=paper.paper_id,
                action="download",
                status="failed",
                target_path=str(target_path),
                error=error,
                task_id=task.task_id,
            ), task
        task.mark(DownloadTaskStatus.VERIFIED)
        return DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="download",
            status="downloaded",
            target_path=str(target_path),
            task_id=task.task_id,
            storage_ref=storage_ref,
        ), task
    except httpx.HTTPStatusError as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        if paper.doi and exc.response.status_code in {401, 403, 429}:
            return _execute_cdp_download(
                config,
                paper,
                dry_run,
                task,
                prior_error=error,
            )
        task.mark(DownloadTaskStatus.FAILED, error=error)
        task.schedule_retry(config.download_retry.base_delay_seconds)
        return DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="download",
            status="failed",
            target_path=str(target_path),
            error=error,
            task_id=task.task_id,
        ), task
    except httpx.HTTPError as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        if paper.doi and _should_try_cdp_after_open_access_http_error(paper, exc):
            return _execute_cdp_download(
                config,
                paper,
                dry_run,
                task,
                prior_error=error,
            )
        task.mark(DownloadTaskStatus.FAILED, error=error)
        task.schedule_retry(config.download_retry.base_delay_seconds)
        return DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="download",
            status="failed",
            target_path=str(target_path),
            error=error,
            task_id=task.task_id,
        ), task


def _execute_cdp_download(
    config: LitTraceConfig,
    paper: PaperMetadata,
    dry_run: bool,
    task: DownloadTask,
    *,
    prior_error: str | None = None,
) -> tuple[DownloadExecutionItem, DownloadTask]:
    target_path = _target_pdf_path(config, paper)
    if dry_run:
        return DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="cdp_publisher_download",
            status="planned",
            target_path=str(target_path),
        ), task
    task.requires_login = True
    task.attempt_count += 1
    task.mark(DownloadTaskStatus.DOWNLOADING)
    result = download_paper_via_cdp(config, paper.doi or paper.paper_id, target_path)
    error = result.error or prior_error
    storage_ref: dict[str, object] | None = None
    if result.downloaded and target_path.exists():
        try:
            storage_ref = _store_pdf_artifact(
                config,
                task,
                paper,
                target_path.read_bytes(),
                content_type="application/pdf",
            )
            task.mark(DownloadTaskStatus.VERIFIED)
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            task.mark(DownloadTaskStatus.FAILED, error=error)
            task.schedule_retry(config.download_retry.base_delay_seconds)
    elif result.requires_user_action:
        task.mark(DownloadTaskStatus.AUTH_REQUIRED, error=error)
    else:
        task.mark(DownloadTaskStatus.FAILED, error=error)
        task.schedule_retry(config.download_retry.base_delay_seconds)
    return DownloadExecutionItem(
        paper_id=paper.paper_id,
        action="cdp_publisher_download",
        status="downloaded"
        if result.downloaded
        else ("requires_login" if result.requires_user_action else "failed"),
        target_path=str(target_path),
        task_id=task.task_id,
        storage_ref=storage_ref,
        login_instructions=[result.user_action] if result.user_action else [],
        error=error,
    ), task


def _target_pdf_path(config: LitTraceConfig, paper: PaperMetadata) -> Path:
    return target_pdf_path(config, paper)


def _looks_like_pdf_bytes(content: bytes) -> bool:
    return content.lstrip()[:5] == b"%PDF-"


def _looks_like_human_verification_response(response: httpx.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type and "text" not in content_type:
        return False
    sample = response.content[:8192].decode("utf-8", errors="ignore").lower()
    markers = [
        "recaptcha",
        "captcha",
        "checking your browser",
        "verify you are human",
        "human verification",
        "just a moment",
        "cloudflare",
        "__cf_chl",
        "are you a robot",
        "access denied",
        "blocked for possible abuse",
        "misuse.ncbi.nlm.nih.gov",
        "请验证",
        "正在进行安全验证",
    ]
    return any(marker in sample for marker in markers)


def _should_try_cdp_fallback(response: httpx.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    if _looks_like_human_verification_response(response):
        return True
    if "html" in content_type or "text" in content_type:
        return True
    return False


def _should_try_cdp_after_open_access_http_error(
    paper: PaperMetadata,
    exc: httpx.HTTPError,
) -> bool:
    if paper.access_type != AccessType.OPEN_ACCESS or not paper.doi:
        return False
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return identify_publisher(paper.doi) != "unknown"
    return True


class _PdfLinkHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() not in {"a", "link", "iframe", "embed"}:
            return
        for name, value in attrs:
            if name.lower() in {"href", "src"} and value:
                self.links.append(value)


def _extract_pdf_links_from_html(base_url: str, html: str) -> list[str]:
    parser = _PdfLinkHTMLParser()
    try:
        parser.feed(html)
    except Exception:
        return []
    candidates: list[str] = []
    seen: set[str] = set()
    for raw in parser.links:
        lowered = raw.lower()
        if not any(marker in lowered for marker in [".pdf", "/pdf", "download", "article/download"]):
            continue
        candidate = urljoin(base_url, raw)
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def make_download_retry_handler(config: LitTraceConfig):
    async def handler(task: DownloadTask) -> DownloadTask:
        paper = PaperMetadata(
            paper_id=task.paper_id,
            title=task.paper_id,
            doi=task.doi,
            pdf_url=task.source_url if task.source_url else None,
            access_type=task.access_type,
        )
        timeout = httpx.Timeout(config.api.request_timeout_seconds)
        headers = {"User-Agent": config.api.user_agent}
        async with httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
        ) as client:
            _, updated = await _execute_one(client, config, paper, False, task)
        return updated

    return handler


def _store_pdf_artifact(
    config: LitTraceConfig,
    task: DownloadTask,
    paper: PaperMetadata,
    data: bytes,
    *,
    content_type: str,
) -> dict[str, object]:
    task.mark(DownloadTaskStatus.UPLOADING_TO_OBJECT_STORAGE)
    store = artifact_store_from_config(config)
    artifact_id = f"paper_pdf:{paper.paper_id}"
    object_key = build_artifact_object_key(
        config,
        ArtifactKeyContext(
            user_id=task.user_id,
            session_id=task.session_id,
            kind="paper_pdf",
            artifact_id=artifact_id,
            filename="paper.pdf",
            paper_id=paper.paper_id,
        ),
    )
    ref = store.put_bytes(
        object_key,
        data,
        content_type=content_type or "application/pdf",
        metadata={
            "user_id": task.user_id,
            "session_id": task.session_id,
            "paper_id": paper.paper_id,
            "kind": "paper_pdf",
            "doi": paper.doi or "",
        },
    )
    record = artifact_registry_from_config(config).upsert(
        ArtifactRecord.from_blob_ref(
            ref,
            artifact_id=artifact_id,
            user_id=task.user_id,
            session_id=task.session_id,
            kind="paper_pdf",
            paper_id=paper.paper_id,
            metadata={
                "doi": paper.doi,
                "source_name": task.source_name,
                "source_url": task.source_url,
            },
        )
    )
    _register_download_state(config, task, record)
    task.artifact_id = artifact_id
    task.target_bucket = ref.bucket
    task.target_object_key = ref.object_key
    task.sha256 = ref.sha256
    task.size_bytes = ref.size_bytes
    task.mark(DownloadTaskStatus.STORED)
    return ref.model_dump(mode="json")


def _register_download_state(
    config: LitTraceConfig,
    task: DownloadTask,
    record: ArtifactRecord,
) -> None:
    state_store = state_store_from_config(config)
    if state_store is None:
        return
    state_store.grant_permission(
        SessionPermissionGrant(
            scope="artifact",
            resource_key=f"{task.session_id}:{record.artifact_id}",
            owner_user_id=task.user_id,
            grantee_user_id=task.user_id,
            role="owner",
            granted_by=task.user_id,
            metadata={"kind": record.kind, "paper_id": record.paper_id},
        )
    )
    if not config.rag.enabled:
        return
    state_store.enqueue_embedding_job(
        EmbeddingJobRecord(
            job_id=_embedding_job_id(
                task.user_id,
                task.session_id,
                record.artifact_id,
                record.sha256,
                record.revision,
            ),
            profile_id=f"pending:{task.user_id}:{task.session_id}",
            user_id=task.user_id,
            session_id=task.session_id,
            artifact_id=record.artifact_id,
            source_revision=record.revision,
            content_sha256=record.sha256,
        )
    )


def _embedding_job_id(
    user_id: str,
    session_id: str,
    artifact_id: str,
    content_sha256: str | None,
    source_revision: str | None,
) -> str:
    digest = sha256(
        "\0".join(
            [
                user_id,
                session_id,
                artifact_id,
                content_sha256 or "",
                source_revision or "",
            ]
        ).encode("utf-8")
    ).hexdigest()[:24]
    return f"emb:{digest}"
