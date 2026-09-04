from __future__ import annotations

import asyncio
import hashlib
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import httpx

from littrace.artifact_store import (
    ArtifactKeyContext,
    BlobRef,
    artifact_store_from_config,
    build_artifact_object_key,
)
from littrace.artifact_registry import ArtifactRecord, artifact_registry_from_config
from littrace.access_layer.cdp import (
    check_cdp_status,
    download_paper_via_cdp,
    identify_publisher,
)
from littrace.access_layer.paths import build_download_plan, target_pdf_path
from littrace.config import LitTraceConfig
from littrace.download_tasks import (
    DownloadTask,
    DownloadTaskStatus,
    download_task_store_from_config,
)
from littrace.lifecycle import enqueue_embedding_outbox, record_lifecycle_event
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
    session_id = request.session_id or "adhoc"
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        # Round 28: build ONE shared CDP browser for the whole batch
        # so gated papers reuse a single Chrome tab instead of
        # creating 45 phantom ``about:blank`` tabs. The browser is
        # synchronised with the asyncio semaphore so concurrent
        # plan items still serialize through one tab — that's
        # actually what we want for CF / SSO flows which need the
        # page state to settle between navigations.
        from littrace.access_layer.cdp_core import CDPBrowser
        shared_browser: "CDPBrowser | None" = None
        shared_browser_lock = asyncio.Lock()
        if not request.dry_run and any(
            p.access_type == AccessType.REQUIRES_LOGIN
            for p in target_papers
        ):
            try:
                status = check_cdp_status(config)
                from littrace.chrome_profiles import (
                    cdp_uses_configured_profile,
                    launch_chrome_for_cdp,
                )

                private_profile = status.available and cdp_uses_configured_profile(config)
                if (not status.available or not private_profile) and config.cdp_downloader.auto_launch_chrome:

                    launch = await asyncio.to_thread(
                        launch_chrome_for_cdp,
                        config,
                        headless=False,
                    )
                    if launch.error and launch.process is not None:
                        try:
                            launch.process.terminate()
                        except Exception:
                            pass
                    if launch.cdp_status is not None:
                        status = launch.cdp_status
                if not status.available:
                    raise RuntimeError(
                        f"CDP browser unavailable at {config.cdp_downloader.cdp_url}"
                    )
                if not private_profile and not config.cdp_downloader.auto_launch_chrome:
                    raise RuntimeError(
                        "Configured CDP endpoint is not owned by LitTrace's private Chrome profile"
                    )
                shared_browser = await asyncio.to_thread(
                    CDPBrowser,
                    config.cdp_downloader.cdp_url,
                    reconnect_attempts=(
                        config.cdp_downloader.websocket_reconnect_attempts
                    ),
                    command_timeout_seconds=(
                        config.cdp_downloader.command_timeout_seconds
                    ),
                )
                await asyncio.to_thread(shared_browser.connect_new_tab)
            except Exception:
                # If the shared browser can't start up, fall back
                # to the per-paper one — the previous behaviour.
                if shared_browser is not None:
                    try:
                        await asyncio.to_thread(shared_browser.close_tab)
                    except Exception:
                        pass
                shared_browser = None

        async def run_plan_item(plan_item):
            paper = next(paper for paper in target_papers if paper.paper_id == plan_item.paper_id)
            task = DownloadTask.from_paper(
                config,
                paper,
                session_id=session_id,
            )
            _record_discovered_relevant(config, task, paper)
            _record_task_lifecycle(config, task, "acquisition_queued")
            if not request.dry_run and config.download_retry.enabled:
                task_store.upsert(task)
            item, task = await _execute_one(
                client,
                config,
                paper,
                request.dry_run,
                task,
                write_local=(request.target != "storage_only"),
                shared_browser=shared_browser,
                shared_browser_lock=shared_browser_lock,
            )
            _record_terminal_acquisition_event(config, task)
            if not request.dry_run and config.download_retry.enabled:
                task_store.upsert(task)
            return item

        semaphore = asyncio.Semaphore(config.paper_download.max_concurrent_downloads)

        async def run_bounded(plan_item):
            async with semaphore:
                return await run_plan_item(plan_item)

        try:
            items = list(await asyncio.gather(*(run_bounded(item) for item in plan.items)))
        finally:
            # Close the target as well as the WebSocket, including cancellation
            # and exception paths. ``close()`` alone leaves about:blank open.
            if shared_browser is not None:
                try:
                    await asyncio.to_thread(shared_browser.close_tab)
                except Exception:
                    pass

    # Keep this field aligned with the download plan: it describes selected
    # papers whose publisher requires authentication, regardless of whether
    # this attempt already reached a terminal auth-required state.
    requires_login_count = sum(
        paper.access_type == AccessType.REQUIRES_LOGIN for paper in target_papers
    )
    return DownloadExecutionResult(
        items=items,
        downloaded_count=sum(item.status == "downloaded" for item in items),
        requires_login_count=requires_login_count,
        skipped_count=sum(item.status == "skipped" for item in items),
    )


async def _execute_one(
    client: httpx.AsyncClient,
    config: LitTraceConfig,
    paper: PaperMetadata,
    dry_run: bool,
    task: DownloadTask,
    *,
    write_local: bool = True,
    shared_browser: "CDPBrowser | None" = None,
    shared_browser_lock: asyncio.Lock | None = None,
) -> tuple[DownloadExecutionItem, DownloadTask]:
    _record_task_lifecycle(config, task, "acquisition_started")

    async def run_cdp(prior_error: str | None = None):
        async def execute():
            return await _execute_cdp_download_async(
                config,
                paper,
                dry_run,
                task,
                prior_error=prior_error,
                browser=shared_browser,
            )

        if shared_browser is not None and shared_browser_lock is not None:
            async with shared_browser_lock:
                return await execute()
        return await execute()

    if paper.access_type == AccessType.REQUIRES_LOGIN and paper.doi:
        return await run_cdp()
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

    target_path = target_pdf_path(config, paper)
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
                        if paper.doi and exc.response.status_code in {401, 403, 418, 429}:
                            return await run_cdp(
                                f"{exc.__class__.__name__}: {exc}"
                            )
                        continue
                    except httpx.HTTPError:
                        continue
                    if not _looks_like_pdf_bytes(candidate_response.content):
                        continue
                    if write_local:
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
                return await run_cdp(error)
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
        if write_local:
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
        if paper.doi and exc.response.status_code in {401, 403, 418, 429}:
            return await run_cdp(error)
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
            return await run_cdp(error)
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
    browser=None,
) -> tuple[DownloadExecutionItem, DownloadTask]:
    target_path = target_pdf_path(config, paper)
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
    result = download_paper_via_cdp(
        config, paper.doi or paper.paper_id, target_path,
        browser=browser,
    )
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


async def _execute_cdp_download_async(
    config: LitTraceConfig,
    paper: PaperMetadata,
    dry_run: bool,
    task: DownloadTask,
    *,
    prior_error: str | None = None,
    browser=None,
) -> tuple[DownloadExecutionItem, DownloadTask]:
    """Run the blocking CDP client without blocking other download workers.

    Round 28: caller can pass a shared ``browser`` instance so a
    single Chrome tab is reused across all gated papers in the
    batch. Without the shared browser, each call creates a fresh
    ``about:blank`` tab and leaves it open after the download —
    45 gated papers → 45 phantom tabs the user has to close by
    hand. Reusing the tab keeps Chrome's tab bar clean and cuts
    the per-paper CDP connection setup cost.
    """
    return await asyncio.to_thread(
        _execute_cdp_download,
        config,
        paper,
        dry_run,
        task,
        prior_error=prior_error,
        browser=browser,
    )


# _target_pdf_path internal alias removed; callers use
# littrace.access_layer.paths.target_pdf_path directly.


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
        _record_terminal_acquisition_event(config, updated)
        return updated

    return handler


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    registry = artifact_registry_from_config(config)
    artifact_id = f"paper_pdf:{paper.paper_id}"
    object_key = build_artifact_object_key(
        config,
        ArtifactKeyContext(
            session_id=task.session_id,
            kind="paper_pdf",
            artifact_id=artifact_id,
            filename="paper.pdf",
            paper_id=paper.paper_id,
        ),
    )
    # Idempotency: if a registry row already exists for this
    # (session_id, artifact_id) and the byte sha256 matches, skip the PUT
    # and the lifecycle event. This protects against double-download from
    # chat intent + auto_resume + a retry worker all racing.
    prior = registry.find_in_session(artifact_id, session_id=task.session_id)
    prior_ref = None
    if prior is not None and prior.sha256 and prior.sha256 == _sha256_hex(data):
        prior_ref = BlobRef(
            backend=prior.backend,
            bucket=prior.bucket,
            object_key=prior.object_key,
            sha256=prior.sha256,
            size_bytes=prior.size_bytes,
            content_type=prior.content_type,
            uri=prior.bucket and f"s3://{prior.bucket}/{prior.object_key}" or prior.object_key,
        )
        if store.exists(prior_ref):
            ref = prior_ref
            record = prior
        else:
            prior_ref = None
    if prior_ref is None:
        ref = store.put_bytes(
            object_key,
            data,
            content_type=content_type or "application/pdf",
            metadata={
                "session_id": task.session_id,
                "paper_id": paper.paper_id,
                "kind": "paper_pdf",
                "doi": paper.doi or "",
            },
        )
        record = registry.upsert(
            ArtifactRecord.from_blob_ref(
                ref,
                artifact_id=artifact_id,
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
        record_lifecycle_event(
            config, session_id=task.session_id, paper_id=paper.paper_id,
            event_type="artifact_stored", task_id=task.task_id, artifact_id=artifact_id,
            payload={"artifact_id": artifact_id, "sha256": ref.sha256, "source_revision": record.revision},
        )
        if config.rag.enabled:
            enqueue_embedding_outbox(
                config, session_id=task.session_id, artifact_id=artifact_id,
                content_sha256=ref.sha256, payload={"source_revision": record.revision},
            )
    task.artifact_id = artifact_id
    task.target_bucket = ref.bucket
    task.target_object_key = ref.object_key
    task.sha256 = ref.sha256
    task.size_bytes = ref.size_bytes
    task.mark(DownloadTaskStatus.STORED)
    return ref.model_dump(mode="json")


def _record_task_lifecycle(config: LitTraceConfig, task: DownloadTask, event_type: str) -> None:
    record_lifecycle_event(
        config, session_id=task.session_id, paper_id=task.paper_id,
        event_type=event_type, task_id=task.task_id, artifact_id=task.artifact_id,
        payload={"attempt_count": task.attempt_count, "status": task.status.value},
    )


def _record_discovered_relevant(
    config: LitTraceConfig,
    task: DownloadTask,
    paper: PaperMetadata,
) -> None:
    """Selected download candidates have passed the session relevance gate."""
    record_lifecycle_event(
        config,
        session_id=task.session_id,
        paper_id=paper.paper_id,
        event_type="discovered_relevant",
        task_id=task.task_id,
        payload={
            "relevance_score": paper.relevance_score,
            "access_type": paper.access_type.value,
            "source_name": task.source_name,
        },
    )


def _record_terminal_acquisition_event(config: LitTraceConfig, task: DownloadTask) -> None:
    if task.status == DownloadTaskStatus.VERIFIED:
        event_type = "acquisition_verified"
    elif task.status == DownloadTaskStatus.AUTH_REQUIRED:
        event_type = "acquisition_auth_required"
    elif task.status == DownloadTaskStatus.FAILED:
        event_type = "acquisition_failed_terminal" if task.attempt_count >= task.max_attempts else "acquisition_failed_retryable"
    else:
        return
    _record_task_lifecycle(config, task, event_type)
