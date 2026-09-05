#!/usr/bin/env python3
"""Run a real topic-based download + parse + RAG E2E."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from littrace.access_layer.cdp import check_cdp_status
from littrace.artifact_registry import artifact_registry_from_config
from littrace.artifact_store import BlobRef, artifact_store_from_config
from littrace.config import ArtifactStorageConfig, LitTraceConfig, MetadataStoreConfig, StorageConfig, load_config
from littrace.downloads import execute_downloads
from littrace.models import (
    AccessType,
    DownloadExecutionItem,
    DownloadExecutionRequest,
    LiteratureWorkspace,
    PaperMetadata,
    PaperSearchRequest,
)
from littrace.rag_jobs import run_pending_embedding_jobs
from littrace.retrieval.rag_profile import load_session_rag_profile
from littrace.retrieval.rag_search import search_session_rag
from littrace.retrieval.search import build_query_variants
from littrace.session import create_chat_session, save_workspace
from littrace.skill_runner import parse_workspace_skill, search_papers_skill
from littrace.state_db import state_store_from_config


def _ensure_bucket(config: LitTraceConfig) -> None:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=config.artifact_storage.endpoint_url,
        region_name=config.artifact_storage.region,
    )
    try:
        client.head_bucket(Bucket=config.artifact_storage.bucket)
    except Exception:
        client.create_bucket(Bucket=config.artifact_storage.bucket)


def _configure_minio_env() -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "littrace")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "littrace123")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def _execute_downloads_with_progress(
    config: LitTraceConfig,
    papers: list[PaperMetadata],
    *,
    session_id: str,
) -> list[DownloadExecutionItem]:
    items: list[DownloadExecutionItem] = []
    for index, paper in enumerate(papers, start=1):
        started = time.perf_counter()
        print(
            json.dumps(
                {
                    "stage": "download_item_start",
                    "ts": _now(),
                    "index": index,
                    "total": len(papers),
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "access_type": str(paper.access_type),
                    "pdf_url": str(paper.pdf_url) if paper.pdf_url else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        result = await execute_downloads(
            config,
            [paper],
            DownloadExecutionRequest(
                paper_ids=[paper.paper_id],
                session_id=session_id,
                dry_run=False,
            ),
        )
        item = result.items[0] if result.items else DownloadExecutionItem(
            paper_id=paper.paper_id,
            action="download",
            status="failed",
            error="No download item was returned.",
        )
        items.append(item)
        print(
            json.dumps(
                {
                    "stage": "download_item_done",
                    "ts": _now(),
                    "paper_id": paper.paper_id,
                    "action": item.action,
                    "status": item.status,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "has_storage_ref": bool(item.storage_ref),
                    "error": item.error,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return items


async def _run(topic: str, limit: int) -> int:
    config = load_config()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    topic_slug = "".join(ch if ch.isalnum() else "_" for ch in topic)[:80] or "topic"
    work_root = Path("/private/tmp/littrace-e2e") / f"{run_id}-{topic_slug}"
    config.storage = StorageConfig(
        paper_library_dir=work_root / "papers",
        metadata_dir=work_root / "metadata",
        cache_dir=work_root / "cache",
        sessions_dir=work_root / "sessions",
    )
    config.artifact_storage = ArtifactStorageConfig(
        backend="s3",
        bucket="littrace-e2e",
        endpoint_url="http://127.0.0.1:9000",
        region="us-east-1",
        path_prefix="e2e",
    )
    config.metadata_store = MetadataStoreConfig(
        backend="postgres",
        postgres_dsn="postgresql://littrace:littrace@localhost:5433/littrace",
        schema_name="littrace_e2e",
    )
    config.rag.enabled = True
    config.rag.backend = "pgvector"
    config.rag.postgres_dsn = "postgresql://littrace:littrace@localhost:5433/littrace"
    config.rag.schema_name = "littrace_rag_e2e"
    config.rag.collection_prefix = "littrace_e2e"
    config.rag.auto_refresh_enabled = False
    # Keep the E2E bounded, while preserving DOI metadata so blocked publisher
    # requests can enter the normal CDP / repository fallback path.
    config.api.request_timeout_seconds = min(config.api.request_timeout_seconds, 12.0)
    config.download_retry.max_attempts = 1
    config.api.enable_live_search = True
    config.cdp_downloader.auto_launch_chrome = True
    config.cdp_downloader.cloudflare_wait_seconds = 12.0
    config.cdp_downloader.user_action_wait_seconds = 8.0
    config.cdp_downloader.command_timeout_seconds = 20.0

    _configure_minio_env()
    _ensure_bucket(config)
    cdp_status = check_cdp_status(config)
    print(
        json.dumps(
            {
                "stage": "cdp_preflight",
                "available": cdp_status.available,
                "cdp_url": cdp_status.cdp_url,
                "browser": cdp_status.browser,
                "error": cdp_status.error,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not config.rag.embedding_base_url or not config.rag.embedding_api_key:
        raise RuntimeError(
            "A real embedding endpoint is required. Configure "
            "LITTRACE_RAG_EMBEDDING_BASE_URL and LITTRACE_RAG_EMBEDDING_API_KEY."
        )

    try:
        request = PaperSearchRequest(
            topic=topic,
            year_min=2021,
            # Search a wider pool, then select exactly `limit` real OA PDFs.
            limit=max(limit * 3, 50),
            wants_recent=True,
            live=True,
            query_variants=build_query_variants(topic),
        )
        search = await search_papers_skill(request, config)
        papers = search.result.papers
        eligible = [
            paper
            for paper in papers
            if paper.access_type == AccessType.OPEN_ACCESS and paper.pdf_url
        ]
        def downloadability_score(paper: PaperMetadata) -> tuple[int, int]:
            url = str(paper.pdf_url or "").lower()
            direct = int("doi.org/" not in url)
            host_priority = 0
            if "arxiv.org" in url or "ace.ewapub.com" in url:
                host_priority = 4
            elif "iopscience.org" in url or "link.springer.com/content/pdf" in url:
                host_priority = 3
            elif "mdpi.com" in url:
                host_priority = 2
            return host_priority, direct

        selected = sorted(eligible, key=downloadability_score, reverse=True)[:limit]
        if len(selected) < limit:
            raise RuntimeError(
                f"Real source search returned only {len(selected)} open PDF candidates; need {limit}."
            )
        print(
            json.dumps(
                {
                    "stage": "search",
                    "searched": len(papers),
                    "eligible": len(eligible),
                    "selected_for_download": len(selected),
                    "open_access": sum(paper.access_type == AccessType.OPEN_ACCESS for paper in papers),
                    "requires_login": sum(paper.access_type == AccessType.REQUIRES_LOGIN for paper in papers),
                    "titles": [
                        {
                            "paper_id": paper.paper_id,
                            "title": paper.title,
                            "access_type": str(paper.access_type),
                            "pdf_url": str(paper.pdf_url) if paper.pdf_url else None,
                        }
                        for paper in papers
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        session = create_chat_session(config)
        workspace = LiteratureWorkspace()
        workspace.papers = {paper.paper_id: paper for paper in selected}
        workspace.context.active_papers = [paper.paper_id for paper in selected]
        workspace.context.filters.research_background = topic
        workspace.context.filters.topic = topic
        save_workspace(session, workspace, config=config)

        download_items = await _execute_downloads_with_progress(
            config,
            selected,
            session_id=session.session_id,
        )
        print(
            json.dumps(
                {
                    "stage": "download",
                    "downloaded_count": sum(item.status == "downloaded" for item in download_items),
                    "requires_login_count": sum(
                        item.action == "cdp_publisher_download" or item.status == "requires_login"
                        for item in download_items
                    ),
                    "cdp_attempted_count": sum(
                        item.action == "cdp_publisher_download" for item in download_items
                    ),
                    "cdp_downloaded_count": sum(
                        item.action == "cdp_publisher_download" and item.status == "downloaded"
                        for item in download_items
                    ),
                    "items": [item.model_dump(mode="json") for item in download_items],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        config.parsing.default_parser = "docling"
        config.parsing.parse_strategy = "text_only"
        config.parsing.docling_workers = 1
        config.parsing.paddleocr.max_pages = 2
        workspace, parse_report = await parse_workspace_skill(workspace, config)
        print(json.dumps({"stage": "parse", **parse_report}, ensure_ascii=False), flush=True)
        save_workspace(session, workspace, config=config)

        state_store = state_store_from_config(config)
        pending_before = (
            state_store.list_async_tasks(
                session_id=session.session_id,
                status="queued",
                kind="embedding_job",
                limit=20,
            )
            if state_store
            else []
        )
        embedding_processed = 0
        for _ in range(6):
            report = await run_pending_embedding_jobs(config)
            embedding_processed += report.processed
            pending_after = (
                state_store.list_async_tasks(
                    session_id=session.session_id,
                    status="queued",
                    kind="embedding_job",
                    limit=20,
                )
                if state_store
                else []
            )
            if state_store is None or not pending_after:
                break

        profile = load_session_rag_profile(session, config=config)
        rag_hits = []
        if profile is not None:
            rag_result = await search_session_rag(config, session, "柔性薄膜压阻传感器", top_k=5)
            rag_hits = rag_result.hits if rag_result is not None else []

        object_store = artifact_store_from_config(config)
        storage_refs = [item.storage_ref for item in download_items if item.storage_ref]
        summary = {
            "stage": "summary",
            "topic": topic,
            "session_id": session.session_id,
            "work_root": str(work_root),
            "searched": len(papers),
            "eligible": len(eligible),
            "downloaded_count": sum(item.status == "downloaded" for item in download_items),
            "requires_login_count": sum(
                item.action == "cdp_publisher_download" or item.status == "requires_login"
                for item in download_items
            ),
            "cdp_attempted_count": sum(
                item.action == "cdp_publisher_download" for item in download_items
            ),
            "cdp_downloaded_count": sum(
                item.action == "cdp_publisher_download" and item.status == "downloaded"
                for item in download_items
            ),
            "storage_refs": storage_refs,
            "object_exists": [
                object_store.exists(BlobRef.model_validate(ref))
                for ref in storage_refs
            ],
            "registry_count": len(
                artifact_registry_from_config(config).list_for_session(
                    session_id=session.session_id
                )
            ),
            "parsed_count": parse_report.get("parsed_count"),
            "pending_embedding_jobs_before": len(pending_before),
            "embedding_processed": embedding_processed,
            "rag_profile_loaded": profile is not None,
            "rag_hits": len(rag_hits),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args.topic, args.limit))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
