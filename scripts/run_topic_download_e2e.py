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
from littrace.topic_search import run_topic_search


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
        config.parsing.default_parser = "docling"
        config.parsing.parse_strategy = "text_only"
        config.parsing.docling_workers = 1
        config.parsing.paddleocr.max_pages = 2
        request = PaperSearchRequest(
            topic=topic,
            year_min=2021,
            year_max=2026,
            limit=limit,
            retrieval_limit=min(40, max(limit * 3, limit + 5, 2)),
            source_limit=min(20, max(limit * 2, 3)),
            wants_recent=True,
            live=True,
            query_variants=build_query_variants(topic),
        )
        session = create_chat_session(config)

        def on_progress(payload: dict[str, object]) -> None:
            if payload.get("stage") in {
                "search_finished", "search_unpaywall_finished",
                "download_finished", "download_retry_finished",
                "parse_finished", "parse_retry_finished",
                "rag_finished", "rag_retry_finished",
            }:
                print(json.dumps(payload, ensure_ascii=False), flush=True)

        run = await run_topic_search(
            config,
            session,
            request,
            requested_rag_ready=limit,
            canonical_topic=topic,
            progress_callback=on_progress,
        )
        workspace = run.workspace
        profile = load_session_rag_profile(session, config=config)
        rag_hits = []
        if profile is not None:
            rag_result = await search_session_rag(
                config, session, topic, top_k=5
            )
            rag_hits = rag_result.hits if rag_result is not None else []

        object_store = artifact_store_from_config(config)
        records = artifact_registry_from_config(config).list_for_session(
            session_id=session.session_id
        )
        storage_refs = [
            {
                "backend": record.backend,
                "bucket": record.bucket,
                "object_key": record.object_key,
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
                "content_type": record.content_type,
            }
            for record in records
            if record.kind == "paper_pdf"
        ]
        summary = {
            "stage": "summary",
            "topic": topic,
            "session_id": session.session_id,
            "work_root": str(work_root),
            "status": run.status,
            "searched": run.candidate_count,
            "downloaded_count": run.downloaded_count,
            "parsed_count": run.parsed_count,
            "rag_ready_count": run.rag_ready_count,
            "active_context_count": len(workspace.context.active_papers),
            "requires_login_count": run.requires_login_count,
            "failed_download_count": run.failed_download_count,
            "storage_refs": storage_refs,
            "object_exists": [
                object_store.exists(BlobRef.model_validate(ref))
                for ref in storage_refs
            ],
            "registry_count": len(records),
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
