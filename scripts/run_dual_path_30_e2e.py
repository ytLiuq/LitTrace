#!/usr/bin/env python3
"""Run real 30-paper E2E experiments for direct download and daily_update."""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from littrace.artifact_registry import artifact_registry_from_config
from littrace.artifact_store import BlobRef, artifact_store_from_config
from littrace.config import ArtifactStorageConfig, MetadataStoreConfig, StorageConfig, load_config
from littrace.downloads import execute_downloads
from littrace.models import AccessType, DownloadExecutionRequest, LiteratureWorkspace, PaperSearchRequest
from littrace.rag_jobs import run_daily_rag_maintenance, run_pending_embedding_jobs
from littrace.research_background import assess_research_background, set_workspace_research_background
from littrace.retrieval.search import LiveSearchClient, build_query_variants, filter_papers_by_retrieval_policy
from littrace.session import create_chat_session, load_workspace, save_workspace
from littrace.session_metrics import build_session_knowledge_metrics
from littrace.skill_runner import parse_workspace_skill, search_papers_skill


TOPIC = "flexible piezoresistive pressure sensor"
ROOT = Path("/private/tmp/littrace-e2e") / f"dual-30-{uuid4().hex[:10]}"
RUN_ID = uuid4().hex[:12]


class RunMonitor:
    def __init__(self, path: str):
        self.path = path
        self.started = time.monotonic()
        self.last = {"stage": "startup", "status": "running"}
        self.events_path = ROOT / "run_events.jsonl"
        ROOT.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, status: str = "running", **fields: object) -> None:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "path": self.path,
            "stage": stage,
            "status": status,
            "elapsed_seconds": round(time.monotonic() - self.started, 2),
            **fields,
        }
        self.last = event
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)

    async def heartbeat(self):
        while True:
            await asyncio.sleep(10)
            self.emit("watchdog", "running", current_stage=self.last.get("stage"),
                      current_source=self.last.get("source"), current_variant=self.last.get("variant"))


def configure():
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "littrace")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "littrace123")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    c = load_config()
    c.storage = StorageConfig(
        paper_library_dir=ROOT / "papers", metadata_dir=ROOT / "metadata",
        cache_dir=ROOT / "cache", sessions_dir=ROOT / "sessions",
    )
    c.artifact_storage = ArtifactStorageConfig(
        backend="s3", bucket="littrace-e2e", endpoint_url="http://127.0.0.1:9000",
        region="us-east-1", path_prefix=f"dual-30/{RUN_ID}",
    )
    dsn = "postgresql://littrace:littrace@localhost:5433/littrace"
    c.metadata_store = MetadataStoreConfig(backend="postgres", postgres_dsn=dsn, schema_name=f"littrace_dual_{RUN_ID}")
    c.rag.enabled = True
    c.rag.backend = "pgvector"
    c.rag.postgres_dsn = dsn
    c.rag.schema_name = f"littrace_dual_rag_{RUN_ID}"
    c.rag.collection_prefix = f"dual_{RUN_ID}"
    c.rag.auto_refresh_enabled = True
    c.rag.auto_download_open_access = True
    c.api.enable_live_search = True
    c.cdp_downloader.auto_launch_chrome = True
    c.cdp_downloader.command_timeout_seconds = 120
    c.cdp_downloader.repository_download_timeout_seconds = 120
    c.parsing.default_parser = "docling"
    c.parsing.parse_strategy = "text_only"
    c.parsing.docling_workers = 2
    c.literature_context.active_context_limit = 30
    return c


async def process_embeddings(config, session_id: str):
    total = 0
    failed = 0
    for _ in range(8):
        report = await run_pending_embedding_jobs(config, limit=60)
        total += report.processed
        failed += report.failed
        if not report.job_ids:
            break
    return {"processed": total, "failed": failed}


async def direct_download(config):
    # This is the user-facing "download by topic" flow. An explicitly supplied
    # DOI can still bypass discovery, but a topic batch must use the session's
    # LLM-generated policy before any PDF is selected for acquisition.
    monitor = RunMonitor("user_download")
    heartbeat = asyncio.create_task(monitor.heartbeat())
    monitor.emit("config_ready", "finished")
    session = create_chat_session(config)
    workspace = load_workspace(session)
    monitor.emit("llm_assessment_started")
    assessment = await assess_research_background(
        "近五年柔性薄膜压阻压力传感器的材料、微结构与长期稳定性", config,
    )
    monitor.emit("llm_assessment_finished", "finished", accepted=assessment.accepted)
    if not assessment.accepted or assessment.retrieval_policy is None:
        raise RuntimeError(
            f"topic gate rejected direct topic flow: {assessment.reason or 'missing_retrieval_policy'}"
        )
    set_workspace_research_background(
        workspace,
        assessment.background or TOPIC,
        topic=assessment.topic,
        retrieval_policy=assessment.retrieval_policy,
    )
    workspace.context.filters.year_min = 2021
    save_workspace(session, workspace, config=config)

    policy = assessment.retrieval_policy
    request = PaperSearchRequest(
        topic=policy.canonical_topic or TOPIC,
        year_min=2021,
        limit=150,
        wants_recent=True,
        live=True,
        query_variants=policy.query_variants or build_query_variants(TOPIC),
    )
    monitor.emit("search_started")
    client = LiveSearchClient(config, progress_callback=lambda event: monitor.emit(**event))
    try:
        search_result = await client.fetch(request)
        search_ok = True
    except Exception as exc:
        search_result = None
        search_ok = False
        search_error = f"{exc.__class__.__name__}: {exc}"
    if not search_ok:
        diagnostics = {
            "searched": 0,
            "tool_error": search_error,
            "source_counts": client.diagnostics.source_counts,
            "source_errors": client.diagnostics.errors,
        }
        monitor.emit("search_failed", "failed", error=search_error)
        print(json.dumps({"stage": "direct_search_failed", **diagnostics}, ensure_ascii=False), flush=True)
        return {"path": "user_download", "session_id": session.session_id, "status": "search_failed", "diagnostics": diagnostics}
    monitor.emit("candidate_filter_started")
    accepted, rejected = filter_papers_by_retrieval_policy(search_result.papers, policy)
    accepted_ids = {paper.paper_id for paper in accepted}
    filter_reasons = {
        paper_id: "policy_rejected" for paper_id in rejected
    }
    for paper in accepted:
        if not paper.pdf_url:
            filter_reasons[paper.paper_id] = "no_pdf_url"
        elif paper.access_type != AccessType.OPEN_ACCESS:
            filter_reasons[paper.paper_id] = "not_open_access"
    selected = [
        paper
        for paper in search_result.papers
        if paper.paper_id in accepted_ids
        and paper.access_type == AccessType.OPEN_ACCESS
        and paper.pdf_url
    ][:30]
    diagnostics = {
        "searched": len(search_result.papers),
        "policy_accepted": len(accepted),
        "selected_open_access_pdf": len(selected),
        "filter_counts": {
            reason: sum(value == reason for value in filter_reasons.values())
            for reason in ("policy_rejected", "no_pdf_url", "not_open_access")
        },
        "source_counts": client.diagnostics.source_counts,
    }
    monitor.emit("candidate_filter_finished", "finished", **diagnostics)
    print(json.dumps({"stage": "direct_candidates", **diagnostics}, ensure_ascii=False), flush=True)
    workspace.papers = {paper.paper_id: paper for paper in search_result.papers}
    workspace.context.active_papers = [paper.paper_id for paper in selected]
    workspace.context.filters.search_diagnostics = diagnostics
    save_workspace(session, workspace, config=config)
    if len(selected) < 30:
        monitor.emit(
            "run_failed", "failed", reason="candidate_shortage",
            selected_open_access_pdf=len(selected), required=30,
        )
        heartbeat.cancel()
        return {
            "path": "user_download",
            "session_id": session.session_id,
            "status": "candidate_shortage",
            "diagnostics": diagnostics,
        }
    workspace.parsed_papers = {}
    workspace.context.filters.topic = policy.canonical_topic or TOPIC
    workspace.context.filters.research_retrieval_policy = policy
    save_workspace(session, workspace, config=config)
    monitor.emit("download_started", count=len(selected))
    result = await execute_downloads(
        config, selected,
        DownloadExecutionRequest(paper_ids=[p.paper_id for p in selected], session_id=session.session_id),
    )
    monitor.emit("download_finished", "finished", downloaded=result.downloaded_count,
                 requires_login=result.requires_login_count)
    monitor.emit("parse_started")
    workspace, parse_report = await parse_workspace_skill(workspace, config)
    monitor.emit("parse_finished", "finished", parsed=parse_report.get("parsed_count", 0))
    save_workspace(session, workspace, config=config)
    monitor.emit("embedding_started")
    embedding = await process_embeddings(config, session.session_id)
    monitor.emit("embedding_finished", "finished", **embedding)
    heartbeat.cancel()
    monitor.emit("run_finished", "finished")
    return summarize(config, "user_download", session, result, parse_report, embedding)


async def daily_update(config):
    monitor = RunMonitor("daily_update")
    heartbeat = asyncio.create_task(monitor.heartbeat())
    monitor.emit("daily_update_started")
    session = create_chat_session(config)
    workspace = load_workspace(session)
    monitor.emit("llm_assessment_started")
    assessment = await assess_research_background(
        "近五年柔性薄膜压阻压力传感器的材料、微结构与长期稳定性", config,
    )
    monitor.emit("llm_assessment_finished", "finished", accepted=assessment.accepted)
    if not assessment.accepted or assessment.retrieval_policy is None:
        raise RuntimeError(f"daily topic gate rejected: {assessment.rejection_reason}")
    set_workspace_research_background(
        workspace, assessment.background or TOPIC, topic=assessment.topic,
        retrieval_policy=assessment.retrieval_policy,
    )
    workspace.context.filters.year_min = 2021
    save_workspace(session, workspace, config=config)
    monitor.emit("background_ready", "finished", session_id=session.session_id)
    monitor.emit("daily_search_started")
    # The scheduled runner owns all daily work; this callback exposes its
    # real retrieval progress without changing the production execution path.
    original_sync = __import__("littrace.rag_jobs", fromlist=["run_session_research_background_sync"]).run_session_research_background_sync
    async def monitored_sync(sync_config, sync_session, sync_workspace=None):
        return await original_sync(
            sync_config, sync_session, sync_workspace,
            progress_callback=lambda event: monitor.emit(**event),
        )
    import littrace.rag_jobs as rag_jobs_module
    rag_jobs_module.run_session_research_background_sync = monitored_sync
    daily_report = await run_daily_rag_maintenance(config)
    for session_report in daily_report.session_reports:
        monitor.emit(
            "daily_update_session_finished", "finished",
            searched_count=getattr(session_report, "searched_count", None),
            downloaded_count=getattr(session_report, "downloaded_count", None),
            parsed_count=getattr(session_report, "parsed_count", None),
        )
    monitor.emit(
        "daily_update_finished", "finished",
        sessions_refreshed=daily_report.sessions_refreshed,
        sessions_failed=daily_report.sessions_failed,
        embedding_jobs_processed=daily_report.embedding_jobs_processed,
    )
    monitor.emit("embedding_started")
    embedding = await process_embeddings(config, session.session_id)
    monitor.emit("embedding_finished", "finished", **embedding)
    heartbeat.cancel()
    monitor.emit("run_finished", "finished")
    return summarize(config, "daily_update", session, None, None, embedding, daily_report=daily_report)


def summarize(config, path, session, download, parse_report, embedding, daily_report=None):
    metrics = build_session_knowledge_metrics(config, session.session_id)
    records = artifact_registry_from_config(config).list_for_session(session_id=session.session_id)
    store = artifact_store_from_config(config)
    existing = sum(
        1 for r in records
        if store.exists(BlobRef(
            backend=r.backend, bucket=r.bucket, object_key=r.object_key,
            sha256=r.sha256, size_bytes=r.size_bytes, content_type=r.content_type,
        ))
    )
    return {
        "path": path, "session_id": session.session_id,
        "download": download.model_dump(mode="json") if download else None,
        "daily_report": daily_report.model_dump(mode="json") if daily_report else None,
        "parse": parse_report, "embedding": embedding,
        "object_records": len(records), "objects_present": existing,
        "metrics": metrics.model_dump(mode="json"),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("direct", "daily", "both"), default="both")
    args = parser.parse_args()
    config = configure()
    import boto3
    client = boto3.client("s3", endpoint_url=config.artifact_storage.endpoint_url, region_name=config.artifact_storage.region)
    try:
        client.head_bucket(Bucket=config.artifact_storage.bucket)
    except Exception:
        client.create_bucket(Bucket=config.artifact_storage.bucket)
    results = []
    if args.path in {"direct", "both"}:
        results.append(await direct_download(config))
    if args.path in {"daily", "both"}:
        results.append(await daily_update(config))
    print(json.dumps({"root": str(ROOT), "run_id": RUN_ID, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
