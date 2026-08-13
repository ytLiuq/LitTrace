"""Run a real single-paper PDF -> figure enrichment -> RAG -> chat E2E."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from littrace.artifact_registry import artifact_registry_from_config
from littrace.artifact_store import BlobRef, artifact_store_from_config
from littrace.config import ArtifactStorageConfig, LitTraceConfig, MetadataStoreConfig, StorageConfig
from littrace.chat import handle_chat
from littrace.evidence.tables import extract_performance_cells
from littrace.llm import chat_completion
from littrace.lifecycle import dispatch_embedding_outbox
from littrace.models import ChatRequest, LiteratureWorkspace, PaperMetadata, coerce_parsed
from littrace.rag_jobs import run_pending_embedding_jobs
from littrace.research_writer import write_evidence_grounded_answer
from littrace.retrieval.rag_search import rag_hits_to_evidence_spans, search_session_rag
from littrace.session import create_chat_session, load_workspace, save_workspace
from littrace.state_db import state_store_from_config


PAPER_ID = "10.1021_acsabm.2c00348"
SOURCE_ROOT = Path("/private/tmp/littrace-e2e/dual-30-12fac33c5c")
SOURCE_SESSION = SOURCE_ROOT / "sessions/20260809-183038-be70219e/workspace.json"
QUESTION = os.environ.get(
    "LITTRACE_E2E_QUESTION",
    "请直接提取 DOI 10.1021/acsabm.2c00348 中三种气凝胶样品的材料组成、制备步骤，以及各自表现出的刚性或弹性。",
)


def configure_minio(config: LitTraceConfig) -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "littrace")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "littrace123")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


async def main() -> None:
    source = json.loads(SOURCE_SESSION.read_text(encoding="utf-8"))
    paper = PaperMetadata.model_validate(source["papers"][PAPER_ID])
    parsed = coerce_parsed(source["parsed_papers"][PAPER_ID])
    figure_root = SOURCE_ROOT / "papers/2022/10.1021_acsabm.2c00348/docling_assets"
    for figure in parsed.figures:
        if isinstance(figure, dict):
            figure_id = str(figure.get("figure_id") or "").removeprefix("F")
            asset_path = figure_root / f"figure-{figure_id}.png"
            if asset_path.is_file():
                figure["asset_path"] = str(asset_path)
    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=Path("/private/tmp/littrace-single-e2e/papers"),
            metadata_dir=Path("/private/tmp/littrace-single-e2e/metadata"),
            cache_dir=Path("/private/tmp/littrace-single-e2e/cache"),
            sessions_dir=Path("/private/tmp/littrace-single-e2e/sessions"),
        ),
        artifact_storage=ArtifactStorageConfig(
            backend="s3", bucket="littrace-e2e", endpoint_url="http://127.0.0.1:9000",
            region="us-east-1", path_prefix="single-paper",
        ),
        metadata_store=MetadataStoreConfig(
            backend="postgres", postgres_dsn="postgresql://littrace:littrace@localhost:5433/littrace",
            schema_name="littrace_e2e",
        ),
    )
    config.rag.enabled = True
    config.rag.backend = "pgvector"
    config.rag.postgres_dsn = "postgresql://littrace:littrace@localhost:5433/littrace"
    config.rag.schema_name = "littrace_rag_e2e"
    config.rag.collection_prefix = "littrace_single_e2e"
    config.rag.embedding_base_url = os.environ["LITTRACE_RAG_EMBEDDING_BASE_URL"]
    config.rag.embedding_api_key = os.environ["LITTRACE_RAG_EMBEDDING_API_KEY"]
    config.rag.embedding_model = os.environ.get("LITTRACE_RAG_EMBEDDING_MODEL", "text-embedding-v3")
    config.rag.embedding_dimension = int(os.environ.get("LITTRACE_RAG_EMBEDDING_DIMENSION", "1024"))
    config.rag.auto_refresh_enabled = False
    config.llm.enabled = True
    config.llm.api_key = os.environ["LITTRACE_REAL_LLM_API_KEY"]
    config.llm.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    config.llm.model = "qwen-plus"
    config.llm.request_timeout_seconds = float(
        os.environ.get("LITTRACE_E2E_LLM_TIMEOUT_SECONDS", "120")
    )
    config.figure_enrichment.enabled = os.environ.get("LITTRACE_E2E_SKIP_VISION") != "1"
    config.figure_enrichment.api_key = os.environ["LITTRACE_REAL_LLM_API_KEY"]
    config.figure_enrichment.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    config.figure_enrichment.model = "qwen-vl-plus"
    # Keep the real E2E bounded while still exercising both vision gates.
    config.figure_enrichment.max_figures_per_job = 1
    config.parsing.docling.describe_figures = False
    configure_minio(config)

    pdf_target = config.storage.paper_library_dir / "2022" / PAPER_ID / "paper.pdf"
    pdf_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_ROOT / "papers/2022/10.1021_acsabm.2c00348/paper.pdf", pdf_target)
    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    workspace.papers = {PAPER_ID: paper}
    workspace.parsed_papers = {PAPER_ID: parsed}
    workspace.context.active_papers = [PAPER_ID]
    workspace.context.filters.topic = paper.title
    workspace.context.filters.search_mode = "live"
    workspace.context.filters.research_background = paper.title
    workspace, performance_report = await extract_performance_cells(workspace, config)
    save_workspace(session, workspace, config=config)

    store = state_store_from_config(config)
    jobs_before = len(store.list_embedding_jobs(session_id=session.session_id, limit=100))
    embedding_reports = []
    for _ in range(4):
        report = await run_pending_embedding_jobs(config, limit=20)
        embedding_reports.append(report.model_dump(mode="json"))
        pending = [
            job for job in store.list_pending_embedding_jobs(limit=100)
            if job.session_id == session.session_id
        ]
        if not pending:
            break

    workspace = load_workspace(session)
    rag = await search_session_rag(config, session, QUESTION, top_k=8)
    chat_response, workspace = await handle_chat(
        ChatRequest(message=QUESTION, session_id=session.session_id), workspace, config
    )
    answer_question = f"{QUESTION}\n只引用已检索证据中的事实。"
    writer_evidence = (
        rag_hits_to_evidence_spans(rag.profile, rag.hits, query=answer_question)
        if rag is not None else []
    )
    writer_reply = await write_evidence_grounded_answer(
        config, answer_question, workspace, rag_evidence=writer_evidence,
    )
    evidence_text = "\n\n".join(
        f"[{hit.section}] {hit.text}" for hit in (rag.hits if rag is not None else [])
    )
    draft_reply = await chat_completion(
        config,
        "你是论文事实核对助手。请只依据下面提供的论文证据，用中文回答问题。"
        "如果证据不足要明确说不足，不要补充证据之外的步骤。\n\n论文证据：\n" + evidence_text,
        answer_question,
    )
    parsed_after = coerce_parsed(workspace.parsed_papers[PAPER_ID])
    figure_statuses = []
    for figure in parsed_after.figures:
        if isinstance(figure, dict):
            figure_statuses.append({
                "figure_id": figure.get("figure_id"),
                "status": figure.get("enrichment_status"),
                "context_confirmed": figure.get("context_confirmed"),
                "confidence": figure.get("context_confirmation_confidence"),
                "summary": figure.get("summary") or figure.get("visual_summary"),
            })
    records = artifact_registry_from_config(config).list_for_session(session_id=session.session_id)
    object_store = artifact_store_from_config(config)
    print(json.dumps({
        "test": "real_single_paper_chat_e2e",
        "acquisition_scope": "seeded_real_pdf: download acquisition is intentionally out of scope",
        "session_id": session.session_id,
        "paper": {"paper_id": paper.paper_id, "title": paper.title, "doi": paper.doi},
        "question": QUESTION,
        "artifacts": [{
            "id": r.artifact_id, "kind": r.kind, "object_key": r.object_key,
            "exists": object_store.exists(BlobRef(
                backend=r.backend, bucket=r.bucket, object_key=r.object_key,
                sha256=r.sha256, size_bytes=r.size_bytes, content_type=r.content_type,
            )),
        } for r in records],
        "embedding_jobs_before": jobs_before,
        "performance_extraction": {
            "cell_count": len(workspace.performance_cells),
            "score": performance_report.score,
            "passed": performance_report.passed,
        },
        "embedding_reports": embedding_reports,
        "embedding_jobs_after": [j.model_dump(mode="json") for j in store.list_embedding_jobs(session_id=session.session_id, limit=100)],
        "rag_hits": [{"paper_id": h.paper_id, "section": h.section, "score": h.score, "text": h.text[:1000], "metadata": h.metadata} for h in (rag.hits if rag else [])],
        "figure_enrichment": figure_statuses,
        "agent": {
            "input": answer_question,
            "action": chat_response.action,
            "router_reply": chat_response.reply,
            "writer_used_llm": writer_reply.used_llm,
            "reply": writer_reply.text,
            "writer_error": writer_reply.error,
            "unpublished_draft_used_llm": draft_reply.used_llm,
            "unpublished_draft": draft_reply.text,
            "unpublished_draft_error": draft_reply.error,
            "citations": [c.model_dump(mode="json") for c in chat_response.citations],
            "warnings": chat_response.warnings,
        },
        "finished_at": datetime.now(UTC).isoformat(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
