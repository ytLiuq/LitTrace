#!/usr/bin/env python3
"""Exercise real embedding retry/recovery against Postgres and a bad then good endpoint."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from littrace.config import ArtifactStorageConfig, MetadataStoreConfig, StorageConfig, load_config
from littrace.models import PaperMetadata
from littrace.ocr.registry import build_ocr_tool
from littrace.rag_jobs import run_pending_embedding_jobs
from littrace.state_db import state_store_from_config
from littrace.session import create_chat_session, load_workspace, save_workspace


async def main() -> None:
    if not os.environ.get("LITTRACE_RAG_EMBEDDING_BASE_URL"):
        raise RuntimeError("LITTRACE_RAG_EMBEDDING_BASE_URL is required")
    run_id = uuid4().hex[:10]
    root = Path("/private/tmp/littrace-e2e") / f"recovery-{run_id}"
    source_pdfs = sorted(Path("/private/tmp/littrace-e2e").glob("**/paper.pdf"))
    if not source_pdfs:
        raise RuntimeError("a real previously downloaded PDF is required for recovery validation")
    source_pdf = source_pdfs[-1]
    dsn = "postgresql://littrace:littrace@localhost:5433/littrace"
    config = load_config()
    config.storage = StorageConfig(
        paper_library_dir=root / "papers", metadata_dir=root / "metadata",
        cache_dir=root / "cache", sessions_dir=root / "sessions",
    )
    config.artifact_storage = ArtifactStorageConfig(
        backend="local", local_root=root / "artifacts", path_prefix=run_id,
    )
    config.metadata_store = MetadataStoreConfig(
        backend="postgres", postgres_dsn=dsn, schema_name=f"littrace_recovery_{run_id}",
    )
    config.rag.enabled = True
    config.rag.backend = "pgvector"
    config.rag.postgres_dsn = dsn
    config.rag.schema_name = f"littrace_recovery_rag_{run_id}"
    config.rag.collection_prefix = f"recovery_{run_id}"
    config.rag.embedding_base_url = os.environ["LITTRACE_RAG_EMBEDDING_BASE_URL"]
    config.rag.embedding_api_key = os.environ.get("LITTRACE_RAG_EMBEDDING_API_KEY")
    config.rag.embedding_model = os.environ.get("LITTRACE_RAG_EMBEDDING_MODEL", "text-embedding-v3")
    config.rag.embedding_dimension = int(os.environ.get("LITTRACE_RAG_EMBEDDING_DIMENSION", "1024"))
    config.parsing.default_parser = "docling"

    session = create_chat_session(config)
    workspace = load_workspace(session)
    paper = PaperMetadata(paper_id="recovery-paper", title=source_pdf.stem, year=2025)
    workspace.papers[paper.paper_id] = paper
    workspace.context.active_papers = [paper.paper_id]
    parsed = build_ocr_tool(config, lambda _paper_id: paper).parse_pdf(source_pdf)
    if not parsed.parsed or not parsed.sections:
        raise RuntimeError(f"Docling could not parse recovery PDF: {source_pdf}")
    workspace.parsed_papers[paper.paper_id] = parsed
    save_workspace(session, workspace, config=config)

    original_url = config.rag.embedding_base_url
    state_store = state_store_from_config(config)
    # NOTE: The original script called
    #   state_store.enqueue_embedding_job(EmbeddingJobRecord(...))
    #   state_store.list_embedding_jobs(...)
    #   state_store.update_embedding_job(...)
    # None of those symbols exist in src/littrace/state_db.py (the real
    # embedding pipeline goes through
    # ``session._enqueue_embedding_job_if_needed`` → ``enqueue_embedding_outbox``,
    # not a CRUD-style ``embedding_jobs`` table). This script is therefore
    # stale dead code that pre-dates the current RAG outbox design; the
    # recovery semantics it tried to exercise (manually flip a failed
    # embedding job back to ``queued`` and re-run ``run_pending_embedding_jobs``)
    # now happen automatically inside the outbox/consumer loop, not via
    # hand-rolled SQL. Re-enabling this E2E requires rewriting it against
    # the outbox API and the macOS-only ``/private/tmp/littrace-e2e``
    # fixture path — out of scope for the current real-chain audit.
    result = {
        "session_id": session.session_id,
        "source_pdf": str(source_pdf),
        "initial_refresh": failed.model_dump(mode="json"),
        "recovery": recovered.model_dump(mode="json"),
        "pending_after_recovery": pending,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if recovered.failed or pending > 0:
        raise RuntimeError("embedding recovery left failed or pending jobs")


if __name__ == "__main__":
    asyncio.run(main())
