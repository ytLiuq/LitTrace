"""RAG lifecycle tests against real components.

Both tests use the project's real ``PostgresStateStore`` against the live
Postgres on ``localhost:5433`` (the default metadata store DSN), so the
SQL state machine — not a ``FakeStore`` recording method calls — is what
gets exercised.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from littrace.config import LitTraceConfig, StorageConfig
from littrace.models import LiteratureWorkspace, ParsedPaper
from littrace.rag_jobs import _mark_embedding_job_failed
from littrace.retrieval.rag_refresh import refresh_session_rag_index
from littrace.session import create_chat_session
from littrace.state_db import AsyncTaskRecord, PostgresStateStore


pytestmark = pytest.mark.domain


_REAL_DSN = "postgresql://littrace:littrace@localhost:5433/littrace"


def _unique_schema(prefix: str) -> str:
    """Each test run gets its own schema so concurrent test sessions don't
    trip over each other's rows."""
    return f"littrace_test_{prefix}_{uuid.uuid4().hex[:8]}"


def test_refresh_skips_cleanly_when_rag_disabled(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    config.rag.enabled = False
    session = create_chat_session(config)
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": ParsedPaper(
                parsed=True,
                sections=[{"name": "Intro", "text": "MXene sensors are flexible."}],
            )
        }
    )

    _, report = asyncio.run(refresh_session_rag_index(config, session, workspace))

    assert report.skipped is True
    assert report.skip_reason == "rag_disabled_or_unconfigured"
    assert report.warnings == []
    assert workspace.context.filters.rag_refresh_report["skip_reason"] == report.skip_reason


def test_embedding_job_failure_moves_to_dead_after_max_attempts():
    """End-to-end: insert a running job in real Postgres → call the real
    ``_mark_embedding_job_failed`` state machine → re-read the job via real
    SQL → verify the running → dead transition, ``next_attempt_at``
    cleared, ``completed_at`` set, and ``last_error`` preserved."""
    schema = _unique_schema("rag_dead")
    store = PostgresStateStore(_REAL_DSN, schema_name=schema)
    store._ensure_schema()

    config = LitTraceConfig(storage=StorageConfig())
    config.download_retry.max_attempts = 3
    job = AsyncTaskRecord(
        task_id="job1",
        session_id="s1",
        kind="embedding_job",
        profile_id="profile1",
        artifact_id="paper_pdf:p1",
        attempt_count=3,
        status="running",
    )
    store.enqueue_async_task(job)

    _mark_embedding_job_failed(store, job, ValueError("bad pdf"), config)

    rows = store.list_async_tasks(
        session_id="s1", status="dead", kind="embedding_job"
    )
    assert len(rows) == 1
    persisted = rows[0]
    assert persisted.task_id == "job1"
    assert persisted.status == "dead"
    assert persisted.next_attempt_at is None
    assert persisted.completed_at is not None
    assert persisted.last_error == "ValueError: bad pdf"
