"""Real-Postgres acceptance for App Server domain command transactions."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from littrace.artifact_registry import ArtifactRecord, artifact_registry_from_config
from littrace.codex_runtime.gateway import LitTraceToolGateway
from littrace.config import LitTraceConfig, MetadataStoreConfig
from littrace.download_jobs import run_pending_download_jobs
from littrace.models import (
    DownloadExecutionItem,
    DownloadExecutionResult,
    EvidenceSpan,
    LiteratureWorkspace,
    PaperMetadata,
    ParsedPaper,
    PerformanceCell,
)
from littrace.parse_jobs import ParseExecutionOutput, run_pending_parse_jobs
from littrace.state_db import (
    AgentThreadBindingRecord,
    SessionStateRecord,
    state_store_from_config,
)
from littrace.table_jobs import TableExecutionOutput, run_pending_table_jobs

pytestmark = pytest.mark.domain

_REAL_DSN = "postgresql://littrace:littrace@localhost:5433/littrace"


def test_app_server_commands_commit_cas_audit_and_idempotency(
    monkeypatch,
) -> None:
    schema = f"littrace_test_agent_{uuid.uuid4().hex[:8]}"
    config = LitTraceConfig(
        metadata_store=MetadataStoreConfig(
            backend="postgres",
            postgres_dsn=_REAL_DSN,
            schema_name=schema,
        )
    )
    store = state_store_from_config(config)
    session_id = f"agent-{uuid.uuid4().hex[:8]}"
    thread_id = f"thread-{uuid.uuid4().hex[:8]}"
    initial = LiteratureWorkspace(
        papers={"paper-1": PaperMetadata(paper_id="paper-1", title="Initial")}
    )
    initial.context.active_papers = ["paper-1"]
    initial.context.filters.research_background = "Flexible sensing materials"
    store.upsert_session_state(
        SessionStateRecord(
            session_id=session_id,
            workspace_json=initial.model_dump(mode="json"),
        )
    )
    store.upsert_agent_thread_binding(
        AgentThreadBindingRecord(
            session_id=session_id,
            codex_thread_id=thread_id,
        )
    )
    gateway = LitTraceToolGateway(config, store)

    selection_args = {
        "mode": "replace",
        "paper_ids": ["paper-1"],
        "expected_revision": 0,
        "idempotency_key": "postgres-selection-1",
    }
    selected = asyncio.run(
        gateway.call(
            "set_download_selection",
            selection_args,
            codex_thread_id=thread_id,
        )
    )
    replay = asyncio.run(
        gateway.call(
            "set_download_selection",
            selection_args,
            codex_thread_id=thread_id,
        )
    )

    async def fake_search(request, _config):
        result = LiteratureWorkspace(
            papers={
                "paper-2": PaperMetadata(
                    paper_id="paper-2",
                    title=f"Result for {request.topic}",
                    year=2025,
                )
            }
        )
        result.context.active_papers = ["paper-2"]
        result.context.filters.topic = request.topic
        result.context.filters.search_mode = "mock"
        return result

    monkeypatch.setattr("littrace.workflow.run_search_preview", fake_search)
    searched = asyncio.run(
        gateway.call(
            "search_papers",
            {
                "topic": "MXene sensors",
                "live": False,
                "expected_revision": 1,
                "idempotency_key": "postgres-search-1",
            },
            codex_thread_id=thread_id,
        )
    )
    with pytest.raises(RuntimeError, match="CAS mismatch"):
        asyncio.run(
            gateway.call(
                "enqueue_download",
                {
                    "paper_ids": ["paper-2"],
                    "expected_revision": 1,
                    "idempotency_key": "postgres-download-stale",
                },
                codex_thread_id=thread_id,
            )
        )
    assert store.list_async_tasks(session_id=session_id, kind="download_job") == []

    download_args = {
        "paper_ids": ["paper-2"],
        "target": "storage_only",
        "expected_revision": 2,
        "idempotency_key": "postgres-download-1",
    }
    queued = asyncio.run(
        gateway.call(
            "enqueue_download",
            download_args,
            codex_thread_id=thread_id,
        )
    )
    queue_replay = asyncio.run(
        gateway.call(
            "enqueue_download",
            download_args,
            codex_thread_id=thread_id,
        )
    )

    async def fake_download(_config, workspace, request):
        assert workspace.context.active_papers == ["paper-2"]
        assert request.session_id == session_id
        assert request.target == "storage_only"
        return DownloadExecutionResult(
            items=[
                DownloadExecutionItem(
                    paper_id="paper-2",
                    action="download",
                    status="downloaded",
                )
            ],
            downloaded_count=1,
        )

    worker_report = asyncio.run(
        run_pending_download_jobs(
            config,
            state_store=store,
            executor=fake_download,
            worker_id="domain-worker",
        )
    )
    retry_queued = asyncio.run(
        gateway.call(
            "enqueue_download",
            {
                "paper_ids": ["paper-2"],
                "target": "storage_only",
                "expected_revision": 3,
                "idempotency_key": "postgres-download-retry",
            },
            codex_thread_id=thread_id,
        )
    )

    async def unavailable(_config, _workspace, _request):
        raise RuntimeError("temporary storage outage")

    failed_report = asyncio.run(
        run_pending_download_jobs(
            config,
            state_store=store,
            executor=unavailable,
            worker_id="domain-worker-failure",
        )
    )
    retry_job = next(
        job
        for job in store.list_async_tasks(session_id=session_id, kind="download_job")
        if job.task_id == retry_queued["task_id"]
    )
    assert failed_report.failed == 1
    assert retry_job.status == "failed"
    retry_job.next_attempt_at = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    store.update_async_task(retry_job)
    recovered_report = asyncio.run(
        run_pending_download_jobs(
            config,
            state_store=store,
            executor=fake_download,
            worker_id="domain-worker-recovery",
        )
    )
    source_sha = "d" * 64
    artifact_registry_from_config(config).upsert(
        ArtifactRecord(
            artifact_id="paper_pdf:paper-2",
            session_id=session_id,
            kind="paper_pdf",
            paper_id="paper-2",
            object_key=f"sessions/{session_id}/papers/paper-2/paper.pdf",
            backend="local",
            sha256=source_sha,
        )
    )
    parse_args = {
        "paper_ids": ["paper-2"],
        "parse_strategy": "text_only",
        "expected_revision": 4,
        "idempotency_key": "postgres-parse-1",
    }
    parse_queued = asyncio.run(
        gateway.call(
            "enqueue_parse",
            parse_args,
            codex_thread_id=thread_id,
        )
    )
    parse_replay = asyncio.run(
        gateway.call(
            "enqueue_parse",
            parse_args,
            codex_thread_id=thread_id,
        )
    )

    async def fake_parse(_config, _session_id, _papers, _sources, strategy):
        assert strategy == "text_only"
        return ParseExecutionOutput(
            parsed_papers={
                "paper-2": ParsedPaper(
                    title="Parsed paper",
                    parsed=True,
                    sections=[{"name": "Results", "text": "Sensitivity 12.5 kPa-1"}],
                )
            },
            source_sha256={"paper-2": source_sha},
            report={"parsed_count": 1, "failed_count": 0},
        )

    parse_worker_report = asyncio.run(
        run_pending_parse_jobs(
            config,
            state_store=store,
            executor=fake_parse,
            source_sha_lookup=lambda _session, _paper: source_sha,
            worker_id="domain-parse-worker",
        )
    )
    table_args = {
        "paper_ids": ["paper-2"],
        "expected_revision": 6,
        "idempotency_key": "postgres-table-1",
    }
    table_queued = asyncio.run(
        gateway.call(
            "enqueue_table_extraction",
            table_args,
            codex_thread_id=thread_id,
        )
    )
    table_replay = asyncio.run(
        gateway.call(
            "enqueue_table_extraction",
            table_args,
            codex_thread_id=thread_id,
        )
    )

    async def fake_table(_config, _workspace):
        return TableExecutionOutput(
            performance_cells=[
                PerformanceCell(
                    paper_id="paper-2",
                    metric="sensitivity",
                    value=12.5,
                    unit="kPa-1",
                    evidence=EvidenceSpan(
                        paper_id="paper-2",
                        section="Results",
                        snippet="Sensitivity 12.5 kPa-1",
                    ),
                )
            ],
            structured_artifacts=[
                {
                    "paper_id": "paper-2",
                    "artifact_type": "table",
                    "label": "Table 1",
                    "text": "Sensitivity 12.5 kPa-1",
                }
            ],
            harness={"passed": True, "score": 1.0},
        )

    table_worker_report = asyncio.run(
        run_pending_table_jobs(
            config,
            state_store=store,
            executor=fake_table,
            worker_id="domain-table-worker",
        )
    )

    state = store.get_session_state(session_id)
    binding = store.get_agent_thread_binding(session_id)
    assert state is not None
    assert binding is not None
    canonical = LiteratureWorkspace.model_validate(state.workspace_json)
    events = store.list_chat_events(session_id)

    assert selected["workspace_revision"] == 1
    assert replay["idempotency_reused"] is True
    assert searched["workspace_revision"] == 2
    assert queued["workspace_revision"] == 3
    assert queue_replay["idempotency_reused"] is True
    assert worker_report.processed == 1
    assert worker_report.downloaded == 1
    assert recovered_report.processed == 1
    assert parse_queued["workspace_revision"] == 5
    assert parse_replay["idempotency_reused"] is True
    assert parse_worker_report.processed == 1
    assert parse_worker_report.parsed == 1
    assert table_queued["workspace_revision"] == 7
    assert table_replay["idempotency_reused"] is True
    assert table_worker_report.processed == 1
    assert table_worker_report.performance_cells == 1
    jobs = store.list_async_tasks(session_id=session_id, kind="download_job")
    assert len(jobs) == 2
    assert {job.task_id for job in jobs} == {
        queued["task_id"],
        retry_queued["task_id"],
    }
    assert all(job.status == "completed" for job in jobs)
    recovered_job = next(job for job in jobs if job.task_id == retry_queued["task_id"])
    assert recovered_job.attempt_count == 2
    parse_jobs = store.list_async_tasks(session_id=session_id, kind="parse_job")
    assert len(parse_jobs) == 1
    assert parse_jobs[0].status == "completed"
    table_jobs = store.list_async_tasks(session_id=session_id, kind="table_job")
    assert len(table_jobs) == 1
    assert table_jobs[0].status == "completed"
    assert state.revision == 8
    assert binding.workspace_revision == 8
    assert canonical.context.active_papers == ["paper-2"]
    assert canonical.parsed_papers["paper-2"].parsed is True
    assert canonical.performance_cells[0].metric == "sensitivity"
    assert canonical.context.filters.research_background == "Flexible sensing materials"
    tool_events = [event for event in events if event.get("type") == "agent_tool_committed"]
    assert [event["tool"] for event in tool_events[-6:]] == [
        "set_download_selection",
        "search_papers",
        "enqueue_download",
        "enqueue_download",
        "enqueue_parse",
        "enqueue_table_extraction",
    ]
    assert events[-1]["type"] == "async_workspace_job_committed"
    assert events[-1]["kind"] == "table_job"
