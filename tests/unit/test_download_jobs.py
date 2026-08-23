from __future__ import annotations

import asyncio

from littrace.config import LitTraceConfig
from littrace.download_jobs import run_pending_download_jobs
from littrace.models import DownloadExecutionItem, DownloadExecutionResult, PaperMetadata
from littrace.state_db import AsyncTaskRecord


class _TaskStore:
    def __init__(self, task: AsyncTaskRecord) -> None:
        self.task = task
        self.updated: list[AsyncTaskRecord] = []

    def claim_pending_async_tasks(self, **kwargs):
        assert kwargs["kind"] == "download_job"
        if self.task.status not in {"queued", "failed"}:
            return []
        self.task.status = "running"
        self.task.attempt_count += 1
        self.task.lease_owner = kwargs["worker_id"]
        return [self.task]

    def update_async_task(self, task):
        self.task = task
        self.updated.append(task.model_copy(deep=True))
        return task


def _job(*, attempt_count: int = 0) -> AsyncTaskRecord:
    paper = PaperMetadata(paper_id="paper-1", title="One")
    return AsyncTaskRecord(
        task_id="download:one",
        session_id="session-1",
        kind="download_job",
        artifact_id="download_batch:one",
        event_type="download_requested",
        status="queued",
        attempt_count=attempt_count,
        result_json={
            "schema_version": "littrace.download_job.v1",
            "command": {
                "paper_ids": ["paper-1"],
                "target": "storage_only",
                "expected_revision": 2,
                "papers": [paper.model_dump(mode="json")],
            },
        },
    )


def test_download_worker_executes_snapshot_and_completes_job() -> None:
    store = _TaskStore(_job())

    async def execute(_config, workspace, request):
        assert workspace.context.active_papers == ["paper-1"]
        assert request.session_id == "session-1"
        assert request.target == "storage_only"
        return DownloadExecutionResult(
            items=[
                DownloadExecutionItem(
                    paper_id="paper-1",
                    action="download",
                    status="downloaded",
                )
            ],
            downloaded_count=1,
        )

    report = asyncio.run(
        run_pending_download_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=execute,
            worker_id="worker-1",
        )
    )

    assert report.processed == 1
    assert report.downloaded == 1
    assert store.task.status == "completed"
    assert store.task.result_json["execution"]["downloaded_count"] == 1
    assert store.task.lease_owner is None


def test_download_worker_persists_retryable_failure() -> None:
    store = _TaskStore(_job())

    async def fail(_config, _workspace, _request):
        raise RuntimeError("object store unavailable")

    report = asyncio.run(
        run_pending_download_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=fail,
            worker_id="worker-1",
        )
    )

    assert report.failed == 1
    assert store.task.status == "failed"
    assert store.task.next_attempt_at is not None
    assert store.task.last_error == "RuntimeError: object store unavailable"
    assert store.task.lease_owner is None
