import asyncio

import pytest

from littrace.config import LitTraceConfig, MetadataStoreConfig
from littrace.download_tasks import (
    DownloadRetryWorker,
    DownloadTask,
    DownloadTaskStatus,
    PostgresDownloadTaskStore,
    download_task_store_from_config,
)
from littrace.models import AccessType


class InMemoryTaskStore:
    def __init__(self) -> None:
        self.tasks: dict[str, DownloadTask] = {}

    def upsert(self, task: DownloadTask) -> DownloadTask:
        self.tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> DownloadTask | None:
        return self.tasks.get(task_id)

    def list_retryable(self, *, limit: int = 20) -> list[DownloadTask]:
        return [task for task in self.tasks.values() if task.retryable][:limit]


def test_retry_worker_runs_retryable_tasks_once():
    store = InMemoryTaskStore()
    task = DownloadTask(
        session_id="s1",
        paper_id="p1",
        access_type=AccessType.OPEN_ACCESS,
        status=DownloadTaskStatus.FAILED,
        attempt_count=1,
        max_attempts=3,
    )

    store.upsert(task)

    async def handler(candidate: DownloadTask) -> DownloadTask:
        await asyncio.sleep(0)
        candidate.mark(DownloadTaskStatus.VERIFIED)
        return candidate

    worker = DownloadRetryWorker(store, handler, interval_seconds=0.01)

    assert worker.run_once() == 1
    assert store.get(task.task_id).status == DownloadTaskStatus.VERIFIED


def test_download_task_uses_configured_retry_attempts():
    config = LitTraceConfig()
    config.download_retry.max_attempts = 7

    task = DownloadTask(
        session_id="s1",
        paper_id="p1",
        access_type=AccessType.OPEN_ACCESS,
        max_attempts=config.download_retry.max_attempts,
    )

    assert task.max_attempts == 7


def test_download_task_store_factory_requires_postgres_dsn():
    config = LitTraceConfig(metadata_store=MetadataStoreConfig(backend="postgres", postgres_dsn=None))

    with pytest.raises(ValueError, match="postgres_dsn"):
        download_task_store_from_config(config)


def test_download_task_store_factory_builds_postgres_store():
    config = LitTraceConfig(
        metadata_store=MetadataStoreConfig(
            backend="postgres",
            postgres_dsn="postgresql://example/littrace",
            schema_name="custom_schema",
        )
    )

    store = download_task_store_from_config(config)

    assert isinstance(store, PostgresDownloadTaskStore)
    assert store.schema_name == "custom_schema"
