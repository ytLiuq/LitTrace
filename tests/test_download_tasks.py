import asyncio

import pytest

from littrace.config import LitTraceConfig, MetadataStoreConfig, StorageConfig
from littrace.download_tasks import (
    DownloadRetryWorker,
    DownloadTask,
    DownloadTaskStatus,
    LocalDownloadTaskStore,
    PostgresDownloadTaskStore,
    download_task_store_from_config,
)
from littrace.models import AccessType


def test_local_download_task_store_lists_failed_retryable_tasks(tmp_path):
    store = LocalDownloadTaskStore(tmp_path / "download_tasks.json")
    task = DownloadTask(
        user_id="u1",
        session_id="s1",
        paper_id="p1",
        access_type=AccessType.OPEN_ACCESS,
        status=DownloadTaskStatus.FAILED,
        attempt_count=1,
        max_attempts=3,
    )

    store.upsert(task)

    retryable = store.list_retryable()
    assert [item.task_id for item in retryable] == [task.task_id]


def test_download_retry_worker_runs_retryable_tasks_once(tmp_path):
    store = LocalDownloadTaskStore(tmp_path / "download_tasks.json")
    task = DownloadTask(
        user_id="u1",
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


def test_download_task_uses_configured_retry_attempts(tmp_path):
    config = LitTraceConfig(
        storage=StorageConfig(metadata_dir=tmp_path, default_user_id="u2")
    )
    config.download_retry.max_attempts = 7

    task = DownloadTask(
        user_id=config.storage.default_user_id,
        session_id="s1",
        paper_id="p1",
        access_type=AccessType.OPEN_ACCESS,
        max_attempts=config.download_retry.max_attempts,
    )

    assert task.max_attempts == 7


def test_download_task_store_factory_uses_local_json_by_default(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(metadata_dir=tmp_path))

    store = download_task_store_from_config(config)

    assert isinstance(store, LocalDownloadTaskStore)


def test_download_task_store_factory_requires_postgres_dsn():
    config = LitTraceConfig(metadata_store=MetadataStoreConfig(backend="postgres"))

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
