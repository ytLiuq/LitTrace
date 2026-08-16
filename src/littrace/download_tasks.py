from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Awaitable, Callable, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.models import AccessType, PaperMetadata


class DownloadTaskStatus(StrEnum):
    PLANNED = "planned"
    QUEUED = "queued"
    FETCHING_SOURCE = "fetching_source"
    AUTH_REQUIRED = "auth_required"
    SOURCE_RESOLVED = "source_resolved"
    DOWNLOADING = "downloading"
    UPLOADING_TO_OBJECT_STORAGE = "uploading_to_object_storage"
    STORED = "stored"
    VERIFIED = "verified"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DownloadTask(BaseModel):
    task_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    paper_id: str
    source_name: str | None = None
    source_url: str | None = None
    doi: str | None = None
    access_type: AccessType = AccessType.UNAVAILABLE
    requires_login: bool = False
    status: DownloadTaskStatus = DownloadTaskStatus.PLANNED
    attempt_count: int = 0
    max_attempts: int = 3
    retry_after: str | None = None
    last_error: str | None = None
    target_bucket: str | None = None
    target_object_key: str | None = None
    artifact_id: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None

    @classmethod
    def from_paper(
        cls,
        config: LitTraceConfig,
        paper: PaperMetadata,
        *,
        session_id: str,
    ) -> "DownloadTask":
        return cls(
            session_id=session_id,
            paper_id=paper.paper_id,
            source_name=_source_name_for_paper(paper),
            source_url=str(paper.pdf_url or paper.source_urls[0]) if paper.pdf_url or paper.source_urls else None,
            doi=paper.doi,
            access_type=paper.access_type,
            requires_login=paper.access_type == AccessType.REQUIRES_LOGIN,
            status=DownloadTaskStatus.QUEUED,
            max_attempts=config.download_retry.max_attempts,
        )

    @property
    def retryable(self) -> bool:
        if self.status not in {DownloadTaskStatus.FAILED, DownloadTaskStatus.QUEUED}:
            return False
        if self.requires_login and self.status == DownloadTaskStatus.AUTH_REQUIRED:
            return False
        if self.attempt_count >= self.max_attempts:
            return False
        if self.retry_after is None:
            return True
        try:
            return datetime.fromisoformat(self.retry_after) <= datetime.now(UTC)
        except ValueError:
            return True

    def mark(self, status: DownloadTaskStatus, *, error: str | None = None) -> "DownloadTask":
        self.status = status
        self.last_error = error
        self.updated_at = datetime.now(UTC).isoformat()
        if status in {DownloadTaskStatus.STORED, DownloadTaskStatus.VERIFIED}:
            self.completed_at = self.updated_at
        return self

    def schedule_retry(self, base_delay_seconds: float) -> "DownloadTask":
        import random as _random

        base = min(base_delay_seconds * (2 ** max(self.attempt_count - 1, 0)), 3600)
        # Jitter ±20% to break thundering-herd alignment between concurrent
        # retry workers and across many failed papers.
        jitter = base * (0.8 + 0.4 * _random.random())
        delay = int(jitter)
        self.retry_after = (datetime.now(UTC) + timedelta(seconds=delay)).isoformat()
        self.updated_at = datetime.now(UTC).isoformat()
        return self


class DownloadTaskStore(Protocol):
    def upsert(self, task: DownloadTask) -> DownloadTask:
        ...

    def get(self, task_id: str) -> DownloadTask | None:
        ...

    def list_retryable(self, *, limit: int = 20) -> list[DownloadTask]:
        ...

    def list_for_session(self, session_id: str) -> list[DownloadTask]:
        ...


class PostgresDownloadTaskStore:
    def __init__(self, dsn: str, *, schema_name: str = "littrace") -> None:
        self.dsn = dsn
        self.schema_name = _safe_identifier(schema_name)
        self._initialized = False

    def upsert(self, task: DownloadTask) -> DownloadTask:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.download_tasks (
                    task_id, session_id, paper_id, source_name, source_url, doi,
                    access_type, requires_login, status, attempt_count, max_attempts,
                    retry_after, last_error, target_bucket, target_object_key, artifact_id,
                    sha256, size_bytes, created_at, updated_at, completed_at, payload
                )
                VALUES (
                    %(task_id)s, %(session_id)s, %(paper_id)s,
                    %(source_name)s, %(source_url)s, %(doi)s, %(access_type)s,
                    %(requires_login)s, %(status)s, %(attempt_count)s, %(max_attempts)s,
                    %(retry_after)s, %(last_error)s, %(target_bucket)s,
                    %(target_object_key)s, %(artifact_id)s, %(sha256)s, %(size_bytes)s,
                    %(created_at)s, %(updated_at)s, %(completed_at)s, %(payload)s
                )
                ON CONFLICT (session_id, paper_id) DO UPDATE SET
                    source_name = EXCLUDED.source_name,
                    source_url = EXCLUDED.source_url,
                    doi = EXCLUDED.doi,
                    access_type = EXCLUDED.access_type,
                    requires_login = EXCLUDED.requires_login,
                    status = EXCLUDED.status,
                    attempt_count = EXCLUDED.attempt_count,
                    max_attempts = EXCLUDED.max_attempts,
                    retry_after = EXCLUDED.retry_after,
                    last_error = EXCLUDED.last_error,
                    target_bucket = EXCLUDED.target_bucket,
                    target_object_key = EXCLUDED.target_object_key,
                    artifact_id = EXCLUDED.artifact_id,
                    sha256 = EXCLUDED.sha256,
                    size_bytes = EXCLUDED.size_bytes,
                    updated_at = EXCLUDED.updated_at,
                    completed_at = EXCLUDED.completed_at,
                    payload = EXCLUDED.payload
                """,
                _task_row(task),
            )
            conn.commit()
        return task

    def get(self, task_id: str) -> DownloadTask | None:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.download_tasks
                WHERE task_id = %(task_id)s
                """,
                {"task_id": task_id},
            ).fetchone()
        if row is None:
            return None
        return _task_from_payload(row[0])

    def list_retryable(self, *, limit: int = 20) -> list[DownloadTask]:
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload
                FROM {self.schema_name}.download_tasks
                WHERE status IN ('queued', 'failed')
                  AND attempt_count < max_attempts
                  AND (retry_after IS NULL OR retry_after <= now())
                ORDER BY updated_at ASC
                LIMIT %(limit)s
                """,
                {"limit": limit},
            ).fetchall()
        return [_task_from_payload(row[0]) for row in rows]

    def list_for_session(self, session_id: str) -> list[DownloadTask]:
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload FROM {self.schema_name}.download_tasks
                WHERE session_id = %(session_id)s
                ORDER BY updated_at DESC
                """,
                {"session_id": session_id},
            ).fetchall()
        return [_task_from_payload(row[0]) for row in rows]

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._connect() as conn:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.download_tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    source_name TEXT,
                    source_url TEXT,
                    doi TEXT,
                    access_type TEXT NOT NULL,
                    requires_login BOOLEAN NOT NULL DEFAULT FALSE,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    retry_after TIMESTAMPTZ,
                    last_error TEXT,
                    target_bucket TEXT,
                    target_object_key TEXT,
                    artifact_id TEXT,
                    sha256 TEXT,
                    size_bytes BIGINT,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ,
                    payload JSONB NOT NULL,
                    CONSTRAINT download_tasks_session_paper_uk
                        UNIQUE (session_id, paper_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS download_tasks_retry_idx
                ON {self.schema_name}.download_tasks (status, retry_after, updated_at)
                WHERE status IN ('queued', 'failed')
                """
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.download_tasks DROP COLUMN IF EXISTS user_id CASCADE"
            )
            conn.commit()
        self._initialized = True

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "Postgres download task storage requires the optional storage extra: "
                "pip install -e '.[storage]'"
            ) from exc
        return psycopg.connect(self.dsn)


DownloadRetryHandler = Callable[[DownloadTask], Awaitable[DownloadTask]]


class DownloadRetryWorker:
    def __init__(
        self,
        store: DownloadTaskStore,
        handler: DownloadRetryHandler,
        *,
        interval_seconds: float = 30.0,
        batch_size: int = 10,
    ) -> None:
        self.store = store
        self.handler = handler
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout)

    def run_once(self) -> int:
        tasks = self.store.list_retryable(limit=self.batch_size)
        for task in tasks:
            updated = asyncio.run(self.handler(task))
            self.store.upsert(updated)
        return len(tasks)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()
            self._stop_event.wait(self.interval_seconds)


def download_task_store_from_config(config: LitTraceConfig) -> DownloadTaskStore:
    if config.metadata_store.backend != "postgres":
        raise ValueError("metadata_store.backend must be 'postgres' for download tasks.")
    dsn = config.metadata_store.postgres_dsn
    if not dsn:
        raise ValueError("metadata_store.postgres_dsn is required for Postgres task storage.")
    return PostgresDownloadTaskStore(dsn, schema_name=config.metadata_store.schema_name)


def _source_name_for_paper(paper: PaperMetadata) -> str | None:
    if paper.publisher:
        return paper.publisher
    if paper.pdf_url:
        return "pdf_url"
    if paper.source_urls:
        return "source_url"
    if paper.doi:
        return "doi"
    return None


def _safe_identifier(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in value)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"littrace_{cleaned}"
    return cleaned[:63]


def _task_row(task: DownloadTask) -> dict[str, object]:
    row = task.model_dump(mode="json")
    row["access_type"] = str(task.access_type)
    row["status"] = str(task.status)
    try:
        from psycopg.types.json import Jsonb
    except ImportError:
        row["payload"] = json.dumps(task.model_dump(mode="json"))
    else:
        row["payload"] = Jsonb(task.model_dump(mode="json"))
    return row


def _task_from_payload(payload: object) -> DownloadTask:
    if isinstance(payload, str):
        return DownloadTask.model_validate_json(payload)
    return DownloadTask.model_validate(payload)
