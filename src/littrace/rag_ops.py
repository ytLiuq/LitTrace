from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.artifact_store import artifact_store_from_config
from littrace.config import LitTraceConfig
from littrace.state_db import EmbeddingJobQueueReport, EmbeddingJobRecord, state_store_from_config


class RagDoctorCheck(BaseModel):
    name: str
    status: str
    detail: str | None = None
    latency_ms: float | None = None


class RagDoctorReport(BaseModel):
    schema_version: str = "littrace.rag_doctor_report.v1"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    checks: list[RagDoctorCheck] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.status in {"ok", "skipped"} for check in self.checks)


class RagJobsStatusReport(BaseModel):
    schema_version: str = "littrace.rag_jobs_status_report.v1"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    configured: bool = False
    queue: EmbeddingJobQueueReport | None = None
    jobs: list[EmbeddingJobRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def build_rag_jobs_status_report(
    config: LitTraceConfig,
    *,
    status: str | None = None,
    session_id: str | None = None,
    limit: int = 20,
) -> RagJobsStatusReport:
    report = RagJobsStatusReport()
    store = state_store_from_config(config)
    if store is None:
        report.warnings.append("metadata_store.backend is not postgres; embedding queue is unavailable")
        return report
    report.configured = True
    report.queue = store.embedding_job_queue_report()
    report.jobs = store.list_embedding_jobs(status=status, session_id=session_id, limit=limit)
    return report


def requeue_dead_rag_jobs(
    config: LitTraceConfig,
    *,
    session_id: str | None = None,
    limit: int = 20,
) -> int:
    store = state_store_from_config(config)
    if store is None:
        return 0
    return store.requeue_dead_embedding_jobs(session_id=session_id, limit=limit)


def run_rag_doctor(config: LitTraceConfig) -> RagDoctorReport:
    report = RagDoctorReport()
    report.checks.append(_check_metadata_postgres(config))
    report.checks.append(_check_rag_pgvector(config))
    report.checks.append(_check_artifact_storage(config))
    report.checks.append(_check_embedding_endpoint_config(config))
    return report


def _check_metadata_postgres(config: LitTraceConfig) -> RagDoctorCheck:
    if config.metadata_store.backend != "postgres":
        return RagDoctorCheck(
            name="metadata_postgres",
            status="skipped",
            detail="metadata_store.backend is not postgres",
        )
    if not config.metadata_store.postgres_dsn:
        return RagDoctorCheck(
            name="metadata_postgres",
            status="error",
            detail="metadata_store.postgres_dsn is required",
        )
    started = datetime.now(UTC)
    try:
        store = state_store_from_config(config)
        assert store is not None
        store._ensure_schema()
        with store._connect() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception as exc:
        return _failed_check("metadata_postgres", started, exc)
    return _ok_check("metadata_postgres", started, "schema reachable")


def _check_rag_pgvector(config: LitTraceConfig) -> RagDoctorCheck:
    if not config.rag.enabled:
        return RagDoctorCheck(name="rag_pgvector", status="skipped", detail="rag.enabled is false")
    if config.rag.backend != "pgvector":
        return RagDoctorCheck(
            name="rag_pgvector",
            status="skipped",
            detail=f"rag.backend is {config.rag.backend}",
        )
    if not config.rag.postgres_dsn:
        return RagDoctorCheck(
            name="rag_pgvector",
            status="error",
            detail="rag.postgres_dsn is required",
        )
    started = datetime.now(UTC)
    try:
        import psycopg

        with psycopg.connect(config.rag.postgres_dsn) as conn:
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
    except Exception as exc:
        return _failed_check("rag_pgvector", started, exc)
    if row is None:
        return RagDoctorCheck(
            name="rag_pgvector",
            status="error",
            detail="pgvector extension is not installed in the RAG database",
            latency_ms=_elapsed_ms(started),
        )
    return _ok_check("rag_pgvector", started, f"pgvector extension {row[0]}")


def _check_artifact_storage(config: LitTraceConfig) -> RagDoctorCheck:
    started = datetime.now(UTC)
    try:
        store = artifact_store_from_config(config)
        if config.artifact_storage.backend == "local":
            root = config.artifact_storage.local_root
            Path(root).mkdir(parents=True, exist_ok=True)
            return _ok_check("artifact_storage", started, f"local root {root}")
        if config.artifact_storage.backend == "s3":
            client = store.client  # type: ignore[attr-defined]
            client.head_bucket(Bucket=config.artifact_storage.bucket)
            return _ok_check(
                "artifact_storage",
                started,
                f"s3 bucket {config.artifact_storage.bucket} reachable",
            )
    except Exception as exc:
        return _failed_check("artifact_storage", started, exc)
    return RagDoctorCheck(
        name="artifact_storage",
        status="error",
        detail=f"unsupported backend {config.artifact_storage.backend}",
        latency_ms=_elapsed_ms(started),
    )


def _check_embedding_endpoint_config(config: LitTraceConfig) -> RagDoctorCheck:
    if not config.rag.enabled:
        return RagDoctorCheck(
            name="embedding_endpoint",
            status="skipped",
            detail="rag.enabled is false",
        )
    missing = []
    if not config.rag.embedding_base_url:
        missing.append("rag.embedding_base_url")
    if not config.rag.embedding_api_key:
        missing.append("rag.embedding_api_key")
    if config.rag.embedding_dimension <= 0:
        missing.append("rag.embedding_dimension")
    if missing:
        return RagDoctorCheck(
            name="embedding_endpoint",
            status="error",
            detail="missing " + ", ".join(missing),
        )
    return RagDoctorCheck(
        name="embedding_endpoint",
        status="ok",
        detail=f"{config.rag.embedding_provider}:{config.rag.embedding_model} dim={config.rag.embedding_dimension}",
    )


def _ok_check(name: str, started: datetime, detail: str) -> RagDoctorCheck:
    return RagDoctorCheck(name=name, status="ok", detail=detail, latency_ms=_elapsed_ms(started))


def _failed_check(name: str, started: datetime, exc: Exception) -> RagDoctorCheck:
    return RagDoctorCheck(
        name=name,
        status="error",
        detail=f"{exc.__class__.__name__}: {exc}",
        latency_ms=_elapsed_ms(started),
    )


def _elapsed_ms(started: datetime) -> float:
    return round((datetime.now(UTC) - started).total_seconds() * 1000, 2)
