from littrace.config import ArtifactStorageConfig, LitTraceConfig, StorageConfig
from littrace.rag_ops import build_rag_jobs_status_report, run_rag_doctor


def test_rag_jobs_status_reports_unconfigured_metadata_store(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path / "sessions"))

    report = build_rag_jobs_status_report(config)

    assert report.configured is False
    assert report.queue is None
    assert report.jobs == []
    assert report.warnings


def test_rag_doctor_checks_local_artifact_storage(tmp_path):
    config = LitTraceConfig(
        storage=StorageConfig(sessions_dir=tmp_path / "sessions"),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )

    report = run_rag_doctor(config)

    checks = {check.name: check for check in report.checks}
    assert checks["metadata_postgres"].status == "skipped"
    assert checks["rag_pgvector"].status == "skipped"
    assert checks["artifact_storage"].status == "ok"
    assert checks["embedding_endpoint"].status == "skipped"
