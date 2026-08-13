from littrace.artifact_registry import ArtifactRecord, artifact_registry_from_config
from littrace.artifact_store import BlobRef
from littrace.config import ArtifactStorageConfig, LitTraceConfig, StorageConfig
from littrace.models import LiteratureWorkspace, PaperMetadata, SourceRecord
from littrace.session import create_chat_session, save_workspace
from littrace.models import ParsedPaper
from littrace.session_metrics import _acquisition_metric, build_session_knowledge_metrics
from littrace.state_db import EmbeddingJobRecord, PaperLifecycleEventRecord


def test_session_metrics_reports_readiness_and_acquisition(tmp_path):
    config = LitTraceConfig(
        storage=StorageConfig(
            sessions_dir=tmp_path / "sessions",
            metadata_dir=tmp_path / "metadata",
        ),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    session = create_chat_session(config)
    workspace = LiteratureWorkspace(
        papers={
            "p1": PaperMetadata(paper_id="p1", title="Paper 1"),
            "p2": PaperMetadata(paper_id="p2", title="Paper 2"),
        }
    )
    workspace.context.filters.research_background = "flexible pressure sensor films"
    workspace.context.filters.research_background_status = "accepted"
    workspace.context.filters.valid_candidate_count = 2
    workspace.source_records = {
        "source:p1": SourceRecord(
            source_record_id="source:p1",
            paper_id="p1",
            source_name="openalex",
        ),
        "source:p2": SourceRecord(
            source_record_id="source:p2",
            paper_id="p2",
            source_name="openalex",
        ),
    }
    save_workspace(session, workspace, config=config)

    object_key = f"sessions/{session.session_id}/papers/p1/paper.pdf"
    object_path = config.artifact_storage.local_root / object_key
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(b"%PDF-1.4\npaper")
    artifact_registry_from_config(config).upsert(
        ArtifactRecord.from_blob_ref(
            BlobRef(
                backend="local",
                object_key=object_key,
                content_type="application/pdf",
                size_bytes=14,
                sha256="sha-p1",
            ),
            artifact_id="paper_pdf:p1",
            session_id=session.session_id,
            kind="paper_pdf",
            paper_id="p1",
        )
    )
    artifact_registry_from_config(config).upsert(
        ArtifactRecord.from_blob_ref(
            BlobRef(
                backend="local",
                object_key=f"sessions/{session.session_id}/papers/p2/paper.pdf",
                content_type="application/pdf",
                size_bytes=14,
                sha256="sha-p2",
            ),
            artifact_id="paper_pdf:p2",
            session_id=session.session_id,
            kind="paper_pdf",
            paper_id="p2",
        )
    )

    report = build_session_knowledge_metrics(config, session.session_id)

    assert report.discovery.value == 2
    assert report.discovery.status == "measured"
    assert report.acquisition.value == 0.5
    assert report.readiness == "pdf_ready"
    assert report.rag.stale_count == 1
    assert report.consistency.numerator is not None
    assert report.consistency.denominator is not None
    assert report.consistency.missing_count is not None
    assert report.consistency.numerator + report.consistency.missing_count == report.consistency.denominator
    assert report.consistency.missing_count > 0
    assert report.artifact_audit is not None
    assert report.artifact_audit.missing_object_count > 0


def test_session_metrics_uses_completed_embedding_jobs_for_rag_freshness(monkeypatch, tmp_path):
    config = LitTraceConfig(
        storage=StorageConfig(
            sessions_dir=tmp_path / "sessions",
            metadata_dir=tmp_path / "metadata",
        ),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    session = create_chat_session(config)
    workspace = LiteratureWorkspace(
        papers={"p1": PaperMetadata(paper_id="p1", title="Paper 1")}
    )
    workspace.context.filters.research_background = "flexible pressure sensor films"
    workspace.context.filters.research_background_status = "accepted"
    workspace.context.filters.valid_candidate_count = 1
    save_workspace(session, workspace, config=config)

    object_key = f"sessions/{session.session_id}/papers/p1/paper.pdf"
    object_path = config.artifact_storage.local_root / object_key
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(b"%PDF-1.4\npaper")
    artifact_registry_from_config(config).upsert(
        ArtifactRecord.from_blob_ref(
            BlobRef(
                backend="local",
                object_key=object_key,
                content_type="application/pdf",
                size_bytes=14,
                sha256="sha-p1",
            ),
            artifact_id="paper_pdf:p1",
            session_id=session.session_id,
            kind="paper_pdf",
            paper_id="p1",
        )
    )

    class FakeStateStore:
        def list_embedding_jobs(self, *, status=None, session_id=None, limit=20):
            return [
                EmbeddingJobRecord(
                    job_id="job1",
                    profile_id="profile1",
                    session_id=session.session_id,
                    artifact_id="paper_pdf:p1",
                    content_sha256="sha-p1",
                    status="completed",
                )
            ]

    monkeypatch.setattr(
        "littrace.session_metrics.state_store_from_config",
        lambda _config: FakeStateStore(),
    )

    report = build_session_knowledge_metrics(config, session.session_id)

    assert report.rag.status == "measured"
    assert report.rag.value == 1.0
    assert report.rag.stale_count == 0
    assert report.readiness == "rag_ready"


def test_acquisition_metric_uses_lifecycle_terminal_states():
    events = [
        PaperLifecycleEventRecord(session_id="s1", paper_id="p1", task_id="t1", event_type="acquisition_verified"),
        PaperLifecycleEventRecord(session_id="s1", paper_id="p2", task_id="t2", event_type="acquisition_auth_required"),
        PaperLifecycleEventRecord(session_id="s1", paper_id="p3", task_id="t3", event_type="acquisition_failed_terminal"),
        PaperLifecycleEventRecord(session_id="s1", paper_id="p4", task_id="t4", event_type="acquisition_failed_retryable"),
    ]
    metric, status = _acquisition_metric(events, stored_pdf_count=99, relevant_count=99, truncated=False)
    assert status == "measured"
    assert metric[:3] == (0.3333, 1, 3)
    assert "auth_required=1" in metric[3]


def test_structured_document_bytes_are_stable_across_workspace_saves(tmp_path):
    config = LitTraceConfig(
        storage=StorageConfig(sessions_dir=tmp_path / "sessions"),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    workspace.parsed_papers["p1"] = ParsedPaper(
        title="Paper 1",
        parsed=True,
        structured_document={"z": 1, "a": {"y": 2, "x": 3}},
    )

    save_workspace(session, workspace, config=config)
    first = (session.structured_documents_dir / "p1.json").read_bytes()
    save_workspace(session, workspace, config=config)
    second = (session.structured_documents_dir / "p1.json").read_bytes()

    assert first == second
