import asyncio

from littrace.config import LitTraceConfig, StorageConfig
from littrace.models import (
    AccessType,
    LiteratureWorkspace,
    PaperMetadata,
    PaperSearchRequest,
    PaperSearchResult,
    TopicRetrievalPolicy,
)
from littrace.rag_jobs import (
    _mark_embedding_job_failed,
    iter_sentinel_watchlist_ids,
    iter_workspace_session_ids,
    run_daily_rag_maintenance,
)
from littrace.state_db import EmbeddingJobRecord
from littrace.state_db import ArtifactOutboxRecord
from littrace.lifecycle import dispatch_embedding_outbox
from littrace.artifact_ops import reconcile_session_artifacts
from littrace.artifact_registry import ArtifactRecord
from littrace.session import create_chat_session, save_workspace
from littrace.rag_jobs import run_session_research_background_sync


def test_iterators_find_sessions_and_watchlists(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    sentinel_dir = tmp_path / "sentinel" / "mxene"
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    (sentinel_dir / "watchlist.yaml").write_text("watchlist_id: mxene\n", encoding="utf-8")

    assert iter_workspace_session_ids(tmp_path) == [session.session_id]
    assert iter_sentinel_watchlist_ids(tmp_path) == ["mxene"]


def test_daily_maintenance_runs_watchlists_and_sessions(monkeypatch, tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    config.rag.enabled = True
    config.rag.auto_refresh_enabled = True
    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    save_workspace(session, workspace, config=config)
    sentinel_dir = tmp_path / "sentinel" / "mxene"
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    (sentinel_dir / "watchlist.yaml").write_text("watchlist_id: mxene\n", encoding="utf-8")

    calls: list[str] = []

    async def fake_run_sentinel(config, watchlist_id, topic=None):
        calls.append(f"sentinel:{watchlist_id}")

    async def fake_refresh(config, session_obj, workspace_obj=None):
        calls.append(f"refresh:{session_obj.session_id}")

        class DummyReport:
            warnings = []
            chunk_count = 0
            upserted_count = 0
            skipped = True
            skip_reason = "mock"

        return None, DummyReport()

    monkeypatch.setattr("littrace.rag_jobs.run_sentinel", fake_run_sentinel)
    monkeypatch.setattr("littrace.rag_jobs.refresh_session_rag_index", fake_refresh)
    monkeypatch.setattr("littrace.rag_jobs.iter_rag_session_ids", lambda _config: [session.session_id])
    monkeypatch.setattr("littrace.rag_jobs.reconcile_session_artifacts", lambda *_args, **_kwargs: type("Report", (), {"checked": 0, "missing": 0, "requeued": 0, "warnings": []})())

    report = asyncio.run(run_daily_rag_maintenance(config))

    assert report.sentinel_watchlists == 1
    assert report.sessions_refreshed == 1
    assert report.sessions_skipped == 0
    assert calls == ["sentinel:mxene", f"refresh:{session.session_id}"]


def test_daily_maintenance_skips_sessions_when_profile_auto_refresh_disabled(
    monkeypatch,
    tmp_path,
):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    config.rag.enabled = True
    config.rag.auto_refresh_enabled = False
    session = create_chat_session(config)
    calls: list[str] = []

    async def fake_refresh(config, session_obj, workspace_obj=None):
        calls.append(f"refresh:{session_obj.session_id}")

    monkeypatch.setattr("littrace.rag_jobs.refresh_session_rag_index", fake_refresh)
    monkeypatch.setattr("littrace.rag_jobs.iter_rag_session_ids", lambda _config: [session.session_id])
    monkeypatch.setattr("littrace.rag_jobs.reconcile_session_artifacts", lambda *_args, **_kwargs: type("Report", (), {"checked": 0, "missing": 0, "requeued": 0, "warnings": []})())

    report = asyncio.run(run_daily_rag_maintenance(config))

    assert report.sessions_refreshed == 0
    assert report.sessions_skipped == 1
    assert calls == []


def test_embedding_job_failure_moves_to_dead_after_max_attempts():
    config = LitTraceConfig(storage=StorageConfig())
    config.download_retry.max_attempts = 3
    job = EmbeddingJobRecord(
        job_id="job1",
        profile_id="profile1",
        session_id="s1",
        artifact_id="paper_pdf:p1",
        attempt_count=3,
        status="running",
    )
    updated: list[EmbeddingJobRecord] = []

    class FakeStore:
        def update_embedding_job(self, record):
            updated.append(record.model_copy(deep=True))

    _mark_embedding_job_failed(FakeStore(), job, ValueError("bad pdf"), config)

    assert updated[0].status == "dead"
    assert updated[0].next_attempt_at is None
    assert updated[0].completed_at is not None
    assert updated[0].last_error == "ValueError: bad pdf"


def test_outbox_dispatch_creates_embedding_job_then_completes_outbox(monkeypatch):
    config = LitTraceConfig(storage=StorageConfig())
    outbox = ArtifactOutboxRecord(
        outbox_id="outbox-1", session_id="s1", artifact_id="paper_pdf:p1", content_sha256="sha",
    )
    accepted = []
    completed = []

    class FakeStore:
        def claim_artifact_outbox(self, **_kwargs):
            outbox.status = "running"
            outbox.attempt_count = 1
            return [outbox]
        def enqueue_embedding_job(self, record):
            accepted.append(record)
        def update_artifact_outbox(self, record):
            completed.append(record.model_copy(deep=True))

    monkeypatch.setattr("littrace.lifecycle.state_store_from_config", lambda _config: FakeStore())
    dispatched, failed, warnings = dispatch_embedding_outbox(config)
    assert dispatched == 1
    assert failed == 0
    assert not warnings
    assert accepted[0].artifact_id == "paper_pdf:p1"
    assert completed[0].status == "completed"


def test_background_sync_uses_llm_policy_for_queries_and_download_gate(monkeypatch, tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    config.rag.auto_download_open_access = True
    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    workspace.context.filters.research_background = "柔性薄膜压阻压力传感器的长期稳定性"
    workspace.context.filters.research_background_status = "accepted"
    workspace.context.filters.research_retrieval_policy = TopicRetrievalPolicy(
        canonical_topic="flexible piezoresistive pressure sensor stability",
        query_variants=["flexible piezoresistive pressure sensor stability"],
        required_concept_groups=[["flexible"], ["thin-film"], ["pressure"], ["piezoresistive", "resistive"]],
        excluded_concepts=["capacitive", "optical"],
    )
    captured = {}
    papers = [
        PaperMetadata(paper_id="piezo", title="Flexible thin film piezoresistive pressure sensor with stability", access_type=AccessType.OPEN_ACCESS),
        PaperMetadata(paper_id="capacitive", title="Flexible capacitive pressure sensor", access_type=AccessType.OPEN_ACCESS),
        PaperMetadata(paper_id="optical", title="Flexible optical pressure sensor", access_type=AccessType.OPEN_ACCESS),
    ]

    async def fake_search(request, _config):
        captured["request"] = request
        return type("Search", (), {"result": PaperSearchResult(request=request, papers=papers)})()

    async def fake_download(_config, _workspace, request):
        captured["download_ids"] = request.paper_ids
        return type("Result", (), {"downloaded_count": 0, "requires_login_count": 0})()

    async def fake_refresh(_config, _session, workspace):
        return None, type("Report", (), {"skipped": True})()

    async def fake_run_pending(_config, *, limit=20):
        return type(
            "Report",
            (),
            {"outbox_dispatched": 1, "processed": 1, "failed": 0, "warnings": []},
        )()

    monkeypatch.setattr("littrace.rag_jobs.search_papers_skill", fake_search)
    monkeypatch.setattr("littrace.rag_jobs.execute_downloads_skill", fake_download)
    monkeypatch.setattr("littrace.rag_jobs.refresh_session_rag_index", fake_refresh)
    monkeypatch.setattr("littrace.rag_jobs.run_pending_embedding_jobs", fake_run_pending)

    report = asyncio.run(run_session_research_background_sync(config, session, workspace))

    assert captured["request"].topic == "flexible piezoresistive pressure sensor stability"
    assert captured["request"].query_variants == ["flexible piezoresistive pressure sensor stability"]
    assert captured["download_ids"] == ["piezo"]
    assert workspace.context.active_papers == ["piezo"]
    assert set(workspace.context.excluded_papers) == {"capacitive", "optical"}
    assert report.policy_rejected_count == 2
    assert report.outbox_dispatched == 1
    assert report.embedding_jobs_processed == 1


def test_background_sync_does_not_fallback_to_legacy_heuristic_without_policy(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    workspace.context.filters.research_background = "柔性薄膜压阻压力传感器的长期稳定性"
    workspace.context.filters.research_background_status = "accepted"

    report = asyncio.run(run_session_research_background_sync(config, session, workspace))

    assert report.skipped is True
    assert report.skip_reason == "missing_retrieval_policy"


def test_outbox_dispatch_marks_dead_after_retry_budget(monkeypatch):
    config = LitTraceConfig(storage=StorageConfig())
    config.download_retry.max_attempts = 2
    outbox = ArtifactOutboxRecord(
        outbox_id="outbox-1", session_id="s1", artifact_id="paper_pdf:p1",
        attempt_count=1,
    )
    updated = []

    class FakeStore:
        def claim_artifact_outbox(self, **_kwargs):
            outbox.status = "running"
            outbox.attempt_count = 2
            return [outbox]
        def enqueue_embedding_job(self, _record):
            raise ValueError("queue unavailable")
        def update_artifact_outbox(self, record):
            updated.append(record.model_copy(deep=True))

    monkeypatch.setattr("littrace.lifecycle.state_store_from_config", lambda _config: FakeStore())
    dispatched, failed, warnings = dispatch_embedding_outbox(config)
    assert dispatched == 0
    assert failed == 1
    assert warnings
    assert updated[0].status == "dead"
    assert updated[0].completed_at is not None


def test_reconciliation_records_missing_and_requeues_stale(monkeypatch):
    config = LitTraceConfig(storage=StorageConfig())
    records = [
        ArtifactRecord(artifact_id="paper_pdf:missing", session_id="s1", paper_id="missing", kind="paper_pdf", object_key="missing.pdf", backend="local"),
        ArtifactRecord(artifact_id="paper_pdf:stale", session_id="s1", paper_id="stale", kind="paper_pdf", object_key="stale.pdf", backend="local", sha256="sha", revision="r1"),
    ]
    events = []
    queued = []

    class Registry:
        def list_for_session(self, **_kwargs):
            return records

    class ObjectStore:
        def exists(self, ref):
            return ref.object_key == "stale.pdf"

    class StateStore:
        def list_paper_lifecycle_events(self, _session_id):
            return []

        def list_embedding_jobs(self, **_kwargs):
            return []

    monkeypatch.setattr("littrace.artifact_ops.load_existing_session", lambda *_args: type("Session", (), {"session_id": "s1"})())
    monkeypatch.setattr("littrace.artifact_ops.artifact_registry_from_config", lambda _config: Registry())
    monkeypatch.setattr("littrace.artifact_ops.artifact_store_from_config", lambda _config: ObjectStore())
    monkeypatch.setattr("littrace.artifact_ops.state_store_from_config", lambda _config: StateStore())
    monkeypatch.setattr("littrace.artifact_ops.record_lifecycle_event", lambda *_args, **kwargs: events.append(kwargs))
    monkeypatch.setattr("littrace.artifact_ops.enqueue_embedding_outbox", lambda *_args, **kwargs: queued.append(kwargs))

    report = reconcile_session_artifacts(config, "s1")
    assert (report.checked, report.missing, report.requeued) == (2, 1, 1)
    assert events[0]["event_type"] == "artifact_missing"
    assert events[1]["event_type"] == "embedding_requeued"
    assert queued[0]["artifact_id"] == "paper_pdf:stale"
