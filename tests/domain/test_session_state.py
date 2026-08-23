"""Session lifecycle tests — real session + metadata store + RAG profile.

The other session/metrics/auto-resume tests were removed: most were mock-
driven auto-resume / metric plumbing. The Postgres state-store tests in
particular used ``FakeConnection`` substitutes and never touched real SQL.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
import uuid

import pytest

from littrace.artifact_registry import ArtifactRecord
from littrace.artifact_store import LocalArtifactStore
from littrace.config import (
    ArtifactStorageConfig,
    LitTraceConfig,
    MetadataStoreConfig,
    StorageConfig,
)
from littrace.models import ChatRequest, LiteratureWorkspace, PaperMetadata
from littrace.runtime.memory import load_session_memory
from littrace.session import (
    append_message,
    create_chat_session,
    delete_chat_session,
    list_chat_sessions,
    load_or_create_session,
    load_workspace,
    save_workspace,
    _session_state_store,
)
from littrace.state_db import SessionStateRecord, state_store_from_config


pytestmark = pytest.mark.domain


_REAL_DSN = "postgresql://littrace:littrace@localhost:5433/littrace"


def test_session_folder_persists_workspace_and_messages(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    workspace = LiteratureWorkspace(papers={"p1": PaperMetadata(paper_id="p1", title="Paper")})

    save_workspace(session, workspace)
    append_message(session, "user", ChatRequest(message="hello"))

    assert session.workspace_path.exists()
    assert session.workspace_dir == session.root / "workspace"
    assert session.structured_documents_dir == session.root / "workspace" / "structured_documents"
    assert session.structured_documents_dir.exists()
    assert session.artifact_index_path.exists()
    assert session.snapshots_dir.exists()
    assert session.evidence_dir.exists()
    assert session.releases_dir.exists()
    # Messages are persisted to the metadata store only — no local messages.jsonl.
    assert load_workspace(session).papers["p1"].title == "Paper"

    summaries = list_chat_sessions(config)
    assert summaries
    assert summaries[0].session_id == session.session_id
    assert summaries[0].topic == "hello"
    assert summaries[0].message_count == 1
    assert summaries[0].paper_count == 0


def test_session_persists_user_scoped_rag_profile(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    config.rag.enabled = True
    config.rag.postgres_dsn = "postgresql://littrace:littrace@localhost:5432/littrace"
    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    workspace.context.filters.topic = "MXene pressure sensor"

    save_workspace(session, workspace, config=config)

    profile_path = session.workspace_dir / "rag" / "profile.json"
    manifest_path = session.workspace_dir / "manifest.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert profile["session_id"] == session.session_id
    assert profile["backend"] == "pgvector"
    assert profile["topic"] == "MXene pressure sensor"
    assert profile["collection_name"].startswith("littrace_")
    assert manifest["rag_enabled"] is True
    assert manifest["rag"]["profile_id"] == profile["profile_id"]
    assert workspace.context.filters.rag_profile["profile_id"] == profile["profile_id"]


def test_session_persists_memory_json(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    workspace.context.filters.pending_intent = {"actions": ["search"], "topic": "MXene"}
    workspace.context.filters.search_mode = "live"

    save_workspace(session, workspace)

    # Memory is persisted to the metadata store only — no local memory.json.
    memory = load_session_memory(session)
    assert memory.session_id == session.session_id
    assert memory.working.pending_intent == {"actions": ["search"], "topic": "MXene"}


def test_cas_strict_for_real_writers(tmp_path):
    """CAS must raise when a zero-writer tries to overwrite a real revision.

    The carve-out allows expected=0 saves against revision<=1 (the
    baseline seed and the _ensure_chat_trail placeholder), but at
    revision>=2 the row reflects a real writer's output. A caller that
    hands in a fresh LiteratureWorkspace() (workspace_revision=0) at
    that point must be rejected — otherwise concurrent saves can
    silently drop a writer's output.
    """
    schema = f"littrace_test_cas_strict_{uuid.uuid4().hex[:8]}"
    config = LitTraceConfig(
        storage=StorageConfig(sessions_dir=tmp_path / "sessions"),
        metadata_store=MetadataStoreConfig(
            backend="postgres",
            postgres_dsn=_REAL_DSN,
            schema_name=schema,
            allow_schema_reset=True,
        ),
    )
    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    workspace.papers["p1"] = PaperMetadata(paper_id="p1", title="Real")
    save_workspace(session, workspace)
    # session is now at revision=1. Manually bump to revision=2 to
    # simulate "a real writer has committed".
    state_store = _session_state_store(session, config)
    state_store.upsert_session_state(
        SessionStateRecord(
            session_id=session.session_id,
            workspace_json={"revision": 2},
            manifest_json={"revision": 2},
            revision=2,
        )
    )
    # A fresh LiteratureWorkspace (default revision=0) must now raise.
    with pytest.raises(RuntimeError, match="Workspace revision mismatch"):
        save_workspace(session, LiteratureWorkspace(), config=config)


def test_chat_trail_fk_survives_background_first(tmp_path):
    """record_lifecycle_event before save_workspace must not raise FK.

    _ensure_chat_trail backfills a revision=0 session_state row so the
    chat_trail FK holds. Background paths (downloads, embedding outbox)
    hit chat_trail before the chat path has saved anything. The
    backfill must succeed silently and leave the row in a state the
    chat path can then bump via save_workspace.
    """
    schema = f"littrace_test_fk_{uuid.uuid4().hex[:8]}"
    config = LitTraceConfig(
        storage=StorageConfig(sessions_dir=tmp_path / "sessions"),
        metadata_store=MetadataStoreConfig(
            backend="postgres",
            postgres_dsn=_REAL_DSN,
            schema_name=schema,
            allow_schema_reset=True,
        ),
    )
    state_store = state_store_from_config(config)
    new_session_id = f"bg-{uuid.uuid4().hex[:12]}"
    # Simulate the background path: append a lifecycle event for a
    # session that has no Postgres row yet.
    state_store.append_chat_event(
        new_session_id,
        {
            "event_id": "bg-1",
            "paper_id": "p1",
            "event_type": "download.completed",
            "occurred_at": "2026-01-01T00:00:00Z",
            "task_id": None,
            "artifact_id": None,
            "payload": {},
        },
    )
    # The chat path then creates a session with this id and saves a
    # workspace — the backfilled row must be overwritten, not conflict.
    session = load_or_create_session(config, new_session_id)
    save_workspace(session, LiteratureWorkspace(), config=config)
    assert load_workspace(session).context.filters.workspace_revision == 1
    schema = f"littrace_test_cas_{uuid.uuid4().hex[:8]}"
    config = LitTraceConfig(
        storage=StorageConfig(sessions_dir=tmp_path / "sessions"),
        metadata_store=MetadataStoreConfig(
            backend="postgres",
            postgres_dsn=_REAL_DSN,
            schema_name=schema,
            allow_schema_reset=True,
        ),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    session = create_chat_session(config)
    writer_a = load_workspace(session)
    writer_b = load_workspace(session)
    writer_a.context.filters.topic = "writer-a"
    writer_b.context.filters.topic = "writer-b"
    start = Barrier(2)

    def persist(workspace: LiteratureWorkspace) -> str | None:
        start.wait()
        try:
            save_workspace(session, workspace, config=config)
        except RuntimeError as exc:
            return str(exc)
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(persist, (writer_a, writer_b)))

    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum("Workspace revision mismatch" in (outcome or "") for outcome in outcomes) == 1

    canonical = load_workspace(session)
    materialized = LiteratureWorkspace.model_validate_json(
        session.workspace_path.read_text(encoding="utf-8")
    )
    manifest = json.loads((session.workspace_dir / "manifest.json").read_text(encoding="utf-8"))
    assert canonical.context.filters.workspace_revision == 2
    assert materialized.context.filters.workspace_revision == 2
    assert materialized.context.filters.topic == canonical.context.filters.topic
    assert canonical.context.filters.topic in {"writer-a", "writer-b"}
    assert manifest["revision"] == 2
    assert len(list(session.snapshots_dir.glob("workspace-*.json"))) == 2


def test_delete_session_reports_object_storage_failures(monkeypatch, tmp_path):
    """Real ``PostgresArtifactRegistry`` + real ``LocalArtifactStore`` with
    a ``monkeypatch``-injected ``OSError`` on the real store's ``delete``
    method. The registry path (``list_for_session`` / ``delete_for_session``)
    runs real SQL; only the store's ``delete`` is faulted.
    """
    schema = f"littrace_test_del_{uuid.uuid4().hex[:8]}"
    config = LitTraceConfig(
        storage=StorageConfig(
            sessions_dir=tmp_path / "sessions",
            metadata_dir=tmp_path / "metadata",
        ),
        metadata_store=MetadataStoreConfig(
            backend="postgres",
            postgres_dsn=_REAL_DSN,
            schema_name=schema,
        ),
        artifact_storage=__import__(
            "littrace.config", fromlist=["ArtifactStorageConfig"]
        ).ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    session = create_chat_session(config)
    save_workspace(session, LiteratureWorkspace(), config=config)

    real_store = LocalArtifactStore(config.artifact_storage.local_root)
    object_key = f"sessions/{session.session_id}/papers/p1/paper.pdf"
    real_store.put_bytes(object_key, b"%PDF-1.4\n", content_type="application/pdf")
    record = ArtifactRecord.from_blob_ref(
        real_store.ref_for_path(
            real_store._path_for_key(object_key),  # type: ignore[attr-defined]
            object_key,
            content_type="application/pdf",
        ),
        artifact_id="paper_pdf:p1",
        session_id=session.session_id,
        kind="paper_pdf",
        paper_id="p1",
    )

    # Real registry wires itself from the same config — the artifact row
    # hits real Postgres on ``localhost:5433``.
    registry = __import__("littrace.artifact_registry", fromlist=["artifact_registry_from_config"]).artifact_registry_from_config(config)
    registry.upsert(record)

    # Fault only the real store's delete call — the registry still runs
    # real SQL against Postgres, the orchestration in ``delete_chat_session``
    # is the real one, and the LocalArtifactStore itself is the real class.
    def boom(_ref):
        raise OSError(f"cannot delete {_ref.object_key}")

    monkeypatch.setattr(real_store, "delete", boom)
    monkeypatch.setattr("littrace.session.artifact_store_from_config", lambda _c: real_store)

    report = delete_chat_session(config, session.session_id)

    assert report.deleted is True
    assert report.object_deleted_count == 0
    assert any(
        failure["artifact_id"] == "paper_pdf:p1"
        for failure in report.object_delete_failures
    )
    assert report.warnings == [f"object_delete_failed:{len(report.object_delete_failures)}"]
    # Real SQL: registry rows are actually gone after delete_for_session.
    assert registry.list_for_session(session_id=session.session_id) == []
