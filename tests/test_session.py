import json

from littrace.config import LitTraceConfig, StorageConfig
from littrace.document_composer import build_research_document_report
from littrace.models import (
    ChatRequest,
    EvidenceSpan,
    LiteratureWorkspace,
    PaperMetadata,
    ParsedPaper,
    PerformanceCell,
)
from littrace.evidence.claims import register_evidence
from littrace.session import (
    append_message,
    create_chat_session,
    list_chat_sessions,
    load_workspace,
    load_or_create_session,
    save_workspace,
    load_artifact_index,
)
from littrace.runtime.memory import load_session_memory


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
    assert session.messages_path.exists()
    assert load_workspace(session).papers["p1"].title == "Paper"

    summaries = list_chat_sessions(config)
    assert summaries
    assert summaries[0].session_id == session.session_id
    assert summaries[0].topic == "hello"
    assert summaries[0].message_count == 1
    assert summaries[0].paper_count == 0


def test_session_persists_user_scoped_rag_profile(tmp_path):
    config = LitTraceConfig(
        storage=StorageConfig(sessions_dir=tmp_path, default_user_id="u1")
    )
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

    assert profile["user_id"] == "u1"
    assert profile["session_id"] == session.session_id
    assert profile["backend"] == "pgvector"
    assert profile["topic"] == "MXene pressure sensor"
    assert profile["collection_name"].startswith("littrace_u1_")
    assert manifest["rag_enabled"] is True
    assert manifest["rag"]["profile_id"] == profile["profile_id"]
    assert workspace.context.filters.rag_profile["profile_id"] == profile["profile_id"]


def test_load_or_create_session_recovers_user_id_from_manifest(tmp_path):
    config = LitTraceConfig(
        storage=StorageConfig(sessions_dir=tmp_path, default_user_id="fallback-user")
    )
    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    save_workspace(session, workspace, config=config)

    manifest_path = session.workspace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["user_id"] = "u2"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_or_create_session(config, session.session_id)

    assert loaded.user_id == "u2"
    assert loaded.session_id == session.session_id


def test_session_persists_structured_documents_in_workspace(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    workspace = LiteratureWorkspace(
        papers={"p1": PaperMetadata(paper_id="p1", title="Paper")},
        parsed_papers={
            "p1": ParsedPaper(
                title="Paper",
                parsed=True,
                structured_document={"schema": "test", "markdown": "# Paper"},
            )
        },
    )

    save_workspace(session, workspace)

    structured_path = session.structured_documents_dir / "p1.json"
    assert structured_path.exists()
    assert (session.workspace_dir / "manifest.json").exists()
    assert session.artifact_index_path.exists()
    assert list(session.snapshots_dir.glob("workspace-*.json"))
    artifact_index = load_artifact_index(session)
    assert artifact_index["schema"] == "littrace.session_artifact_index.v1"
    assert any(item["kind"] == "structured_document" for item in artifact_index["artifacts"])
    loaded = load_workspace(session)
    assert loaded.context.filters.structured_document_count == 1
    assert "p1" in loaded.context.filters.structured_document_paths


def test_session_persists_memory_json(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    workspace.context.filters.pending_intent = {"actions": ["search"], "topic": "MXene"}
    workspace.context.filters.search_mode = "live"

    save_workspace(session, workspace)

    memory_path = session.workspace_dir / "memory.json"
    memory = load_session_memory(session)
    assert memory_path.exists()
    assert memory.session_id == session.session_id
    assert memory.working.pending_intent == {"actions": ["search"], "topic": "MXene"}


def test_session_persists_evidence_and_release_snapshots(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    workspace = LiteratureWorkspace(
        papers={"p1": PaperMetadata(paper_id="p1", title="Paper")},
        performance_cells=[
            PerformanceCell(
                paper_id="p1",
                metric="sensitivity",
                value=12.5,
                unit="kPa-1",
                evidence=EvidenceSpan(
                    paper_id="p1", page=1, snippet="Sensitivity reached 12.5 kPa-1."
                ),
            )
        ],
    )
    register_evidence(workspace, [workspace.performance_cells[0].evidence])
    build_research_document_report(workspace, config)

    save_workspace(session, workspace)

    assert (session.evidence_dir / "spans.json").exists()
    assert (session.evidence_dir / "claims.json").exists()
    assert (session.evidence_dir / "verification.json").exists()
    assert list(session.releases_dir.glob("release_*.json"))


def test_session_retains_bounded_snapshots_and_recovers_from_corrupt_workspace(tmp_path):
    config = LitTraceConfig(
        storage=StorageConfig(sessions_dir=tmp_path, workspace_snapshot_limit=2)
    )
    session = create_chat_session(config)
    workspace = LiteratureWorkspace(papers={"p1": PaperMetadata(paper_id="p1", title="Paper")})

    save_workspace(session, workspace)
    save_workspace(session, workspace)
    save_workspace(session, workspace)

    snapshots = list(session.snapshots_dir.glob("workspace-*.json"))
    assert len(snapshots) == 2
    assert workspace.context.filters.workspace_revision >= 4

    session.workspace_path.write_text("{not-json", encoding="utf-8")
    recovered = load_workspace(session)

    assert recovered.papers["p1"].title == "Paper"
