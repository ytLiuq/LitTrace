from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata, ParsedPaper
from littrace.runtime.memory import (
    append_episode_from_agent_result,
    build_memory_view,
    build_session_memory,
    load_session_memory,
    save_session_memory,
)
from littrace.runtime.messages import AgentArtifact, AgentRunResult, ReActStep, ReActTrace
from littrace.config import LitTraceConfig, StorageConfig
from littrace.session import create_chat_session


def test_build_session_memory_splits_four_layers():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper 1")],
    )
    workspace.context.filters.pending_intent = {"actions": ["search"]}
    workspace.context.filters.search_mode = "live"
    workspace.context.filters.source_routes = ["OpenAlex"]
    workspace.context.filters.structured_document_paths = {"p1": "/tmp/p1.json"}
    workspace.context.filters.docling_quality_reports = {"p1": {"score": 0.8}}
    workspace.parsed_papers["p1"] = ParsedPaper(
        title="Paper 1",
        parsed=True,
        structured_document={"markdown": "# Paper"},
        sections=[{"name": "Intro", "text": "Body"}],
    )

    memory = build_session_memory(
        workspace,
        session_id="s1",
        artifact_index={
            "artifacts": [{"kind": "workspace_snapshot", "id": "snap1", "path": "/tmp/snap.json"}]
        },
    )

    assert memory.session_id == "s1"
    assert memory.working.pending_intent == {"actions": ["search"]}
    assert memory.document.records[0].content["paper_id"] == "p1"
    assert memory.preference.values["preferred_parser"] == "docling"
    assert memory.episodic.records[0].content["artifact_id"] == "snap1"


def test_build_memory_view_limits_records_and_surfaces_warnings():
    workspace = LiteratureWorkspace()
    workspace.context.filters.pending_intent = {"actions": ["search"]}

    view = build_memory_view(workspace, purpose="synthesis")

    assert view.working.pending_intent == {"actions": ["search"]}
    assert "pending_intent_active" in view.warnings
    assert "empty_active_context" in view.warnings
    assert "no_document_memory" in view.warnings


def test_memory_records_agent_react_trace_and_artifacts():
    result = AgentRunResult(
        agent="retrieval",
        artifacts=[AgentArtifact(kind="paper_search_result", producer="retrieval")],
        react_trace=ReActTrace(
            stop_reason="completed",
            steps=[
                ReActStep(
                    step_index=1,
                    thought="Need papers",
                    action="search_papers",
                    observation="retrieved 3",
                    tool="search_papers",
                    next_action="finish",
                )
            ],
        ),
    )

    memory = append_episode_from_agent_result(build_session_memory(LiteratureWorkspace()), result)

    assert any(record.source == "react:retrieval" for record in memory.episodic.records)
    assert any("agent_artifact" in record.tags for record in memory.episodic.records)


def test_session_memory_can_roundtrip(tmp_path):
    session = create_chat_session(
        LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path / "sessions"))
    )
    memory = build_session_memory(LiteratureWorkspace(), session_id=session.session_id)

    path = save_session_memory(session, memory)
    loaded = load_session_memory(session)

    assert path.exists()
    assert loaded.session_id == session.session_id
