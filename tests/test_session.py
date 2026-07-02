from littrace.config import LitTraceConfig, StorageConfig
from littrace.models import ChatRequest, LiteratureWorkspace, PaperMetadata
from littrace.session import (
    append_message,
    create_chat_session,
    list_chat_sessions,
    load_workspace,
    save_workspace,
)


def test_session_folder_persists_workspace_and_messages(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    workspace = LiteratureWorkspace(
        papers={"p1": PaperMetadata(paper_id="p1", title="Paper")}
    )

    save_workspace(session, workspace)
    append_message(session, "user", ChatRequest(message="hello"))

    assert session.workspace_path.exists()
    assert session.messages_path.exists()
    assert load_workspace(session).papers["p1"].title == "Paper"

    summaries = list_chat_sessions(config)
    assert summaries
    assert summaries[0].session_id == session.session_id
    assert summaries[0].topic == "hello"
    assert summaries[0].message_count == 1
    assert summaries[0].paper_count == 0
