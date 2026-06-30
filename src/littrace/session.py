from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from littrace.config import LitTraceConfig
from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace


class ChatSession(BaseModel):
    session_id: str
    root: Path
    workspace_path: Path
    messages_path: Path
    artifacts_dir: Path


def create_chat_session(config: LitTraceConfig) -> ChatSession:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"{timestamp}-{uuid4().hex[:8]}"
    root = config.storage.sessions_dir / session_id
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    session = ChatSession(
        session_id=session_id,
        root=root,
        workspace_path=root / "workspace.json",
        messages_path=root / "messages.jsonl",
        artifacts_dir=artifacts_dir,
    )
    save_workspace(session, LiteratureWorkspace())
    return session


def load_or_create_session(config: LitTraceConfig, session_id: str | None = None) -> ChatSession:
    if session_id:
        root = config.storage.sessions_dir / session_id
        if root.exists():
            return ChatSession(
                session_id=session_id,
                root=root,
                workspace_path=root / "workspace.json",
                messages_path=root / "messages.jsonl",
                artifacts_dir=root / "artifacts",
            )
    return create_chat_session(config)


def load_workspace(session: ChatSession) -> LiteratureWorkspace:
    if not session.workspace_path.exists():
        return LiteratureWorkspace()
    return LiteratureWorkspace.model_validate_json(session.workspace_path.read_text(encoding="utf-8"))


def save_workspace(session: ChatSession, workspace: LiteratureWorkspace) -> None:
    session.root.mkdir(parents=True, exist_ok=True)
    session.workspace_path.write_text(
        workspace.model_dump_json(indent=2),
        encoding="utf-8",
    )


def append_message(session: ChatSession, role: str, payload: ChatRequest | ChatResponse | str) -> None:
    session.root.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        content = payload
    else:
        content = payload.model_dump(mode="json")
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "content": content,
    }
    with session.messages_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
