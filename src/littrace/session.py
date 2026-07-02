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


class ChatSessionSummary(BaseModel):
    session_id: str
    root: Path
    updated_at: str
    topic: str = "未命名主题"
    message_count: int = 0
    paper_count: int = 0


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


def list_chat_sessions(config: LitTraceConfig, limit: int = 20) -> list[ChatSessionSummary]:
    root = config.storage.sessions_dir
    if not root.exists():
        return []
    summaries: list[ChatSessionSummary] = []
    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        workspace_path = session_dir / "workspace.json"
        messages_path = session_dir / "messages.jsonl"
        updated_source = messages_path if messages_path.exists() else session_dir
        try:
            updated_at = datetime.fromtimestamp(updated_source.stat().st_mtime).isoformat(
                timespec="minutes"
            )
        except OSError:
            updated_at = "unknown"
        message_count = 0
        if messages_path.exists():
            try:
                message_lines = messages_path.read_text(encoding="utf-8").splitlines()
                message_count = len(message_lines)
            except OSError:
                message_lines = []
                message_count = 0
        else:
            message_lines = []
        paper_count = 0
        topic = "未命名主题"
        for raw_line in message_lines:
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if record.get("role") != "user":
                continue
            content = record.get("content")
            if isinstance(content, dict):
                content = content.get("message")
            if isinstance(content, str) and content.strip():
                topic = _summarize_topic(content)
                break
        if workspace_path.exists():
            try:
                summary_session = ChatSession(
                    session_id=session_dir.name,
                    root=session_dir,
                    workspace_path=workspace_path,
                    messages_path=messages_path,
                    artifacts_dir=session_dir / "artifacts",
                )
                paper_count = len(load_workspace(summary_session).context.active_papers)
            except Exception:
                paper_count = 0
        summaries.append(
            ChatSessionSummary(
                session_id=session_dir.name,
                root=session_dir,
                updated_at=updated_at,
                topic=topic,
                message_count=message_count,
                paper_count=paper_count,
            )
        )
    summaries.sort(key=lambda item: item.updated_at, reverse=True)
    return summaries[:limit]


def _summarize_topic(text: str, max_length: int = 24) -> str:
    cleaned = " ".join(text.split()).strip(" ：:，,。.")
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].rstrip() + "..."
