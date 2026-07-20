from __future__ import annotations

import json
import os
from hashlib import sha256
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from pydantic import BaseModel

from littrace.config import LitTraceConfig
from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.runtime.memory import build_session_memory, load_session_memory, save_session_memory


class ChatSession(BaseModel):
    session_id: str
    root: Path
    workspace_dir: Path
    workspace_path: Path
    messages_path: Path
    artifacts_dir: Path
    artifact_index_path: Path
    snapshots_dir: Path
    structured_documents_dir: Path
    evidence_dir: Path
    releases_dir: Path
    snapshot_limit: int = 30

    @classmethod
    def from_root(
        cls,
        root: Path,
        session_id: str | None = None,
        snapshot_limit: int = 30,
    ) -> "ChatSession":
        root = Path(root)
        workspace_dir = root / "workspace"
        return cls(
            session_id=session_id or root.name,
            root=root,
            workspace_dir=workspace_dir,
            workspace_path=root / "workspace.json",
            messages_path=root / "messages.jsonl",
            artifacts_dir=root / "artifacts",
            artifact_index_path=workspace_dir / "artifact_index.json",
            snapshots_dir=workspace_dir / "snapshots",
            structured_documents_dir=workspace_dir / "structured_documents",
            evidence_dir=workspace_dir / "evidence",
            releases_dir=workspace_dir / "releases",
            snapshot_limit=snapshot_limit,
        )


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
    workspace_dir = root / "workspace"
    artifacts_dir = root / "artifacts"
    structured_documents_dir = workspace_dir / "structured_documents"
    snapshots_dir = workspace_dir / "snapshots"
    (workspace_dir / "evidence").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "releases").mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    structured_documents_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    session = ChatSession.from_root(
        root,
        session_id,
        snapshot_limit=config.storage.workspace_snapshot_limit,
    )
    save_workspace(session, LiteratureWorkspace())
    return session


def load_or_create_session(config: LitTraceConfig, session_id: str | None = None) -> ChatSession:
    if session_id:
        root = config.storage.sessions_dir / session_id
        if root.exists():
            return ChatSession.from_root(
                root,
                session_id,
                snapshot_limit=config.storage.workspace_snapshot_limit,
            )
    return create_chat_session(config)


def load_workspace(session: ChatSession) -> LiteratureWorkspace:
    if not session.workspace_path.exists():
        return LiteratureWorkspace()
    raw = session.workspace_path.read_text(encoding="utf-8")
    manifest_path = session.workspace_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        manifest = {}
    expected_hash = manifest.get("workspace_sha256")
    if expected_hash and expected_hash != sha256(raw.encode("utf-8")).hexdigest():
        recovered = _load_latest_snapshot(session)
        if recovered is not None:
            return recovered
        raise ValueError("Workspace integrity check failed and no valid snapshot is available.")
    return LiteratureWorkspace.model_validate_json(raw)


def save_workspace(session: ChatSession, workspace: LiteratureWorkspace) -> None:
    session.root.mkdir(parents=True, exist_ok=True)
    session.workspace_dir.mkdir(parents=True, exist_ok=True)
    session.structured_documents_dir.mkdir(parents=True, exist_ok=True)
    session.snapshots_dir.mkdir(parents=True, exist_ok=True)
    session.evidence_dir.mkdir(parents=True, exist_ok=True)
    session.releases_dir.mkdir(parents=True, exist_ok=True)
    workspace.context.filters.workspace_revision = _next_workspace_revision(session)
    _persist_structured_documents(session, workspace)
    _persist_evidence_and_releases(session, workspace)
    snapshot_path = _persist_workspace_snapshot(session, workspace)
    artifact_index = _build_artifact_index(session, workspace, snapshot_path)
    workspace.context.filters.artifact_index = artifact_index
    memory = build_session_memory(
        workspace,
        session_id=session.session_id,
        artifact_index=artifact_index,
    )
    workspace_json = workspace.model_dump_json(indent=2)
    _atomic_write(session.workspace_path, workspace_json)
    manifest = {
        "schema": "littrace.session_workspace.v2",
        "session_id": session.session_id,
        "revision": workspace.context.filters.workspace_revision,
        "workspace_sha256": sha256(workspace_json.encode("utf-8")).hexdigest(),
        "storage_mode": "session-workspace",
        "rag_enabled": False,
        "workspace_path": str(session.workspace_path),
        "structured_documents_dir": str(session.structured_documents_dir),
        "artifact_index_path": str(session.artifact_index_path),
        "snapshots_dir": str(session.snapshots_dir),
        "structured_document_count": workspace.context.filters.structured_document_count,
        "workspace_snapshot_count": workspace.context.filters.workspace_snapshot_count,
    }
    _atomic_write(
        session.artifact_index_path,
        json.dumps(artifact_index, ensure_ascii=False, indent=2),
    )
    save_session_memory(session, memory)
    # Manifest is the commit marker and is written only after every artifact.
    _atomic_write(
        session.workspace_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )


def append_message(
    session: ChatSession, role: str, payload: ChatRequest | ChatResponse | str
) -> None:
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
                summary_session = ChatSession.from_root(session_dir)
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


def _persist_structured_documents(session: ChatSession, workspace: LiteratureWorkspace) -> None:
    paths: dict[str, str] = {}
    for paper_id, parsed in workspace.parsed_papers.items():
        if not parsed.parsed:
            continue
        if not parsed.structured_document and not parsed.sections and not parsed.tables:
            continue
        target = session.structured_documents_dir / f"{_safe_filename(paper_id)}.json"
        _atomic_write(target, parsed.model_dump_json(indent=2))
        paths[paper_id] = str(target)
    workspace.context.filters.structured_document_count = len(paths)
    workspace.context.filters.structured_document_paths = paths


def _persist_evidence_and_releases(session: ChatSession, workspace: LiteratureWorkspace) -> None:
    _atomic_write(
        session.evidence_dir / "spans.json",
        json.dumps(
            {key: span.model_dump(mode="json") for key, span in workspace.evidence_records.items()},
            ensure_ascii=False,
            indent=2,
        ),
    )
    _atomic_write(
        session.evidence_dir / "claims.json",
        json.dumps(
            [claim.model_dump(mode="json") for claim in workspace.claims],
            ensure_ascii=False,
            indent=2,
        ),
    )
    _atomic_write(
        session.evidence_dir / "verification.json",
        json.dumps(
            [report.model_dump(mode="json") for report in workspace.claim_verification_reports],
            ensure_ascii=False,
            indent=2,
        ),
    )
    for snapshot in workspace.release_snapshots:
        target = session.releases_dir / f"{_safe_filename(snapshot.snapshot_id)}.json"
        _atomic_write(target, snapshot.model_dump_json(indent=2))


def _persist_workspace_snapshot(session: ChatSession, workspace: LiteratureWorkspace) -> Path:
    existing = sorted(session.snapshots_dir.glob("workspace-*.json"))
    limit = max(1, int(getattr(session, "snapshot_limit", 30)))
    for stale in existing[: max(0, len(existing) - limit + 1)]:
        stale.unlink(missing_ok=True)
    workspace.context.filters.workspace_snapshot_count = min(len(existing) + 1, limit)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = session.snapshots_dir / f"workspace-{timestamp}.json"
    _atomic_write(target, workspace.model_dump_json(indent=2))
    return target


def _next_workspace_revision(session: ChatSession) -> int:
    manifest_path = session.workspace_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return int(manifest.get("revision", 0)) + 1
    except (OSError, ValueError, json.JSONDecodeError):
        return 1


def _load_latest_snapshot(session: ChatSession) -> LiteratureWorkspace | None:
    for snapshot in sorted(session.snapshots_dir.glob("workspace-*.json"), reverse=True):
        try:
            return LiteratureWorkspace.model_validate_json(snapshot.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def _atomic_write(path: Path, content: str) -> None:
    """Durably replace one workspace artifact without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _build_artifact_index(
    session: ChatSession,
    workspace: LiteratureWorkspace,
    snapshot_path: Path,
) -> dict[str, object]:
    artifacts: list[dict[str, object]] = [
        {
            "kind": "workspace",
            "id": "current",
            "path": str(session.workspace_path),
            "format": "json",
        },
        {
            "kind": "workspace_snapshot",
            "id": snapshot_path.stem,
            "path": str(snapshot_path),
            "format": "json",
        },
    ]
    if session.messages_path.exists():
        artifacts.append(
            {
                "kind": "messages",
                "id": "messages",
                "path": str(session.messages_path),
                "format": "jsonl",
            }
        )
    for paper_id, path in workspace.context.filters.structured_document_paths.items():
        artifacts.append(
            {
                "kind": "structured_document",
                "id": paper_id,
                "path": path,
                "format": "json",
            }
        )
    if workspace.context.filters.document_report:
        artifacts.append({"kind": "document_report", "id": "latest", "format": "inline"})
    if workspace.context.filters.autonomous_loop_report:
        artifacts.append({"kind": "autonomous_loop_report", "id": "latest", "format": "inline"})
    return {
        "schema": "littrace.session_artifact_index.v1",
        "session_id": session.session_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": artifacts,
        "counts": {
            "artifacts": len(artifacts),
            "structured_documents": workspace.context.filters.structured_document_count,
            "snapshots": workspace.context.filters.workspace_snapshot_count,
        },
    }


def load_artifact_index(session: ChatSession) -> dict[str, object]:
    if not session.artifact_index_path.exists():
        return {}
    try:
        value = json.loads(session.artifact_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_filename(value: str) -> str:
    return (
        "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)[:120]
        or "paper"
    )


def load_memory(session: ChatSession):
    return load_session_memory(session)
