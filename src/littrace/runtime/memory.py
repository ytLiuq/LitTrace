from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from littrace.models import LiteratureWorkspace, coerce_parsed
from littrace.runtime.messages import AgentRunResult, ReActTrace

if TYPE_CHECKING:
    from littrace.session import ChatSession


MemoryKind = Literal["working", "episodic", "document", "preference"]
MemoryScope = Literal["turn", "session", "workspace"]


class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: MemoryKind
    scope: MemoryScope = "session"
    source: str
    content: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    expires_at: str | None = None
    tags: list[str] = Field(default_factory=list)


class WorkingMemory(BaseModel):
    pending_intent: dict[str, Any] | None = None
    active_paper_ids: list[str] = Field(default_factory=list)
    selected_for_download: list[str] = Field(default_factory=list)
    search_mode: str | None = None
    topic: str | None = None
    structured_document_count: int = 0
    workspace_snapshot_count: int = 0


class EpisodicMemory(BaseModel):
    records: list[MemoryRecord] = Field(default_factory=list)


class DocumentMemory(BaseModel):
    records: list[MemoryRecord] = Field(default_factory=list)


class PreferenceMemory(BaseModel):
    records: list[MemoryRecord] = Field(default_factory=list)
    values: dict[str, Any] = Field(default_factory=dict)


class SessionMemory(BaseModel):
    schema_version: str = "littrace.session_memory.v1"
    session_id: str | None = None
    working: WorkingMemory = Field(default_factory=WorkingMemory)
    episodic: EpisodicMemory = Field(default_factory=EpisodicMemory)
    document: DocumentMemory = Field(default_factory=DocumentMemory)
    preference: PreferenceMemory = Field(default_factory=PreferenceMemory)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class MemoryView(BaseModel):
    purpose: str
    working: WorkingMemory
    preferences: dict[str, Any] = Field(default_factory=dict)
    recent_episodes: list[MemoryRecord] = Field(default_factory=list)
    document_refs: list[MemoryRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def build_session_memory(
    workspace: LiteratureWorkspace,
    *,
    session_id: str | None = None,
    artifact_index: dict[str, Any] | None = None,
    agent_results: list[AgentRunResult] | None = None,
) -> SessionMemory:
    memory = SessionMemory(
        session_id=session_id,
        working=_working_memory_from_workspace(workspace),
        document=DocumentMemory(records=_document_records_from_workspace(workspace)),
        preference=PreferenceMemory(
            records=_preference_records_from_workspace(workspace),
            values=_preference_values_from_workspace(workspace),
        ),
        episodic=EpisodicMemory(
            records=[
                *_episode_records_from_artifact_index(artifact_index or {}),
                *_episode_records_from_agent_results(agent_results or []),
            ]
        ),
    )
    return memory


def build_memory_view(
    workspace: LiteratureWorkspace,
    *,
    purpose: str = "planning",
    session_memory: SessionMemory | None = None,
    max_episodes: int = 8,
    max_documents: int = 8,
) -> MemoryView:
    memory = session_memory or build_session_memory(workspace)
    warnings: list[str] = []
    if memory.working.pending_intent:
        warnings.append("pending_intent_active")
    if not memory.working.active_paper_ids and purpose in {"planning", "synthesis", "review"}:
        warnings.append("empty_active_context")
    if purpose in {"synthesis", "review"} and not memory.document.records:
        warnings.append("no_document_memory")
    return MemoryView(
        purpose=purpose,
        working=memory.working,
        preferences=memory.preference.values,
        recent_episodes=memory.episodic.records[-max_episodes:],
        document_refs=memory.document.records[:max_documents],
        warnings=warnings,
    )


def memory_path_for_session(session: "ChatSession") -> Path:
    return session.workspace_dir / "memory.json"


def save_session_memory(session: "ChatSession", memory: SessionMemory) -> Path:
    path = memory_path_for_session(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(memory.model_dump_json(indent=2))
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)
    return path


def load_session_memory(session: "ChatSession") -> SessionMemory:
    path = memory_path_for_session(session)
    if not path.exists():
        return SessionMemory(session_id=session.session_id)
    try:
        return SessionMemory.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return SessionMemory(session_id=session.session_id)


def append_episode_from_agent_result(
    memory: SessionMemory,
    result: AgentRunResult,
) -> SessionMemory:
    memory.episodic.records.extend(_episode_records_from_agent_results([result]))
    memory.generated_at = datetime.now().isoformat(timespec="seconds")
    return memory


def _working_memory_from_workspace(workspace: LiteratureWorkspace) -> WorkingMemory:
    filters = workspace.context.filters
    return WorkingMemory(
        pending_intent=filters.pending_intent,
        active_paper_ids=list(workspace.context.active_papers),
        selected_for_download=list(workspace.context.selected_for_download),
        search_mode=filters.search_mode,
        topic=filters.topic,
        structured_document_count=filters.structured_document_count,
        workspace_snapshot_count=filters.workspace_snapshot_count,
    )


def _document_records_from_workspace(workspace: LiteratureWorkspace) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for paper_id, raw in workspace.parsed_papers.items():
        parsed = coerce_parsed(raw)
        if not parsed.parsed:
            continue
        quality = workspace.context.filters.docling_quality_reports.get(paper_id, {})
        records.append(
            MemoryRecord(
                kind="document",
                scope="workspace",
                source="parsed_papers",
                confidence=float(quality.get("score", 1.0) or 1.0),
                tags=["structured_document", str(parsed.title or paper_id)],
                content={
                    "paper_id": paper_id,
                    "title": parsed.title,
                    "section_count": len(parsed.sections),
                    "table_count": len(parsed.tables),
                    "figure_count": len(parsed.figures),
                    "structured_document_path": workspace.context.filters.structured_document_paths.get(
                        paper_id
                    ),
                    "quality": quality,
                },
            )
        )
    return records


def _preference_records_from_workspace(workspace: LiteratureWorkspace) -> list[MemoryRecord]:
    values = _preference_values_from_workspace(workspace)
    if not values:
        return []
    return [
        MemoryRecord(
            kind="preference",
            scope="session",
            source="workspace_context",
            tags=["user_preference"],
            content=values,
        )
    ]


def _preference_values_from_workspace(workspace: LiteratureWorkspace) -> dict[str, Any]:
    filters = workspace.context.filters
    values: dict[str, Any] = {}
    if workspace.context.visible_to_user is False:
        values["show_context"] = False
    if workspace.context.selected_for_download:
        values["download_selection_mode"] = "selected"
    if filters.search_mode:
        values["last_search_mode"] = filters.search_mode
    if filters.source_routes:
        values["preferred_source_routes"] = list(filters.source_routes)
    if filters.docling_quality_reports:
        values["preferred_parser"] = "docling"
    return values


def _episode_records_from_artifact_index(index: dict[str, Any]) -> list[MemoryRecord]:
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    records: list[MemoryRecord] = []
    for artifact in artifacts[-20:]:
        if not isinstance(artifact, dict):
            continue
        records.append(
            MemoryRecord(
                kind="episodic",
                scope="session",
                source="artifact_index",
                tags=[str(artifact.get("kind") or "artifact")],
                content={
                    "artifact_id": artifact.get("id"),
                    "kind": artifact.get("kind"),
                    "path": artifact.get("path"),
                    "format": artifact.get("format"),
                },
            )
        )
    return records


def _episode_records_from_agent_results(results: list[AgentRunResult]) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for result in results:
        if result.react_trace:
            records.extend(_episode_records_from_react_trace(result.agent, result.react_trace))
        for artifact in result.artifacts:
            records.append(
                MemoryRecord(
                    kind="episodic",
                    scope="turn",
                    source=f"agent:{result.agent}",
                    tags=["agent_artifact", artifact.kind],
                    confidence=1.0 if result.status == "completed" else 0.5,
                    content={
                        "artifact_id": artifact.artifact_id,
                        "kind": artifact.kind,
                        "producer": artifact.producer,
                    },
                )
            )
    return records


def _episode_records_from_react_trace(agent: str, trace: ReActTrace) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for step in trace.steps:
        records.append(
            MemoryRecord(
                kind="episodic",
                scope="turn",
                source=f"react:{agent}",
                tags=["react_step", step.action],
                confidence=1.0 if step.ok else 0.4,
                content={
                    "step_index": step.step_index,
                    "thought": step.thought,
                    "decision": step.decision,
                    "action": step.action,
                    "observation": step.observation,
                    "tool": step.tool,
                    "next_action": step.next_action,
                    "stop_reason": trace.stop_reason,
                },
            )
        )
    return records
