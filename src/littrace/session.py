from __future__ import annotations

import json
import os
import shutil
from hashlib import sha256
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from pydantic import BaseModel, Field

from littrace.artifact_registry import ArtifactRecord, artifact_registry_from_config
from littrace.artifact_store import (
    ArtifactKeyContext,
    BlobRef,
    artifact_store_from_config,
    build_artifact_object_key,
)
from littrace.access_layer.paths import target_pdf_path
from littrace.config import LitTraceConfig
from littrace.lifecycle import enqueue_embedding_outbox
from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.state_db import (
    SessionStateRecord,
    SessionStateSnapshotRecord,
    state_store_from_config,
)
from littrace.retrieval.pgvector_store import PgvectorRagStore
from littrace.retrieval.rag_profile import (
    load_session_rag_profile,
    save_session_rag_profile,
    session_rag_profile_path,
)
from littrace.runtime.memory import build_session_memory, load_session_memory


class ChatSession(BaseModel):
    session_id: str
    metadata_store_backend: str = "postgres"
    metadata_postgres_dsn: str | None = None
    metadata_schema_name: str = "littrace"
    root: Path
    workspace_dir: Path
    # workspace_path / artifact_index_path on the JSON mirror
    # are gone in round 3 topic B — Postgres is the source of truth.
    # We still carry artifact_index_path so the artifact_index dict
    # can be persisted to disk for parse outputs, but the JSON
    # workspace.json / artifact_index.json / manifest.json files
    # are no longer written here. messages_path remains as a dead
    # field (never written) for now.
    messages_path: Path
    artifacts_dir: Path
    artifact_index_path: Path
    snapshots_dir: Path
    structured_documents_dir: Path
    evidence_dir: Path
    releases_dir: Path
    rag_dir: Path
    snapshot_limit: int = 30

    @classmethod
    def from_root(
        cls,
        root: Path,
        session_id: str | None = None,
        snapshot_limit: int = 30,
        config: LitTraceConfig | None = None,
    ) -> "ChatSession":
        root = Path(root)
        workspace_dir = root / "workspace"
        return cls(
            session_id=session_id or root.name,
            metadata_store_backend=config.metadata_store.backend if config is not None else "postgres",
            metadata_postgres_dsn=config.metadata_store.postgres_dsn if config is not None else None,
            metadata_schema_name=config.metadata_store.schema_name if config is not None else "littrace",
            root=root,
            workspace_dir=workspace_dir,
            messages_path=root / "messages.jsonl",
            artifacts_dir=root / "artifacts",
            artifact_index_path=workspace_dir / "artifact_index.json",
            snapshots_dir=workspace_dir / "snapshots",
            structured_documents_dir=workspace_dir / "structured_documents",
            evidence_dir=workspace_dir / "evidence",
            releases_dir=workspace_dir / "releases",
            rag_dir=workspace_dir / "rag",
            snapshot_limit=snapshot_limit,
        )


class ChatSessionSummary(BaseModel):
    session_id: str
    root: Path
    updated_at: str
    topic: str = "未命名主题"
    message_count: int = 0
    paper_count: int = 0


def session_path_for(config: LitTraceConfig, session_id: str) -> Path:
    """Reconstruct the on-disk session root from session_id (root_path is no
    longer stored in Postgres; the disk path is deterministic from config +
    session_id)."""
    return config.storage.sessions_dir / session_id


def create_chat_session(config: LitTraceConfig) -> ChatSession:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"{timestamp}-{uuid4().hex[:8]}"
    session = _build_session(config, session_id)
    # Seed a draft placeholder so chat_trail / artifact_registry FK
    # references resolve immediately. _sync_session_state flips this to
    # status='active' the first time save_workspace commits real data.
    state_store = _session_state_store(session, config)
    state_store.upsert_session_state(
        SessionStateRecord(
            session_id=session.session_id,
            revision=0,
            status="draft",
        )
    )
    return session


def _build_session(config: LitTraceConfig, session_id: str) -> ChatSession:
    """Create the on-disk directory layout for ``session_id`` and return a
    ``ChatSession`` handle. No Postgres row is written here — the caller
    (create_chat_session or load_or_create_session) seeds the row via
    save_workspace or via the revision=0 placeholder in
    ``_ensure_chat_trail``.
    """
    root = config.storage.sessions_dir / session_id
    workspace_dir = root / "workspace"
    artifacts_dir = root / "artifacts"
    structured_documents_dir = workspace_dir / "structured_documents"
    snapshots_dir = workspace_dir / "snapshots"
    (workspace_dir / "evidence").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "releases").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "rag").mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    structured_documents_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    return ChatSession.from_root(
        root,
        session_id,
        snapshot_limit=config.storage.workspace_snapshot_limit,
        config=config,
    )


def load_or_create_session(
    config: LitTraceConfig,
    session_id: str | None = None,
) -> ChatSession:
    state_store = state_store_from_config(config)
    if session_id:
        record = state_store.get_session_state(session_id)
        # Archived sessions are treated as if they don't exist — callers
        # fall through to the "no row" branch and create a fresh session
        # rather than reviving a deleted workspace.
        if record is not None and record.status != "archived":
            return ChatSession.from_root(
                session_path_for(config, session_id),
                session_id,
                snapshot_limit=config.storage.workspace_snapshot_limit,
                config=config,
            )
        # Caller asked for a specific id but Postgres has no row (or it
        # was archived) — honor the id (so chat_trail FKs and the
        # caller's headers line up) instead of silently minting a new
        # timestamp-based one.
        session = _build_session(config, session_id)
        # Seed a draft placeholder so chat_trail FK references resolve.
        # Caller's first save will flip status to 'active'.
        state_store.upsert_session_state(
            SessionStateRecord(
                session_id=session.session_id,
                revision=0,
                status="draft",
            )
        )
        return session
    return create_chat_session(config)


def load_existing_session(
    config: LitTraceConfig,
    session_id: str,
) -> ChatSession | None:
    state_store = state_store_from_config(config)
    record = state_store.get_session_state(session_id)
    if record is not None and record.status != "archived":
        return ChatSession.from_root(
            session_path_for(config, session_id),
            session_id,
            snapshot_limit=config.storage.workspace_snapshot_limit,
            config=config,
        )
    return None


def load_workspace(session: ChatSession) -> LiteratureWorkspace:
    record = _load_session_record(session)
    try:
        if record is None:
            return LiteratureWorkspace()
        return LiteratureWorkspace.model_validate(record.workspace_json)
    except Exception as exc:
        raise ValueError(f"Invalid Postgres workspace state for session {session.session_id}") from exc


def save_workspace(
    session: ChatSession,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig | None = None,
) -> None:
    expected_revision = _workspace_revision(workspace)
    state_store = _session_state_store(session, config)
    with state_store.session_write_lock(session.session_id):
        _assert_workspace_revision(
            state_store,
            session_id=session.session_id,
            expected_revision=expected_revision,
        )
        _save_workspace_locked(
            session,
            workspace,
            config=config,
            state_store=state_store,
            expected_revision=expected_revision,
        )


def _ensure_session_dirs(session: ChatSession) -> None:
    """mkdir the on-disk directories that downstream artifact fallouts
    (parse, evidence, rag, releases) write into. The session root is
    also created so that callers running on a fresh checkout do not
    trip a ``FileNotFoundError`` before the first save.
    """
    for path in (
        session.root,
        session.workspace_dir,
        session.structured_documents_dir,
        session.snapshots_dir,
        session.evidence_dir,
        session.releases_dir,
        session.rag_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _bump_workspace_revision(
    workspace: LiteratureWorkspace, expected_revision: int,
) -> int:
    """Set ``workspace.context.filters.workspace_revision`` to
    ``expected_revision + 1`` and return the new value. The increment
    is the only place that touches the revision counter, so the
    caller knows the post-condition exactly.
    """
    new_revision = expected_revision + 1
    workspace.context.filters.workspace_revision = new_revision
    return new_revision


def _compute_workspace_sha256(workspace: LiteratureWorkspace) -> str:
    """Return a stable content hash for the workspace JSON.

    Used to detect a CAS mismatch and to index the snapshot side
    table. The hash covers the full JSON dump including
    ``context.filters.workspace_revision`` so the same logical
    workspace always hashes the same way.
    """
    return sha256(workspace.model_dump_json(indent=2).encode("utf-8")).hexdigest()


def _apply_rag_metadata_to_workspace(
    workspace: LiteratureWorkspace,
    rag_profile: Any,
    config: LitTraceConfig,
) -> None:
    """Copy the relevant fields from a built ``RagProfile`` into the
    workspace's ``context.filters`` so downstream consumers can read
    them from a single source.
    """
    workspace.context.filters.rag_profile = rag_profile.model_dump(mode="json")
    workspace.context.filters.rag_enabled = config.rag.enabled
    workspace.context.filters.rag_backend = config.rag.backend
    workspace.context.filters.rag_source_routes = list(rag_profile.source_routes)
    workspace.context.filters.rag_last_refreshed_at = rag_profile.last_refreshed_at


def _build_save_manifest(
    session: ChatSession,
    workspace: LiteratureWorkspace,
    *,
    artifact_index: dict[str, object],
    memory: Any,
    rag_profile: Any,
    config: LitTraceConfig | None,
    workspace_sha256: str,
    revision: int,
) -> dict[str, object]:
    """Assemble the manifest dict that the JSON mirror used to hold.

    Postgres is the source of truth, but the same dict is embedded
    into ``session_state.manifest_json`` for callers that want a
    single round-trip read.
    """
    return {
        "schema": "littrace.session_workspace.v2",
        "session_id": session.session_id,
        "revision": revision,
        "workspace_sha256": workspace_sha256,
        "storage_mode": "session-workspace",
        "artifact_storage": artifact_index.get("storage"),
        "rag_enabled": bool(rag_profile and config and config.rag.enabled),
        "rag": rag_profile.model_dump(mode="json") if rag_profile is not None else None,
        "rag_profile_path": str(session_rag_profile_path(session)),
        "structured_documents_dir": str(session.structured_documents_dir),
        "artifact_index_path": str(session.artifact_index_path),
        "snapshots_dir": str(session.snapshots_dir),
        "structured_document_count": workspace.context.filters.structured_document_count,
        "workspace_snapshot_count": workspace.context.filters.workspace_snapshot_count,
    }


def _register_postgres_artifacts(
    session: ChatSession,
    workspace: LiteratureWorkspace,
    manifest: dict[str, object],
    artifact_index: dict[str, object],
    memory: Any,
    rag_profile: Any,
    *,
    config: LitTraceConfig | None,
    state_store: Any,
    expected_revision: int,
) -> None:
    """The three post-CAS steps in a fixed order:

      1. ``_sync_session_state`` — the CAS update / INSERT that makes
         the new revision visible to other readers.
      2. ``_capture_workspace_snapshot`` — append the per-revision
         row to the snapshot side table.
      3. ``_register_artifacts`` — register blob references with the
         artifact registry so future RAG rebuilds can find the files.

    Steps 1 and 2 must run in that order — the snapshot must not
    exist for a revision that is not yet visible in
    ``session_state``. Step 3 is last because it is the only one
    that talks to ``artifact_store`` and should not happen for a
    revision that did not actually commit.
    """
    _sync_session_state(
        session,
        workspace,
        manifest,
        artifact_index,
        memory,
        rag_profile,
        state_store=state_store,
        expected_revision=expected_revision,
    )
    _capture_workspace_snapshot(
        session,
        workspace,
        state_store=state_store,
        workspace_sha256=manifest["workspace_sha256"],
    )
    _register_artifacts(session, artifact_index, config, rag_profile=rag_profile)


def _save_workspace_locked(
    session: ChatSession,
    workspace: LiteratureWorkspace,
    *,
    config: LitTraceConfig | None,
    state_store,
    expected_revision: int,
) -> None:
    """Compose the save pipeline from the single-purpose helpers.

    Each step is a one-call expression so a future change to any
    one (e.g. a new ``_build_manifest`` field, or a different
    directory layout) lands in a clearly bounded function.
    """
    _ensure_session_dirs(session)
    _persist_structured_documents(session, workspace)
    _persist_evidence_and_releases(session, workspace)
    rag_profile = (
        save_session_rag_profile(config, session, workspace)
        if config is not None
        else None
    )
    if rag_profile is not None and config is not None:
        _apply_rag_metadata_to_workspace(workspace, rag_profile, config)
    new_revision = _bump_workspace_revision(workspace, expected_revision)
    workspace_sha256 = _compute_workspace_sha256(workspace)
    artifact_index = _build_artifact_index(
        session, workspace, snapshot_path=None, config=config,
    )
    workspace.context.filters.artifact_index = artifact_index
    memory = build_session_memory(
        workspace,
        session_id=session.session_id,
        artifact_index=artifact_index,
    )
    manifest = _build_save_manifest(
        session,
        workspace,
        artifact_index=artifact_index,
        memory=memory,
        rag_profile=rag_profile,
        config=config,
        workspace_sha256=workspace_sha256,
        revision=new_revision,
    )
    _register_postgres_artifacts(
        session,
        workspace,
        manifest,
        artifact_index,
        memory,
        rag_profile,
        config=config,
        state_store=state_store,
        expected_revision=expected_revision,
    )


def append_message(
    session: ChatSession, role: str, payload: ChatRequest | ChatResponse | str
) -> None:
    if isinstance(payload, str):
        content = payload
    else:
        content = payload.model_dump(mode="json")
    state_store = _session_state_store(session)
    now = datetime.now().isoformat(timespec="seconds")
    content_json = content if isinstance(content, dict) else {"message": content}
    content_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    message_id = uuid4().hex
    state_store.append_chat_message(
        session.session_id,
        {
            "message_id": message_id,
            "role": role,
            "content_json": content_json if isinstance(content_json, dict) else {},
            "content_text": content_text,
            "created_at": now,
            "updated_at": now,
        },
    )


def list_chat_sessions(config: LitTraceConfig, limit: int = 20) -> list[ChatSessionSummary]:
    state_store = state_store_from_config(config)
    if state_store is not None:
        summaries: list[ChatSessionSummary] = []
        for record in state_store.list_session_states(limit=limit):
            topic = "未命名主题"
            # topic is extracted in SQL via the ->>'topic' JSONB op so the
            # full workspace_json never crosses the wire for the sidebar.
            if isinstance(record.topic, str) and record.topic.strip():
                topic = record.topic
            if topic == "未命名主题":
                for message in state_store.list_chat_messages(record.session_id):
                    if message.get("role") != "user":
                        continue
                    # Prefer the inner ChatRequest.message field when present;
                    # content_text may be the JSON dump of a dict payload, which
                    # would otherwise become the entire topic string.
                    content_json = message.get("content_json") or {}
                    inner = content_json.get("message") if isinstance(content_json, dict) else None
                    content = (
                        inner
                        if isinstance(inner, str) and inner.strip()
                        else message.get("content_text")
                    )
                    if isinstance(content, str) and content.strip():
                        topic = _summarize_topic(content)
                        break
            message_count = len(state_store.list_chat_messages(record.session_id))
            summaries.append(
                ChatSessionSummary(
                    session_id=record.session_id,
                    root=session_path_for(config, record.session_id),
                    updated_at=record.updated_at,
                    topic=topic,
                    message_count=message_count,
                    # active_paper_count is extracted in SQL via
                    # jsonb_array_length(workspace_json->'context'->'active_papers')
                    # so the sidebar doesn't have to fetch workspace_json.
                    paper_count=record.active_paper_count,
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
        # Stable bytes keep artifact SHA and embedding-job version aligned when
        # workers save the workspace again after refreshing RAG.
        payload = parsed.model_dump(mode="json")
        _atomic_write(
            target,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
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


def _capture_workspace_snapshot(
    session: ChatSession,
    workspace: LiteratureWorkspace,
    *,
    state_store,
    workspace_sha256: str,
) -> None:
    """Append the workspace to the Postgres snapshots side table.

    The on-disk ``<session.root>/workspace/snapshots/workspace-*.json``
    files were dropped in round 3 topic B; this function is the new
    home for per-revision history. ``state_store.upsert_session_snapshot``
    is INSERT ON CONFLICT DO NOTHING, so re-saving the same revision
    is a no-op rather than overwriting prior history.
    """
    snapshot_count = state_store.list_session_snapshots(session.session_id, limit=1000)
    limit = max(1, int(getattr(session, "snapshot_limit", 30)))
    workspace.context.filters.workspace_snapshot_count = min(len(snapshot_count) + 1, limit)
    state_store.upsert_session_snapshot(
        SessionStateSnapshotRecord(
            session_id=session.session_id,
            revision=workspace.context.filters.workspace_revision,
            workspace_sha256=workspace_sha256,
            workspace_json=json.loads(workspace.model_dump_json()),
        )
    )


def _workspace_revision(workspace: LiteratureWorkspace) -> int:
    try:
        revision = int(workspace.context.filters.workspace_revision)
    except (TypeError, ValueError) as exc:
        raise ValueError("workspace_revision must be an integer") from exc
    if revision < 0:
        raise ValueError("workspace_revision must be non-negative")
    return revision


def _assert_workspace_revision(
    state_store,
    *,
    session_id: str,
    expected_revision: int,
) -> None:
    record = state_store.get_session_state(session_id)
    if record is None:
        # Brand-new session with no row yet. expected=0 is the only legal
        # value: the chat path's first save lands revision=1.
        if expected_revision == 0:
            return
        raise RuntimeError(
            f"Workspace revision mismatch for {session_id}: "
            f"expected revision {expected_revision}, got 0"
        )
    # Archived sessions are read-only. The strictness here is deliberate:
    # a delete_chat_session flip to 'archived' is meant to freeze the
    # workspace forever, not just hide it from list_chat_sessions.
    if record.status == "archived":
        raise RuntimeError(
            f"Cannot write to archived session {session_id}"
        )
    # Draft rows are the _ensure_chat_trail placeholder. They have no
    # real workspace yet, so an expected=0 save from the chat path is
    # legitimately the first real writer; let it through and let the
    # caller bump status to 'active' via _sync_session_state.
    if record.status == "draft" and expected_revision == 0:
        return
    if record.revision != expected_revision:
        raise RuntimeError(
            f"Workspace revision mismatch for {session_id}: "
            f"expected revision {expected_revision}, got {record.revision}"
        )


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
    config: LitTraceConfig | None = None,
) -> dict[str, object]:
    store = artifact_store_from_config(config) if config is not None else None
    def artifact_entry(
        *,
        kind: str,
        artifact_id: str,
        path: Path | None = None,
        format: str | None = None,
        filename: str | None = None,
        paper_id: str | None = None,
        revision: str | None = None,
    ) -> dict[str, object]:
        entry: dict[str, object] = {
            "kind": kind,
            "id": artifact_id,
        }
        if paper_id is not None:
            entry["paper_id"] = paper_id
        if revision is not None:
            entry["revision"] = revision
        if path is not None:
            entry["path"] = str(path)
        if format is not None:
            entry["format"] = format
        if store is not None and path is not None and path.exists():
            object_key = build_artifact_object_key(
                config,
                ArtifactKeyContext(
                    session_id=session.session_id,
                    kind=kind,
                    artifact_id=artifact_id,
                    filename=filename or path.name,
                    paper_id=paper_id,
                    revision=revision,
                ),
            )
            ref = store.ref_for_path(
                path,
                object_key,
                content_type=_content_type_for_format(format),
                metadata={
                    "session_id": session.session_id,
                    "kind": kind,
                    "artifact_id": artifact_id,
                },
            )
            entry["storage_ref"] = ref.model_dump(mode="json")
        return entry

    artifacts: list[dict[str, object]] = [
        # workspace.json is no longer on disk (round 3 topic B). The
        # "current" workspace lives in session_state.workspace_json
        # so we record a metadata-only entry that points operators at
        # the Postgres row instead of a file path.
        artifact_entry(
            kind="workspace",
            artifact_id="current",
            path=None,
            format="json",
            filename="current.json",
        ),
    ]
    if snapshot_path is not None:
        artifacts.append(
            artifact_entry(
                kind="workspace_snapshot",
                artifact_id=snapshot_path.stem,
                path=snapshot_path,
                format="json",
                filename=snapshot_path.name,
            )
        )
    if config is not None:
        for paper_id, paper in workspace.papers.items():
            pdf_path = target_pdf_path(config, paper)
            if pdf_path.exists():
                artifacts.append(
                    artifact_entry(
                        kind="paper_pdf",
                        artifact_id=f"paper_pdf:{paper_id}",
                        paper_id=paper_id,
                        path=pdf_path,
                        format="pdf",
                        filename="paper.pdf",
                    )
                )
    if session.messages_path.exists():
        artifacts.append(
            artifact_entry(
                kind="messages",
                artifact_id="messages",
                path=session.messages_path,
                format="jsonl",
                filename="messages.jsonl",
            )
        )
    memory_path = session.workspace_dir / "memory.json"
    if memory_path.exists():
        artifacts.append(
            artifact_entry(
                kind="memory",
                artifact_id="memory",
                path=memory_path,
                format="json",
                filename="memory.json",
            )
        )
    for paper_id, path in workspace.context.filters.structured_document_paths.items():
        artifacts.append(
            artifact_entry(
                kind="structured_document",
                artifact_id=paper_id,
                paper_id=paper_id,
                path=Path(path),
                format="json",
                filename=f"{_safe_filename(paper_id)}.json",
            )
        )
        parsed = workspace.parsed_papers.get(paper_id)
        for figure in getattr(parsed, "figures", []) if parsed is not None else []:
            asset_path = figure.get("asset_path") if isinstance(figure, dict) else None
            if not asset_path or not Path(str(asset_path)).exists():
                continue
            figure_id = str(figure.get("figure_id") or "figure")
            artifacts.append(
                artifact_entry(
                    kind="figure_image",
                    artifact_id=f"figure_image:{paper_id}:{figure_id}",
                    paper_id=paper_id,
                    path=Path(str(asset_path)),
                    format="png",
                    filename=f"figures/{figure_id}.png",
                )
            )
    for paper_id, links in workspace.supplementary_links.items():
        for index, link in enumerate(links):
            path = Path(str(link))
            if not path.exists():
                continue
            artifacts.append(
                artifact_entry(
                    kind="supplementary",
                    artifact_id=f"supplementary:{paper_id}:{index}",
                    paper_id=paper_id,
                    path=path,
                    format=path.suffix.lstrip(".").lower() or "binary",
                    filename=f"supplementary/{path.name}",
                )
            )
    rag_profile_path = session_rag_profile_path(session)
    if rag_profile_path.exists():
        artifacts.append(
            artifact_entry(
                kind="rag_profile",
                artifact_id="profile",
                path=rag_profile_path,
                format="json",
                filename="rag/profile.json",
            )
        )
    for evidence_path in sorted(session.evidence_dir.glob("*.json")):
        artifacts.append(
            artifact_entry(
                kind="evidence",
                artifact_id=evidence_path.stem,
                path=evidence_path,
                format="json",
                filename=f"evidence/{evidence_path.name}",
            )
        )
    for release_path in sorted(session.releases_dir.glob("*.json")):
        artifacts.append(
            artifact_entry(
                kind="release_snapshot",
                artifact_id=release_path.stem,
                path=release_path,
                format="json",
                filename=f"releases/{release_path.name}",
            )
        )
    if workspace.context.filters.document_report:
        artifacts.append({"kind": "document_report", "id": "latest", "format": "inline"})
    if workspace.context.filters.autonomous_loop_report:
        artifacts.append({"kind": "autonomous_loop_report", "id": "latest", "format": "inline"})
    return {
        "schema": "littrace.session_artifact_index.v1",
        "session_id": session.session_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "storage": {
            "backend": config.artifact_storage.backend if config is not None else "local",
            "path_prefix": config.artifact_storage.path_prefix if config is not None else "",
        },
        "artifacts": artifacts,
        "counts": {
            "artifacts": len(artifacts),
            "structured_documents": workspace.context.filters.structured_document_count,
            "snapshots": workspace.context.filters.workspace_snapshot_count,
        },
    }


def _content_type_for_format(format: str | None) -> str | None:
    if format == "json":
        return "application/json"
    if format == "jsonl":
        return "application/x-ndjson"
    if format == "pdf":
        return "application/pdf"
    return None


def load_artifact_index(session: ChatSession) -> dict[str, object]:
    state_store = _session_state_store(session)
    record = state_store.get_session_state(session.session_id)
    if record is None or not isinstance(record.artifact_index_json, dict):
        return {}
    return record.artifact_index_json


def _safe_filename(value: str) -> str:
    return (
        "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)[:120]
        or "paper"
    )


def load_memory(session: ChatSession):  # backward-compat alias
    return load_session_memory(session)


class SessionDeleteReport(BaseModel):
    session_id: str
    root_path: str
    deleted: bool = False
    artifact_count: int = 0
    embedded_chunk_count: int = 0
    state_record_count: int = 0
    object_deleted_count: int = 0
    object_delete_failures: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def delete_chat_session(
    config: LitTraceConfig,
    session_id: str,
    *,
    purge_files: bool = False,
) -> SessionDeleteReport:
    """Archive a chat session.

    ``purge_files=False`` (default) is a soft delete: the session_state
    row flips to status='archived', the on-disk workspace directory is
    left in place (recoverable), and downstream artifacts (object store
    blobs, RAG embeddings) are still deleted so storage does not leak.
    Pass ``purge_files=True`` to also ``shutil.rmtree`` the workspace —
    that path is irreversible.
    """
    session = load_existing_session(config, session_id)
    if session is None:
        return SessionDeleteReport(
            session_id=session_id,
            root_path=str(config.storage.sessions_dir / session_id),
            deleted=False,
            warnings=["session_not_found"],
        )

    artifact_registry = artifact_registry_from_config(config)
    artifact_store = artifact_store_from_config(config)
    state_store = _session_state_store(session, config)
    artifact_records = artifact_registry.list_for_session(
        session_id=session.session_id,
    )
    object_deleted_count = 0
    object_delete_failures: list[dict[str, object]] = []
    for record in artifact_records:
        try:
            artifact_store.delete(
                BlobRef(
                    backend=record.backend,
                    bucket=record.bucket,
                    object_key=record.object_key,
                    uri=None,
                    sha256=record.sha256,
                    size_bytes=record.size_bytes,
                    content_type=record.content_type,
                    metadata={},
                )
            )
            object_deleted_count += 1
        except Exception as exc:
            object_delete_failures.append(
                {
                    "artifact_id": record.artifact_id,
                    "object_key": record.object_key,
                    "backend": record.backend,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )

    artifact_deleted = artifact_registry.delete_for_session(
        session_id=session.session_id,
    )

    embedded_chunk_count = 0
    profile = load_session_rag_profile(session)
    if profile is not None and config.rag.backend == "pgvector" and config.rag.postgres_dsn:
        try:
            embedded_chunk_count = PgvectorRagStore(config, profile).delete_session()
        except Exception:
            pass

    # Soft-delete by default: archive_session_state flips status to
    # 'archived'. The on-disk workspace stays in place so the row can
    # be un-archived later (out of scope today but the contract is
    # ready). purge_files=True removes the directory as well.
    state_record_count = 0
    if state_store is not None:
        try:
            state_record_count = state_store.archive_session_state(
                session.session_id,
            )
        except Exception:
            pass

    if purge_files and session.root.exists():
        shutil.rmtree(session.root, ignore_errors=True)

    return SessionDeleteReport(
        session_id=session.session_id,
        root_path=str(session.root),
        deleted=True,
        artifact_count=artifact_deleted,
        embedded_chunk_count=embedded_chunk_count,
        state_record_count=state_record_count,
        object_deleted_count=object_deleted_count,
        object_delete_failures=object_delete_failures,
        warnings=[
            *([f"object_delete_failed:{len(object_delete_failures)}"] if object_delete_failures else []),
        ],
    )


def _register_artifacts(
    session: ChatSession,
    artifact_index: dict[str, object],
    config: LitTraceConfig | None,
    *,
    rag_profile=None,
) -> None:
    if config is None:
        return
    registry = artifact_registry_from_config(config)
    artifacts = artifact_index.get("artifacts")
    if not isinstance(artifacts, list):
        return
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        ref = item.get("storage_ref")
        if not isinstance(ref, dict):
            continue
        artifact_id = str(item.get("id") or "")
        kind = str(item.get("kind") or "artifact")
        if not artifact_id:
            continue
        blob_ref = BlobRef.model_validate(ref)
        record = ArtifactRecord.from_blob_ref(
            blob_ref,
            artifact_id=artifact_id,
            session_id=session.session_id,
            kind=kind,
            paper_id=str(item.get("paper_id")) if item.get("paper_id") else None,
            revision=str(item.get("revision")) if item.get("revision") else None,
        )
        registry.upsert(record)
        _enqueue_embedding_job_if_needed(session, item, record, config, rag_profile)


def _enqueue_embedding_job_if_needed(
    session: ChatSession,
    item: dict[str, object],
    record: ArtifactRecord,
    config: LitTraceConfig,
    rag_profile,
) -> None:
    if not config.rag.enabled:
        return
    kind = str(item.get("kind") or "")
    if kind not in {"structured_document", "paper_pdf", "supplementary"}:
        return
    enqueue_embedding_outbox(
        config,
        session_id=session.session_id,
        artifact_id=record.artifact_id,
        content_sha256=record.sha256,
        payload={"source_revision": record.revision, "reason": "artifact_registry_sync"},
    )


def _sync_session_state(
    session: ChatSession,
    workspace: LiteratureWorkspace,
    manifest: dict[str, object],
    artifact_index: dict[str, object],
    memory,
    rag_profile,
    *,
    state_store,
    expected_revision: int,
) -> None:
    workspace_json = json.loads(workspace.model_dump_json())
    new_revision = int(manifest.get("revision", 0) or 0)
    record = SessionStateRecord(
        session_id=session.session_id,
        workspace_sha256=manifest.get("workspace_sha256"),
        workspace_json=workspace_json if isinstance(workspace_json, dict) else {},
        manifest_json=manifest,
        artifact_index_json=artifact_index,
        memory_view_json=memory.model_dump(mode="json") if hasattr(memory, "model_dump") else {},
        rag_profile_json=rag_profile.model_dump(mode="json") if rag_profile is not None else {},
        revision=new_revision,
        # First save_workspace promotes the row from the
        # _ensure_chat_trail 'draft' placeholder into 'active'. Later
        # saves keep it 'active'. delete_chat_session is what flips to
        # 'archived' (handled separately, never via _sync_session_state).
        status="active",
        )
    state_store.upsert_session_state(record, expected_revision=expected_revision)


def _load_session_record(session: ChatSession) -> SessionStateRecord | None:
    state_store = _session_state_store(session)
    if state_store is None:
        return None
    return state_store.get_session_state(session.session_id)


def _session_state_store(
    session: ChatSession,
    config: LitTraceConfig | None = None,
):
    from littrace.state_db import session_state_store

    return session_state_store(session, config)
