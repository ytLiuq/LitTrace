from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.sentinel.state import AccessTask, SentinelState, Watchlist


@dataclass
class SentinelStore:
    config: LitTraceConfig
    watchlist_id: str
    root: Path
    workspace_dir: Path
    # workspace_path / messages_path removed in round 3 topic B —
    # Postgres is the source of truth. Sentinel subscribes to chat
    # traffic through the same StateStore surface the chat path uses.
    artifacts_dir: Path
    artifact_index_path: Path
    snapshots_dir: Path
    structured_documents_dir: Path
    evidence_dir: Path
    releases_dir: Path
    state_path: Path
    access_queue_path: Path
    digest_dir: Path
    evidence_base_dir: Path

    @property
    def session_id(self) -> str:
        return f"sentinel:{self.watchlist_id}"


def sentinel_root(config: LitTraceConfig, watchlist_id: str) -> Path:
    return config.storage.sessions_dir / "sentinel" / watchlist_id


def get_sentinel_store(config: LitTraceConfig, watchlist_id: str) -> SentinelStore:
    root = sentinel_root(config, watchlist_id)
    workspace_dir = root / "workspace"
    artifacts_dir = root / "artifacts"
    snapshots_dir = workspace_dir / "snapshots"
    structured_documents_dir = workspace_dir / "structured_documents"
    evidence_dir = workspace_dir / "evidence"
    releases_dir = workspace_dir / "releases"
    evidence_base_dir = root / "evidence_base"
    store = SentinelStore(
        config=config,
        watchlist_id=watchlist_id,
        root=root,
        workspace_dir=workspace_dir,
        artifacts_dir=artifacts_dir,
        artifact_index_path=workspace_dir / "artifact_index.json",
        snapshots_dir=snapshots_dir,
        structured_documents_dir=structured_documents_dir,
        evidence_dir=evidence_dir,
        releases_dir=releases_dir,
        state_path=root / "state.json",
        access_queue_path=root / "access_queue.json",
        digest_dir=root / "digests",
        evidence_base_dir=evidence_base_dir,
    )
    return store


def ensure_sentinel_store(config: LitTraceConfig, watchlist: Watchlist) -> SentinelStore:
    store = get_sentinel_store(config, watchlist.watchlist_id)
    store.root.mkdir(parents=True, exist_ok=True)
    store.workspace_dir.mkdir(parents=True, exist_ok=True)
    store.artifacts_dir.mkdir(parents=True, exist_ok=True)
    store.snapshots_dir.mkdir(parents=True, exist_ok=True)
    store.structured_documents_dir.mkdir(parents=True, exist_ok=True)
    store.evidence_dir.mkdir(parents=True, exist_ok=True)
    store.releases_dir.mkdir(parents=True, exist_ok=True)
    store.digest_dir.mkdir(parents=True, exist_ok=True)
    store.evidence_base_dir.mkdir(parents=True, exist_ok=True)
    from littrace.session import ChatSession, save_workspace
    from littrace.state_db import state_store_from_config

    if state_store_from_config(config).get_session_state(store.session_id) is None:
        save_workspace(
            ChatSession.from_root(store.root, store.session_id, config=config),
            LiteratureWorkspace(),
            config=config,
        )
    return store


def load_watchlist(store: SentinelStore) -> Watchlist:
    record = _sentinel_record(store)
    raw = record.manifest_json.get("watchlist") if record else None
    return Watchlist.model_validate(raw or {"watchlist_id": store.watchlist_id, "topic": store.watchlist_id})


def save_watchlist(store: SentinelStore, watchlist: Watchlist) -> Path:
    state = load_sentinel_state(store).model_copy(update={"watchlist": watchlist})
    save_sentinel_state(store, state)
    return store.root


def load_sentinel_state(store: SentinelStore) -> SentinelState:
    record = _sentinel_record(store)
    raw = record.manifest_json.get("sentinel_state") if record else None
    return SentinelState.model_validate(raw or {"watchlist": {"watchlist_id": store.watchlist_id, "topic": store.watchlist_id}})


def save_sentinel_state(store: SentinelStore, state: SentinelState) -> Path:
    record = _sentinel_record(store)
    if record is None:
        raise ValueError(f"Sentinel session is missing: {store.watchlist_id}")
    manifest = dict(record.manifest_json)
    manifest["watchlist"] = state.watchlist.model_dump(mode="json")
    manifest["sentinel_state"] = state.model_dump(mode="json")
    record.manifest_json = manifest
    _state_store(store).upsert_session_state(record)
    return store.root


def load_access_queue(store: SentinelStore) -> list[AccessTask]:
    return load_sentinel_state(store).access_queue


def save_access_queue(store: SentinelStore, tasks: list[AccessTask]) -> Path:
    state = load_sentinel_state(store)
    save_sentinel_state(store, state.model_copy(update={"access_queue": tasks}))
    return store.root


def save_sentinel_workspace(
    store: SentinelStore,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig | None = None,
) -> None:
    session_like = type(
        "SentinelWorkspaceSession",
        (),
        {
            "session_id": store.session_id,
            "root": store.root,
            "workspace_dir": store.workspace_dir,
            "messages_path": store.root / "messages.jsonl",  # never written
            "artifacts_dir": store.artifacts_dir,
            "artifact_index_path": store.artifact_index_path,
            "snapshots_dir": store.snapshots_dir,
            "structured_documents_dir": store.structured_documents_dir,
            "evidence_dir": store.evidence_dir,
            "releases_dir": store.releases_dir,
            "rag_dir": store.workspace_dir / "rag",
            "metadata_store_backend": store.config.metadata_store.backend,
            "metadata_postgres_dsn": store.config.metadata_store.postgres_dsn,
            "metadata_schema_name": store.config.metadata_store.schema_name,
        },
    )()
    from littrace.session import save_workspace

    save_workspace(session_like, workspace, config=config)


def load_sentinel_workspace(store: SentinelStore) -> LiteratureWorkspace:
    session_like = type(
        "SentinelWorkspaceSession",
        (),
        {
            "session_id": store.session_id,
            "root": store.root,
            "workspace_dir": store.workspace_dir,
            "messages_path": store.root / "messages.jsonl",  # never written
            "artifacts_dir": store.artifacts_dir,
            "artifact_index_path": store.artifact_index_path,
            "snapshots_dir": store.snapshots_dir,
            "structured_documents_dir": store.structured_documents_dir,
            "evidence_dir": store.evidence_dir,
            "releases_dir": store.releases_dir,
            "rag_dir": store.workspace_dir / "rag",
            "metadata_store_backend": store.config.metadata_store.backend,
            "metadata_postgres_dsn": store.config.metadata_store.postgres_dsn,
            "metadata_schema_name": store.config.metadata_store.schema_name,
        },
    )()
    from littrace.session import load_workspace

    return load_workspace(session_like)


def _state_store(store: SentinelStore):
    from littrace.state_db import state_store_from_config

    return state_store_from_config(store.config)


def _sentinel_record(store: SentinelStore):
    return _state_store(store).get_session_state(store.session_id)


def touch_run_dir(store: SentinelStore, run_id: str) -> Path:
    run_dir = store.root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_evidence_base(
    store: SentinelStore,
    run_id: str,
    workspace: LiteratureWorkspace,
    resource_pack: Any,
    quality_report: Any | None = None,
) -> Path:
    run_dir = store.evidence_base_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        run_dir / "papers.jsonl", [paper.model_dump(mode="json") for paper in resource_pack.papers]
    )
    _write_jsonl(
        run_dir / "citation_records.jsonl",
        [record.model_dump(mode="json") for record in resource_pack.citation_records],
    )
    _write_jsonl(
        run_dir / "performance_cells.jsonl",
        [cell.model_dump(mode="json") for cell in workspace.performance_cells],
    )
    _write_jsonl(
        run_dir / "full_text_reports.jsonl",
        [report.model_dump(mode="json") for report in workspace.full_text_reports.values()],
    )
    structured_documents = {
        paper_id: parsed.model_dump(mode="json")
        for paper_id, parsed in workspace.parsed_papers.items()
        if parsed.parsed or parsed.structured_document
    }
    (run_dir / "structured_documents.json").write_text(
        json.dumps(structured_documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if quality_report is not None:
        quality_dir = store.evidence_base_dir / "quality_reports"
        quality_dir.mkdir(parents=True, exist_ok=True)
        (quality_dir / f"{run_id}.json").write_text(
            quality_report.model_dump_json(indent=2), encoding="utf-8"
        )
    latest_path = store.evidence_base_dir / "latest_run.txt"
    latest_path.write_text(run_id, encoding="utf-8")
    return run_dir


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")
