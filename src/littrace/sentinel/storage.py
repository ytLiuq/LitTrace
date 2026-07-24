from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.sentinel.state import AccessTask, SentinelState, Watchlist


@dataclass
class SentinelStore:
    watchlist_id: str
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
    state_path: Path
    access_queue_path: Path
    digest_dir: Path
    evidence_base_dir: Path
    user_id: str = "local-user"

    @property
    def session_id(self) -> str:
        return self.watchlist_id


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
        watchlist_id=watchlist_id,
        root=root,
        workspace_dir=workspace_dir,
        workspace_path=root / "workspace.json",
        messages_path=root / "messages.jsonl",
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
        user_id=config.storage.default_user_id,
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
    return store


def load_watchlist(store: SentinelStore) -> Watchlist:
    path = store.root / "watchlist.yaml"
    if not path.exists():
        return Watchlist(watchlist_id=store.watchlist_id, topic=store.watchlist_id)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    raw.setdefault("watchlist_id", store.watchlist_id)
    raw.setdefault("topic", store.watchlist_id)
    return Watchlist.model_validate(raw)


def save_watchlist(store: SentinelStore, watchlist: Watchlist) -> Path:
    store.root.mkdir(parents=True, exist_ok=True)
    path = store.root / "watchlist.yaml"
    path.write_text(
        yaml.safe_dump(watchlist.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return path


def load_sentinel_state(store: SentinelStore) -> SentinelState:
    if not store.state_path.exists():
        state = SentinelState(watchlist=load_watchlist(store))
        state.access_queue = load_access_queue(store)
        return state
    try:
        state = SentinelState.model_validate_json(store.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        state = SentinelState(watchlist=load_watchlist(store))
    persisted_queue = load_access_queue(store)
    if persisted_queue:
        state.access_queue = persisted_queue
    return state


def save_sentinel_state(store: SentinelStore, state: SentinelState) -> Path:
    store.root.mkdir(parents=True, exist_ok=True)
    state_path = store.state_path
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    save_access_queue(store, state.access_queue)
    return state_path


def load_access_queue(store: SentinelStore) -> list[AccessTask]:
    if not store.access_queue_path.exists():
        return []
    try:
        raw = json.loads(store.access_queue_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    tasks: list[AccessTask] = []
    for item in raw:
        try:
            tasks.append(AccessTask.model_validate(item))
        except ValueError:
            continue
    return tasks


def save_access_queue(store: SentinelStore, tasks: list[AccessTask]) -> Path:
    store.root.mkdir(parents=True, exist_ok=True)
    store.access_queue_path.write_text(
        json.dumps([task.model_dump(mode="json") for task in tasks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return store.access_queue_path


def save_sentinel_workspace(
    store: SentinelStore,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig | None = None,
) -> None:
    session_like = type(
        "SentinelWorkspaceSession",
        (),
        {
            "session_id": store.watchlist_id,
            "user_id": store.user_id,
            "root": store.root,
            "workspace_dir": store.workspace_dir,
            "workspace_path": store.workspace_path,
            "messages_path": store.messages_path,
            "artifacts_dir": store.artifacts_dir,
            "artifact_index_path": store.artifact_index_path,
            "snapshots_dir": store.snapshots_dir,
            "structured_documents_dir": store.structured_documents_dir,
            "evidence_dir": store.evidence_dir,
            "releases_dir": store.releases_dir,
            "rag_dir": store.workspace_dir / "rag",
        },
    )()
    from littrace.session import save_workspace

    save_workspace(session_like, workspace, config=config)


def load_sentinel_workspace(store: SentinelStore) -> LiteratureWorkspace:
    session_like = type(
        "SentinelWorkspaceSession",
        (),
        {
            "session_id": store.watchlist_id,
            "user_id": store.user_id,
            "root": store.root,
            "workspace_dir": store.workspace_dir,
            "workspace_path": store.workspace_path,
            "messages_path": store.messages_path,
            "artifacts_dir": store.artifacts_dir,
            "artifact_index_path": store.artifact_index_path,
            "snapshots_dir": store.snapshots_dir,
            "structured_documents_dir": store.structured_documents_dir,
            "evidence_dir": store.evidence_dir,
            "releases_dir": store.releases_dir,
            "rag_dir": store.workspace_dir / "rag",
        },
    )()
    from littrace.session import load_workspace

    return load_workspace(session_like)


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
