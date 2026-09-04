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
    # Round 24: when the GUI starts sentinel via littrace-qt, this
    # is the GUI's chat session id. ``save_*`` / ``load_*`` write
    # paper metadata, RAG chunks, and artifacts into *that* session
    # so the user sees the new papers immediately. Without it
    # (standalone CLI) ``target_session_id`` falls back to the
    # legacy ``sentinel:<watchlist>`` id so previous behaviour is
    # preserved.
    target_session_id: str = ""

    @property
    def session_id(self) -> str:
        return f"sentinel:{self.watchlist_id}"

    @property
    def resolved_session_id(self) -> str:
        """The Postgres state_db session row sentinel reads / writes
        paper data through. Round 24: defaults to ``target_session_id``
        when provided (the GUI case) so the user sees sentinel
        results in their main workspace immediately. Without
        ``target_session_id`` we fall back to the legacy
        ``sentinel:<watchlist>`` session for CLI usage."""
        return self.target_session_id or self.session_id


def sentinel_root(config: LitTraceConfig, watchlist_id: str) -> Path:
    return config.storage.sessions_dir / "sentinel" / watchlist_id


def get_sentinel_store(
    config: LitTraceConfig,
    watchlist_id: str,
    *,
    main_session_id: str | None = None,
) -> SentinelStore:
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
        target_session_id=main_session_id or "",
    )
    return store


def ensure_sentinel_store(
    config: LitTraceConfig,
    watchlist: Watchlist,
    *,
    main_session_id: str | None = None,
) -> SentinelStore:
    store = get_sentinel_store(
        config, watchlist.watchlist_id, main_session_id=main_session_id,
    )
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

    # Round 24: only create the legacy sentinel row when no main
    # session is provided. When the GUI is in the loop, sentinel
    # writes into the user's main session, so creating a parallel
    # ``sentinel:<watchlist>`` row would just re-introduce the
    # "two-session" problem we just fixed.
    if not main_session_id:
        sentinel_sid = store.session_id
        if state_store_from_config(config).get_session_state(sentinel_sid) is None:
            save_workspace(
                ChatSession.from_root(store.root, sentinel_sid, config=config),
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
    if record is None:
        return SentinelState.model_validate({
            "watchlist": {
                "watchlist_id": store.watchlist_id,
                "topic": store.watchlist_id,
            }
        })
    manifest = record.manifest_json or {}
    if store.target_session_id:
        # Round 24: main-session path reads from
        # ``sentinel_runs[<watchlist_id>]`` so multiple watchlists
        # can coexist in the same main session.
        runs = manifest.get("sentinel_runs") or {}
        raw = runs.get(store.watchlist_id)
    else:
        raw = manifest.get("sentinel_state")
    return SentinelState.model_validate(
        raw or {"watchlist": {
            "watchlist_id": store.watchlist_id,
            "topic": store.watchlist_id,
        }}
    )


def save_sentinel_state(store: SentinelStore, state: SentinelState) -> Path:
    record = _sentinel_record(store)
    if record is None:
        # Round 24: when invoked from littrace-qt the target is
        # the GUI's main chat session, which always exists (the
        # shell controller creates it on startup). The "missing
        # sentinel row" path only fires for the standalone CLI
        # without ``--main-session-id`` — and that path always
        # falls through to the legacy ``sentinel:<watchlist>``
        # session that ``ensure_sentinel_store`` provisioned.
        #
        # Round 29: in real-world runs the GUI session row can
        # still be missing for transient reasons (DB unreachable
        # on the GUI side at startup, ``main_session_id`` was
        # passed before the GUI thread finished seeding the row,
        # the row was archived). Raising here was killing the
        # sentinel subprocess *before* any search happened, so
        # the GUI's daily dialog kept reporting 0/0/0 with no
        # clue why. Instead, when ``target_session_id`` is set
        # and the row is genuinely absent, seed a placeholder
        # row from the sentinel side and retry. This is a
        # write-only path that does not mutate the GUI's view
        # of the session — it just gives us a writable row so
        # the rest of the run can proceed.
        if store.target_session_id:
            from littrace.state_db import state_store_from_config
            from littrace.session import SessionStateRecord
            state_store = state_store_from_config(store.config)
            existing = state_store.get_session_state(store.target_session_id)
            if existing is not None:
                # Race: row appeared between our last check and now.
                record = existing
            else:
                state_store.upsert_session_state(SessionStateRecord(
                    session_id=store.target_session_id,
                    revision=0,
                    status="draft",
                ))
                record = state_store.get_session_state(
                    store.target_session_id
                )
        if record is None:
            raise ValueError(
                f"Sentinel session row is missing: {store.resolved_session_id}"
            )
    manifest = dict(record.manifest_json)
    manifest["watchlist"] = state.watchlist.model_dump(mode="json")
    # Round 24: when writing into the user's main session we
    # namespace the sentinel bookkeeping under a ``sentinel_runs``
    # sub-key so multiple watchlists can coexist in the same main
    # session without clobbering each other.
    if store.target_session_id:
        runs = manifest.get("sentinel_runs") or {}
        runs[store.watchlist_id] = state.model_dump(mode="json")
        manifest["sentinel_runs"] = runs
    else:
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
    # Round 24: when ``target_session_id`` is set (littrace-qt
    # case) sentinel writes paper metadata, RAG chunks, and
    # artifacts into the user's *main* workspace. The on-disk
    # workspace dir under ``sentinel/<watchlist_id>`` still exists
    # (for evidence_base / digest / run_summary files) but is no
    # longer the source of truth for paper data. The CLI case
    # (no ``main_session_id``) keeps the legacy behaviour.
    if store.target_session_id:
        from littrace.session import ChatSession, save_workspace
        # Use the user's main chat session — ChatSession.from_root
        # constructs an object whose ``session_id`` and ``root``
        # point at the main session row, so ``save_workspace``
        # routes paper data through the main Postgres row.
        main_root = (
            Path(store.config.storage.sessions_dir)
            / store.target_session_id
        )
        main_root.mkdir(parents=True, exist_ok=True)
        main_session_like = ChatSession.from_root(
            main_root, store.target_session_id, config=store.config,
        )
        # Round 28: the main session is also live in the GUI
        # thread — the user might be pinning a paper, editing a
        # filter, etc. while sentinel is running. ``save_workspace``
        # uses a revision optimistic-lock and raises
        # ``RuntimeError("Workspace revision mismatch")`` on conflict.
        # Retry up to 3 times: re-read the current workspace,
        # re-apply sentinel's paper set on top, save again. The
        # paper set is monotonic (we only add) so re-apply is
        # idempotent and never overwrites user changes like
        # ``active_papers`` because we merge field-by-field, not
        # by replacing the whole dict.
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                save_workspace(
                    main_session_like, workspace, config=config,
                )
                return
            except RuntimeError as exc:
                last_exc = exc
                if "revision mismatch" not in str(exc):
                    # Genuine error (e.g. malformed workspace) — no
                    # point in retrying.
                    raise
                # Re-read the current main session, re-apply
                # sentinel's data on top, retry. ``load_workspace``
                # via the same ChatSession object returns the
                # latest server-side state.
                from littrace.session import load_workspace as _load
                latest = _load(main_session_like)
                for key in (
                    "papers", "parsed_papers", "performance_cells",
                    "guard_reports", "resolution_decisions",
                    "claims", "claim_verification_reports",
                    "release_snapshots",
                ):
                    sentinel_block = getattr(workspace, key, None)
                    if not isinstance(sentinel_block, dict):
                        continue
                    latest_block = getattr(latest, key, None) or {}
                    if not isinstance(latest_block, dict):
                        latest_block = {}
                    latest_block.update(sentinel_block)
                    setattr(latest, key, latest_block)
                # active_papers: union, sentinel first so it doesn't
                # accidentally drop user-pinned papers.
                existing_active = list(latest.context.active_papers or [])
                sentinel_active = list(
                    workspace.context.active_papers or []
                )
                merged = list(dict.fromkeys(
                    sentinel_active + existing_active
                ))
                latest.context.active_papers = merged
                workspace = latest
        # All 3 attempts hit revision mismatch — give up
        # gracefully. The user will see fewer papers than expected
        # for this run; the next run will retry and fill in.
        import logging
        logging.warning(
            "sentinel: gave up writing to main session after 3 "
            "revision-mismatch retries: %s",
            last_exc,
        )
        return
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
    if store.target_session_id:
        # Round 24: main-session path reads the user's main
        # workspace directly so we never have to copy data between
        # two ``workspace_json`` blobs.
        from littrace.session import ChatSession, load_workspace
        main_root = (
            Path(store.config.storage.sessions_dir)
            / store.target_session_id
        )
        if not main_root.exists():
            return LiteratureWorkspace()
        main_session_like = ChatSession.from_root(
            main_root, store.target_session_id, config=store.config,
        )
        return load_workspace(main_session_like)
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
    # Round 24: when ``target_session_id`` is set we read / write
    # through that row instead of the legacy
    # ``sentinel:<watchlist>`` row. The legacy path is preserved
    # for CLI invocations that don't pass ``--main-session-id``.
    return _state_store(store).get_session_state(store.resolved_session_id)


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
