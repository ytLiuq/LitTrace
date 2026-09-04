from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from littrace.config import LitTraceConfig
from littrace.models import AccessType, DownloadExecutionRequest, LiteratureWorkspace, PaperSearchRequest
from littrace.sentinel.digest import build_digest_markdown, save_digest
from littrace.sentinel.resource_pack import ResourcePack, build_resource_pack, save_resource_pack
from littrace.sentinel.state import AccessTask, SentinelRunSummary, SentinelState, Watchlist
from littrace.session import save_workspace as _save_workspace_to_session
from littrace.state_db import state_store_from_config
from littrace.sentinel.storage import (
    SentinelStore,
    ensure_sentinel_store,
    load_sentinel_state,
    load_sentinel_workspace,
    save_evidence_base,
    save_sentinel_state,
    save_sentinel_workspace,
    save_watchlist,
    timestamp_run_id,
    touch_run_dir,
)
from littrace.retrieval.rag_refresh import refresh_session_rag_index
from littrace.skill_runner import (
    build_quality_report_skill,
    execute_downloads_skill,
    extract_tables_skill,
    parse_workspace_skill,
    resolve_workspace_full_text_skill,
    search_papers_skill,
)
from littrace.context import add_ranked_candidate_papers
from littrace.evidence.tables import extract_structured_artifacts


@dataclass
class SentinelRunResult:
    store: SentinelStore
    state: SentinelState
    workspace: LiteratureWorkspace
    resource_pack: ResourcePack
    summary: SentinelRunSummary
    digest_path: str | None = None
    run_dir: str | None = None


class LiteratureSentinel:
    def __init__(
        self,
        config: LitTraceConfig,
        watchlist: Watchlist,
        *,
        main_session_id: str | None = None,
    ):
        self.config = config
        self.watchlist = watchlist
        # Round 24: when invoked from littrace-qt, ``main_session_id``
        # is the GUI's chat session id and we want every paper,
        # parsed output, RAG chunk, and artifact to land in *that*
        # session so the user sees them immediately in their main
        # workspace. Without a main session (standalone CLI) we
        # fall back to the legacy ``sentinel:<watchlist>`` session.
        self.main_session_id = main_session_id
        # The on-disk store still keeps the per-watchlist layout
        # (evidence_base / digests / run summaries) so the per-
        # watchlist evidence trail stays organised on disk. The
        # *workspace* + RAG + artifact metadata writes go through
        # ``self.target_session_id`` instead.
        self.store = ensure_sentinel_store(
            config, watchlist, main_session_id=main_session_id,
        )
        self.target_session_id = main_session_id or self.store.session_id
        save_watchlist(self.store, watchlist)

    def load_state(self) -> SentinelState:
        return load_sentinel_state(self.store)

    def load_workspace(self) -> LiteratureWorkspace:
        return load_sentinel_workspace(self.store)

    async def run(self) -> SentinelRunResult:
        run_id = timestamp_run_id()
        started_at = datetime.now().isoformat(timespec="seconds")
        state = self.load_state()
        workspace = self.load_workspace()
        state.watchlist = self.watchlist
        run_dir = touch_run_dir(self.store, run_id)

        request = PaperSearchRequest(
            topic=self.watchlist.topic,
            discipline="materials chemistry",
            year_min=self.watchlist.year_min,
            # Round 17: thread the user-selected year upper bound
            # through to the retrieval layer. ``None`` means "no
            # upper bound", which is also the backward-compat
            # default when ``Watchlist.year_max`` is unset.
            year_max=self.watchlist.year_max,
            live=self.config.api.enable_live_search,
            query_variants=list(self.watchlist.query_variants),
        )

        search = await search_papers_skill(request, self.config)

        # Round 24: route the RAG collection into the user's main
        # session so the agent can search it later via
        # ``search_workspace_rag``. ``refresh_session_rag_index``
        # builds the collection name as
        # ``{prefix}_{getattr(session, 'session_id')}`` — we need the
        # session id to read as ``self.target_session_id`` so we
        # pass a lightweight shim that exposes just ``.session_id``
        # and the bits ``save_session_rag_profile`` reaches into.
        rag_session = _TargetSessionShim(
            target_session_id=self.target_session_id,
            store=self.store,
        )
        candidate_papers = search.result.papers
        if candidate_papers:
            workspace = add_ranked_candidate_papers(
                workspace,
                candidate_papers,
                request,
                active_limit=self.config.literature_context.active_context_limit,
            )

        # Round 23: every paper returned by the search this run is
        # reported as a candidate — the user explicitly asked for the
        # semantics "sentinel retrieves papers for a topic", not "only
        # papers I haven't seen before". ``seen_paper_ids`` is still
        # updated so the dialog can show new-vs-seen breakdown, but
        # neither the download stage nor the run summary gate on it.
        seen = set(state.seen_paper_ids)
        new_candidates = [
            paper for paper in candidate_papers
            if paper.paper_id not in seen
        ]
        for paper in new_candidates:
            state.seen_paper_ids.append(paper.paper_id)
        seen_candidates = [
            paper for paper in candidate_papers
            if paper.paper_id in seen
        ]

        # Full-text resolution is incremental. Re-probing every publisher URL on
        # every daily tick causes needless 429s/timeouts for papers already seen.
        unresolved_ids = [
            paper_id
            for paper_id in workspace.context.active_papers
            if paper_id not in workspace.full_text_reports
        ]
        if unresolved_ids:
            probe_workspace = workspace.model_copy(deep=True)
            probe_workspace.context.active_papers = unresolved_ids
            probe_workspace = await resolve_workspace_full_text_skill(
                probe_workspace, self.config
            )
            for paper_id in unresolved_ids:
                workspace.full_text_reports[paper_id] = probe_workspace.full_text_reports[
                    paper_id
                ]
                workspace.papers[paper_id] = probe_workspace.papers[paper_id]

        access_tasks = _build_access_tasks(workspace, state)
        state.access_queue = _merge_access_tasks(state.access_queue, access_tasks)

        # Sentinel auto-downloads to the object store only — never to the
        # user's working directory. Same posture as daily_update: bytes go
        # to the artifact backend, leaving paper_library_dir clean.
        downloadable_ids = [
            paper.paper_id
            for paper in workspace.papers.values()
            if paper.access_type == AccessType.OPEN_ACCESS
            and paper.paper_id in workspace.context.active_papers
        ]
        downloaded_count = 0
        download_warnings: list[str] = []
        if downloadable_ids:
            try:
                download_result = await execute_downloads_skill(
                    self.config,
                    workspace,
                    DownloadExecutionRequest(
                        paper_ids=downloadable_ids,
                        session_id=self.store.session_id if hasattr(self.store, "session_id") else None,
                        target="storage_only",
                    ),
                )
                downloaded_count = download_result.downloaded_count
                # Round 17: a zero-download count on a non-empty
                # candidate list is almost always a transport / auth
                # failure that used to be silent. Surface it as a
                # warning so the user can see "下载 0 篇" in the
                # status strip instead of wondering why
                # ``downloaded=0`` while ``new_candidates>0``.
                if downloadable_ids and downloaded_count == 0:
                    download_warnings.append(
                        f"下载阶段：{len(downloadable_ids)} 篇候选全部失败（详见 digest.md）"
                    )
            except Exception as exc:
                # Round 17: previously an exception inside
                # ``execute_downloads_skill`` would propagate up and
                # abort the whole sentinel run (no digest, no
                # quality report, no resource pack). Now we capture
                # it as a warning so the user at least sees the
                # rest of the run results — search hits, candidates,
                # parse status — instead of an opaque traceback.
                download_warnings.append(
                    f"下载阶段失败：{exc.__class__.__name__}: {exc}"
                )

        # Round 25: parse + table metrics now run unconditionally
        # at the end of every sentinel run. Previously the parse step
        # was gated on ``config.sentinel.parse_on_daily`` (default
        # False) so most runs shipped PDFs to the object store
        # without RAG chunks. The user explicitly asked for
        # "silent parse on download", so the download -> parse
        # chain is now in-band: ``execute_downloads_skill``
        # triggers parse right after each download, and the table
        # extraction here catches anything that landed before the
        # parse step ran. ``parse_workspace_skill`` is imported
        # above for legacy callers; we don't need to call it again.
        try:
            workspace, table_harness = await extract_tables_skill(workspace, self.config)
        except Exception as exc:
            workspace, table_harness = extract_structured_artifacts(workspace)
            table_harness.warnings.append(
                f"Performance metric extraction skipped: {exc.__class__.__name__}: {exc}"
            )
        parse_report = {
            # Round 29: ``workspace.context.filters`` is a
            # ``WorkspaceFilters`` (defined in models.py line ~461).
            # It does NOT have a ``parsed_paper_count`` field —
            # the count lives at ``parsed_full_text_count`` (or on
            # the ``WorkspaceSummary`` projection). The previous
            # field access raised ``AttributeError`` here, which
            # propagated up and killed the subprocess *before*
            # ``cli.py`` could print the ``run_id:`` /
            # ``new_candidates:`` / ``downloaded:`` summary lines
            # — so the GUI's daily dialog showed 0/0/0 with no
            # hint about the real cause.
            #
            # Guard the attribute access so an unexpected rename
            # degrades to a 0 count instead of an unrecoverable
            # crash.
            "parsed_count": (
                getattr(
                    workspace.context.filters,
                    "parsed_full_text_count",
                    0,
                )
                if hasattr(workspace.context, "filters")
                else 0
            ),
            "warnings": [],
        }
        quality_report = build_quality_report_skill(self.config, workspace)
        quality_warnings = [
            *quality_report.warnings,
            *table_harness.warnings,
            *parse_report.get("warnings", []),
            # Round 17: download warnings are surfaced alongside
            # quality / parse warnings so the digest and the GUI
            # status strip both see them.
            *download_warnings,
        ]
        try:
            _, rag_refresh_report = await refresh_session_rag_index(
                self.config, rag_session, workspace
            )
            quality_warnings = [*quality_warnings, *rag_refresh_report.warnings]
        except Exception as exc:
            quality_warnings.append(
                f"RAG refresh skipped: {exc.__class__.__name__}: {exc}"
            )
        resource_pack = build_resource_pack(
            workspace,
            state,
            quality_warnings=quality_warnings,
        )
        resource_pack_file = save_resource_pack(self.store.root, resource_pack, run_id=run_id)
        save_evidence_base(self.store, run_id, workspace, resource_pack, quality_report)
        digest_markdown = build_digest_markdown(resource_pack, state)
        digest_path, digest_record = save_digest(self.store.root, run_id, digest_markdown, resource_pack, state)
        finished_at = datetime.now().isoformat(timespec="seconds")
        state.last_run_at = finished_at
        state.warnings = quality_warnings
        save_sentinel_workspace(self.store, workspace, config=self.config)
        save_sentinel_state(self.store, state)
        # Round 24: paper metadata, RAG chunks, and artifacts were
        # already written into the user's main session above (via
        # ``save_sentinel_workspace`` and
        # ``refresh_session_rag_index(rag_session, …)`` which both
        # route through ``self.target_session_id``). The previous
        # Round-23 mirror step is no longer needed and is removed
        # to avoid the duplicate write that confused diagnostics.
        # The legacy ``_mirror_workspace_to_main_session`` helper
        # is kept on disk (below) so any in-flight subprocesses
        # that ``import`` it don't break at import time.

        summary = SentinelRunSummary(
            run_id=run_id,
            watchlist_id=self.watchlist.watchlist_id,
            topic=self.watchlist.topic,
            started_at=started_at,
            finished_at=finished_at,
            # Round 23: report all three counters — total returned,
            # already-seen subset, new-only subset — so the dialog
            # can show "X papers this run, Y of them new" rather
            # than only the (often zero) new-only count.
            candidate_count=len(candidate_papers),
            new_candidate_count=len(new_candidates),
            seen_candidate_count=len(seen_candidates),
            downloaded_count=downloaded_count,
            parsed_count=int(parse_report.get("parsed_count", 0)),
            access_task_count=len(state.access_queue),
            digest_path=str(digest_path),
            resource_pack_path=str(resource_pack_file),
            quality_score=quality_report.metrics.get("overall_score"),
            warnings=quality_warnings,
        )
        (run_dir / "run_summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        (run_dir / "digest.md").write_text(digest_markdown, encoding="utf-8")
        return SentinelRunResult(
            store=self.store,
            state=state,
            workspace=workspace,
            resource_pack=resource_pack,
            summary=summary,
            digest_path=str(digest_path),
            run_dir=str(run_dir),
        )

    def access_review(self) -> list[AccessTask]:
        return list(self.load_state().access_queue)

    async def resume_after_login(self) -> SentinelRunResult:
        run_id = timestamp_run_id()
        started_at = datetime.now().isoformat(timespec="seconds")
        state = self.load_state()
        workspace = self.load_workspace()
        queued_tasks = [task for task in state.access_queue if task.retry_after_login]
        login_papers = [task.paper_id for task in queued_tasks if task.paper_id in workspace.papers]
        downloaded_count = 0
        # resume_after_login still doesn't auto-download to local — the user
        # must re-trigger the explicit manual download path after auth.
        # We only surface the queued papers so the user can act on them.
        parse_report = {"parsed_count": 0, "warnings": []}
        completed_ids = {
            paper_id
            for paper_id in login_papers
            if paper_id in workspace.parsed_papers and workspace.parsed_papers[paper_id].parsed
        }
        if completed_ids:
            state.access_queue = [task for task in state.access_queue if task.paper_id not in completed_ids]
        state.warnings = [*state.warnings, *parse_report.get("warnings", [])]
        _, rag_refresh_report = await refresh_session_rag_index(self.config, rag_session, workspace)
        state.warnings.extend(rag_refresh_report.warnings)
        finished_at = datetime.now().isoformat(timespec="seconds")
        state.last_run_at = finished_at
        save_sentinel_workspace(self.store, workspace, config=self.config)
        save_sentinel_state(self.store, state)
        resource_pack = build_resource_pack(workspace, state, quality_warnings=list(state.warnings))
        resource_pack_file = save_resource_pack(self.store.root, resource_pack, run_id=run_id)
        save_evidence_base(self.store, run_id, workspace, resource_pack)
        summary = SentinelRunSummary(
            run_id=run_id,
            watchlist_id=self.watchlist.watchlist_id,
            topic=self.watchlist.topic,
            started_at=started_at,
            finished_at=finished_at,
            new_candidates_count=0,
            downloaded_count=downloaded_count,
            parsed_count=int(parse_report.get("parsed_count", 0)),
            access_task_count=len(state.access_queue),
            resource_pack_path=str(resource_pack_file),
            quality_score=None,
            warnings=list(state.warnings),
        )
        return SentinelRunResult(
            store=self.store,
            state=state,
            workspace=workspace,
            resource_pack=resource_pack,
            summary=summary,
            digest_path=str(digest_path),
            run_dir=str(run_dir),
        )


class _TargetSessionShim:
    """Lightweight stand-in for the GUI's main chat session when
    calling ``refresh_session_rag_index`` from sentinel. The RAG
    refresh module reads ``session.session_id`` to build the
    pgvector collection name. Sentinel's own ``SentinelStore.session_id``
    is the legacy ``sentinel:<watchlist>`` id — using it would put
    RAG chunks into a per-watchlist collection the GUI's chat
    ``search_workspace_rag`` never queries. This shim exposes the
    target session id under the same attribute name so the
    collection lands in the user's main workspace.

    Anything else ``refresh_session_rag_index`` reaches into
    (``config`` etc.) is forwarded to ``store.config`` so we
    don't have to fake the full ``ChatSession`` surface.
    """

    def __init__(self, target_session_id: str, store: "SentinelStore") -> None:
        self._target_session_id = target_session_id
        self._store = store

    @property
    def session_id(self) -> str:
        return self._target_session_id

    @property
    def config(self) -> LitTraceConfig:
        return self._store.config

    @property
    def root(self) -> Path:
        return Path(self._store.config.storage.sessions_dir) / self._target_session_id

    @property
    def workspace_dir(self) -> Path:
        # Round 29: ``refresh_session_rag_index`` (and a couple of
        # other RAG helpers) reach for ``session.workspace_dir``
        # directly to write RAG chunk files under the session
        # root. The shim only exposed ``session_id`` / ``config`` /
        # ``root`` previously — the missing ``workspace_dir`` was
        # caught as ``AttributeError`` by the surrounding
        # ``try/except`` and surfaced as "RAG refresh skipped: …".
        # Without that error the chunks would have been silently
        # dropped; with it the user sees a noisy warning on every
        # run. Forward the same path the GUI's main ``ChatSession``
        # uses (``root / "workspace"``) so the shim is a complete
        # enough stand-in.
        return self.root / "workspace"

    @property
    def messages_path(self) -> Path:
        return self.root / "messages.jsonl"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def artifact_index_path(self) -> Path:
        return self.workspace_dir / "artifact_index.json"

    @property
    def snapshots_dir(self) -> Path:
        return self.workspace_dir / "snapshots"

    @property
    def structured_documents_dir(self) -> Path:
        return self.workspace_dir / "structured_documents"

    @property
    def evidence_dir(self) -> Path:
        return self.workspace_dir / "evidence"

    @property
    def releases_dir(self) -> Path:
        return self.workspace_dir / "releases"

    @property
    def rag_dir(self) -> Path:
        return self.workspace_dir / "rag"

    @property
    def metadata_store_backend(self) -> str:
        return self._store.config.metadata_store.backend

    @property
    def metadata_postgres_dsn(self) -> str:
        return self._store.config.metadata_store.postgres_dsn

    @property
    def metadata_schema_name(self) -> str:
        return self._store.config.metadata_store.schema_name


def _build_access_tasks(workspace: LiteratureWorkspace, state: SentinelState) -> list[AccessTask]:
    tasks: list[AccessTask] = []
    for paper_id in workspace.context.active_papers:
        paper = workspace.papers[paper_id]
        report = workspace.full_text_reports.get(paper_id)
        if paper.access_type == AccessType.REQUIRES_LOGIN or (
            report and report.login_required_candidate_count > 0 and not report.best_pdf_url
        ):
            tasks.append(
                AccessTask(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    doi=paper.doi,
                    publisher=paper.publisher,
                    landing_url=str(report.best_landing_url) if report and report.best_landing_url else None,
                    reason="requires_institution_login",
                )
            )
    return tasks


def _merge_access_tasks(existing: list[AccessTask], new_tasks: list[AccessTask]) -> list[AccessTask]:
    by_paper_id = {task.paper_id: task for task in existing}
    for task in new_tasks:
        by_paper_id[task.paper_id] = task
    return list(by_paper_id.values())


def _mirror_workspace_to_main_session(
    *,
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    sentinel_session_id: str,
) -> None:
    """Round 23: write ``workspace`` into the LitTrace main chat
    session's state_db row so the GUI's context panel actually
    surfaces the papers sentinel just found.

    Sentinel historically only wrote into its own
    ``sentinel:<watchlist_id>`` session row (see
    ``store.session_id``), which the GUI's main ``_context_panel``
    — which reads ``self._controller.workspace`` — never saw.
    Symptom: daily-run dialog said "已检索到 X 篇" but
    "打开上下文" landed on an empty list. We now write into
    whichever session row is the "main" one for the running GUI.

    Strategy: pick the most-recently-updated session row that
    isn't the sentinel session itself. This is a heuristic
    (matches what ``littrace-qt`` would have created on startup)
    but avoids hard-coding a session_id the GUI would have to
    pass in. If no other session exists we silently no-op —
    the operator can still browse ``sentinel:<watchlist_id>``
    directly via ``littrace session open``.
    """
    try:
        store = state_store_from_config(config)
    except Exception:
        return
    candidates: list[str] = []
    # The state_db API is ``list_session_states(limit=N)`` returning
    # ``SessionSummaryRecord`` objects. We pull ``session_id`` off
    # each summary and drop the sentinel session itself.
    summaries: list = []
    method = getattr(store, "list_session_states", None)
    if callable(method):
        try:
            summaries = method(limit=50)
        except Exception:
            summaries = []
    candidates = [
        s.session_id for s in summaries
        if getattr(s, "session_id", None)
        and s.session_id != sentinel_session_id
    ]
    if not candidates:
        return
    # Pick the most-recently-updated non-sentinel session so the
    # GUI's "active" workspace wins.
    target_state = None
    target_session_id = candidates[0]
    for sid in candidates:
        try:
            row = store.get_session_state(sid)
        except Exception:
            continue
        if row is None:
            continue
        if target_state is None:
            target_state = row
            target_session_id = sid
            continue
        cur = getattr(row, "updated_at", None)
        prev = getattr(target_state, "updated_at", None)
        if cur and (not prev or str(cur) > str(prev)):
            target_state = row
            target_session_id = sid
    if target_state is None:
        return
    # Merge: take the existing workspace, overlay sentinel's paper
    # / parsed / claims dicts, and append active_papers. We don't
    # overwrite anything that already exists in the GUI session —
    # the operator may have manually pinned/unpinned papers we
    # shouldn't clobber.
    existing = target_state.workspace_json
    if isinstance(existing, str):
        import json as _json
        try:
            existing = _json.loads(existing)
        except Exception:
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    sentinel_ws = workspace.model_dump(mode="json")
    for key in (
        "papers", "parsed_papers", "performance_cells",
        "evidence_records", "claims", "guard_reports",
        "full_text_reports", "resolution_decisions",
        "claim_verification_reports", "release_snapshots",
        "supplementary_links",
    ):
        sentinel_block = sentinel_ws.get(key) or {}
        if not isinstance(sentinel_block, dict):
            continue
        existing_block = existing.get(key)
        if not isinstance(existing_block, dict):
            existing_block = {}
        # Sentinel is the source of truth for papers it surfaced;
        # the operator's UI may have toggled ``active_papers``
        # independently though, so we preserve that set.
        existing_block.update(sentinel_block)
        existing[key] = existing_block
    # Build a LiteratureWorkspace object out of the merged dict so
    # ``save_workspace`` can ``getattr(ws, "context").active_papers``
    # rather than ``ws["context"]["active_papers"]`` (the latter is
    # what made the first cut silently fail — ``save_workspace`` is
    # typed).
    sentinel_active = list(sentinel_ws.get("context", {}).get("active_papers", []))
    ctx = existing.get("context") or {}
    existing_active = list(ctx.get("active_papers", []) if isinstance(ctx, dict) else [])
    merged_active = list(dict.fromkeys(existing_active + sentinel_active))
    if not isinstance(ctx, dict):
        ctx = {}
    ctx["active_papers"] = merged_active
    existing["context"] = ctx
    try:
        from littrace.models import LiteratureWorkspace as _LitWS
        merged_ws = _LitWS.model_validate(existing)
    except Exception:
        # Validation failed — fall back to writing the raw dict and
        # letting the StateStore journal catch any shape drift on
        # the next read.
        try:
            target_state.workspace_json = existing
            store.upsert_session_state(target_state)
        except Exception:
            pass
        return
    try:
        from littrace.session import ChatSession
        # ChatSession.from_root expects a ``root`` Path. Pull it from
        # the state row when available; fall back to ``sessions_dir``
        # + session_id.
        row_root = getattr(target_state, "root", None) or (
            Path(config.storage.sessions_dir) / target_session_id
        )
        target_session_obj = ChatSession.from_root(
            row_root, target_session_id, config=config,
        )
        _save_workspace_to_session(
            target_session_obj, merged_ws, config=config,
        )
    except Exception:
        # Last-ditch fallback if ChatSession.from_root fails (e.g.
        # a Postgres-backed row without a filesystem root).
        try:
            target_state.workspace_json = existing
            store.upsert_session_state(target_state)
        except Exception:
            pass
