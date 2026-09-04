from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any
from dataclasses import dataclass
from pathlib import Path

# Round 22: Python's stdout is line-buffered by default, so a
# ``print()`` call gets held in the buffer until the next newline
# or until the process flushes. When the GUI's daily-run driver
# SIGTERMs the sentinel subprocess via ``proc.terminate()``
# (PER_ROUND_TIMEOUT = 180s), the buffered stdout never reaches
# the parent's pipe — ``proc.stdout.read()`` returns an empty
# string and ``_summarise_sentinel_output`` finds no ``new_candidates:``
# / ``downloaded:`` lines → the dialog displays 0/0/0/1 instead
# of the real counters.
#
# Force unbuffered stdout at process startup so every print() lands
# in the pipe immediately. ``flush=True`` on individual print()s
# would also work but is easy to forget when adding new diagnostics.
sys.stdout.reconfigure(line_buffering=True)

from littrace.attachments import attach_pdf_to_paper, check_download_presence
from littrace.workflow_status import build_workflow_status
from littrace.quality_audits import audit_parser, audit_storyline, audit_tables
from littrace.auto_resume import auto_resume_downloaded_pdfs_async
from littrace.access_layer import (
    browser_login_session_for_paper,
    check_cdp_status,
    launch_login_for_paper,
    publisher_window_session_name_for_chat,
)
from littrace.chat import handle_chat
from littrace.chrome_profiles import (
    build_browser_setup_report,
    discover_chrome_profiles,
    format_shell_command,
    launch_chrome_for_cdp,
)
from littrace.config import load_config
from littrace.config_wizard import write_config_template
from littrace.download_jobs import (
    download_jobs_status,
    requeue_dead_download_jobs,
    run_download_job_daemon,
    run_pending_download_jobs,
)
from littrace.skill_runner import (
    build_quality_report_skill,
    build_research_plan_skill,
    export_session_bundle_skill,
    resolve_workspace_full_text_skill,
)
from littrace.retrieval.full_text import (
    backfill_workspace_by_dois,
    full_text_config_warnings,
)
from littrace.retrieval.rag_refresh import refresh_session_rag_index
from littrace.retrieval.rag_search import search_session_rag
from littrace.research_background import set_workspace_research_background
from littrace.rag_jobs import (
    iter_sentinel_watchlist_ids,
    iter_workspace_session_ids,
    run_daily_rag_daemon,
    run_daily_rag_maintenance,
    run_pending_embedding_jobs,
)
from littrace.rag_ops import (
    build_rag_jobs_status_report,
    requeue_dead_rag_jobs,
    run_rag_doctor,
)
from littrace.session_metrics import build_session_knowledge_metrics
from littrace.artifact_ops import reconcile_session_artifacts
from littrace.evaluation.golden_eval import run_golden_eval
from littrace.models import ChatRequest, LiteratureWorkspace
from littrace.parse_jobs import (
    parse_jobs_status,
    requeue_dead_parse_jobs,
    run_parse_job_daemon,
    run_pending_parse_jobs,
)
from littrace.evaluation.pdf_benchmark import benchmark_pdf_parsing
from littrace.publisher_connectors import build_publisher_search_plan
from littrace.publisher_retrieval import (
    fetch_publisher_search_results,
    merge_retrieval_result_into_workspace,
)
from littrace.publisher_session import build_publisher_session_e2e_report
from littrace.publisher_e2e import run_interactive_publisher_e2e
from littrace.rerank_learning import learn_rerank_policy_from_golden
from littrace.evaluation.retrieval_eval import run_retrieval_golden_eval
from littrace.sentinel.cli import access_review as sentinel_access_review
from littrace.sentinel.cli import init_sentinel, run_sentinel
from littrace.sentinel.cli import resume_after_login as sentinel_resume_after_login
from littrace.sentinel.state import Watchlist
from littrace.sentinel.storage import (
    ensure_sentinel_store,
    get_sentinel_store,
    load_sentinel_state,
    load_watchlist,
)
from littrace.session import (
    append_message,
    create_chat_session,
    load_or_create_session,
    load_workspace,
    save_workspace,
)
from littrace.publication import render_publication_storyline
from littrace.evidence.storyline_review import review_storyline
from littrace.supplementary import attach_supplementary_file
from littrace.table_jobs import (
    requeue_dead_table_jobs,
    run_pending_table_jobs,
    run_table_job_daemon,
    table_jobs_status,
)
from littrace.evidence.tables import decide_artifact_extraction_need


@dataclass
class ShellState:
    workspace: LiteratureWorkspace
    session_id: str
    session_root: str
    context_visible: bool = True


def main() -> None:
    # A second positional arg that is not a recognised subcommand should
    # fail loud (exit 1) instead of silently dropping into the REPL shell.
    #
    # The shell collapses unquoted words into one argv token (e.g.
    # ``littrace sentinel status`` arrives as ``sys.argv[1] == "sentinel status"``).
    # Split sys.argv[1] on whitespace and rewrite sys.argv so the per-subcommand
    # handlers can keep using ``sys.argv[2]`` for the action and ``sys.argv[2:]``
    # for the rest. After this rewrite, ``sys.argv == ["littrace", "sentinel",
    # "status", ...]`` regardless of how the user quoted the command.
    if len(sys.argv) > 1:
        head, *rest = sys.argv[1].split()
        sys.argv = [sys.argv[0], head, *rest, *sys.argv[2:]]
    first_token = sys.argv[1] if len(sys.argv) > 1 else ""
    if first_token.startswith("-"):
        # Flag-like argv (e.g. --help) goes to the REPL shell.
        pass
    elif first_token and first_token not in {
        "sentinel",
        "rag",
        "doctor",
        "jobs",
        "metrics",
        "setup-browser",
        "publisher-e2e",
        "compaction",
        "eval-from-rollout",
        "plugin",
    }:
        print(
            f"Unknown subcommand: {first_token!r}. "
            "Run `littrace` with no args for the interactive shell, or pass "
            "one of: sentinel, rag, jobs, doctor, metrics, setup-browser, publisher-e2e.",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(sys.argv) > 1 and sys.argv[1] == "sentinel":
        config = load_config()
        asyncio.run(_run_sentinel_command(config))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "rag":
        config = load_config()
        asyncio.run(_run_rag_command(config))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "compaction":
        config = load_config()
        asyncio.run(_run_compaction_command(config))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "eval-from-rollout":
        _run_eval_from_rollout_command(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "plugin":
        _run_plugin_command(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "jobs":
        config = load_config()
        asyncio.run(_run_jobs_command(config))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        config = load_config()
        _print_doctor(config)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "metrics":
        config = load_config()
        _run_metrics_command(config)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "setup-browser":
        config = load_config()
        profile = _arg_value("--profile")
        launch = "--no-launch" not in sys.argv and (
            "--launch" in sys.argv or config.cdp_downloader.auto_launch_chrome
        )
        _print_browser_setup(config, profile_name=profile, launch=launch)
        return
    if len(sys.argv) > 2 and sys.argv[1] == "publisher-e2e":
        config = load_config()
        doi = sys.argv[2]
        timeout = _arg_float("--timeout", 900.0)
        poll = _arg_float("--poll", 5.0)
        max_reopens = int(_arg_float("--max-browser-reopens", 2.0))
        wait_user_action = "--no-wait-user-action" not in sys.argv
        user_action_timeout = _arg_float_or_none("--user-action-timeout")
        asyncio.run(
            _run_publisher_e2e_command(
                config,
                doi,
                timeout,
                poll,
                wait_user_action,
                user_action_timeout,
                max_reopens,
            )
        )
        return
    asyncio.run(run_shell())


async def _run_publisher_e2e_command(
    config,
    doi: str,
    timeout: float,
    poll: float,
    wait_user_action: bool,
    user_action_timeout: float | None,
    max_browser_reopens: int,
) -> None:
    report = await run_interactive_publisher_e2e(
        config.model_copy(
            update={
                "browser": config.browser.model_copy(
                    update={"allow_confirm_browser_fallback": True}
                )
            }
        ),
        doi,
        timeout_seconds=timeout,
        poll_interval_seconds=poll,
        wait_for_user_action=wait_user_action,
        user_action_timeout_seconds=user_action_timeout,
        max_browser_reopens=max_browser_reopens,
    )
    print(
        f"Publisher E2E: completed={report.completed}, downloaded={report.downloaded_pdf}, parsed={report.parsed_full_text}"
    )
    print(f"session={report.session_name}")
    print(f"target={report.target_path}")
    print(
        f"attempts={report.attempts}, elapsed={report.elapsed_seconds}s, access={report.last_access_state}"
    )
    if report.institutional_login_opened:
        print("institutional_login_opened=true")
    if report.needs_user_action and report.user_action_message:
        print(f"user_action={report.user_action_message}")
    if report.last_error:
        print(f"error={report.last_error}")
    if report.warnings:
        print("warnings=" + "；".join(report.warnings))


async def _run_compaction_command(config) -> None:
    from littrace.codex_runtime.compaction import (
        run_pending_compaction,
        run_pending_compaction_daemon,
    )

    if len(sys.argv) > 2 and sys.argv[2] == "daemon":
        interval = 60.0
        for arg in sys.argv[3:]:
            if arg.startswith("--interval-seconds="):
                interval = float(arg.split("=", 1)[1])
        await run_pending_compaction_daemon(
            config, interval_seconds=interval, run_immediately=True,
        )
        return
    report = await run_pending_compaction(config)
    print(
        f"compaction finished: enqueued={report.enqueued} "
        f"succeeded={report.succeeded} failed={report.failed}"
    )
    for warning in report.warnings:
        print(f"  - {warning}")


async def _run_sentinel_command(config) -> None:
    if len(sys.argv) < 3:
        print("用法：littrace sentinel init|run|status|access-review|resume-after-login ...")
        return
    action = sys.argv[2]
    watchlist_id = _arg_value("--watchlist") or _arg_value("--watchlist-id") or "mxene_sensor"
    topic = _arg_value("--topic") or _arg_value("--objective")
    # Round 17: thread the user-selected year range and minimum
    # target through to the watchlist so ``run_sentinel`` picks
    # them up. ``None`` / unset falls back to the watchlist's
    # existing value (or the Watchlist default for new fields).
    year_min_arg = _arg_int("--year-min", None)
    year_max_arg = _arg_int("--year-max", None)
    target_papers_arg = _arg_int("--target-papers", None)

    if action == "init":
        root = init_sentinel(config, watchlist_id, topic or watchlist_id)
        print(f"sentinel initialized: {root}")
        return
    if action == "run":
        # Apply the optional override flags by patching the
        # watchlist BEFORE constructing ``LiteratureSentinel``
        # (which calls ``save_watchlist`` in its constructor — the
        # override therefore persists across the run).
        from littrace.sentinel.storage import (
            ensure_sentinel_store,
            load_watchlist,
        )
        store = ensure_sentinel_store(
            config,
            Watchlist(watchlist_id=watchlist_id, topic=topic or watchlist_id),
        )
        watchlist = load_watchlist(store)
        updates: dict[str, Any] = {}
        if topic:
            updates["topic"] = topic
            updates["objective"] = topic
        if year_min_arg is not None:
            updates["year_min"] = year_min_arg
        if year_max_arg is not None:
            updates["year_max"] = year_max_arg
        if target_papers_arg is not None:
            updates["target_papers"] = target_papers_arg
        if updates:
            watchlist = watchlist.model_copy(update=updates)
        # Round 24: thread the GUI main session id through so
        # sentinel writes paper metadata + RAG chunks directly into
        # the main workspace, not into a separate
        # ``sentinel:<watchlist_id>`` session. The mirror step we
        # added in Round 23 was a band-aid; the proper fix is for
        # the user-facing workspace to be the single source of
        # truth. We also accept the legacy
        # ``LITTRACE_SENTINEL_MAIN_SESSION`` env var so the
        # standalone ``littrace sentinel run`` (no GUI) keeps
        # working — it falls back to the sentinel session when
        # nothing is passed.
        main_session_id = _arg_value("--main-session-id") or os.environ.get(
            "LITTRACE_SENTINEL_MAIN_SESSION"
        )
        # Round 29: defensive wrap. ``run_sentinel`` can fail
        # partway (the most recent occurrence was an
        # ``AttributeError`` on a renamed ``WorkspaceFilters``
        # field that crashed the subprocess *before* the
        # summary lines were printed — leaving the GUI's daily
        # dialog at 0/0/0 with no diagnostic). Now we catch any
        # unexpected exception, still emit the counter lines
        # (zeros + a warning), so the GUI parses a coherent
        # result and surfaces the real error in the warning
        # line instead of silently dropping the run.
        try:
            result = await run_sentinel(
                config, watchlist, main_session_id=main_session_id,
            )
        except Exception as sentinel_exc:
            import traceback
            tb = traceback.format_exc(limit=4)
            run_id = f"error-{int(time.time())}"
            print(f"run_id: {run_id}", flush=True)
            print(f"watchlist: {watchlist.watchlist_id}", flush=True)
            print(f"topic: {watchlist.topic}", flush=True)
            print(f"year_range: {watchlist.year_min}-{watchlist.year_max or 'now'}", flush=True)
            print(f"target_papers: {watchlist.target_papers}", flush=True)
            print("new_candidates: 0", flush=True)
            print("candidates_total: 0", flush=True)
            print("candidates_seen: 0", flush=True)
            print("downloaded: 0", flush=True)
            print("parsed: 0", flush=True)
            print("access_tasks: 0", flush=True)
            print(
                "warnings: sentinel crashed before printing "
                f"summary: {sentinel_exc.__class__.__name__}: {sentinel_exc} | "
                f"trace: {tb.replace(chr(10), ' | ')}",
                flush=True,
            )
            return
        # Round22: every line ends with ``flush=True`` because the
        # GUI's daily-run driver SIGTERMs us on PER_ROUND_TIMEOUT
        # (180s); without explicit flush the buffered stdout is lost
        # mid-run and ``_summarise_sentinel_output`` returns 0/0/0/0
        # to the dialog. We tried ``sys.stdout.reconfigure`` at
        # module import (still in place below) but it doesn't survive
        # the asyncio.run + nest-of-callbacks path on every Python
        # version — the explicit flush is the belt-and-braces.
        print(f"run_id: {result.summary.run_id}", flush=True)
        print(f"watchlist: {result.summary.watchlist_id}", flush=True)
        print(f"topic: {result.summary.topic}", flush=True)
        print(f"year_range: {watchlist.year_min}-{watchlist.year_max or 'now'}", flush=True)
        print(f"target_papers: {watchlist.target_papers}", flush=True)
        print(f"new_candidates: {result.summary.new_candidate_count}", flush=True)
        print(f"candidates_total: {result.summary.candidate_count}", flush=True)
        print(f"candidates_seen: {result.summary.seen_candidate_count}", flush=True)
        print(f"downloaded: {result.summary.downloaded_count}", flush=True)
        print(f"parsed: {result.summary.parsed_count}", flush=True)
        print(f"access_tasks: {result.summary.access_task_count}", flush=True)
        if result.summary.digest_path:
            print(f"digest: {result.summary.digest_path}", flush=True)
        if result.summary.warnings:
            print("warnings: " + "；".join(result.summary.warnings), flush=True)
        return
    if action == "status":
        store = get_sentinel_store(config, watchlist_id)
        state = load_sentinel_state(store)
        print(f"watchlist: {state.watchlist.watchlist_id}")
        print(f"topic: {state.watchlist.topic}")
        print(f"last_run_at: {state.last_run_at or 'never'}")
        print(f"seen: {len(state.seen_paper_ids)}")
        print(f"access_tasks: {len(state.access_queue)}")
        print(f"retry_tasks: {len(state.retry_queue)}")
        print(f"digest_history: {len(state.digest_history)}")
        return
    if action == "access-review":
        tasks = sentinel_access_review(config, watchlist_id, topic)
        print(f"access_tasks: {len(tasks)}")
        for task in tasks[:20]:
            print(f"- {task.paper_id}: {task.reason} | {task.title}")
        return
    if action == "resume-after-login":
        result = await sentinel_resume_after_login(config, watchlist_id, topic)
        print(f"resumed run_id: {result.summary.run_id}")
        print(f"downloaded: {result.summary.downloaded_count}")
        print(f"parsed: {result.summary.parsed_count}")
        print(f"remaining_access_tasks: {result.summary.access_task_count}")
        return
    print(f"未知 sentinel 动作: {action}")


async def _run_rag_command(config) -> None:
    if len(sys.argv) < 3:
        print("用法：littrace rag refresh --session SESSION_ID | refresh-all | daily | daemon | jobs | doctor")
        return
    action = sys.argv[2]
    if action == "refresh":
        session_id = _arg_value("--session")
        if session_id is None and len(sys.argv) > 3 and not sys.argv[3].startswith("--"):
            session_id = sys.argv[3]
        if not session_id:
            print("用法：littrace rag refresh --session SESSION_ID")
            return
        session = load_or_create_session(config, session_id)
        workspace = load_workspace(session)
        _, report = await refresh_session_rag_index(config, session, workspace)
        save_workspace(session, workspace, config=config)
        _print_rag_refresh_report(report)
        return
    if action == "search":
        session_id = _arg_value("--session")
        query = _arg_value("--query")
        if not session_id or not query:
            print("用法：littrace rag search --session SESSION_ID --query QUERY")
            return
        session = load_or_create_session(config, session_id)
        result = await search_session_rag(config, session, query, top_k=_arg_int("--top-k", 5))
        if result is None:
            print("rag search: no session profile or rag is disabled")
            return
        print(f"rag profile: {result.profile.profile_id}")
        print(f"hits: {len(result.hits)}")
        for idx, hit in enumerate(result.hits, start=1):
            section = f" section={hit.section}" if hit.section else ""
            page = f" page={hit.page}" if hit.page is not None else ""
            print(f"{idx}. score={hit.score:.4f} paper={hit.paper_id}{section}{page}")
            print(f"   {hit.text[:240]}")
        return
    if action == "refresh-all":
        root = config.storage.sessions_dir
        if not root.exists():
            print("rag refresh-all: sessions_dir does not exist")
            return
        refreshed = 0
        failed = 0
        for session_dir in sorted(root.iterdir()):
            if not session_dir.is_dir() or session_dir.name == "sentinel":
                continue
            # Postgres is the source of truth (round 3 topic B). A
            # session is "live" when it has a session_state row;
            # sessions that have a directory but no Postgres row are
            # pre-migration leftovers and the migration script
            # handles them. Skip them here so refresh-all does not
            # touch cold backups.
            session_id = session_dir.name
            if state_store_from_config(config).get_session_state(session_id) is None:
                continue
            try:
                session = load_or_create_session(config, session_dir.name)
                workspace = load_workspace(session)
                _, report = await refresh_session_rag_index(config, session, workspace)
                save_workspace(session, workspace, config=config)
                refreshed += 1
                print(
                    f"{session.session_id}: chunks={report.chunk_count}, upserted={report.upserted_count}, skipped={report.skipped}"
                )
            except Exception as exc:
                failed += 1
                print(f"{session_dir.name}: failed: {exc.__class__.__name__}: {exc}")
        print(f"rag refresh-all: refreshed={refreshed}, failed={failed}")
        return
    if action == "daily":
        report = await run_daily_rag_maintenance(config)
        _print_daily_rag_report(report)
        return
    if action == "jobs":
        subaction = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "status"
        if subaction == "status":
            report = build_rag_jobs_status_report(
                config,
                status=_arg_value("--status"),
                session_id=_arg_value("--session"),
                limit=_arg_int("--limit", 20),
            )
            _print_rag_jobs_status_report(report)
            return
        if subaction == "run":
            report = await run_pending_embedding_jobs(config, limit=_arg_int("--limit", 20))
            print(f"processed: {report.processed}")
            print(f"failed: {report.failed}")
            print(f"skipped: {report.skipped}")
            if report.job_ids:
                print("job_ids: " + ", ".join(report.job_ids))
            if report.warnings:
                print("warnings: " + "；".join(report.warnings))
            return
        if subaction == "requeue-dead":
            count = requeue_dead_rag_jobs(
                config,
                session_id=_arg_value("--session"),
                limit=_arg_int("--limit", 20),
            )
            print(f"requeued: {count}")
            return
        if subaction == "reconcile":
            session_id = _arg_value("--session")
            if not session_id:
                print("rag jobs reconcile requires --session SESSION_ID")
                return
            report = reconcile_session_artifacts(config, session_id, limit=_arg_int("--limit", 200))
            print(f"checked: {report.checked}; missing: {report.missing}; requeued: {report.requeued}")
            if report.warnings:
                print("warnings: " + "；".join(report.warnings))
            return
        print("用法：littrace rag jobs [status|run|requeue-dead|reconcile] [--session SESSION_ID] [--status STATUS] [--limit N]")
        return
    if action == "doctor":
        _print_rag_doctor_report(run_rag_doctor(config))
        return
    if action == "daemon":
        interval_hours = _arg_float("--interval-hours", 24.0)
        run_immediately = "--no-immediate-run" not in sys.argv
        print(
            f"rag daemon starting: watchlists={len(iter_sentinel_watchlist_ids(config.storage.sessions_dir))}, "
            f"sessions={len(iter_workspace_session_ids(config.storage.sessions_dir))}, "
            f"interval_hours={interval_hours}, run_immediately={run_immediately}"
        )
        await run_daily_rag_daemon(
            config,
            interval_hours=interval_hours,
            run_immediately=run_immediately,
        )
        return
    print(f"未知 rag 动作: {action}")


async def _run_jobs_command(config) -> None:
    if len(sys.argv) < 4 or sys.argv[2] not in {"download", "parse", "table"}:
        print(
            "用法：littrace jobs download|parse|table "
            "run|daemon|status|requeue-dead [--session SESSION_ID] [--limit N]"
        )
        return
    job_kind = sys.argv[2]
    action = sys.argv[3]
    limit = _arg_int("--limit", config.download_retry.batch_size)
    if action == "run":
        if job_kind == "download":
            report = await run_pending_download_jobs(config, limit=limit)
            details = (
                f"downloaded: {report.downloaded}\n"
                f"requires_login: {report.requires_login}"
            )
        elif job_kind == "parse":
            report = await run_pending_parse_jobs(config, limit=limit)
            details = (
                f"parsed: {report.parsed}\n"
                f"parse_failed: {report.parse_failed}\n"
                f"stale: {report.stale}"
            )
        else:
            report = await run_pending_table_jobs(config, limit=limit)
            details = (
                f"performance_cells: {report.performance_cells}\n"
                f"structured_artifacts: {report.structured_artifacts}\n"
                f"stale: {report.stale}"
            )
        print(f"processed: {report.processed}")
        print(f"failed: {report.failed}")
        print(details)
        if report.job_ids:
            print("job_ids: " + ", ".join(report.job_ids))
        if report.warnings:
            print("warnings: " + "；".join(report.warnings))
        return
    if action == "daemon":
        interval = _arg_float(
            "--interval-seconds",
            config.download_retry.interval_seconds,
        )
        print(
            f"{job_kind} job daemon starting: "
            f"interval_seconds={interval}, batch_size={limit}"
        )
        if job_kind == "download":
            await run_download_job_daemon(
                config,
                interval_seconds=interval,
                limit=limit,
            )
        elif job_kind == "parse":
            await run_parse_job_daemon(
                config,
                interval_seconds=interval,
                limit=limit,
            )
        else:
            await run_table_job_daemon(
                config,
                interval_seconds=interval,
                limit=limit,
            )
        return
    if action == "status":
        status_args = {
            "session_id": _arg_value("--session"),
            "status": _arg_value("--status"),
            "limit": limit,
        }
        if job_kind == "download":
            queue, jobs = download_jobs_status(config, **status_args)
        elif job_kind == "parse":
            queue, jobs = parse_jobs_status(config, **status_args)
        else:
            queue, jobs = table_jobs_status(config, **status_args)
        print(
            f"queued={queue.queued} running={queue.running} failed={queue.failed} "
            f"dead={queue.dead} completed={queue.completed} ready={queue.ready_to_claim}"
        )
        for job in jobs:
            print(
                f"- {job.task_id}: status={job.status} attempts={job.attempt_count} "
                f"session={job.session_id} error={job.last_error or '-'}"
            )
        return
    if action == "requeue-dead":
        if job_kind == "download":
            count = requeue_dead_download_jobs(config, limit=limit)
        elif job_kind == "parse":
            count = requeue_dead_parse_jobs(config, limit=limit)
        else:
            count = requeue_dead_table_jobs(config, limit=limit)
        print(f"requeued: {count}")
        return
    print(
        "用法：littrace jobs download|parse|table "
        "run|daemon|status|requeue-dead [--session SESSION_ID] [--limit N]"
    )


def _run_metrics_command(config) -> None:
    if len(sys.argv) < 3 or sys.argv[2] != "session":
        print("用法：littrace metrics session --session SESSION_ID")
        return
    session_id = _arg_value("--session")
    if session_id is None and len(sys.argv) > 3 and not sys.argv[3].startswith("--"):
        session_id = sys.argv[3]
    if not session_id:
        print("用法：littrace metrics session --session SESSION_ID")
        return
    report = build_session_knowledge_metrics(
        config,
        session_id,
        artifact_limit=_arg_int("--artifact-limit", 200),
    )
    print(f"session: {report.session_id}")
    print(f"readiness: {report.readiness}")
    discovery = report.discovery
    acquisition = report.acquisition
    rag = report.rag
    consistency = report.consistency
    print(f"Discovery: 今日相关新增 {discovery.value} 篇 [{discovery.status}]")
    print(
        "Acquisition: PDF 获取率 "
        f"{_format_metric_percent(acquisition.value)} "
        f"({acquisition.numerator}/{acquisition.denominator}) [{acquisition.status}]"
    )
    print(
        "RAG: freshness "
        f"{_format_metric_percent(rag.value)}，stale {rag.stale_count} "
        f"({rag.numerator}/{rag.denominator}) [{rag.status}]"
    )
    print(
        "Consistency: pass "
        f"{_format_metric_percent(consistency.value)}，missing {consistency.missing_count} "
        f"({consistency.numerator}/{consistency.denominator}) [{consistency.status}]"
    )
    if report.artifact_audit is not None:
        audit = report.artifact_audit
        print(
            "artifact_audit: "
            f"artifacts={audit.artifact_count}, checked={audit.checked_count}, "
            f"missing={audit.missing_object_count}, size_bytes={audit.total_size_bytes}"
        )
    if report.warnings:
        print("warnings: " + "；".join(report.warnings))


def _run_plugin_command(argv: list[str]) -> None:
    """CLI driver for ``littrace plugin list|info``.

    Round 13 step 2: read-only introspection of the entry-point
    plugins that ``pip install`` has dropped on the Python
    path. ``plugin list`` prints every entry point grouped by
    distribution; ``plugin info <name>`` prints one entry
    point's full payload (callable module + docstring).
    """
    from littrace.marketplace import list_plugins, plugin_info

    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: littrace plugin list\n"
            "       littrace plugin info <name>\n"
            "\n"
            "Third-party plugins are discovered via Python entry-point\n"
            "groups: littrace.skills, littrace.mcp_servers,\n"
            "littrace.harnesses. Install a plugin with\n"
            "``pip install <package>`` and it shows up here.",
            file=sys.stderr,
        )
        sys.exit(2)
    sub = argv[0]
    if sub == "list":
        result = list_plugins()
        if not result.entries:
            print("no third-party LitTrace plugins installed.")
            return
        # Group by distribution so a single ``pip install``
        # surfaces as one block.
        by_dist: dict[str, list[Any]] = {}
        for entry in result.entries:
            by_dist.setdefault(entry.dist or "(unknown)", []).append(entry)
        for dist, entries in sorted(by_dist.items()):
            print(f"== {dist} ==")
            for entry in sorted(entries, key=lambda e: (e.group, e.name)):
                print(
                    f"  [{entry.group:<22s}] {entry.name:<32s} "
                    f"-> {entry.value.__module__}.{entry.value.__qualname__}"
                )
        if result.failures:
            print()
            print("load failures:")
            for failure in result.failures:
                print(f"  - {failure}")
        return
    if sub == "info":
        if len(argv) < 2:
            print(
                "usage: littrace plugin info <name> "
                "(<name> matches either an entry-point name or "
                "a distribution name)",
                file=sys.stderr,
            )
            sys.exit(2)
        info = plugin_info(argv[1])
        if info is None:
            print(f"plugin not found: {argv[1]!r}", file=sys.stderr)
            sys.exit(1)
        for key, value in info.items():
            print(f"{key}: {value}")
        return
    print(f"unknown plugin subcommand: {sub!r}", file=sys.stderr)
    sys.exit(2)


def _run_eval_from_rollout_command(argv: list[str]) -> None:
    """CLI driver for ``littrace eval-from-rollout <path>``.

    Round 10 step 2: converts rollout JSONL traces to harness
    check items and prints the resulting reports. Defaults to
    the two standard checks (``check_citations`` and
    ``check_retry_health``); callers can pick a subset with
    ``--checks``. ``--report`` writes a JSON dump so CI can
    attach the result as a build artifact.
    """
    from littrace.evaluation.harnesses import HarnessConfig, HarnessEngine
    from littrace.evaluation.rollout_eval import convert_directory, merge_bundles

    if not argv or argv[0].startswith("-"):
        print(
            "usage: littrace eval-from-rollout <rollout-dir-or-file> "
            "[--checks check_citations,check_retry_health] "
            "[--report out.json] [--performance-confidence-threshold F] "
            "[--max-retry-rate F] [--max-failure-rate F]",
            file=sys.stderr,
        )
        sys.exit(2)
    path = Path(argv[0])
    checks_filter = _arg_value("--checks")
    report_path = _arg_value("--report")
    perf_threshold = _arg_float("--performance-confidence-threshold", 0.6)
    max_retry_rate = _arg_float("--max-retry-rate", 0.5)
    max_failure_rate = _arg_float("--max-failure-rate", 0.2)

    bundles = convert_directory(path)
    if not bundles:
        print(f"no rollout JSONL files under {path}", file=sys.stderr)
        sys.exit(1)
    items_map = merge_bundles(bundles)
    config = HarnessConfig(
        performance_confidence_threshold=perf_threshold,
        max_retry_rate=max_retry_rate,
        max_failure_rate=max_failure_rate,
    )
    selected = checks_filter.split(",") if checks_filter else [
        "check_citations",
        "check_retry_health",
    ]
    engine = HarnessEngine(config=config)
    # ``run_with_deps`` is single-target. The CLI runs every
    # selected check independently and we skip any name the
    # registry does not know so an operator can pass an
    # arbitrary subset.
    reports: dict[str, Any] = {}
    for name in selected:
        if engine.registry.get(name) is None:
            print(f"  skip unknown check: {name}")
            continue
        items = items_map.get(name, [])
        reports[name] = engine.run(name, items)

    json_payload: list[dict[str, object]] = []
    for name, report in reports.items():
        print(f"=== {name} ===")
        print(f"  passed: {report.passed}")
        print(f"  score: {report.score:.2f}")
        print(f"  item_count: {report.item_count}")
        for finding in report.findings:
            # ``Severity`` is a plain str subclass, not a
            # ``StrEnum``, so ``str(finding.severity)`` is the
            # canonical way to render the level.
            print(
                f"  - [{finding.severity}] {finding.message}"
                + (f" (paper={finding.paper_id})" if finding.paper_id else "")
            )
        json_payload.append({
            "check": name,
            "passed": report.passed,
            "score": report.score,
            "item_count": report.item_count,
            "errors": report.errors,
            "warnings": report.warnings,
        })
    print()
    print(
        f"sessions: {sum(1 for b in bundles if b.session_id)}  "
        f"turns: {len(items_map.get('__turns__', []))}  "
        f"tool_calls: {len(items_map.get('__tool_calls__', []))}  "
        f"errors: {len(items_map.get('__errors__', []))}"
    )
    if report_path:
        Path(report_path).write_text(
            json.dumps(
                {
                    "rollout_path": str(path),
                    "checks": selected,
                    "reports": json_payload,
                    "summary": {
                        "sessions": sum(1 for b in bundles if b.session_id),
                        "turns": len(items_map.get("__turns__", [])),
                        "tool_calls": len(items_map.get("__tool_calls__", [])),
                        "errors": len(items_map.get("__errors__", [])),
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"report: {report_path}")


def _print_rag_refresh_report(report) -> None:
    print(f"rag profile: {report.profile_id}")
    print(f"session: {report.session_id}")
    print(f"collection: {report.collection_name}")
    print(f"backend: {report.backend}")
    print(f"chunks: {report.chunk_count}")
    print(f"upserted: {report.upserted_count}")
    if report.skipped:
        print("skipped: true")
    if getattr(report, "skip_reason", None):
        print(f"skip_reason: {report.skip_reason}")
    if report.warnings:
        print("warnings: " + "；".join(report.warnings))


def _print_daily_rag_report(report) -> None:
    print(f"started_at: {report.started_at}")
    if report.finished_at:
        print(f"finished_at: {report.finished_at}")
    print(f"sentinel_watchlists: {report.sentinel_watchlists}")
    print(f"sentinel_failed: {report.sentinel_failed}")
    print(f"sessions_refreshed: {report.sessions_refreshed}")
    print(f"sessions_skipped: {report.sessions_skipped}")
    print(f"sessions_failed: {report.sessions_failed}")
    print(f"artifacts_reconciled: {report.artifacts_reconciled}")
    print(f"missing_artifacts: {report.missing_artifacts}")
    print(f"embedding_requeued: {report.embedding_requeued}")
    print(f"outbox_dispatched: {report.outbox_dispatched}")
    print(f"outbox_failed: {report.outbox_failed}")
    if report.warnings:
        print("warnings: " + "；".join(report.warnings))


def _print_rag_jobs_status_report(report) -> None:
    print(f"configured: {report.configured}")
    if report.queue is not None:
        queue = report.queue
        print(
            "queue: "
            f"total={queue.total}, queued={queue.queued}, running={queue.running}, "
            f"failed={queue.failed}, dead={queue.dead}, completed={queue.completed}, "
            f"ready={queue.ready_to_claim}, reclaimable={queue.reclaimable_running}"
        )
        if queue.oldest_ready_at:
            print(f"oldest_ready_at: {queue.oldest_ready_at}")
        if queue.latest_error:
            print(f"latest_error: {queue.latest_error}")
    for job in report.jobs:
        suffix = f" error={job.last_error}" if job.last_error else ""
        print(
            f"{job.job_id} status={job.status} session={job.session_id} "
            f"artifact={job.artifact_id} attempts={job.attempt_count}{suffix}"
        )
    if report.warnings:
        print("warnings: " + "；".join(report.warnings))


def _print_rag_doctor_report(report) -> None:
    print(f"ok: {report.ok}")
    for check in report.checks:
        latency = f" ({check.latency_ms}ms)" if check.latency_ms is not None else ""
        detail = f": {check.detail}" if check.detail else ""
        print(f"{check.name}: {check.status}{latency}{detail}")


def _arg_float(name: str, default: float) -> float:
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    try:
        return float(sys.argv[index + 1])
    except (IndexError, ValueError):
        return default


def _arg_float_or_none(name: str) -> float | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    try:
        return float(sys.argv[index + 1])
    except (IndexError, ValueError):
        return None


def _arg_int(name: str, default: int) -> int:
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    try:
        return int(sys.argv[index + 1])
    except (IndexError, ValueError):
        return default


def _format_metric_percent(value: float | int | str | None) -> str:
    if isinstance(value, (float, int)):
        return f"{value * 100:.1f}%"
    return "N/A"


def _arg_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    try:
        return sys.argv[index + 1]
    except IndexError:
        return None


async def run_shell() -> None:
    config = load_config()
    session = create_chat_session(config)
    state = ShellState(
        workspace=LiteratureWorkspace(),
        session_id=session.session_id,
        session_root=str(session.root),
    )
    print("LitTrace agent shell")
    print(
        "输入研究任务开始。命令：/context /hide-context /show-context /papers "
        "/login N /browser-login N /attach N path.pdf /attach-si N path /full-text /publisher-retrieve family topic /check-downloads /resume-downloads /parse /table /storyline "
        "/dashboard /doctor /setup-browser /quality /agents /workflow /quality-audits /plan topic /init-config /set-bg topic /ocr-choice /storyline-report /storyline-review /benchmark /golden-eval /retrieval-eval /rerank-learn /publisher-session-test /export /quit"
    )
    print("对话例子：选择第 1、3 篇下载；全部下载；取消选择第 2 篇；生成发展脉络。")
    print(f"session: {state.session_id}")
    print(f"folder:  {state.session_root}")
    print()

    while True:
        try:
            message = input("littrace > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not message:
            continue
        if message in {"/quit", "/exit"}:
            print("bye")
            return
        if message == "/hide-context":
            state.context_visible = False
            state.workspace.context.visible_to_user = False
            print("已隐藏上下文窗。")
            continue
        if message == "/show-context":
            state.context_visible = True
            state.workspace.context.visible_to_user = True
            print("已显示上下文窗。")
            print(format_context_panel(state.workspace))
            continue
        if message in {"/context", "/papers"}:
            print(format_context_panel(state.workspace))
            continue
        if message in {"/dashboard", "/tui"}:
            print(format_dashboard(state))
            continue
        if message == "/doctor":
            _print_doctor(config)
            continue
        if message == "/setup-browser":
            _print_browser_setup(
                config,
                profile_name=None,
                launch=config.cdp_downloader.auto_launch_chrome,
            )
            continue
        if message == "/quality":
            report = build_quality_report_skill(config, state.workspace)
            print("Quality metrics:")
            for name, value in report.metrics.items():
                print(f"- {name}: {value}")
            if report.warnings:
                print("注意：" + "；".join(report.warnings[:8]))
            continue
        if message == "/ocr-choice":
            report = decide_artifact_extraction_need(state.workspace)
            print(f"OCR 建议: {report.recommended_parse_strategy}")
            print(f"理由: {report.reason}")
            print("按钮:")
            for button in report.buttons:
                marker = "推荐" if button.get("recommended") == "true" else "可选"
                print(
                    f"- [{marker}] {button['label']} -> parse_strategy={button['parse_strategy']}"
                )
                print(f"  {button['description']}")
            print("你也可以直接输入：只看文字层解析PDF / 强制OCR解析")
            continue
        if message == "/workflow":
            report = build_workflow_status(state.workspace)
            print(
                f"Workflow: ready={report.ready_count}, blocked={report.blocked_count}, complete={report.complete_count}"
            )
            for transition in report.transitions:
                print(
                    f"- {transition.source} -> {transition.target}: {transition.status} | {transition.artifact}"
                )
            if report.recommended_next_steps:
                print("下一步建议：" + "，".join(report.recommended_next_steps))
            continue
        if message == "/quality-audits":
            for report in [
                audit_parser(config, state.workspace),
                audit_tables(state.workspace),
                audit_storyline(state.workspace),
            ]:
                print(
                    f"- {report.component}: {'passed' if report.passed else 'needs work'} ({report.score})"
                )
                for finding in report.findings[:3]:
                    print(f"  - {finding}")
            continue
        if message == "/full-text":
            for warning in full_text_config_warnings(config):
                print(f"配置建议: {warning}")
            state.workspace = await resolve_workspace_full_text_skill(state.workspace, config)
            save_workspace(session, state.workspace, config=config)
            print(f"Full-text reports: {len(state.workspace.full_text_reports)}")
            for paper_id in state.workspace.context.active_papers[:12]:
                report = state.workspace.full_text_reports.get(paper_id)
                if not report:
                    continue
                print(
                    f"- {paper_id}: candidates={len(report.candidates)}, "
                    f"oa={report.open_access_candidate_count}, "
                    f"login={report.login_required_candidate_count}"
                )
                if report.best_pdf_url:
                    print(f"  pdf: {report.best_pdf_url}")
                elif report.best_landing_url:
                    print(f"  landing: {report.best_landing_url}")
                for warning in report.warnings[:2]:
                    print(f"  warning: {warning}")
            continue
        if message.startswith("/backfill-dois "):
            raw = message.removeprefix("/backfill-dois ").strip()
            dois = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
            before = len(state.workspace.context.active_papers)
            state.workspace = await backfill_workspace_by_dois(state.workspace, dois, config)
            save_workspace(session, state.workspace, config=config)
            added = len(state.workspace.context.active_papers) - before
            print(f"DOI backfill: added {added} papers.")
            continue
        if message.startswith("/plan "):
            topic = message.removeprefix("/plan ").strip()
            plan = await build_research_plan_skill(topic, state.workspace)
            print(f"Research plan: {plan.topic}")
            for index, step in enumerate(plan.steps, start=1):
                print(f"{index}. [{step.component}] {step.action} -> {step.expected_output}")
            if plan.warnings:
                print("注意：" + "；".join(plan.warnings))
            continue
        if message == "/init-config":
            result = write_config_template()
            print(f"Config: {'created' if result.created else 'not changed'} at {result.path}")
            if result.warnings:
                print("注意：" + "；".join(result.warnings))
            continue
        if message.startswith("/set-bg "):
            # Set the session's long-term research background so
            # downstream commands (``/search-papers``,
            # ``/downloads/...``, ``/storyline``) skip the
            # fast-gate clarification and the workspace is
            # tagged with a real topic. Mirrors the
            # ``POST /chat`` research-background handler in
            # the FastAPI layer so a TUI user does not have
            # to start the API server to bootstrap a session.
            background = message.removeprefix("/set-bg ").strip()
            if not background:
                print("用法：/set-bg 我研究的是<材料>在<场景>中的<问题>")
                continue
            state.workspace = set_workspace_research_background(
                state.workspace,
                background,
            )
            save_workspace(session, state.workspace, config=config)
            filters = state.workspace.context.filters
            print(f"研究背景已设置 (status={filters.research_background_status})")
            print(f"  主题: {filters.topic}")
            print(f"  时间: {filters.research_background_set_at}")
            print("接下来可以 /search-papers /downloads/plan 等。")
            continue
        if message == "/parse":
            message = "解析当前文献全文"
        if message == "/table":
            message = "生成当前文献性能对比表"
        if message == "/storyline":
            message = "生成当前文献发展脉络"
        if message == "/storyline-report":
            markdown, _ = render_publication_storyline(state.workspace, config)
            print(markdown)
            continue
        if message == "/storyline-review":
            report = review_storyline(state.workspace)
            print(
                f"Storyline review: {'passed' if report.passed else 'needs work'} "
                f"({report.claim_count} claims)"
            )
            for warning in report.warnings:
                print(f"- {warning}")
            continue
        if message.startswith("/login "):
            index = _parse_index_arg(message)
            paper_id = _paper_id_for_index(state.workspace, index) if index else None
            if not paper_id:
                print("没有找到这个编号的文献。")
                continue
            paper = state.workspace.papers[paper_id]
            result = launch_login_for_paper(
                config,
                paper,
                state.workspace.full_text_reports.get(paper_id),
            )
            print(f"登录页: {result.login_url or '无'}")
            print(f"目标路径: {result.target_path or '无'}")
            for instruction in result.instructions:
                print(f"- {instruction}")
            if result.error:
                print(f"错误: {result.error}")
            continue
        if message.startswith("/browser-login "):
            index = _parse_index_arg(message.replace("/browser-login", "/login", 1))
            paper_id = _paper_id_for_index(state.workspace, index) if index else None
            if not paper_id:
                print("没有找到这个编号的文献。")
                continue
            paper = state.workspace.papers[paper_id]
            plan = browser_login_session_for_paper(
                config,
                paper,
                state.workspace.full_text_reports.get(paper_id),
                browser_session_name=publisher_window_session_name_for_chat(state.session_id),
            )
            print(f"浏览器会话: {plan.login_url or '无'}")
            print(f"目标 PDF: {plan.target_path}")
            if plan.pdf_url:
                print(f"授权后后台 PDF: {plan.pdf_url}")
            if plan.browser_act_command:
                print("browser-act 命令:")
                print(" ".join(plan.browser_act_command))
            for step in plan.automation_steps:
                print(f"- {step}")
            if plan.error:
                print(f"错误: {plan.error}")
            continue
        if message.startswith("/attach "):
            parsed = _parse_attach_args(message)
            if not parsed:
                print("用法：/attach N /path/to/paper.pdf")
                continue
            index, source_path = parsed
            paper_id = _paper_id_for_index(state.workspace, index)
            if not paper_id:
                print("没有找到这个编号的文献。")
                continue
            result = attach_pdf_to_paper(config, state.workspace, paper_id, source_path)
            print(f"PDF 绑定: {'成功' if result.attached else '失败'}")
            print(f"目标路径: {result.target_path}")
            if result.error:
                print(f"错误: {result.error}")
            save_workspace(session, state.workspace, config=config)
            continue
        if message.startswith("/attach-si "):
            parsed = _parse_attach_args(message.replace("/attach-si", "/attach", 1))
            if not parsed:
                print("用法：/attach-si N /path/to/supporting-info.pdf")
                continue
            index, source_path = parsed
            paper_id = _paper_id_for_index(state.workspace, index)
            if not paper_id:
                print("没有找到这个编号的文献。")
                continue
            result = attach_supplementary_file(state.workspace, session, paper_id, source_path)
            print(f"SI 绑定: {'成功' if result.attached else '失败'}")
            if result.target_path:
                print(f"目标路径: {result.target_path}")
            if result.error:
                print(f"错误: {result.error}")
            save_workspace(session, state.workspace, config=config)
            continue
        if message.startswith("/publisher-retrieve "):
            parsed = _parse_publisher_retrieve_args(message)
            if not parsed:
                print("用法：/publisher-retrieve acs MXene sensor")
                continue
            family, topic = parsed
            plan_report = build_publisher_search_plan(topic, families=[family])
            if not plan_report.plans:
                print(f"没有 {family} 的 publisher 检索计划。")
                continue
            result = await fetch_publisher_search_results(config, plan_report.plans[0])
            state.workspace = merge_retrieval_result_into_workspace(state.workspace, result)
            save_workspace(session, state.workspace, config=config)
            print(f"Publisher retrieval: {family}, 新增/合并 {len(result.papers)} 篇。")
            if result.warnings:
                print("注意：" + "；".join(result.warnings))
            continue
        if message == "/check-downloads":
            report = check_download_presence(config, state.workspace)
            print(
                f"PDF 检测：{report.ready_to_parse_count} 篇已就绪，{report.missing_count} 篇缺失。"
            )
            for item in report.items[:12]:
                marker = "ok" if item.exists else "missing"
                print(f"- [{marker}] {item.title}: {item.expected_path}")
            if report.ready_to_parse_count:
                print("可运行 /resume-downloads 自动解析已就绪 PDF 并写入 artifacts。")
            continue
        if message == "/resume-downloads":
            state.workspace, result = await auto_resume_downloaded_pdfs_async(
                config, state.workspace, session
            )
            save_workspace(session, state.workspace, config=config)
            print(
                f"自动恢复：ready={result.ready_to_parse_count}, parsed={result.parsed_count}, "
                f"performance_cells={result.performance_cell_count}"
            )
            if result.artifact_paths:
                print("Artifacts:")
                for name, path in result.artifact_paths.items():
                    print(f"- {name}: {path}")
            continue
        if message == "/benchmark":
            report = benchmark_pdf_parsing(state.workspace, config)
            print(
                "PDF/OCR benchmark: "
                f"active={report.active_papers}, local_pdf={report.local_pdf_count}, "
                f"parsed={report.parsed_count}, failed={report.failed_count}, "
                f"page_evidence={report.parsed_with_page_evidence}, "
                f"avg_conf={report.average_evidence_confidence}"
            )
            if report.warnings:
                print("注意：" + "；".join(report.warnings))
            continue
        if message == "/golden-eval":
            report = run_golden_eval(config, state.workspace)
            print(f"Golden eval: cases={report.case_count}, dir={report.golden_set_dir}")
            for name, value in report.metrics.items():
                print(f"- {name}: {value}")
            for failure in report.failures[:8]:
                print(f"- missing[{failure['agent']}:{failure['field']}]: {failure['missing']}")
            if report.warnings:
                print("注意：" + "；".join(report.warnings))
            continue
        if message in {"/retrieval-eval", "/retrieval-eval-live"}:
            report = await run_retrieval_golden_eval(config, live=True)
            print(f"Retrieval golden eval: cases={report.case_count}, live={report.live}")
            for name, value in report.metrics.items():
                print(f"- {name}: {value}")
            for case in report.cases:
                print(
                    f"- {case.case_id}: active_recall={case.active_recall}, "
                    f"candidate_recall={case.candidate_recall}, mrr={case.mrr}, "
                    f"active={case.active_count}, candidates={case.candidate_count}"
                )
                if case.warnings:
                    print(f"  warnings: {'；'.join(case.warnings[:3])}")
            if report.warnings:
                print("注意：" + "；".join(report.warnings))
            continue
        if message == "/rerank-learn":
            report = await learn_rerank_policy_from_golden(config, live=True)
            print(
                f"Rerank learning: candidates={report.candidate_count}, best={report.best_candidate}, score={report.best_score}"
            )
            for item in report.results:
                print(f"- {item['name']}: score={item['score']} metrics={item['metrics']}")
            if report.warnings:
                print("注意：" + "；".join(report.warnings))
            continue
        if message.startswith("/publisher-session-test"):
            family = message.removeprefix("/publisher-session-test").strip() or None
            state.workspace, report = build_publisher_session_e2e_report(
                config,
                state.workspace,
                session=session,
                publisher_family=family,
                timeout_seconds=5.0,
            )
            save_workspace(session, state.workspace, config=config)
            print(
                f"Publisher session E2E: planned={report.planned_count}, "
                f"completed={report.completed}, parsed={report.parsed_count}"
            )
            for path in report.target_paths[:8]:
                print(f"- target: {path}")
            if report.warnings:
                print("注意：" + "；".join(report.warnings))
            continue
        if message == "/export":
            paths = await export_session_bundle_skill(session, state.workspace, config)
            print("已导出研究包：")
            for name, path in paths.items():
                print(f"- {name}: {path}")
            continue

        response, workspace = await handle_chat(
            ChatRequest(message=message, session_id=state.session_id),
            state.workspace,
            config,
        )
        response.session_id = state.session_id
        response.session_root = state.session_root
        state.workspace = workspace
        state.context_visible = state.workspace.context.visible_to_user
        save_workspace(session, state.workspace, config=config)
        append_message(session, "user", message)
        append_message(session, "assistant", response)
        print()
        print(f"LitTrace: {response.reply}")
        if response.download_plan:
            print(
                f"下载计划：{response.download_plan.downloadable_count} 篇可处理，"
                f"{response.download_plan.requires_login_count} 篇需要登录。"
            )
        if response.publisher_routes:
            routes = response.publisher_routes.get("routes", [])
            login_count = sum(1 for route in routes if route.get("requires_login"))
            print(f"出版商路线：{len(routes)} 条，{login_count} 条可能需要登录。")
        if response.comparison_matrix:
            print(f"性能矩阵：{len(response.comparison_matrix.matrices)} 个指标组。")
        if response.warnings:
            print("注意：" + "；".join(response.warnings[:3]))
        if state.context_visible:
            print()
            print(format_context_panel(state.workspace))
        print()


def format_context_panel(workspace: LiteratureWorkspace) -> str:
    ids = workspace.context.active_papers
    if not ids:
        return "[上下文窗] 当前没有文献。"
    selected = set(workspace.context.selected_for_download)
    visibility = "显示" if workspace.context.visible_to_user else "隐藏"
    lines = [
        f"[上下文窗:{visibility}] 当前文献 {len(ids)} 篇，已选下载 {len(selected)} 篇",
        "提示：可输入“选择第 1、3 篇下载”“全部下载”“取消选择第 2 篇”。",
    ]
    for index, paper_id in enumerate(ids[:12], start=1):
        paper = workspace.papers[paper_id]
        year = paper.year or "n.d."
        source = paper.journal or paper.publisher or "unknown source"
        marker = "*" if paper_id in selected else " "
        lines.append(
            f"{marker} {index}. {paper.title} "
            f"({year}, {source}, {paper.access_type}, id={paper.paper_id})"
        )
    if len(ids) > 12:
        lines.append(f"... 还有 {len(ids) - 12} 篇")
    return "\n".join(lines)


def _print_doctor(config) -> None:
    status = check_cdp_status(config)
    print(f"cdp downloader: {'ok' if status.available else 'missing'}")
    print(f"cdp url: {status.cdp_url}")
    if status.browser:
        print(f"browser: {status.browser}")
    if status.web_socket_debugger_url:
        print("websocket: available")
    if status.error:
        print(f"error: {status.error}")
    setup = build_browser_setup_report(config)
    if setup.discovery.executable:
        print(f"chrome: {setup.discovery.executable}")
    if setup.discovery.user_data_dir:
        print(f"user data dir: {setup.discovery.user_data_dir}")
        # Surface whether LitTrace is sharing Chrome with the user's
        # day-to-day browser (collision risk) or running in a private
        # profile (the default).
        from pathlib import Path as _Path
        resolved = _Path(setup.discovery.user_data_dir).expanduser()
        is_private = not (
            resolved.resolve() == _Path.home() / "Library/Application Support/Google/Chrome"
            or resolved.resolve() == _Path.home() / ".config/google-chrome"
        )
        if is_private:
            print(
                "profile mode: private (LitTrace's Chrome is isolated from your "
                "day-to-day browser; sign in to each publisher once)"
            )
        else:
            print(
                "profile mode: shared (LitTrace is reusing your day-to-day Chrome "
                "profile — quit Chrome before launching LitTrace, or set "
                "cdp_downloader.chrome_user_data_dir to a fresh path)"
            )
    if setup.selected_profile:
        cookies = ", ".join(setup.selected_profile.publisher_cookie_domains) or "none detected"
        print(
            f"profile: {setup.selected_profile.name} "
            f"({setup.selected_profile.display_name or 'unnamed'}), publisher cookies: {cookies}"
        )
    if setup.warnings:
        for warning in setup.warnings:
            print(f"warning: {warning}")
    if setup.instructions:
        print("browser setup:")
        for instruction in setup.instructions:
            print(f"- {instruction}")
    _print_parser_diagnostics(config)
    _print_codex_auth_diagnostics(config)


def _print_parser_diagnostics(config) -> None:
    """Probe every parser backend declared in ``config.parsing.preferred_engines``
    plus the configured default, and report which heavy dependencies are
    actually importable. The original ``littrace doctor`` only checked CDP /
    Chrome and silently treated paddleocr as available even when the
    ``[parsers]`` extra was not installed — leaving every /parse call to fail
    with ``PaddleOCR is not installed`` at runtime.
    """
    import importlib

    declared = [config.parsing.default_parser, *config.parsing.preferred_engines]
    declared = [name for name in dict.fromkeys(declared) if name]
    deps_for = {
        "paddleocr": ("paddleocr", "pypdfium2"),
        "docling": ("docling", "pypdfium2"),
        "marker": ("marker_pdf",),
        "grobid": ("grobid_client",),
    }
    print("\nparsers:")
    for name in declared:
        deps = deps_for.get(name, ())
        missing = [dep for dep in deps if not _safe_import(importlib, dep)]
        if not missing:
            print(f"  {name}: ok")
        else:
            print(f"  {name}: missing deps {', '.join(missing)}")


def _safe_import(importlib, module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def _print_codex_auth_diagnostics(config) -> None:
    """Show whether the configured codex app-server command is reachable and
    whether the active ``CODEX_HOME`` looks authenticated. The original
    doctor never surfaced this, so users only learned that codex calls
    failed with ``401 Unauthorized`` after pressing Enter in the chat box.
    """
    cmd = config.agent_runtime.codex_command
    binary = cmd[0] if cmd else ""
    from pathlib import Path

    binary_ok = False
    if binary:
        binary_path = Path(binary)
        binary_ok = binary_path.exists() or bool(__import__("shutil").which(binary))
    print("\ncodex:")
    print(f"  command: {' '.join(cmd) or '(empty)'}")
    print(f"  binary: {'ok' if binary_ok else 'missing'}")
    if config.agent_runtime.codex_home_mode.value == "shared":
        home = Path.home() / ".codex"
        home_raw = "~/.codex"
    else:
        raw = Path(config.agent_runtime.codex_home).expanduser()
        # Resolve relative paths against the config.yaml directory so the
        # reported location stays stable regardless of the cwd the user
        # launched from (Finder, IDE, parent directory, …).
        if not raw.is_absolute():
            base = Path(getattr(config, "_config_path", "").parent or Path.cwd())
            home = (base / raw).resolve()
            home_raw = f"{raw} (resolved from {base})"
        else:
            home = raw.resolve()
            home_raw = str(home)
    auth = home / "auth.json"
    print(f"  codex_home: {home_raw} (mode={config.agent_runtime.codex_home_mode.value})")
    print(f"  codex_home resolved: {home}")
    print(f"  auth.json: {'present' if auth.exists() else 'missing'}")
    if binary_ok:
        print("  tip: littrace-qt uses codex for chat. If chat fails with 401,")
        print("       check that auth.json above points to a valid OpenAI API key,")
        print("       or set LITTRACE_CODEX_HOME_MODE=shared to use the default ~/.codex.")


def _print_browser_setup(config, profile_name: str | None, launch: bool) -> None:
    discovery = discover_chrome_profiles(config)
    selected_name = profile_name or config.cdp_downloader.chrome_profile_name
    print("Chrome profiles:")
    print(f"platform: {discovery.platform}")
    print(f"executable: {discovery.executable or 'not found'}")
    print(f"user data dir: {discovery.user_data_dir or 'not found'}")
    if discovery.profiles:
        for profile in discovery.profiles:
            marker = "*" if profile.name == selected_name else " "
            cookies = ", ".join(profile.publisher_cookie_domains) or "none detected"
            print(
                f"{marker} {profile.name} ({profile.display_name or 'unnamed'}), cookies={cookies}"
            )
    else:
        print("profiles: none found")
    for warning in discovery.warnings:
        print(f"warning: {warning}")
    if launch:
        # Round 18: ``littrace setup-browser --launch`` is the
        # explicit "I want a visible Chrome window I can sign in
        # to" path — opt out of the headless default that
        # ``littrace-qt`` uses for its sentinel companion. The
        # ``report.launch_plan`` below keeps the default for the
        # printed hint so users still see what the embedded Qt
        # shell will spawn.
        result = launch_chrome_for_cdp(
            config, profile_name=profile_name, headless=False
        )
        if result.already_available:
            print("cdp: already available")
        elif result.launched:
            print("cdp: launched and available")
        else:
            print("cdp: launch failed")
            if result.error:
                print(f"error: {result.error}")
        if result.command:
            print("command: " + format_shell_command(result.command))
    else:
        report = build_browser_setup_report(config)
        if report.launch_plan:
            print("start command: " + format_shell_command(report.launch_plan.command))


def format_dashboard(state: ShellState) -> str:
    workspace = state.workspace
    active = len(workspace.context.active_papers)
    selected = len(workspace.context.selected_for_download)
    parsed = len(workspace.parsed_papers)
    cells = len(workspace.performance_cells)
    visible = "显示" if workspace.context.visible_to_user else "隐藏"
    lines = [
        "[LitTrace Dashboard]",
        f"session: {state.session_id}",
        f"folder:  {state.session_root}",
        f"context: {active} papers, panel={visible}, selected_downloads={selected}",
        f"parsing: {parsed} parsed records, performance_cells={cells}",
        "commands: /context /login N /attach N path.pdf /attach-si N path /check-downloads "
        "/resume-downloads /parse /table /doctor /setup-browser /storyline-report /storyline-review /benchmark /export",
    ]
    return "\n".join(lines)


def _parse_index_arg(message: str) -> int | None:
    parts = message.split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _parse_attach_args(message: str) -> tuple[int, str] | None:
    parts = message.split(maxsplit=2)
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), parts[2]
    except ValueError:
        return None


def _parse_publisher_retrieve_args(message: str) -> tuple[str, str] | None:
    parts = message.split(maxsplit=2)
    if len(parts) != 3:
        return None
    return parts[1], parts[2]


def _paper_id_for_index(workspace: LiteratureWorkspace, index: int | None) -> str | None:
    if index is None:
        return None
    position = index - 1
    if position < 0 or position >= len(workspace.context.active_papers):
        return None
    return workspace.context.active_papers[position]


if __name__ == "__main__":
    main()
