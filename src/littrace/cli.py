from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

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
from littrace.sentinel.storage import get_sentinel_store, load_sentinel_state
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
from littrace.evidence.tables import decide_artifact_extraction_need


@dataclass
class ShellState:
    workspace: LiteratureWorkspace
    session_id: str
    session_root: str
    context_visible: bool = True


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "sentinel":
        config = load_config()
        asyncio.run(_run_sentinel_command(config))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "rag":
        config = load_config()
        asyncio.run(_run_rag_command(config))
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


async def _run_sentinel_command(config) -> None:
    if len(sys.argv) < 3:
        print("用法：littrace sentinel init|run|status|access-review|resume-after-login ...")
        return
    action = sys.argv[2]
    watchlist_id = _arg_value("--watchlist") or _arg_value("--watchlist-id") or "mxene_sensor"
    topic = _arg_value("--topic") or _arg_value("--objective")

    if action == "init":
        root = init_sentinel(config, watchlist_id, topic or watchlist_id)
        print(f"sentinel initialized: {root}")
        return
    if action == "run":
        result = await run_sentinel(config, watchlist_id, topic)
        print(f"run_id: {result.summary.run_id}")
        print(f"watchlist: {result.summary.watchlist_id}")
        print(f"topic: {result.summary.topic}")
        print(f"new_candidates: {result.summary.new_candidates_count}")
        print(f"downloaded: {result.summary.downloaded_count}")
        print(f"parsed: {result.summary.parsed_count}")
        print(f"access_tasks: {result.summary.access_task_count}")
        if result.summary.digest_path:
            print(f"digest: {result.summary.digest_path}")
        if result.summary.warnings:
            print("warnings: " + "；".join(result.summary.warnings))
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
            if not (session_dir / "workspace.json").exists():
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
        "/dashboard /doctor /setup-browser /quality /agents /workflow /quality-audits /plan topic /init-config /ocr-choice /storyline-report /storyline-review /benchmark /golden-eval /retrieval-eval /rerank-learn /publisher-session-test /export /quit"
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
        result = launch_chrome_for_cdp(config, profile_name=profile_name)
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
