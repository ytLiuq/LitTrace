"""Round 5: thread compaction worker.

codex-harness's Codex App Server exposes a ``thread/compact`` RPC
that compresses a thread's conversation history (similar to
context-window summarisation). ``AppServerClient.compact_thread``
already wraps the RPC; this module is what actually drives it on
a schedule.

Two entry points:

  - ``CompactionWorker`` — a ``threading.Thread`` daemon (same
    pattern as ``DownloadRetryWorker``) that polls every
    ``interval_seconds`` and enqueues ``compaction_job`` rows for
    sessions whose ``turn_count`` / ``last_total_tokens`` crossed
    the configured thresholds. Cheap, lock-free, periodic.
  - ``run_pending_compaction`` — an async batch driver that claims
    enqueued rows, calls the App Server's ``thread/compact`` RPC
    via the shared runtime manager, and records the result. Used
    by both the worker (background) and the CLI
    (``littrace compaction run``).

Threading model note: ``runtime_manager.use`` is sync (it submits
to a daemon-thread event loop and blocks), so the worker thread
can call it without crossing an async boundary.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from littrace.config import LitTraceConfig
from littrace.state_db import (
    AsyncTaskRecord,
    SessionStateRecord,
    state_store_from_config,
)

if TYPE_CHECKING:
    from littrace.codex_runtime.runtime import CodexAppServerRuntimeManager
    from littrace.state_db import StateStore


log = logging.getLogger("compaction")


@dataclass
class CompactionJobReport:
    started_at: str
    finished_at: str | None = None
    enqueued: int = 0
    succeeded: int = 0
    failed: int = 0
    warnings: list[str] = None.__class__(list) if False else None  # placeholder

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


class CompactionWorker:
    """Threading daemon that enqueues compaction_job rows.

    The worker does **not** call ``thread/compact`` itself — that
    happens in ``run_pending_compaction`` once a row is dequeued. The
    split keeps the daemon short (it just scans the state table and
    drops rows) and lets the actual RPC run on the App Server's
    event loop via the shared runtime manager.
    """

    def __init__(
        self,
        state_store: "StateStore",
        *,
        interval_seconds: float = 60.0,
        batch_size: int = 5,
        threshold_turns: int = 30,
        threshold_tokens: int = 50_000,
    ) -> None:
        self.state_store = state_store
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self.threshold_turns = threshold_turns
        self.threshold_tokens = threshold_tokens
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="compaction-worker",
        )
        self._thread.start()
        log.info("compaction_worker_started")

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None
        log.info("compaction_worker_stopped")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                log.exception("compaction_worker_iteration_failed")
            self._stop_event.wait(self.interval_seconds)

    def run_once(self) -> list[str]:
        """Enqueue compaction jobs for due sessions.

        Returns the session_ids processed. The list is useful for
        tests and for the in-line ``run_pending_compaction`` driver
        that consumes what the worker enqueued.
        """
        due = self.state_store.compaction_due_sessions(
            threshold_turns=self.threshold_turns,
            threshold_tokens=self.threshold_tokens,
            limit=self.batch_size,
        )
        processed: list[str] = []
        for session_id, thread_id, _turns, _tokens in due:
            try:
                self.state_store.enqueue_async_task(
                    AsyncTaskRecord(
                        task_id=f"compaction-{session_id}-{thread_id}",
                        session_id=session_id,
                        kind="compaction_job",
                        artifact_id=thread_id,
                        attempt_count=0,
                        next_attempt_at=datetime.now(UTC).isoformat(),
                    )
                )
                processed.append(session_id)
            except Exception:
                log.exception(
                    "compaction_enqueue_failed session=%s thread=%s",
                    session_id, thread_id,
                )
        return processed


def _mark_completed(task: AsyncTaskRecord) -> AsyncTaskRecord:
    return task.model_copy(
        update={
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "last_error": None,
        }
    )


def _mark_failed(
    task: AsyncTaskRecord,
    *,
    max_attempts: int,
    base_delay_seconds: float,
    error: str,
) -> AsyncTaskRecord:
    """Back-off the failed task or push it to dead.

    The back-off formula mirrors ``download_tasks.py``: delay doubles
    each attempt up to 1 hour, capped at ``max_attempts``. Once
    ``max_attempts`` is reached the task goes to ``dead`` and stops
    re-trying; an operator can requeue it via
    ``requeue_dead_async_tasks(kind="compaction_job")``.
    """
    new_attempt = task.attempt_count + 1
    if new_attempt >= max_attempts:
        return task.model_copy(
            update={
                "status": "dead",
                "attempt_count": new_attempt,
                "last_error": error,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
    delay = min(base_delay_seconds * (2 ** max(new_attempt - 1, 0)), 3600)
    next_at = (
        datetime.now(UTC).timestamp() + delay
    )
    return task.model_copy(
        update={
            "status": "failed",
            "attempt_count": new_attempt,
            "last_error": error,
            "next_attempt_at": datetime.fromtimestamp(
                next_at, tz=UTC
            ).isoformat(),
        }
    )


async def _compact_one(
    config: LitTraceConfig,
    binding: Any,
    state_store: "StateStore",
    runtime_manager: "CodexAppServerRuntimeManager",
    request_timeout: float,
) -> None:
    """Run a single ``thread/compact`` RPC and stamp the result."""
    client = await runtime_manager.get_client()
    if client is None:
        raise RuntimeError(
            f"no codex runtime client available for session {binding.session_id}"
        )
    try:
        await client.compact_thread(
            binding.codex_thread_id, timeout=request_timeout,
        )
    finally:
        # codex is not async-context-managed here; release immediately
        # so the runtime manager can hand the same client out to the
        # next chat call.
        await runtime_manager.release_client(client)
    state_store.upsert_agent_thread_binding(
        binding.model_copy(
            update={
                "last_compacted_at": datetime.now(UTC).isoformat(),
            }
        )
    )


async def run_pending_compaction(
    config: LitTraceConfig,
    *,
    limit: int = 10,
) -> CompactionJobReport:
    """Drive one batch of pending compaction jobs to completion."""
    from littrace.codex_runtime.runtime import shared_runtime_manager

    state_store = state_store_from_config(config)
    compaction = config.compaction
    report = CompactionJobReport(started_at=datetime.now(UTC).isoformat())
    if not compaction.enabled:
        log.info("compaction_disabled_skipping")
        report.finished_at = datetime.now(UTC).isoformat()
        return report

    manager = shared_runtime_manager(
        (
            ("codex", "app-server"),
            config.agent_runtime.startup_timeout_seconds,
            config.agent_runtime.request_timeout_seconds,
            tuple(),
        ),
        ("codex", "app-server"),
        client_options={},
    )

    claimed = state_store.claim_pending_async_tasks(
        worker_id="compaction",
        kind="compaction_job",
        limit=limit,
        lease_seconds=300.0,
    )
    for task in claimed:
        report.enqueued += 1
        binding = state_store.get_agent_thread_binding(task.session_id)
        if binding is None or binding.status not in ("active", "idle"):
            state_store.update_async_task(_mark_completed(task))
            report.warnings.append(
                f"compaction skipped session={task.session_id} status={binding.status if binding else 'no-binding'}"
            )
            continue
        try:
            await _compact_one(
                config, binding, state_store, manager,
                request_timeout=compaction.request_timeout_seconds,
            )
            state_store.update_async_task(_mark_completed(task))
            report.succeeded += 1
        except Exception as exc:
            state_store.update_async_task(_mark_failed(
                task,
                max_attempts=compaction.max_attempts,
                base_delay_seconds=compaction.base_delay_seconds,
                error=f"{exc.__class__.__name__}: {exc}",
            ))
            report.failed += 1
            log.exception(
                "compaction_failed session=%s task=%s",
                task.session_id, task.task_id,
            )
    report.finished_at = datetime.now(UTC).isoformat()
    return report


async def run_pending_compaction_daemon(
    config: LitTraceConfig,
    *,
    interval_seconds: float = 60.0,
    run_immediately: bool = True,
) -> None:
    """Long-running asyncio loop that drives compaction in batches.

    Mirror of ``run_daily_rag_daemon``. Used by the
    ``littrace compaction daemon`` CLI subcommand.
    """
    if run_immediately:
        try:
            await run_pending_compaction(config)
        except Exception:
            log.exception("compaction_daemon_initial_run_failed")
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await run_pending_compaction(config)
        except Exception:
            log.exception("compaction_daemon_iteration_failed")
