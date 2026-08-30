"""Round 20 (Phase 6h): worker-level tests for the autonomous review queue.

The MCP gateway test (``tests/unit/test_codex_gateway.py``) covers
``enqueue_autonomous_review`` end-to-end through the gateway. This file
covers the worker side: claim → execute → CAS-commit → requeue-dead.
Mirrors ``tests/unit/test_table_jobs.py`` and ``test_document_jobs.py``
so the durable queue contract is locked for ``autonomous_review_job``.
"""

from __future__ import annotations

import asyncio
import json
from hashlib import sha256

from littrace.autonomous_loop_jobs import (
    AutonomousReviewExecutionOutput,
    run_pending_autonomous_review_jobs,
)
from littrace.config import LitTraceConfig
from littrace.models import (
    LiteratureWorkspace,
    PaperMetadata,
    ParsedPaper,
    ReviewLoopReport,
)
from littrace.state_db import AsyncTaskRecord, SessionStateRecord


class _TaskStore:
    def __init__(self, task: AsyncTaskRecord, workspace: LiteratureWorkspace) -> None:
        self.task = task
        self.state = SessionStateRecord(
            session_id=task.session_id,
            workspace_json=workspace.model_dump(mode="json"),
            revision=workspace.context.filters.workspace_revision,
        )
        self.commit_calls = 0
        self.conflict_once = False

    def claim_pending_async_tasks(self, **kwargs):
        assert kwargs["kind"] == "autonomous_review_job"
        if self.task.status not in {"queued", "failed"}:
            return []
        self.task.status = "running"
        self.task.attempt_count += 1
        self.task.lease_owner = kwargs["worker_id"]
        return [self.task]

    def get_session_state(self, session_id):
        return self.state if session_id == self.state.session_id else None

    def commit_async_workspace_result(self, **kwargs):
        self.commit_calls += 1
        if self.conflict_once and self.commit_calls == 1:
            current = LiteratureWorkspace.model_validate(self.state.workspace_json)
            current.context.selected_for_download = ["paper-1"]
            current.context.filters.workspace_revision += 1
            self.state = self.state.model_copy(
                update={
                    "workspace_json": current.model_dump(mode="json"),
                    "revision": self.state.revision + 1,
                }
            )
            raise RuntimeError("SessionState CAS mismatch")
        assert kwargs["expected_revision"] == self.state.revision
        self.state = self.state.model_copy(
            update={
                "workspace_json": kwargs["workspace_json"],
                "workspace_sha256": kwargs["workspace_sha256"],
                "revision": kwargs["expected_revision"] + 1,
            }
        )
        self.task.status = "completed"
        self.task.result_json = kwargs["result_json"]
        self.task.lease_owner = None
        return self.task

    def update_async_task(self, task):
        self.task = task
        return task


def _source_hash(paper: PaperMetadata, parsed: ParsedPaper) -> str:
    """Hash the combined paper+parsed payload the way the worker does.

    See ``autonomous_loop_jobs._source_sha256``. The worker rejects
    inputs whose hash doesn't match what the gateway enqueued, so the
    test fixture must produce the same digest the worker recomputes on
    claim.
    """
    payload = {
        "paper": paper.model_dump(mode="json"),
        "parsed": parsed.model_dump(mode="json"),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fixture(
    *,
    auto_replan: bool = False,
) -> tuple[AsyncTaskRecord, LiteratureWorkspace, str]:
    paper = PaperMetadata(paper_id="paper-1", title="One")
    parsed = ParsedPaper(
        title="One",
        parsed=True,
        sections=[{"name": "Results", "text": "Sensitivity 12.5 kPa-1"}],
    )
    digest = _source_hash(paper, parsed)
    workspace = LiteratureWorkspace(
        papers={"paper-1": paper},
        parsed_papers={"paper-1": parsed},
    )
    workspace.context.active_papers = ["paper-1"]
    workspace.context.filters.workspace_revision = 2
    task = AsyncTaskRecord(
        task_id="autoreview:one",
        session_id="session-1",
        kind="autonomous_review_job",
        artifact_id="autonomous_review_batch:one",
        event_type="autonomous_review_requested",
        result_json={
            "schema_version": "littrace.autonomous_review_job.v1",
            "command": {
                "paper_ids": ["paper-1"],
                "auto_replan": auto_replan,
                "expected_revision": 1,
                "source_sha256": {"paper-1": digest},
            },
        },
    )
    return task, workspace, digest


async def _execute(_config, workspace, _auto_replan):
    return AutonomousReviewExecutionOutput(
        report=ReviewLoopReport(
            objective="validate sensitivity claim",
            final_answer="claim holds within stated tolerance",
            rounds=[],
            passed=True,
            score=0.95,
            release_ready=True,
        ),
        auto_replan=False,
        source_sha256={},
    )


def test_autonomous_review_worker_commits_report_into_workspace() -> None:
    """Round 20: ``auto_replan=False`` is the safe default — the
    worker only writes the report into the workspace, never mutates
    parsed_papers / performance_cells / storyline."""
    task, workspace, _digest = _fixture(auto_replan=False)
    store = _TaskStore(task, workspace)

    report = asyncio.run(
        run_pending_autonomous_review_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="autoreview-worker",
        )
    )

    canonical = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert report.processed == 1
    assert report.passed is True
    assert report.score == 0.95
    assert report.release_ready is True
    assert store.task.status == "completed"
    assert canonical.context.filters.autonomous_loop_report is not None
    assert (
        canonical.context.filters.autonomous_loop_report["final_answer"]
        == "claim holds within stated tolerance"
    )


def test_autonomous_review_worker_with_auto_replan_passes_flag() -> None:
    """Round 20: ``auto_replan=True`` is propagated to the executor;
    the executor decides whether to call replan actions. The worker
    itself doesn't gate on the flag, it just forwards."""
    task, workspace, _digest = _fixture(auto_replan=True)
    store = _TaskStore(task, workspace)

    observed = {"auto_replan": None}

    async def executor(_config, _workspace, auto_replan):
        observed["auto_replan"] = auto_replan
        return AutonomousReviewExecutionOutput(
            report=ReviewLoopReport(
                objective="validate",
                final_answer="ok",
                rounds=[],
                passed=True,
                score=1.0,
                release_ready=True,
            ),
            auto_replan=auto_replan,
            source_sha256={},
        )

    report = asyncio.run(
        run_pending_autonomous_review_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=executor,
            worker_id="autoreview-worker",
        )
    )

    assert observed["auto_replan"] is True
    assert report.processed == 1
    assert store.task.status == "completed"


def test_autonomous_review_worker_remerges_after_cas_conflict() -> None:
    """CAS retry: another writer raced us between snapshot and commit;
    the worker must reload canonical state and try again — same
    pattern as ``document_jobs`` and ``table_jobs``."""
    task, workspace, _digest = _fixture()
    store = _TaskStore(task, workspace)
    store.conflict_once = True

    report = asyncio.run(
        run_pending_autonomous_review_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="autoreview-worker",
        )
    )

    canonical = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert report.processed == 1
    assert store.commit_calls == 2
    assert canonical.context.selected_for_download == ["paper-1"]
    assert canonical.context.filters.autonomous_loop_report is not None


def test_autonomous_review_worker_does_not_execute_against_stale_input() -> None:
    """Source hashes are the staleness contract. If ``parsed_papers``
    has changed since the gateway enqueued the task, the worker must
    skip execution rather than emit a review based on stale inputs."""
    task, workspace, _digest = _fixture()
    workspace.parsed_papers["paper-1"].sections[0]["text"] = "Changed parse"
    store = _TaskStore(task, workspace)

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("stale parsed input must not reach the reviewer")

    report = asyncio.run(
        run_pending_autonomous_review_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=unexpected,
            worker_id="autoreview-worker",
        )
    )

    assert report.processed == 1
    assert report.stale == 1
    assert store.task.status == "completed"
    assert store.task.result_json["execution"]["stale_paper_ids"] == ["paper-1"]


def test_autonomous_review_worker_rejects_nonbool_auto_replan() -> None:
    """Round 20: ``auto_replan`` must be a strict bool. If the gateway
    ever sends ``auto_replan="yes"`` the worker must fail loud rather
    than treat it as truthy."""
    task, workspace, _digest = _fixture()
    task.result_json["command"]["auto_replan"] = "yes"  # type: ignore[index]
    store = _TaskStore(task, workspace)

    report = asyncio.run(
        run_pending_autonomous_review_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="autoreview-worker",
        )
    )

    assert report.processed == 0
    assert report.failed == 1
    assert any(
        "auto_replan" in warning and "bool" in warning
        for warning in report.warnings
    )


def test_autonomous_review_worker_handles_no_jobs_to_claim() -> None:
    """Empty queue short-circuit: the worker must finish cleanly without
    raising when there is nothing to do."""
    task, workspace, _digest = _fixture()
    task.status = "completed"  # nothing queued
    store = _TaskStore(task, workspace)

    report = asyncio.run(
        run_pending_autonomous_review_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="autoreview-worker",
        )
    )

    assert report.processed == 0
    assert report.failed == 0
    assert report.job_ids == []
    assert store.commit_calls == 0
