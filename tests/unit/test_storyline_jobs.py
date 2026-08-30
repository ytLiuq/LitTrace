"""Round 20 (Phase 6g follow-up): worker-level tests for the storyline queue.

The MCP gateway test (``tests/unit/test_codex_gateway.py``) covers
``enqueue_storyline`` end-to-end through the gateway. This file covers
the worker side: claim → execute → CAS-commit. Mirrors
``tests/unit/test_table_jobs.py``, ``test_document_jobs.py``, and
``test_autonomous_loop_jobs.py`` so the durable queue contract is
locked for ``storyline_job``.
"""

from __future__ import annotations

import asyncio
import json
from hashlib import sha256

from littrace.config import LitTraceConfig
from littrace.models import (
    EvidenceSpan,
    LiteratureWorkspace,
    PaperMetadata,
    ParsedPaper,
    StorylineClaim,
)
from littrace.state_db import AsyncTaskRecord, SessionStateRecord
from littrace.storyline_jobs import (
    StorylineExecutionOutput,
    run_pending_storyline_jobs,
)


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
        assert kwargs["kind"] == "storyline_job"
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
    """Hash the combined paper+parsed payload the way ``storyline_jobs``
    does. The worker rejects inputs whose hash doesn't match what the
    gateway enqueued, so the test fixture must produce the same digest
    the worker recomputes on claim.
    """
    payload = {
        "paper": paper.model_dump(mode="json"),
        "parsed": parsed.model_dump(mode="json"),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fixture() -> tuple[AsyncTaskRecord, LiteratureWorkspace, str]:
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
        task_id="storyline:one",
        session_id="session-1",
        kind="storyline_job",
        artifact_id="storyline_batch:one",
        event_type="storyline_requested",
        result_json={
            "schema_version": "littrace.storyline_job.v1",
            "command": {
                "paper_ids": ["paper-1"],
                "expected_revision": 1,
                "source_sha256": {"paper-1": digest},
            },
        },
    )
    return task, workspace, digest


async def _execute(_config, workspace):
    assert workspace.context.active_papers == ["paper-1"]
    return StorylineExecutionOutput(
        storyline_claims=[
            StorylineClaim(
                claim="MXene sensors reach 12.5 kPa-1 sensitivity",
                claim_type="performance",
                evidence=[
                    EvidenceSpan(
                        paper_id="paper-1",
                        section="Results",
                        snippet="Sensitivity 12.5 kPa-1",
                    )
                ],
                confidence=0.9,
            )
        ],
        source_sha256={},
    )


def test_storyline_worker_commits_claims_into_workspace() -> None:
    """Round 20: a happy-path run ends with the storyline claims
    committed into the workspace and the task marked completed."""
    task, workspace, _digest = _fixture()
    store = _TaskStore(task, workspace)

    report = asyncio.run(
        run_pending_storyline_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="storyline-worker",
        )
    )

    canonical = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert report.processed == 1
    assert report.claims == 1
    assert report.failed == 0
    assert store.task.status == "completed"
    assert len(canonical.claims) == 1
    assert canonical.claims[0].claim_origin == "storyline"


def test_storyline_worker_remerges_after_cas_conflict() -> None:
    """CAS retry: another writer raced us between snapshot and commit;
    the worker must reload canonical state and try again — same
    pattern as ``document_jobs`` / ``autonomous_loop_jobs``."""
    task, workspace, _digest = _fixture()
    store = _TaskStore(task, workspace)
    store.conflict_once = True

    report = asyncio.run(
        run_pending_storyline_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="storyline-worker",
        )
    )

    canonical = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert report.processed == 1
    assert store.commit_calls == 2
    # The conflicting writer's selection must survive the re-merge.
    assert canonical.context.selected_for_download == ["paper-1"]
    assert len(canonical.claims) == 1


def test_storyline_worker_does_not_execute_against_stale_input() -> None:
    """Source hashes are the staleness contract. If ``parsed_papers``
    has changed since the gateway enqueued the task, the worker must
    skip execution rather than emit claims based on stale inputs."""
    task, workspace, _digest = _fixture()
    workspace.parsed_papers["paper-1"].sections[0]["text"] = "Changed parse"
    store = _TaskStore(task, workspace)

    async def unexpected(*_args):
        raise AssertionError("stale parsed input must not reach the storyteller")

    report = asyncio.run(
        run_pending_storyline_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=unexpected,
            worker_id="storyline-worker",
        )
    )

    assert report.processed == 1
    assert report.stale == 1
    assert store.task.status == "completed"
    assert store.task.result_json["execution"]["stale_paper_ids"] == ["paper-1"]
    assert store.task.result_json["execution"]["storyline_claim_count"] == 0


def test_storyline_worker_handles_no_jobs_to_claim() -> None:
    """Empty queue short-circuit: the worker must finish cleanly without
    raising when there is nothing to do."""
    task, workspace, _digest = _fixture()
    task.status = "completed"  # nothing queued
    store = _TaskStore(task, workspace)

    report = asyncio.run(
        run_pending_storyline_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="storyline-worker",
        )
    )

    assert report.processed == 0
    assert report.failed == 0
    assert report.job_ids == []
    assert store.commit_calls == 0
