"""Round 20 (Phase 6g): worker-level tests for the document job queue.

The MCP gateway test (``tests/unit/test_codex_gateway.py``) covers
``enqueue_document`` end-to-end through the gateway. This file covers
the worker side: claim → execute → CAS-commit → requeue-dead. Mirrors
``tests/unit/test_table_jobs.py`` so the durable queue contract is
locked for the new ``document_job`` kind.
"""

from __future__ import annotations

import asyncio
import json
from hashlib import sha256

from littrace.config import LitTraceConfig
from littrace.document_jobs import (
    DocumentExecutionOutput,
    run_pending_document_jobs,
)
from littrace.models import (
    LiteratureWorkspace,
    PaperMetadata,
    ParsedPaper,
    ResearchDocumentReport,
    ResearchDocumentSection,
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
        assert kwargs["kind"] == "document_job"
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


def _parsed_hash(parsed: ParsedPaper) -> str:
    return sha256(
        json.dumps(
            parsed.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _source_hash(paper: PaperMetadata, parsed: ParsedPaper) -> str:
    """Hash the combined paper+parsed payload the way ``document_jobs``
    does (``_source_sha256``). The worker rejects inputs whose hash
    doesn't match what the gateway enqueued, so the test fixture must
    produce the same digest the worker recomputes on claim.
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
        task_id="document:one",
        session_id="session-1",
        kind="document_job",
        artifact_id="document_batch:one",
        event_type="document_requested",
        result_json={
            "schema_version": "littrace.document_job.v1",
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
    return DocumentExecutionOutput(
        report=ResearchDocumentReport(
            title="MXene Sensor Review",
            markdown="# Findings\nSensitivity 12.5 kPa-1.",
            sections=[
                ResearchDocumentSection(
                    title="Findings",
                    body="Sensitivity 12.5 kPa-1.",
                )
            ],
            citation_records=[],
            release_ready=True,
        ),
        source_sha256={},
    )


def test_document_worker_commits_report_into_workspace() -> None:
    task, workspace, _digest = _fixture()
    store = _TaskStore(task, workspace)

    report = asyncio.run(
        run_pending_document_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="document-worker",
        )
    )

    canonical = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert report.processed == 1
    assert report.sections == 1
    assert report.failed == 0
    assert store.task.status == "completed"
    assert canonical.context.filters.document_report["title"] == "MXene Sensor Review"
    assert store.task.result_json["execution"]["release_ready"] is True


def test_document_worker_remerges_after_cas_conflict() -> None:
    """CAS retry: another writer raced us between snapshot and commit;
    the worker must reload canonical state and try again."""
    task, workspace, _digest = _fixture()
    store = _TaskStore(task, workspace)
    store.conflict_once = True

    report = asyncio.run(
        run_pending_document_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="document-worker",
        )
    )

    canonical = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert report.processed == 1
    assert store.commit_calls == 2
    # The conflicting writer's selection must survive the re-merge.
    assert canonical.context.selected_for_download == ["paper-1"]
    assert canonical.context.filters.document_report is not None


def test_document_worker_does_not_execute_against_changed_parsed_input() -> None:
    """Source hashes are the staleness contract — if ``parsed_papers``
    has changed since the gateway enqueued the task, we must skip
    execution rather than emit a report based on stale inputs."""
    task, workspace, _digest = _fixture()
    workspace.parsed_papers["paper-1"].sections[0]["text"] = "Changed parse"
    store = _TaskStore(task, workspace)

    async def unexpected(*_args):
        raise AssertionError("stale parsed input must not reach the composer")

    report = asyncio.run(
        run_pending_document_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=unexpected,
            worker_id="document-worker",
        )
    )

    assert report.processed == 1
    assert report.stale == 1
    assert store.task.status == "completed"
    assert store.task.result_json["execution"]["stale_paper_ids"] == ["paper-1"]
    # And we did NOT execute the executor — only one CAS commit, no
    # document_report was written.
    assert store.task.result_json["execution"]["section_count"] == 0


def test_document_worker_handles_no_jobs_to_claim() -> None:
    """Empty queue short-circuit: the worker must finish cleanly without
    raising when there is nothing to do."""
    task, workspace, _digest = _fixture()
    task.status = "completed"  # nothing queued
    store = _TaskStore(task, workspace)

    report = asyncio.run(
        run_pending_document_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="document-worker",
        )
    )

    assert report.processed == 0
    assert report.failed == 0
    assert report.job_ids == []
    assert store.commit_calls == 0
