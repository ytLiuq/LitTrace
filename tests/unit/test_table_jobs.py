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
    PerformanceCell,
)
from littrace.state_db import AsyncTaskRecord, SessionStateRecord
from littrace.table_jobs import (
    TableExecutionOutput,
    _execute_table_job,
    run_pending_table_jobs,
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
        assert kwargs["kind"] == "table_job"
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


def _fixture() -> tuple[AsyncTaskRecord, LiteratureWorkspace, str]:
    paper = PaperMetadata(paper_id="paper-1", title="One")
    parsed = ParsedPaper(
        title="One",
        parsed=True,
        sections=[{"name": "Results", "text": "Sensitivity 12.5 kPa-1"}],
    )
    digest = _parsed_hash(parsed)
    workspace = LiteratureWorkspace(
        papers={"paper-1": paper},
        parsed_papers={"paper-1": parsed},
    )
    workspace.context.active_papers = ["paper-1"]
    workspace.context.filters.workspace_revision = 2
    task = AsyncTaskRecord(
        task_id="table:one",
        session_id="session-1",
        kind="table_job",
        artifact_id="table_batch:one",
        event_type="table_extraction_requested",
        result_json={
            "schema_version": "littrace.table_job.v1",
            "command": {
                "paper_ids": ["paper-1"],
                "expected_revision": 1,
                "parsed_sha256": {"paper-1": digest},
            },
        },
    )
    return task, workspace, digest


async def _execute(_config, workspace):
    assert workspace.context.active_papers == ["paper-1"]
    return TableExecutionOutput(
        performance_cells=[
            PerformanceCell(
                paper_id="paper-1",
                metric="sensitivity",
                value=12.5,
                unit="kPa-1",
                evidence=EvidenceSpan(
                    paper_id="paper-1",
                    section="Results",
                    snippet="Sensitivity 12.5 kPa-1",
                ),
            )
        ],
        structured_artifacts=[
            {
                "paper_id": "paper-1",
                "artifact_type": "table",
                "label": "Table 1",
                "text": "Sensitivity 12.5 kPa-1",
                "evidence": {
                    "paper_id": "paper-1",
                    "snippet": "Sensitivity 12.5 kPa-1",
                },
                "confidence": 0.8,
            }
        ],
        harness={"passed": True, "score": 1.0},
    )


def test_table_worker_merges_cells_and_structured_artifacts() -> None:
    task, workspace, _digest = _fixture()
    store = _TaskStore(task, workspace)

    report = asyncio.run(
        run_pending_table_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="table-worker",
        )
    )

    canonical = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert report.processed == 1
    assert report.performance_cells == 1
    assert store.task.status == "completed"
    assert canonical.performance_cells[0].metric == "sensitivity"
    assert canonical.context.filters.structured_artifacts[0]["label"] == "Table 1"


def test_table_worker_remerges_after_cas_conflict() -> None:
    task, workspace, _digest = _fixture()
    store = _TaskStore(task, workspace)
    store.conflict_once = True

    report = asyncio.run(
        run_pending_table_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            worker_id="table-worker",
        )
    )

    canonical = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert report.processed == 1
    assert store.commit_calls == 2
    assert canonical.context.selected_for_download == ["paper-1"]
    assert canonical.performance_cells[0].value == 12.5


def test_table_worker_does_not_execute_against_changed_parsed_input() -> None:
    task, workspace, _digest = _fixture()
    workspace.parsed_papers["paper-1"].sections[0]["text"] = "Changed parse"
    store = _TaskStore(task, workspace)

    async def unexpected(*_args):
        raise AssertionError("stale parsed input must not reach the extractor")

    report = asyncio.run(
        run_pending_table_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=unexpected,
            worker_id="table-worker",
        )
    )

    assert report.processed == 1
    assert report.stale == 1
    assert store.task.status == "completed"
    assert store.task.result_json["execution"]["stale_paper_ids"] == ["paper-1"]


def test_default_table_executor_uses_domain_extractor_with_regex_fallback() -> None:
    _task, workspace, _digest = _fixture()
    config = LitTraceConfig()
    config.llm.enabled = False

    output = asyncio.run(_execute_table_job(config, workspace))

    assert any(cell.metric == "sensitivity" for cell in output.performance_cells)
    assert output.harness
