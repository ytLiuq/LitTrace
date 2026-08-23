from __future__ import annotations

import asyncio
from hashlib import sha256

from littrace.artifact_registry import ArtifactRecord
from littrace.artifact_store import artifact_store_from_config
from littrace.config import ArtifactStorageConfig, LitTraceConfig
from littrace.models import LiteratureWorkspace, PaperMetadata, ParsedPaper
from littrace.parse_jobs import (
    ParseExecutionOutput,
    _execute_parse_job,
    run_pending_parse_jobs,
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
        assert kwargs["kind"] == "parse_job"
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
        assert kwargs["lease_owner"] == self.task.lease_owner
        committed = kwargs["expected_revision"] + 1
        self.state = self.state.model_copy(
            update={
                "workspace_json": kwargs["workspace_json"],
                "workspace_sha256": kwargs["workspace_sha256"],
                "revision": committed,
            }
        )
        self.task.status = "completed"
        self.task.result_json = kwargs["result_json"]
        self.task.lease_owner = None
        return self.task

    def update_async_task(self, task):
        self.task = task
        return task


def _fixture() -> tuple[AsyncTaskRecord, LiteratureWorkspace, ArtifactRecord]:
    paper = PaperMetadata(paper_id="paper-1", title="One")
    source = ArtifactRecord(
        artifact_id="paper_pdf:paper-1",
        session_id="session-1",
        kind="paper_pdf",
        paper_id="paper-1",
        object_key="sessions/session-1/papers/paper-1/paper.pdf",
        backend="local",
        sha256="b" * 64,
    )
    workspace = LiteratureWorkspace(papers={paper.paper_id: paper})
    workspace.context.active_papers = [paper.paper_id]
    workspace.context.filters.workspace_revision = 1
    task = AsyncTaskRecord(
        task_id="parse:one",
        session_id="session-1",
        kind="parse_job",
        artifact_id="parse_batch:one",
        event_type="parse_requested",
        result_json={
            "schema_version": "littrace.parse_job.v1",
            "command": {
                "paper_ids": [paper.paper_id],
                "parse_strategy": "text_only",
                "expected_revision": 0,
                "papers": [paper.model_dump(mode="json")],
                "sources": [source.model_dump(mode="json")],
            },
        },
    )
    return task, workspace, source


async def _execute(_config, _session_id, papers, sources, strategy):
    assert [paper.paper_id for paper in papers] == ["paper-1"]
    assert [source.paper_id for source in sources] == ["paper-1"]
    assert strategy == "text_only"
    return ParseExecutionOutput(
        parsed_papers={
            "paper-1": ParsedPaper(
                title="One",
                parsed=True,
                sections=[{"name": "Results", "text": "Sensitivity 12.5 kPa-1"}],
            )
        },
        source_sha256={"paper-1": "b" * 64},
        report={"parsed_count": 1, "failed_count": 0},
    )


def test_parse_worker_atomically_merges_result() -> None:
    task, workspace, _source = _fixture()
    store = _TaskStore(task, workspace)

    report = asyncio.run(
        run_pending_parse_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            source_sha_lookup=lambda _session, _paper: "b" * 64,
            worker_id="parse-worker",
        )
    )

    canonical = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert report.processed == 1
    assert report.parsed == 1
    assert store.task.status == "completed"
    assert store.state.revision == 2
    assert canonical.parsed_papers["paper-1"].parsed is True
    assert canonical.context.filters.parsed_full_text_count == 1


def test_parse_worker_remerges_after_workspace_cas_conflict() -> None:
    task, workspace, _source = _fixture()
    store = _TaskStore(task, workspace)
    store.conflict_once = True

    report = asyncio.run(
        run_pending_parse_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            source_sha_lookup=lambda _session, _paper: "b" * 64,
            worker_id="parse-worker",
        )
    )

    canonical = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert report.processed == 1
    assert store.commit_calls == 2
    assert store.state.revision == 3
    assert canonical.context.selected_for_download == ["paper-1"]
    assert canonical.parsed_papers["paper-1"].parsed is True


def test_parse_worker_drops_stale_pdf_result_without_retrying() -> None:
    task, workspace, _source = _fixture()
    store = _TaskStore(task, workspace)

    report = asyncio.run(
        run_pending_parse_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            source_sha_lookup=lambda _session, _paper: "c" * 64,
            worker_id="parse-worker",
        )
    )

    canonical = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert report.processed == 1
    assert report.stale == 1
    assert canonical.parsed_papers == {}
    assert store.task.result_json["execution"]["stale_paper_ids"] == ["paper-1"]


def test_parse_worker_recovers_after_retryable_executor_failure() -> None:
    task, workspace, _source = _fixture()
    store = _TaskStore(task, workspace)

    async def unavailable(_config, _session_id, _papers, _sources, _strategy):
        raise RuntimeError("parser worker unavailable")

    first = asyncio.run(
        run_pending_parse_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=unavailable,
            source_sha_lookup=lambda _session, _paper: "b" * 64,
            worker_id="parse-worker-1",
        )
    )
    second = asyncio.run(
        run_pending_parse_jobs(
            LitTraceConfig(),
            state_store=store,
            executor=_execute,
            source_sha_lookup=lambda _session, _paper: "b" * 64,
            worker_id="parse-worker-2",
        )
    )

    assert first.failed == 1
    assert second.processed == 1
    assert store.task.status == "completed"
    assert store.task.attempt_count == 2


def test_parse_executor_materializes_pdf_from_artifact_store(
    monkeypatch,
    tmp_path,
) -> None:
    config = LitTraceConfig(
        artifact_storage=ArtifactStorageConfig(
            backend="local",
            local_root=tmp_path / "artifacts",
        )
    )
    pdf_bytes = b"%PDF-1.4\nobject-store-source"
    ref = artifact_store_from_config(config).put_bytes(
        "sessions/session-1/papers/paper-1/paper.pdf",
        pdf_bytes,
        content_type="application/pdf",
    )
    paper = PaperMetadata(paper_id="paper-1", title="One")
    source = ArtifactRecord.from_blob_ref(
        ref,
        artifact_id="paper_pdf:paper-1",
        session_id="session-1",
        kind="paper_pdf",
        paper_id="paper-1",
    )

    def fake_parse(workspace, parse_config):
        from littrace.access_layer.paths import target_pdf_path

        path = target_pdf_path(parse_config, paper)
        assert path.read_bytes() == pdf_bytes
        workspace.parsed_papers["paper-1"] = ParsedPaper(
            pdf_path=path,
            parsed=True,
            sections=[{"name": "Results", "text": "Stored PDF parsed"}],
        )
        return workspace, {"parsed_count": 1, "failed_count": 0}

    monkeypatch.setattr("littrace.parse_jobs.parse_workspace_papers", fake_parse)
    output = asyncio.run(
        _execute_parse_job(
            config,
            "session-1",
            [paper],
            [source],
            "text_only",
        )
    )

    assert output.source_sha256 == {"paper-1": sha256(pdf_bytes).hexdigest()}
    assert output.parsed_papers["paper-1"].parsed is True
    assert output.parsed_papers["paper-1"].pdf_path is None
