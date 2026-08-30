from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest

from littrace.artifact_registry import ArtifactRecord
from littrace.codex_runtime.gateway import (
    APP_SERVER_TOOL_NAMES,
    ReadOnlyToolGateway,
    app_server_tool_specs,
)
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace, PaperMetadata, ParsedPaper
from littrace.state_db import (
    AgentThreadBindingRecord,
    AgentToolCallRecord,
    AsyncTaskRecord,
    SessionStateRecord,
)


class _GatewayStore:
    def __init__(self) -> None:
        workspace = LiteratureWorkspace(
            papers={"paper-1": PaperMetadata(paper_id="paper-1", title="One")}
        )
        workspace.context.active_papers = ["paper-1"]
        self.binding = AgentThreadBindingRecord(session_id="session-1", codex_thread_id="thread-1")
        self.state = SessionStateRecord(
            session_id="session-1",
            workspace_json=workspace.model_dump(mode="json"),
        )
        self.tool_calls = {}
        self.tasks: dict[str, AsyncTaskRecord] = {}
        self.commit_barrier = None
        self.commit_lock = Lock()

    def get_agent_thread_binding_by_thread_id(self, thread_id: str):
        return self.binding if thread_id == self.binding.codex_thread_id else None

    def get_session_state(self, session_id: str):
        return self.state if session_id == self.state.session_id else None

    def commit_agent_workspace_tool(self, **kwargs):
        if self.commit_barrier is not None:
            self.commit_barrier.wait(timeout=2)
        with self.commit_lock:
            key = (kwargs["session_id"], kwargs["idempotency_key"])
            prior = self.tool_calls.get(key)
            if prior is not None:
                if (
                    prior.tool_name != kwargs["tool_name"]
                    or prior.arguments_sha256 != kwargs["arguments_sha256"]
                ):
                    raise ValueError("idempotency_key conflict")
                return prior.model_copy(update={"reused": True})
            if self.state.revision != kwargs["expected_revision"]:
                raise RuntimeError("SessionState CAS mismatch")
            committed = kwargs["expected_revision"] + 1
            self.state = self.state.model_copy(
                update={
                    "workspace_json": kwargs["workspace_json"],
                    "workspace_sha256": kwargs["workspace_sha256"],
                    "revision": committed,
                }
            )
            record = AgentToolCallRecord(
                session_id=kwargs["session_id"],
                idempotency_key=kwargs["idempotency_key"],
                tool_name=kwargs["tool_name"],
                arguments_sha256=kwargs["arguments_sha256"],
                expected_revision=kwargs["expected_revision"],
                committed_revision=committed,
                result_json=kwargs["result_json"],
            )
            self.tool_calls[key] = record
            task = kwargs.get("async_task")
            if task is not None:
                self.tasks[task.task_id] = task
            return record

    def list_async_tasks(self, **kwargs):
        jobs = list(self.tasks.values())
        if kwargs.get("session_id") is not None:
            jobs = [job for job in jobs if job.session_id == kwargs["session_id"]]
        if kwargs.get("kind") is not None:
            jobs = [job for job in jobs if job.kind == kwargs["kind"]]
        return jobs[: kwargs.get("limit", 20)]


def test_gateway_resolves_thread_to_canonical_session() -> None:
    gateway = ReadOnlyToolGateway(LitTraceConfig(), _GatewayStore())
    result = asyncio.run(
        gateway.call("get_workspace_context", {}, codex_thread_id="thread-1")
    )
    assert result["session_id"] == "session-1"
    assert [paper["paper_id"] for paper in result["papers"]] == ["paper-1"]


def test_gateway_tool_specs_match_the_app_server_allowlist() -> None:
    assert tuple(spec["name"] for spec in app_server_tool_specs()) == APP_SERVER_TOOL_NAMES


def test_gateway_rejects_unbound_threads_and_tools_outside_allowlist() -> None:
    gateway = ReadOnlyToolGateway(LitTraceConfig(), _GatewayStore())
    with pytest.raises(PermissionError, match="not bound"):
        asyncio.run(gateway.call("get_workspace_context", {}, codex_thread_id="other"))
    with pytest.raises(PermissionError, match="App Server allowlist"):
        asyncio.run(gateway.call("parse_full_text", {}, codex_thread_id="thread-1"))


def test_gateway_commits_search_results_and_preserves_research_background(
    monkeypatch,
) -> None:
    store = _GatewayStore()
    workspace = LiteratureWorkspace.model_validate(store.state.workspace_json)
    workspace.context.filters.research_background = "Flexible sensor materials"
    workspace.context.filters.research_background_status = "accepted"
    store.state = store.state.model_copy(
        update={"workspace_json": workspace.model_dump(mode="json")}
    )

    async def fake_search(request, _config):
        assert request.topic == "MXene pressure sensors"
        result = LiteratureWorkspace(
            papers={
                "paper-2": PaperMetadata(
                    paper_id="paper-2",
                    title="MXene Sensor",
                    year=2025,
                )
            }
        )
        result.context.active_papers = ["paper-2"]
        result.context.filters.topic = request.topic
        result.context.filters.search_mode = "mock"
        return result

    monkeypatch.setattr("littrace.workflow.run_search_preview", fake_search)
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)
    result = asyncio.run(
        gateway.call(
            "search_papers",
            {
                "topic": "MXene pressure sensors",
                "year_min": 2020,
                "limit": 20,
                "live": False,
                "expected_revision": 0,
                "idempotency_key": "search-turn-1",
            },
            codex_thread_id="thread-1",
        )
    )

    committed = LiteratureWorkspace.model_validate(store.state.workspace_json)
    assert result["active_paper_count"] == 1
    assert result["workspace_revision"] == 1
    assert committed.context.active_papers == ["paper-2"]
    assert committed.context.filters.research_background == "Flexible sensor materials"
    assert committed.context.filters.research_background_status == "accepted"
    assert committed.context.filters.workspace_revision == 1
    assert store.state.revision == 1


def test_gateway_commits_download_selection_exactly_once() -> None:
    store = _GatewayStore()
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)
    arguments = {
        "mode": "add",
        "paper_ids": ["paper-1"],
        "expected_revision": 0,
        "idempotency_key": "selection-turn-1",
    }

    first = asyncio.run(
        gateway.call(
            "set_download_selection",
            arguments,
            codex_thread_id="thread-1",
        )
    )
    replay = asyncio.run(
        gateway.call(
            "set_download_selection",
            arguments,
            codex_thread_id="thread-1",
        )
    )

    assert first["selected_for_download"] == ["paper-1"]
    assert first["workspace_revision"] == 1
    assert first["idempotency_reused"] is False
    assert replay["idempotency_reused"] is True
    assert store.state.revision == 1
    assert len(store.tool_calls) == 1


def test_gateway_atomically_enqueues_download_exactly_once() -> None:
    store = _GatewayStore()
    workspace = LiteratureWorkspace.model_validate(store.state.workspace_json)
    workspace.context.selected_for_download = ["paper-1"]
    store.state = store.state.model_copy(
        update={"workspace_json": workspace.model_dump(mode="json")}
    )
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)
    arguments = {
        "expected_revision": 0,
        "idempotency_key": "download-turn-1",
    }

    first = asyncio.run(
        gateway.call("enqueue_download", arguments, codex_thread_id="thread-1")
    )
    replay = asyncio.run(
        gateway.call("enqueue_download", arguments, codex_thread_id="thread-1")
    )
    jobs = asyncio.run(
        gateway.call("get_download_jobs", {}, codex_thread_id="thread-1")
    )

    assert first["status"] == "queued"
    assert first["paper_ids"] == ["paper-1"]
    assert first["workspace_revision"] == 1
    assert replay["idempotency_reused"] is True
    assert store.state.revision == 1
    assert list(store.tasks) == [first["task_id"]]
    assert jobs["jobs"][0]["status"] == "queued"
    assert jobs["jobs"][0]["paper_ids"] == ["paper-1"]


def test_gateway_atomically_enqueues_parse_from_registered_pdf(
    monkeypatch,
) -> None:
    store = _GatewayStore()

    class Registry:
        def find_in_session(self, artifact_id, *, session_id):
            assert artifact_id == "paper_pdf:paper-1"
            assert session_id == "session-1"
            return ArtifactRecord(
                artifact_id=artifact_id,
                session_id=session_id,
                kind="paper_pdf",
                paper_id="paper-1",
                object_key="sessions/session-1/papers/paper-1/paper.pdf",
                backend="local",
                sha256="a" * 64,
            )

    monkeypatch.setattr(
        "littrace.artifact_registry.artifact_registry_from_config",
        lambda _config: Registry(),
    )
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)
    arguments = {
        "paper_ids": ["paper-1"],
        "parse_strategy": "text_only",
        "expected_revision": 0,
        "idempotency_key": "parse-turn-1",
    }

    first = asyncio.run(
        gateway.call("enqueue_parse", arguments, codex_thread_id="thread-1")
    )
    replay = asyncio.run(
        gateway.call("enqueue_parse", arguments, codex_thread_id="thread-1")
    )
    jobs = asyncio.run(
        gateway.call("get_parse_jobs", {}, codex_thread_id="thread-1")
    )

    assert first["status"] == "queued"
    assert first["parse_strategy"] == "text_only"
    assert replay["idempotency_reused"] is True
    assert store.state.revision == 1
    task = store.tasks[first["task_id"]]
    assert task.kind == "parse_job"
    assert task.result_json["command"]["sources"][0]["sha256"] == "a" * 64
    assert jobs["jobs"][0]["paper_ids"] == ["paper-1"]


def test_gateway_atomically_enqueues_table_extraction() -> None:
    store = _GatewayStore()
    workspace = LiteratureWorkspace.model_validate(store.state.workspace_json)
    workspace.parsed_papers["paper-1"] = ParsedPaper(
        title="One",
        parsed=True,
        sections=[{"name": "Results", "text": "Sensitivity 12.5 kPa-1"}],
    )
    store.state = store.state.model_copy(
        update={"workspace_json": workspace.model_dump(mode="json")}
    )
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)
    arguments = {
        "paper_ids": ["paper-1"],
        "expected_revision": 0,
        "idempotency_key": "table-turn-1",
    }

    queued = asyncio.run(
        gateway.call(
            "enqueue_table_extraction",
            arguments,
            codex_thread_id="thread-1",
        )
    )
    replay = asyncio.run(
        gateway.call(
            "enqueue_table_extraction",
            arguments,
            codex_thread_id="thread-1",
        )
    )
    jobs = asyncio.run(
        gateway.call("get_table_jobs", {}, codex_thread_id="thread-1")
    )

    assert queued["status"] == "queued"
    assert replay["idempotency_reused"] is True
    assert store.state.revision == 1
    task = store.tasks[queued["task_id"]]
    assert task.kind == "table_job"
    assert len(task.result_json["command"]["parsed_sha256"]["paper-1"]) == 64
    assert jobs["jobs"][0]["paper_ids"] == ["paper-1"]


def test_gateway_atomically_enqueues_document() -> None:
    """Round 20: ``enqueue_document`` mirrors ``enqueue_table_extraction``.

    Asserts CAS atomicity, idempotent replay, schema version, and that
    ``get_document_jobs`` returns the queued task with paper_ids.
    """
    store = _GatewayStore()
    workspace = LiteratureWorkspace.model_validate(store.state.workspace_json)
    workspace.parsed_papers["paper-1"] = ParsedPaper(
        title="One",
        parsed=True,
        sections=[{"name": "Results", "text": "Sensitivity 12.5 kPa-1"}],
    )
    store.state = store.state.model_copy(
        update={"workspace_json": workspace.model_dump(mode="json")}
    )
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)
    arguments = {
        "paper_ids": ["paper-1"],
        "expected_revision": 0,
        "idempotency_key": "document-turn-1",
    }

    queued = asyncio.run(
        gateway.call(
            "enqueue_document",
            arguments,
            codex_thread_id="thread-1",
        )
    )
    replay = asyncio.run(
        gateway.call(
            "enqueue_document",
            arguments,
            codex_thread_id="thread-1",
        )
    )
    jobs = asyncio.run(
        gateway.call("get_document_jobs", {}, codex_thread_id="thread-1")
    )

    assert queued["status"] == "queued"
    assert replay["idempotency_reused"] is True
    assert store.state.revision == 1
    task = store.tasks[queued["task_id"]]
    assert task.kind == "document_job"
    assert task.result_json["schema_version"] == "littrace.document_job.v1"
    assert len(task.result_json["command"]["source_sha256"]["paper-1"]) == 64
    assert jobs["jobs"][0]["paper_ids"] == ["paper-1"]


def test_gateway_rejects_unparsed_papers_for_document() -> None:
    """Round 20: ``enqueue_document`` requires parsed papers."""
    store = _GatewayStore()
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)

    with pytest.raises(
        ValueError, match="Papers must be parsed before document generation"
    ):
        asyncio.run(
            gateway.call(
                "enqueue_document",
                {
                    "paper_ids": ["paper-1"],
                    "expected_revision": 0,
                    "idempotency_key": "document-unparsed",
                },
                codex_thread_id="thread-1",
            )
        )


def test_gateway_get_document_jobs_returns_empty() -> None:
    store = _GatewayStore()
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)
    result = asyncio.run(
        gateway.call("get_document_jobs", {}, codex_thread_id="thread-1")
    )
    assert result["jobs"] == []


def test_gateway_atomically_enqueues_autonomous_review() -> None:
    """Round 20: ``enqueue_autonomous_review`` mirrors ``enqueue_document``.

    Asserts CAS atomicity, idempotent replay, schema version,
    ``auto_replan=True`` propagation, and that ``get_autonomous_review_jobs``
    returns the queued task with paper_ids and auto_replan.
    """
    store = _GatewayStore()
    workspace = LiteratureWorkspace.model_validate(store.state.workspace_json)
    workspace.parsed_papers["paper-1"] = ParsedPaper(
        title="One",
        parsed=True,
        sections=[{"name": "Results", "text": "Sensitivity 12.5 kPa-1"}],
    )
    store.state = store.state.model_copy(
        update={"workspace_json": workspace.model_dump(mode="json")}
    )
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)
    arguments = {
        "paper_ids": ["paper-1"],
        "auto_replan": True,
        "expected_revision": 0,
        "idempotency_key": "autoreview-turn-1",
    }

    queued = asyncio.run(
        gateway.call(
            "enqueue_autonomous_review",
            arguments,
            codex_thread_id="thread-1",
        )
    )
    replay = asyncio.run(
        gateway.call(
            "enqueue_autonomous_review",
            arguments,
            codex_thread_id="thread-1",
        )
    )
    jobs = asyncio.run(
        gateway.call(
            "get_autonomous_review_jobs", {}, codex_thread_id="thread-1"
        )
    )

    assert queued["status"] == "queued"
    assert queued["auto_replan"] is True
    assert replay["idempotency_reused"] is True
    assert store.state.revision == 1
    task = store.tasks[queued["task_id"]]
    assert task.kind == "autonomous_review_job"
    assert task.result_json["schema_version"] == "littrace.autonomous_review_job.v1"
    assert task.result_json["command"]["auto_replan"] is True
    assert len(task.result_json["command"]["source_sha256"]["paper-1"]) == 64
    assert jobs["jobs"][0]["paper_ids"] == ["paper-1"]
    assert jobs["jobs"][0]["auto_replan"] is True


def test_gateway_rejects_nonbool_auto_replan() -> None:
    """Round 20: ``auto_replan`` must be a strict bool, not a string."""
    store = _GatewayStore()
    workspace = LiteratureWorkspace.model_validate(store.state.workspace_json)
    workspace.parsed_papers["paper-1"] = ParsedPaper(
        title="One",
        parsed=True,
        sections=[{"name": "Results", "text": "Sensitivity 12.5 kPa-1"}],
    )
    store.state = store.state.model_copy(
        update={"workspace_json": workspace.model_dump(mode="json")}
    )
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)

    with pytest.raises(TypeError, match="auto_replan must be a boolean"):
        asyncio.run(
            gateway.call(
                "enqueue_autonomous_review",
                {
                    "paper_ids": ["paper-1"],
                    "auto_replan": "yes",
                    "expected_revision": 0,
                    "idempotency_key": "autoreview-bad",
                },
                codex_thread_id="thread-1",
            )
        )


def test_gateway_get_autonomous_review_jobs_returns_empty() -> None:
    store = _GatewayStore()
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)
    result = asyncio.run(
        gateway.call(
            "get_autonomous_review_jobs", {}, codex_thread_id="thread-1"
        )
    )
    assert result["jobs"] == []


def test_gateway_rejects_stale_write_revision() -> None:
    store = _GatewayStore()
    store.state = store.state.model_copy(update={"revision": 3})
    workspace_json = dict(store.state.workspace_json)
    workspace_json["context"]["filters"]["workspace_revision"] = 3
    store.state = store.state.model_copy(update={"workspace_json": workspace_json})
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)

    with pytest.raises(RuntimeError, match="CAS mismatch"):
        asyncio.run(
            gateway.call(
                "set_download_selection",
                {
                    "mode": "clear",
                    "expected_revision": 2,
                    "idempotency_key": "selection-stale",
                },
                codex_thread_id="thread-1",
            )
        )


def test_concurrent_retry_commits_one_selection_change() -> None:
    store = _GatewayStore()
    store.commit_barrier = Barrier(2)
    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)
    arguments = {
        "mode": "replace",
        "paper_ids": ["paper-1"],
        "expected_revision": 0,
        "idempotency_key": "concurrent-selection",
    }

    def invoke():
        return asyncio.run(
            gateway.call(
                "set_download_selection",
                arguments,
                codex_thread_id="thread-1",
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: invoke(), range(2)))

    assert sorted(result["idempotency_reused"] for result in results) == [False, True]
    assert store.state.revision == 1
    assert len(store.tool_calls) == 1
