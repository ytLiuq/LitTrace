"""Round 20 (Phase 6): end-to-end MCP command coverage test.

This is the contract that locks in "every Codex-exposed tool returns a
well-formed envelope when invoked through ``ReadOnlyToolGateway.call``."
It is the last guard against a future spec/drift mismatch (a tool
present in ``APP_SERVER_TOOL_NAMES`` but missing from the dispatch
table, or vice versa) silently breaking the TUI = Codex 唯一对话面
promise.

Coverage goal: every name in ``APP_SERVER_TOOL_NAMES`` is reachable,
returns a dict, and (where the call would otherwise need parsed papers
or an active download selection) the failure mode is a ``ValueError``
or ``RuntimeError`` from the gateway — NOT a ``KeyError`` from a missing
dispatch arm or an ``AttributeError`` from a missing handler.

The test runs against the same ``_GatewayStore`` fake the per-tool
tests use; we pre-seed just enough state for each call to either
succeed or fail with a typed validation error.
"""

from __future__ import annotations

import asyncio
from threading import Lock

import pytest

from littrace.artifact_registry import ArtifactRecord
from littrace.codex_runtime.gateway import (
    APP_SERVER_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    WRITE_TOOL_NAMES,
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


class _FullCoverageStore:
    """In-memory store seeded to make every MCP tool reachable.

    The store mirrors the fields ``ReadOnlyToolGateway`` consults:
    ``get_session_state``, ``commit_agent_workspace_tool``,
    ``list_async_tasks``, ``get_agent_thread_binding_by_thread_id``.
    """

    def __init__(self) -> None:
        workspace = LiteratureWorkspace(
            papers={
                "paper-1": PaperMetadata(paper_id="paper-1", title="One"),
                "paper-2": PaperMetadata(paper_id="paper-2", title="Two"),
            }
        )
        workspace.context.active_papers = ["paper-1"]
        workspace.context.selected_for_download = ["paper-1"]
        workspace.parsed_papers["paper-1"] = ParsedPaper(
            title="One",
            parsed=True,
            sections=[{"name": "Results", "text": "Sensitivity 12.5 kPa-1"}],
        )
        self.binding = AgentThreadBindingRecord(
            session_id="session-1", codex_thread_id="thread-1"
        )
        self.state = SessionStateRecord(
            session_id="session-1",
            workspace_json=workspace.model_dump(mode="json"),
        )
        self.tool_calls: dict[tuple[str, str], AgentToolCallRecord] = {}
        self.tasks: dict[str, AsyncTaskRecord] = {}
        self._lock = Lock()

    def get_agent_thread_binding_by_thread_id(self, thread_id: str):
        return self.binding if thread_id == self.binding.codex_thread_id else None

    def get_session_state(self, session_id: str):
        return self.state if session_id == self.state.session_id else None

    def commit_agent_workspace_tool(self, **kwargs):
        with self._lock:
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
            jobs = [j for j in jobs if j.session_id == kwargs["session_id"]]
        if kwargs.get("kind") is not None:
            jobs = [j for j in jobs if j.kind == kwargs["kind"]]
        return jobs[: kwargs.get("limit", 20)]


@pytest.fixture
def store(monkeypatch):
    store = _FullCoverageStore()

    class Registry:
        def find_in_session(self, artifact_id, *, session_id):
            assert artifact_id == "paper_pdf:paper-1"
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
    return store


def _build_arguments(name: str, *, idempotency_suffix: str) -> dict:
    """Return the minimal valid arguments for ``name``.

    Each call gets a unique ``idempotency_key`` so the store's
    idempotency layer doesn't short-circuit the second invocation.
    """
    if name in READ_ONLY_TOOL_NAMES:
        # Read-only tools have no required arguments.
        return {}
    if name == "set_download_selection":
        return {
            "mode": "replace",
            "paper_ids": ["paper-1"],
            "expected_revision": 0,
            "idempotency_key": f"{name}-{idempotency_suffix}",
        }
    if name == "search_papers":
        return {
            "topic": "MXene pressure sensors",
            "year_min": 2020,
            "limit": 5,
            "live": False,
            "expected_revision": 0,
            "idempotency_key": f"{name}-{idempotency_suffix}",
        }
    if name == "enqueue_download":
        return {
            "expected_revision": 0,
            "idempotency_key": f"{name}-{idempotency_suffix}",
        }
    if name == "enqueue_parse":
        return {
            "paper_ids": ["paper-1"],
            "parse_strategy": "text_only",
            "expected_revision": 0,
            "idempotency_key": f"{name}-{idempotency_suffix}",
        }
    if name == "enqueue_table_extraction":
        return {
            "paper_ids": ["paper-1"],
            "expected_revision": 0,
            "idempotency_key": f"{name}-{idempotency_suffix}",
        }
    if name in {"enqueue_storyline", "enqueue_document"}:
        return {
            "paper_ids": ["paper-1"],
            "expected_revision": 0,
            "idempotency_key": f"{name}-{idempotency_suffix}",
        }
    if name == "enqueue_autonomous_review":
        return {
            "paper_ids": ["paper-1"],
            "auto_replan": False,
            "expected_revision": 0,
            "idempotency_key": f"{name}-{idempotency_suffix}",
        }
    raise AssertionError(f"unknown tool name in coverage harness: {name}")


def test_app_server_tool_names_match_read_and_write_split() -> None:
    """The split is part of the gateway contract — a future refactor
    that drops a name from ``READ_ONLY_TOOL_NAMES`` will silently break
    the dispatch arm and only the per-tool tests would catch it. Lock
    the split here so the regression signal is loud."""
    assert set(APP_SERVER_TOOL_NAMES) == set(READ_ONLY_TOOL_NAMES) | set(
        WRITE_TOOL_NAMES
    )
    assert set(READ_ONLY_TOOL_NAMES) & set(WRITE_TOOL_NAMES) == set()


def test_app_server_tool_names_include_round_20_additions() -> None:
    """Round 20 added ``enqueue_document`` / ``enqueue_autonomous_review``
    and the matching ``get_*_jobs`` readers. Lock their presence in the
    allowlist so a future ``git revert`` of Phase 2 / 3 fails the test
    instead of silently demoting the model to a tool-less Code-only
    surface."""
    assert "enqueue_document" in APP_SERVER_TOOL_NAMES
    assert "enqueue_autonomous_review" in APP_SERVER_TOOL_NAMES
    assert "get_document_jobs" in APP_SERVER_TOOL_NAMES
    assert "get_autonomous_review_jobs" in APP_SERVER_TOOL_NAMES


def test_app_server_tool_specs_cover_every_name_in_allowlist() -> None:
    """Every name in ``APP_SERVER_TOOL_NAMES`` must have a matching spec
    entry — otherwise the App Server will advertise the tool with no
    schema and the model can't reason about it."""
    spec_names = {spec["name"] for spec in app_server_tool_specs()}
    missing = set(APP_SERVER_TOOL_NAMES) - spec_names
    extra = spec_names - set(APP_SERVER_TOOL_NAMES)
    assert not missing, f"specs missing for: {sorted(missing)}"
    assert not extra, f"specs advertise unknown tools: {sorted(extra)}"


@pytest.mark.parametrize("tool_name", APP_SERVER_TOOL_NAMES)
def test_every_mcp_tool_is_dispatchable_and_returns_a_dict(
    tool_name: str, store
) -> None:
    """Walk every name in ``APP_SERVER_TOOL_NAMES`` through the gateway.

    Either the call succeeds and returns a dict-shaped envelope, or it
    raises a typed validation error (``ValueError`` /
    ``RuntimeError`` / ``PermissionError``) — but never
    ``KeyError`` / ``AttributeError`` from a missing dispatch arm.
    """
    from littrace.codex_runtime.gateway import ReadOnlyToolGateway

    gateway = ReadOnlyToolGateway(LitTraceConfig(), store)
    arguments = _build_arguments(tool_name, idempotency_suffix=tool_name)

    try:
        result = asyncio.run(
            gateway.call(tool_name, arguments, codex_thread_id="thread-1")
        )
    except (ValueError, RuntimeError, PermissionError, TypeError):
        # Typed validation failures are acceptable — they confirm the
        # dispatch arm exists and ran the validator. We only reject
        # ``KeyError`` / ``AttributeError`` below.
        return

    assert isinstance(result, dict), (
        f"{tool_name!r} must return a dict envelope; got {type(result).__name__}"
    )
