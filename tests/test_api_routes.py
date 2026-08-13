from types import SimpleNamespace
from pathlib import Path

import pytest

from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.runtime.memory import SessionMemory


@pytest.mark.anyio
async def test_research_chat_route_sets_explicit_background_before_normal_chat(monkeypatch):
    from littrace.api.routes import research as routes

    captured: dict[str, object] = {}

    fake_session = SimpleNamespace(session_id="s1", root=Path("/tmp/littrace-test-session"))

    fake_api = SimpleNamespace(
        load_config=lambda: object(),
        _set_workspace=lambda workspace: captured.__setitem__("workspace", workspace),
        append_trace=lambda *args, **kwargs: None,
        WORKSPACE=LiteratureWorkspace(),
    )

    async def fake_handle_chat(request, workspace, config, session_memory=None):
        captured["handle_chat_called"] = True
        return ChatResponse(reply="should not be used", action="help", workspace=workspace), workspace

    monkeypatch.setattr(routes, "api_app", fake_api)
    monkeypatch.setattr(routes, "load_or_create_session", lambda config, session_id: fake_session)
    monkeypatch.setattr(routes, "load_workspace", lambda session: LiteratureWorkspace())
    monkeypatch.setattr(
        routes,
        "load_memory",
        lambda session: SessionMemory(),
    )
    monkeypatch.setattr(routes, "handle_chat", fake_handle_chat)
    monkeypatch.setattr(routes, "save_workspace", lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, "append_message", lambda session, role, payload: None)

    response = await routes.chat(
        ChatRequest(
            message="我研究柔性压力传感器的器件机理和长期稳定性",
            session_id="s1",
            research_background="我研究柔性压力传感器的器件机理和长期稳定性",
        )
    )

    assert response.action == "research_background_set"
    assert "handle_chat_called" not in captured
    assert "workspace" in captured
    assert response.workspace.context.filters.research_background == "我研究柔性压力传感器的器件机理和长期稳定性"


@pytest.mark.anyio
async def test_research_chat_route_passes_session_memory_after_background_is_set(monkeypatch):
    from littrace.api.routes import research as routes

    captured: dict[str, object] = {}

    fake_session = SimpleNamespace(session_id="s1", root=Path("/tmp/littrace-test-session"))
    workspace = LiteratureWorkspace()
    workspace.context.filters.research_background = "我研究柔性压力传感器的器件机理和长期稳定性"
    workspace.context.filters.research_background_status = "accepted"

    fake_api = SimpleNamespace(
        load_config=lambda: object(),
        _set_workspace=lambda workspace: captured.__setitem__("workspace", workspace),
        append_trace=lambda *args, **kwargs: None,
        WORKSPACE=workspace,
    )

    async def fake_handle_chat(request, workspace, config, session_memory=None):
        captured["session_memory"] = session_memory
        return ChatResponse(reply="ok", action="help", workspace=workspace), workspace

    monkeypatch.setattr(routes, "api_app", fake_api)
    monkeypatch.setattr(routes, "load_or_create_session", lambda config, session_id: fake_session)
    monkeypatch.setattr(routes, "load_workspace", lambda session: workspace)
    monkeypatch.setattr(
        routes,
        "load_memory",
        lambda session: SessionMemory(),
    )
    monkeypatch.setattr(routes, "handle_chat", fake_handle_chat)
    monkeypatch.setattr(routes, "save_workspace", lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, "append_message", lambda session, role, payload: None)

    response = await routes.chat(ChatRequest(message="继续分析最新论文", session_id="s1"))

    assert response.action == "help"
    assert isinstance(captured["session_memory"], SessionMemory)
    assert "workspace" in captured


@pytest.mark.anyio
async def test_research_chat_route_requires_background_first(monkeypatch):
    from littrace.api.routes import research as routes

    fake_session = SimpleNamespace(session_id="s1", root=Path("/tmp/littrace-test-session"))

    fake_api = SimpleNamespace(
        load_config=lambda: object(),
        _set_workspace=lambda workspace: None,
        append_trace=lambda *args, **kwargs: None,
        WORKSPACE=LiteratureWorkspace(),
    )

    monkeypatch.setattr(routes, "api_app", fake_api)
    monkeypatch.setattr(routes, "load_or_create_session", lambda config, session_id: fake_session)
    monkeypatch.setattr(routes, "load_workspace", lambda session: LiteratureWorkspace())
    monkeypatch.setattr(routes, "save_workspace", lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, "append_message", lambda *args, **kwargs: None)

    response = await routes.chat(ChatRequest(message="你好", session_id="s1"))

    assert response.action == "research_background_required"
    assert "研究背景" in response.reply


@pytest.mark.anyio
async def test_research_chat_route_requires_background_even_when_legacy_topic_exists(monkeypatch):
    from littrace.api.routes import research as routes

    fake_session = SimpleNamespace(session_id="s1", root=Path("/tmp/littrace-test-session"))
    workspace = LiteratureWorkspace()
    workspace.context.filters.topic = "legacy topic from search"

    fake_api = SimpleNamespace(
        load_config=lambda: object(),
        _set_workspace=lambda workspace: None,
        append_trace=lambda *args, **kwargs: None,
        WORKSPACE=workspace,
    )

    monkeypatch.setattr(routes, "api_app", fake_api)
    monkeypatch.setattr(routes, "load_or_create_session", lambda config, session_id: fake_session)
    monkeypatch.setattr(routes, "load_workspace", lambda session: workspace)
    monkeypatch.setattr(routes, "save_workspace", lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, "append_message", lambda session, role, payload: None)

    response = await routes.chat(ChatRequest(message="你好", session_id="s1"))

    assert response.action == "research_background_required"
    assert response.workspace.context.filters.research_background is None


@pytest.mark.anyio
async def test_research_chat_route_keeps_existing_background_when_update_is_invalid(monkeypatch):
    from littrace.api.routes import research as routes

    fake_session = SimpleNamespace(session_id="s1", root=Path("/tmp/littrace-test-session"))
    workspace = LiteratureWorkspace()
    workspace.context.filters.research_background = "我研究柔性压力传感器的器件机理和长期稳定性"
    workspace.context.filters.research_background_status = "accepted"

    fake_api = SimpleNamespace(
        load_config=lambda: object(),
        _set_workspace=lambda workspace: None,
        append_trace=lambda *args, **kwargs: None,
        WORKSPACE=workspace,
    )

    monkeypatch.setattr(routes, "api_app", fake_api)
    monkeypatch.setattr(routes, "load_or_create_session", lambda config, session_id: fake_session)
    monkeypatch.setattr(routes, "load_workspace", lambda session: workspace)
    monkeypatch.setattr(routes, "save_workspace", lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, "append_message", lambda *args, **kwargs: None)

    response = await routes.chat(
        ChatRequest(
            message="需要先设置背景",
            session_id="s1",
            research_background="你好",
        )
    )

    assert response.action == "research_background_required"
    assert workspace.context.filters.research_background_status == "accepted"
    assert workspace.context.filters.research_background == "我研究柔性压力传感器的器件机理和长期稳定性"


@pytest.mark.anyio
async def test_artifacts_parse_context_uses_skill_runner(monkeypatch):
    from littrace.api.routes import artifacts as routes

    captured: dict[str, object] = {}
    workspace = LiteratureWorkspace()

    fake_api = SimpleNamespace(
        load_config=lambda: object(),
        _set_workspace=lambda value: captured.__setitem__("workspace", value),
        WORKSPACE=workspace,
    )

    async def fake_parse_workspace_skill(workspace_arg, config_arg):
        captured["skill_workspace"] = workspace_arg
        return workspace_arg, {"parsed_count": 0}

    monkeypatch.setattr(routes, "api_app", fake_api)
    monkeypatch.setattr(routes, "parse_workspace_skill", fake_parse_workspace_skill)

    result = await routes.parse_context()

    assert result is workspace
    assert captured["skill_workspace"] is workspace
    assert captured["workspace"] is workspace


@pytest.mark.anyio
async def test_download_plan_route_uses_skill_runner(monkeypatch):
    from littrace.api.routes import downloads as routes

    captured: dict[str, object] = {}
    workspace = LiteratureWorkspace()

    fake_api = SimpleNamespace(load_config=lambda: object(), WORKSPACE=workspace)

    async def fake_download_plan_skill(config_arg, workspace_arg):
        captured["workspace"] = workspace_arg
        return {"items": []}

    monkeypatch.setattr(routes, "api_app", fake_api)
    monkeypatch.setattr(routes, "build_download_plan_skill", fake_download_plan_skill)

    result = await routes.download_plan()

    assert result == {"items": []}
    assert captured["workspace"] is workspace


@pytest.mark.anyio
async def test_agents_plan_route_uses_skill_runner(monkeypatch):
    from littrace.api.routes import agents as routes
    from littrace.research_planner import ResearchPlan

    captured: dict[str, object] = {}
    workspace = LiteratureWorkspace()

    async def fake_build_research_plan_skill(topic_arg, workspace_arg):
        captured["topic"] = topic_arg
        captured["workspace"] = workspace_arg
        return ResearchPlan(topic=topic_arg, steps=[])

    monkeypatch.setattr(routes, "get_workspace", lambda: workspace)
    monkeypatch.setattr(routes, "build_research_plan_skill", fake_build_research_plan_skill)

    result = await routes.agents_plan("MXene sensor")

    assert result.topic == "MXene sensor"
    assert captured["topic"] == "MXene sensor"
    assert captured["workspace"] is workspace


def test_api_starts_and_stops_download_retry_worker(monkeypatch):
    from littrace.api import app as api_app
    from littrace.config import LitTraceConfig

    events: list[str] = []
    config = LitTraceConfig()
    config.download_retry.background_worker_enabled = True

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            events.append("created")

        def start(self):
            events.append("started")

        def stop(self, timeout=None):
            events.append(f"stopped:{timeout}")

    monkeypatch.setattr(api_app, "load_config", lambda: config)
    monkeypatch.setattr(api_app, "download_task_store_from_config", lambda _config: object())
    monkeypatch.setattr(api_app, "make_download_retry_handler", lambda _config: object())
    monkeypatch.setattr(api_app, "DownloadRetryWorker", FakeWorker)
    api_app.DOWNLOAD_RETRY_WORKER = None

    api_app._start_background_workers()
    api_app._stop_background_workers()

    assert events == ["created", "started", "stopped:5.0"]


def test_artifact_download_link_checks_registry_scope(monkeypatch, tmp_path):
    from littrace.api.routes import artifacts as routes
    from littrace.artifact_registry import ArtifactRecord
    from littrace.artifact_store import BlobRef
    from littrace.config import LitTraceConfig, ArtifactStorageConfig, StorageConfig

    config = LitTraceConfig(
        storage=StorageConfig(metadata_dir=tmp_path / "metadata"),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    record = ArtifactRecord.from_blob_ref(
            BlobRef(
                backend="local",
                object_key="sessions/s1/papers/p1/paper.pdf",
                content_type="application/pdf",
            ),
            artifact_id="paper_pdf:p1",
            session_id="s1",
            kind="paper_pdf",
            paper_id="p1",
    )

    class FakeRegistry:
        def find_in_session(self, artifact_id, *, session_id):
            if artifact_id == record.artifact_id and session_id == record.session_id:
                return record
            return None

    registry = FakeRegistry()
    fake_api = SimpleNamespace(load_config=lambda: config)
    monkeypatch.setattr(routes, "api_app", fake_api)
    monkeypatch.setattr(routes, "artifact_registry_from_config", lambda _config: registry)

    ok = routes.artifact_download_link(
        "paper_pdf:p1",
        session_id="s1",
        x_littrace_session_id="s1",
    )
    denied = routes.artifact_download_link(
        "paper_pdf:p1",
        session_id="s2",
        x_littrace_session_id="s2",
    )

    assert ok["authorized"] is True
    assert ok["download_url"].startswith("file://")
    assert denied == {"found": False, "authorized": False, "artifact_id": "paper_pdf:p1"}


def test_artifact_download_link_rejects_conflicting_session_header(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from littrace.api.routes import artifacts as routes
    from littrace.config import LitTraceConfig, StorageConfig

    config = LitTraceConfig(storage=StorageConfig(metadata_dir=tmp_path / "metadata"))
    fake_api = SimpleNamespace(load_config=lambda: config)
    monkeypatch.setattr(routes, "api_app", fake_api)

    try:
        routes.artifact_download_link(
            "paper_pdf:p1",
            session_id="s1",
            x_littrace_session_id="s2",
        )
    except HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("expected conflicting session scope to be rejected")


def test_session_delete_route_calls_cleanup(monkeypatch):
    from littrace.api.routes import sessions as routes
    from littrace.session import SessionDeleteReport

    fake_api = SimpleNamespace(load_config=lambda: object())
    monkeypatch.setattr(routes, "api_app", fake_api)
    monkeypatch.setattr(
        routes,
        "delete_chat_session",
        lambda config, session_id: SessionDeleteReport(
            session_id=session_id,
            root_path="/tmp/session",
            deleted=True,
            artifact_count=2,
            embedded_chunk_count=3,
            state_record_count=4,
            object_deleted_count=2,
        ),
    )

    result = routes.delete_session("s1")

    assert result.deleted is True
    assert result.session_id == "s1"


def test_session_metrics_route_resolves_session_scope(monkeypatch):
    from littrace.api.routes import sessions as routes
    from littrace.session_metrics import SessionMetric, SessionKnowledgeMetricsReport

    fake_api = SimpleNamespace(load_config=lambda: object())
    monkeypatch.setattr(routes, "api_app", fake_api)

    def fake_metrics(config, session_id, artifact_limit=200):
        return SessionKnowledgeMetricsReport(
            session_id=session_id,
            readiness="search_ready",
            discovery=SessionMetric(
                name="discovery",
                value=3,
            ),
            acquisition=SessionMetric(
                name="acquisition",
                value=0.5,
            ),
            rag=SessionMetric(
                name="rag",
                value=0.0,
                stale_count=3,
            ),
            consistency=SessionMetric(
                name="consistency",
                value=1.0,
                missing_count=0,
            ),
        )

    monkeypatch.setattr(routes, "build_session_knowledge_metrics", fake_metrics)

    result = routes.session_metrics(
        "s1",
        x_littrace_session_id="s1",
    )

    assert result.session_id == "s1"
    assert result.readiness == "search_ready"
