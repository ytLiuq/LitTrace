from types import SimpleNamespace
from pathlib import Path

import pytest

from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.runtime.memory import SessionMemory


@pytest.mark.anyio
async def test_research_chat_route_passes_session_memory(monkeypatch):
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
        captured["session_memory"] = session_memory
        return ChatResponse(reply="ok", action="help", workspace=workspace), workspace

    monkeypatch.setattr(routes, "api_app", fake_api)
    monkeypatch.setattr(routes, "load_or_create_session", lambda config, session_id: fake_session)
    monkeypatch.setattr(routes, "load_workspace", lambda session: LiteratureWorkspace())
    monkeypatch.setattr(
        routes,
        "load_memory",
        lambda session: SessionMemory(),
    )
    monkeypatch.setattr(routes, "handle_chat", fake_handle_chat)
    monkeypatch.setattr(routes, "save_workspace", lambda session, workspace: None)
    monkeypatch.setattr(routes, "append_message", lambda session, role, payload: None)

    response = await routes.chat(ChatRequest(message="你好", session_id="s1"))

    assert response.action == "help"
    assert isinstance(captured["session_memory"], SessionMemory)
    assert "workspace" in captured


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
