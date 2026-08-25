"""SSE contract tests for ``POST /chat/stream``.

The streaming route is new in round 6. The tests use FastAPI's
``TestClient`` to read the ``text/event-stream`` body end-to-end
and assert:

  * the response is ``text/event-stream`` with the SSE framing
    (``event:`` / ``data:`` / blank line),
  * each ``item/agentMessage/delta`` frame becomes a ``delta`` SSE
    event whose ``data.delta`` matches the producer's text,
  * the final ``done`` event carries the same payload shape as
    ``POST /chat`` (a serialised ``ChatResponse``),
  * an exception mid-stream produces a single ``error`` event and
    closes the response cleanly so the client can render a retry
    affordance.

The chat service is monkeypatched with a fake so the test does
not boot an App Server subprocess. The fake mirrors
``CodexAppServerChatService.chat`` enough to fire ``on_delta``
callbacks the way the real client does.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api


@pytest.fixture
def client() -> TestClient:
    # ``make_app`` is the production factory; importing it via
    # the api package is the supported entry point.
    from littrace.api.app import make_app

    return TestClient(make_app())


def _read_sse_events(response) -> list[dict[str, Any]]:
    """Decode a ``text/event-stream`` body into a list of
    ``{event, data}`` dicts.

    The body is read via ``response.read().decode()`` rather than
    ``response.text`` so the test does not depend on the ``httpx``
    streaming shape. SSE comments (``id:`` / retry / etc.) are
    ignored — we only assert on the ``event:`` and ``data:`` lines.
    """
    body = response.read().decode("utf-8")
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for raw_line in body.splitlines():
        if not raw_line or raw_line.startswith(":"):
            if current:
                events.append(current)
                current = {}
            continue
        if raw_line.startswith("event:"):
            current["event"] = raw_line[len("event:"):].strip()
        elif raw_line.startswith("data:"):
            current["data"] = raw_line[len("data:"):].strip()
    if current:
        events.append(current)
    out: list[dict[str, Any]] = []
    for ev in events:
        out.append({"event": ev.get("event"), "data": json.loads(ev["data"])})
    return out


def test_chat_stream_emits_delta_then_done(client, monkeypatch) -> None:
    """A 3-frame delta stream ends with a single ``done`` event."""

    from littrace.api.routes import research as research_route

    async def fake_chat(
        self,
        request,
        workspace,
        session,
        *,
        cancellation=None,
        session_memory=None,
        on_delta=None,
    ):
        # The real service feeds deltas through the callback the
        # way ``AppServerClient`` does in production: one call per
        # server frame. Mirror that here.
        assert on_delta is not None
        for chunk in ("好的，", "按灵敏度", "排序我推荐 5 篇。"):
            on_delta(chunk)
        from littrace.models import ChatResponse

        return (
            ChatResponse(
                reply="好的，按灵敏度排序我推荐 5 篇。",
                action="codex_app_server_chat",
                session_id=session.session_id,
            ),
            workspace,
        )

    monkeypatch.setattr(
        research_route.CodexAppServerChatService, "chat", fake_chat,
    )

    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "MXene 复合材料在柔性压力传感器中的灵敏度对比", "session_id": "sse-smoke-1"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _read_sse_events(response)
    delta_events = [e for e in events if e["event"] == "delta"]
    done_events = [e for e in events if e["event"] == "done"]
    assert [e["data"]["delta"] for e in delta_events] == [
        "好的，", "按灵敏度", "排序我推荐 5 篇。",
    ]
    assert len(done_events) == 1
    assert done_events[0]["data"]["reply"] == "好的，按灵敏度排序我推荐 5 篇。"
    assert done_events[0]["data"]["action"] == "codex_app_server_chat"


def test_chat_stream_emits_error_event_on_failure(client, monkeypatch) -> None:
    """A raising service emits a single ``error`` event then closes."""

    from littrace.api.routes import research as research_route

    async def fake_chat(
        self,
        request,
        workspace,
        session,
        *,
        cancellation=None,
        session_memory=None,
        on_delta=None,
    ):
        if on_delta is not None:
            on_delta("partial reply ")
        raise RuntimeError("boom")

    monkeypatch.setattr(
        research_route.CodexAppServerChatService, "chat", fake_chat,
    )

    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "MXene 复合材料在柔性压力传感器中的灵敏度对比", "session_id": "sse-error-1"},
    ) as response:
        assert response.status_code == 200
        events = _read_sse_events(response)
    error_events = [e for e in events if e["event"] == "error"]
    delta_events = [e for e in events if e["event"] == "delta"]
    done_events = [e for e in events if e["event"] == "done"]
    assert len(delta_events) == 1
    assert delta_events[0]["data"]["delta"] == "partial reply "
    assert len(error_events) == 1
    assert error_events[0]["data"]["code"] == "codex_app_server_chat_failed"
    assert "RuntimeError" in error_events[0]["data"]["message"]
    assert done_events == []


def test_chat_stream_in_openapi_schema() -> None:
    """The new route shows up under the research tag with the right
    operation_id prefix so the auto-generated client can target it."""
    from littrace.api.app import make_app

    schema = make_app().openapi()
    paths = schema.get("paths", {})
    assert "/chat/stream" in paths, paths.keys()
    post = paths["/chat/stream"]["post"]
    assert "research" in post.get("tags", [])
    assert "operationId" in post
