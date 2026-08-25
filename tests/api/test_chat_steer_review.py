"""Round 8 route tests for /chat/steer, /chat/review and
/chat/{turn_id}/cancel.

The chat service is monkeypatched so the test does not boot an
App Server subprocess. The fake surfaces the same contract the
service exposes (``steer`` / ``start_review`` /
``cancel_turn_with_reason``) so the route is exercised
end-to-end through FastAPI's TestClient.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api


@pytest.fixture
def client() -> TestClient:
    from littrace.api.app import make_app

    return TestClient(make_app())


def test_chat_steer_returns_typed_payload(client, monkeypatch) -> None:
    from littrace.api.routes import research as research_route
    from littrace.codex_runtime.client import SteerTurnResult

    seen: dict[str, Any] = {}

    async def fake_steer(
        self, session, turn_id, text, *, client_user_message_id=None,
    ):
        seen["turn_id"] = turn_id
        seen["text"] = text
        seen["client_user_message_id"] = client_user_message_id
        return SteerTurnResult(
            thread_id="thr-r8",
            turn_id=turn_id,
            client_user_message_id=client_user_message_id,
        )

    monkeypatch.setattr(
        research_route.CodexAppServerChatService, "steer", fake_steer,
    )
    response = client.post(
        "/chat/steer",
        json={
            "turn_id": "turn-existing-1",
            "text": "actually focus on the failing tests first",
            "client_user_message_id": "client-msg-r8-1",
            "session_id": "r8-smoke-steer",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["turn_id"] == "turn-existing-1"
    assert body["thread_id"] == "thr-r8"
    assert body["client_user_message_id"] == "client-msg-r8-1"
    assert body["session_id"] == "r8-smoke-steer"
    assert seen["text"] == "actually focus on the failing tests first"


def test_chat_review_returns_typed_payload(client, monkeypatch) -> None:
    from littrace.api.routes import research as research_route
    from littrace.codex_runtime.client import AppServerTurnResult

    seen: dict[str, Any] = {}

    async def fake_start_review(
        self, session, *, target=None, on_review_complete=None,
    ):
        seen["target"] = target
        seen["callback"] = on_review_complete
        # Simulate the on_review_complete hook firing.
        on_review_complete({"type": "exitedReviewMode", "id": "r1"})
        return AppServerTurnResult(
            thread_id="thr-r8",
            turn_id="turn-review-r8",
            status="completed",
            reply="verdict: ship it",
            turn={"id": "turn-review-r8", "status": "completed", "items": []},
        )

    monkeypatch.setattr(
        research_route.CodexAppServerChatService, "start_review",
        fake_start_review,
    )
    response = client.post(
        "/chat/review",
        json={
            "target": {"type": "commit", "sha": "deadbeef"},
            "session_id": "r8-smoke-review",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["turn_id"] == "turn-review-r8"
    assert body["status"] == "completed"
    assert body["review_text"] == "verdict: ship it"
    assert body["exit_item"] == {"type": "exitedReviewMode", "id": "r1"}
    assert body["session_id"] == "r8-smoke-review"
    assert seen["target"] == {"type": "commit", "sha": "deadbeef"}
    assert seen["callback"] is not None


def test_chat_cancel_stamps_reason_on_binding(client, monkeypatch) -> None:
    from littrace.api.routes import research as research_route

    seen: dict[str, Any] = {}

    async def fake_cancel(self, session, turn_id, *, reason):
        seen["turn_id"] = turn_id
        seen["reason"] = reason
        return True

    monkeypatch.setattr(
        research_route.CodexAppServerChatService, "cancel_turn_with_reason",
        fake_cancel,
    )
    response = client.post(
        "/chat/turn-abc/cancel",
        json={"reason": "user_pressed_esc", "session_id": "r8-smoke-cancel"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["turn_id"] == "turn-abc"
    assert body["reason"] == "user_pressed_esc"
    assert body["acknowledged"] is True
    assert body["session_id"] == "r8-smoke-cancel"
    assert seen["reason"] == "user_pressed_esc"


def test_steer_review_cancel_in_openapi_schema() -> None:
    """The three new routes are advertised under the research tag."""
    from littrace.api.app import make_app

    schema = make_app().openapi()
    paths = schema.get("paths", {})
    for path in ("/chat/steer", "/chat/review"):
        assert path in paths, paths.keys()
        post = paths[path]["post"]
        assert "research" in post.get("tags", [])
    # /chat/{turn_id}/cancel uses a path parameter
    cancel_paths = [
        p for p in paths.keys() if p.startswith("/chat/") and p.endswith("/cancel")
    ]
    assert cancel_paths, "expected at least one /chat/.../cancel path"
    cancel_path = cancel_paths[0]
    post = paths[cancel_path]["post"]
    assert "research" in post.get("tags", [])
