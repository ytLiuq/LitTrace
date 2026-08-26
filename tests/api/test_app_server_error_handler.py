"""Round 7 CR fix: the App Server transport raises typed
``AppServerError`` subclasses. Without a handler FastAPI
would surface them as 500 + a generic message. The handler
maps each ``CodexErrorCode`` to the HTTP status the
operator expects.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api


@pytest.fixture
def client() -> TestClient:
    from littrace.api.app import make_app

    return TestClient(make_app())


def test_unauthorized_maps_to_401(client, monkeypatch) -> None:
    from littrace.api.routes import research as research_route
    from littrace.codex_runtime.errors import UnauthorizedError

    async def fake_start_review(self, session, *, target=None, on_review_complete=None):
        raise UnauthorizedError(
            "isolated LitTrace Codex home is not authenticated"
        )

    monkeypatch.setattr(
        research_route.CodexAppServerChatService, "start_review", fake_start_review,
    )
    response = client.post(
        "/chat/review",
        json={"session_id": "r7-err-1"},
    )
    assert response.status_code == 401
    body = response.json()
    assert body["code"] == "unauthorized"
    assert "isolated LitTrace Codex home" in body["message"]


def test_active_turn_not_steerable_maps_to_409(client, monkeypatch) -> None:
    from littrace.api.routes import research as research_route
    from littrace.codex_runtime.errors import ActiveTurnNotSteerableError

    async def fake_steer(
        self, session, turn_id, text, *, client_user_message_id=None,
    ):
        raise ActiveTurnNotSteerableError("review turn cannot be steered")

    monkeypatch.setattr(
        research_route.CodexAppServerChatService, "steer", fake_steer,
    )
    response = client.post(
        "/chat/steer",
        json={"turn_id": "review-1", "text": "abort"},
    )
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "active_turn_not_steerable"


def test_bad_request_maps_to_400(client, monkeypatch) -> None:
    from littrace.api.routes import research as research_route
    from littrace.codex_runtime.errors import BadRequestError

    async def fake_steer(
        self, session, turn_id, text, *, client_user_message_id=None,
    ):
        raise BadRequestError("missing required field: threadId")

    monkeypatch.setattr(
        research_route.CodexAppServerChatService, "steer", fake_steer,
    )
    response = client.post(
        "/chat/steer",
        json={"turn_id": "t-1", "text": "x"},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "bad_request"


def test_internal_server_error_maps_to_500(client, monkeypatch) -> None:
    from littrace.api.routes import research as research_route
    from littrace.codex_runtime.errors import InternalServerError

    async def fake_steer(
        self, session, turn_id, text, *, client_user_message_id=None,
    ):
        raise InternalServerError("app server subprocess crashed")

    monkeypatch.setattr(
        research_route.CodexAppServerChatService, "steer", fake_steer,
    )
    response = client.post(
        "/chat/steer",
        json={"turn_id": "t-1", "text": "x"},
    )
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_server_error"


def test_handler_includes_additional_details(client, monkeypatch) -> None:
    from littrace.api.routes import research as research_route
    from littrace.codex_runtime.errors import BadRequestError

    async def fake_steer(
        self, session, turn_id, text, *, client_user_message_id=None,
    ):
        raise BadRequestError(
            "missing field",
            additional_details={"field": "threadId", "got": None},
        )

    monkeypatch.setattr(
        research_route.CodexAppServerChatService, "steer", fake_steer,
    )
    response = client.post(
        "/chat/steer",
        json={"turn_id": "t-1", "text": "x"},
    )
    body = response.json()
    assert body["additional_details"] == {"field": "threadId", "got": None}


def test_handler_does_not_swallow_unknown_exceptions(client, monkeypatch) -> None:
    """A non-AppServerError exception still falls through
    FastAPI's default unhandled-exception handler. The
    handler is scoped to ``AppServerError`` so an unrelated
    ``ValueError`` or ``RuntimeError`` does NOT get the
    custom ``code`` field — TestClient re-raises the
    underlying exception so the test sees the original
    ValueError, NOT a 500 JSON response.
    """
    import pytest as _pytest
    from littrace.api.routes import research as research_route

    async def fake_steer(
        self, session, turn_id, text, *, client_user_message_id=None,
    ):
        raise ValueError("totally unrelated")

    monkeypatch.setattr(
        research_route.CodexAppServerChatService, "steer", fake_steer,
    )
    with _pytest.raises(ValueError, match="totally unrelated"):
        client.post(
            "/chat/steer",
            json={"turn_id": "t-1", "text": "x"},
        )
