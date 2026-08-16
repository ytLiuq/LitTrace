from littrace.api.app import api_app
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header

from littrace.api.auth import resolve_request_session
from littrace.session_metrics import SessionKnowledgeMetricsReport, build_session_knowledge_metrics
from littrace.session import SessionDeleteReport, delete_chat_session


router = APIRouter()


@router.delete("/sessions/{session_id}", response_model=SessionDeleteReport)
def delete_session(
    session_id: str,
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> SessionDeleteReport:
    config = api_app.load_config()
    auth = resolve_request_session(
        config,
        route_session_id=session_id,
        header_session_id=x_littrace_session_id,
    )
    return delete_chat_session(config, auth.session_id)


@router.get("/sessions/{session_id}/metrics", response_model=SessionKnowledgeMetricsReport)
def session_metrics(
    session_id: str,
    artifact_limit: int = 200,
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> SessionKnowledgeMetricsReport:
    config = api_app.load_config()
    auth = resolve_request_session(
        config,
        route_session_id=session_id,
        header_session_id=x_littrace_session_id,
    )
    return build_session_knowledge_metrics(
        config,
        auth.session_id,
        artifact_limit=artifact_limit,
    )
