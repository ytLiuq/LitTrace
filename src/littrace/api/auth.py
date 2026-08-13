from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException

from littrace.config import LitTraceConfig


@dataclass(frozen=True)
class RequestSession:
    session_id: str
    source: str = "default"


def resolve_request_session(
    config: LitTraceConfig,
    *,
    route_session_id: str | None = None,
    header_session_id: str | None = None,
) -> RequestSession:
    """Resolve the API access scope from session metadata.

    LitTrace currently isolates data by ``session_id``. A future auth layer can
    validate a signed session token here without changing route handlers.
    """

    route_value = (route_session_id or "").strip()
    header_value = (header_session_id or "").strip()
    if route_value and header_value and route_value != header_value:
        raise HTTPException(
            status_code=403,
            detail="Route session_id does not match X-LitTrace-Session-Id.",
        )
    if route_value:
        return RequestSession(session_id=route_value, source="route")
    if header_value:
        return RequestSession(session_id=header_value, source="header")
    return RequestSession(session_id="adhoc", source="default")
