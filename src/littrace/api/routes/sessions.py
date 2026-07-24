from __future__ import annotations

from fastapi import APIRouter

from littrace.session import SessionDeleteReport, delete_chat_session


class _AppProxy:
    def __getattr__(self, name: str):
        from littrace.api import app as api_app

        return getattr(api_app, name)


api_app = _AppProxy()
router = APIRouter()


@router.delete("/sessions/{session_id}", response_model=SessionDeleteReport)
def delete_session(session_id: str) -> SessionDeleteReport:
    config = api_app.load_config()
    return delete_chat_session(config, session_id)
