from __future__ import annotations

from contextvars import ContextVar

from littrace.api import state as api_state
from littrace.models import LiteratureWorkspace


_CURRENT_SESSION_ID: ContextVar[str | None] = ContextVar(
    "littrace_api_session_id", default=None
)


def set_current_session_id(session_id: str | None):
    return _CURRENT_SESSION_ID.set(session_id.strip() if session_id else None)


def reset_current_session_id(token) -> None:
    _CURRENT_SESSION_ID.reset(token)


def current_session_id() -> str | None:
    return _CURRENT_SESSION_ID.get()


class APIBackend:
    """Compatibility facade shared by the legacy API routes.

    Keeping this object outside ``api.app`` avoids a circular import while the
    routes are migrated to request-scoped session resolution.
    """

    @property
    def WORKSPACE(self) -> LiteratureWorkspace:  # noqa: N802 - compatibility API
        # Keep legacy callers that replace ``littrace.api.app.WORKSPACE``
        # working while the migrated routes use the state module underneath.
        # ``_set_workspace`` updates both references, so this does not create
        # a second source of truth in normal operation.
        from littrace.api import app as app_module

        session_id = current_session_id()
        if session_id:
            from littrace.session import load_or_create_session, load_workspace

            config = self.load_config()
            return load_workspace(load_or_create_session(config, session_id))
        return getattr(app_module, "WORKSPACE", api_state.get_workspace())

    @staticmethod
    def load_config(path: str | None = None):
        # Import lazily so legacy tests/operators can still replace
        # ``littrace.api.app.load_config`` without recreating this facade.
        from littrace.api import app as app_module

        if path is None:
            return app_module.load_config()
        return app_module.load_config(path)

    @staticmethod
    def append_trace(config, event: str, payload: dict[str, object]) -> None:
        from littrace.api import app as app_module

        app_module.append_trace(config, event, payload)

    @staticmethod
    def _set_workspace(workspace: LiteratureWorkspace) -> LiteratureWorkspace:
        from littrace.api import app as app_module

        return app_module._set_workspace(workspace)


api_app = APIBackend()
