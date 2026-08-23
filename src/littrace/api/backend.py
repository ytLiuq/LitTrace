from __future__ import annotations

from littrace.api import state as api_state
from littrace.models import LiteratureWorkspace


class APIBackend:
    """Compatibility facade shared by the legacy API routes.

    Keeping this object outside ``api.app`` avoids a circular import while the
    routes are migrated to request-scoped session resolution.
    """

    @property
    def WORKSPACE(self) -> LiteratureWorkspace:  # noqa: N802 - compatibility API
        return api_state.get_workspace()

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
