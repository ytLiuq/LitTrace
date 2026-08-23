"""Codex App Server integration for LitTrace.

The package owns execution transport and thread identity only.  LitTrace's
Postgres workspace remains the canonical scientific state.
"""

from littrace.codex_runtime.client import (
    AppServerClient,
    AppServerError,
    AppServerProtocolError,
    AppServerTurnResult,
)
from littrace.codex_runtime.runtime import (
    CodexAppServerRuntimeManager,
    shutdown_runtime_managers,
)

__all__ = [
    "AppServerClient",
    "AppServerError",
    "AppServerProtocolError",
    "AppServerTurnResult",
    "CodexAppServerRuntimeManager",
    "shutdown_runtime_managers",
]
