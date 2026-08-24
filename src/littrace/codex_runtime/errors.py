"""Structured error vocabulary for the codex-harness transport layer.

Round 4 — codex-harness full alignment.

Before this module the only failure signal was ``AppServerError``
subclassing ``RuntimeError`` with a free-form message string. The
service and gateway layers could not programmatically distinguish
"context window exceeded" from "session budget exceeded" from
"transport died"; they fell back to substring matching on the
message, which is fragile and breaks every time the upstream copy
shifts by a word.

codex-harness exposes a 9-value enum called ``codexErrorInfo``
that tags every transport-level failure with a stable code plus
optional structured ``additionalDetails``. Mirror that here so
the client / service / gateway / MCP stack can ``except`` on a
class instead of grepping a string.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class CodexErrorCode(StrEnum):
    """Mirror of codex-harness ``codexErrorInfo`` enum.

    Each value is a stable string. New values may be added but
    existing values must not be renamed — clients may dispatch on
    the string form.
    """

    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"
    SESSION_BUDGET_EXCEEDED = "session_budget_exceeded"
    USAGE_LIMIT_EXCEEDED = "usage_limit_exceeded"
    ACTIVE_TURN_NOT_STEERABLE = "active_turn_not_steerable"
    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    SANDBOX_ERROR = "sandbox_error"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    OTHER = "other"


class AppServerError(RuntimeError):
    """Base failure raised by the App Server transport.

    ``error_code`` is always populated (defaults to ``OTHER``) so
    callers never have to handle the bare ``RuntimeError`` case.
    ``additional_details`` is a free-form dict for server-supplied
    context — most useful on ``BAD_REQUEST`` / ``SANDBOX_ERROR``
    where the server returns a structured payload.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: CodexErrorCode = CodexErrorCode.OTHER,
        additional_details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.additional_details: dict[str, Any] = additional_details or {}


class AppServerProtocolError(AppServerError):
    """Malformed wire data or a JSON-RPC error response.

    Always carries ``BAD_REQUEST`` because the server did not give
    us a structured error to map against.
    """

    def __init__(
        self,
        message: str,
        *,
        additional_details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=CodexErrorCode.BAD_REQUEST,
            additional_details=additional_details,
        )


class ContextWindowExceededError(AppServerError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(
            message, error_code=CodexErrorCode.CONTEXT_WINDOW_EXCEEDED, **kw
        )


class SessionBudgetExceededError(AppServerError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(
            message, error_code=CodexErrorCode.SESSION_BUDGET_EXCEEDED, **kw
        )


class UsageLimitExceededError(AppServerError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(
            message, error_code=CodexErrorCode.USAGE_LIMIT_EXCEEDED, **kw
        )


class ActiveTurnNotSteerableError(AppServerError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(
            message, error_code=CodexErrorCode.ACTIVE_TURN_NOT_STEERABLE, **kw
        )


class BadRequestError(AppServerError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(
            message, error_code=CodexErrorCode.BAD_REQUEST, **kw
        )


class UnauthorizedError(AppServerError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(
            message, error_code=CodexErrorCode.UNAUTHORIZED, **kw
        )


class SandboxError(AppServerError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(
            message, error_code=CodexErrorCode.SANDBOX_ERROR, **kw
        )


class InternalServerError(AppServerError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(
            message, error_code=CodexErrorCode.INTERNAL_SERVER_ERROR, **kw
        )


__all__ = [
    "CodexErrorCode",
    "AppServerError",
    "AppServerProtocolError",
    "ContextWindowExceededError",
    "SessionBudgetExceededError",
    "UsageLimitExceededError",
    "ActiveTurnNotSteerableError",
    "BadRequestError",
    "UnauthorizedError",
    "SandboxError",
    "InternalServerError",
]
