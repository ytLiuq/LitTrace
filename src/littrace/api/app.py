from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from littrace.api import state as api_state
from littrace.api.backend import reset_current_session_id, set_current_session_id
from littrace.api.routes.agents import router as agents_router
from littrace.api.routes.artifacts import router as artifacts_router
from littrace.api.routes.context import router as context_router
from littrace.api.routes.downloads import router as downloads_router
from littrace.api.routes.eval import router as eval_router
from littrace.api.routes.publishers import router as publishers_router
from littrace.api.routes.research import router as research_router
from littrace.api.routes.sessions import router as sessions_router
from littrace.api.routes.system import router as system_router
from littrace.codex_runtime.errors import AppServerError, CodexErrorCode
from littrace.codex_runtime.runtime import shutdown_runtime_managers
from littrace.config import load_config as _load_config
from littrace.download_tasks import DownloadRetryWorker, download_task_store_from_config
from littrace.downloads import make_download_retry_handler
from littrace.log import get_logger, metrics
from littrace.models import LiteratureWorkspace
from littrace.tracing import append_trace as _append_trace

logger = get_logger("api")

# Round 4 P3 step 12: bump on every breaking change to the HTTP
# surface. Clients pin the value they were written against and can
# reject / migrate on a mismatch.
LITTRACE_API_VERSION = "0.4"

DOWNLOAD_RETRY_WORKER: DownloadRetryWorker | None = None
COMPACTION_WORKER: "CompactionWorker | None" = None  # type: ignore[name-defined]


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _start_background_workers()
    try:
        yield
    finally:
        _stop_background_workers()
        await asyncio.to_thread(shutdown_runtime_managers)


# OpenAPI metadata. /docs, /redoc, and /openapi.json are opt-in
# (set LITTRACE_API_DOCS_ENABLED=1) so production deployments do not
# leak the internal schema by default.
_OPENAPI_TAGS: list[dict[str, str]] = [
    {
        "name": "research",
        "description": "Workspace search, chat and export entry points.",
    },
    {
        "name": "evaluation",
        "description": "Offline eval harness (golden sets, retrieval, PDF, RAG).",
    },
    {
        "name": "downloads",
        "description": "Publisher download orchestration and retry queue.",
    },
    {
        "name": "artifacts",
        "description": "Parse, table extraction, storyline, evidence, reports.",
    },
    {
        "name": "sessions",
        "description": "Session lifecycle (delete, metrics, knowledge surface).",
    },
    {
        "name": "context",
        "description": "Workspace context (selections, citations, full-text).",
    },
    {
        "name": "publishers",
        "description": "Publisher route discovery and enrichment.",
    },
    {
        "name": "system",
        "description": "Health, metrics, RAG jobs, configuration wizard.",
    },
    {
        "name": "agents",
        "description": "Runtime components, plans, workflow, quality audits.",
    },
]


def _build_docs_urls() -> tuple[str | None, str | None, str | None]:
    """Return (docs_url, redoc_url, openapi_url) from the env var.

    Exposed so the test suite can construct a second app instance
    with docs enabled (FastAPI reads these only at construction).
    """
    enabled = os.environ.get("LITTRACE_API_DOCS_ENABLED", "").strip() == "1"
    if enabled:
        return "/docs", "/redoc", "/openapi.json"
    return None, None, None


def make_app() -> FastAPI:
    """Construct a fresh FastAPI instance with all routes wired.

    Used by ``tests/api/test_openapi.py`` to build a docs-enabled
    app on demand. The module-level ``app`` below is the production
    instance, built once at import time.
    """
    docs_url, redoc_url, openapi_url = _build_docs_urls()
    instance = FastAPI(
        title="LitTrace API",
        version="0.1.0",
        description=(
            "LitTrace — evidence-first scientific literature workflow. "
            "Every endpoint operates on a Postgres-backed session_state; "
            "the workspace at `config.storage.sessions_dir/<session_id>` is "
            "the on-disk projection."
        ),
        contact={"name": "LitTrace", "url": "https://github.com/ytLiuq/LitTrace"},
        license_info={"name": "Apache-2.0"},
        openapi_tags=_OPENAPI_TAGS,
        lifespan=_lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )
    instance.include_router(system_router)
    instance.include_router(agents_router)
    instance.include_router(context_router)
    instance.include_router(downloads_router)
    instance.include_router(publishers_router)
    instance.include_router(eval_router)
    instance.include_router(research_router)
    instance.include_router(sessions_router)
    instance.include_router(artifacts_router)

    @instance.middleware("http")
    async def bind_api_session(request, call_next):
        """Bind X-LitTrace-Session-Id to all legacy route workspace access."""
        token = set_current_session_id(
            request.headers.get("X-LitTrace-Session-Id")
            or request.query_params.get("session_id")
        )
        try:
            return await call_next(request)
        finally:
            reset_current_session_id(token)

    @instance.middleware("http")
    async def add_api_version(request, call_next):
        """Stamp every response with ``X-LitTrace-API-Version``.

        Round 4 P3 step 12: clients can pin the version they were
        written against and reject on mismatch. The header is on
        every response (success and error alike) so a 500 still
        tells the client which wire version was hit.
        """
        response = await call_next(request)
        response.headers["X-LitTrace-API-Version"] = LITTRACE_API_VERSION
        return response

    @instance.middleware("http")
    async def log_requests(request, call_next):
        """Log every HTTP request with method, path, status and duration."""
        import time as _time

        start = _time.perf_counter()
        response = await call_next(request)
        duration_ms = round((_time.perf_counter() - start) * 1000, 2)
        metrics.record(
            "http_request_ms",
            duration_ms,
            labels={
                "method": request.method,
                "path": request.url.path,
                "status": str(response.status_code),
            },
        )
        logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    # Round 7 CR pass 2 fix: the App Server transport raises
    # typed ``AppServerError`` subclasses (``UnauthorizedError``,
    # ``BadRequestError``,`` ``ActiveTurnNotSteerableError``,
    # ``ContextWindowExceededError``, etc.). Without this
    # handler FastAPI would surface them as 500 + a generic
    # message, losing the typed-error value the round 4
    # design documented. The mapping below turns every
    # CodexErrorCode into the HTTP status the operator
    # expects:
    #
    #   400 — bad_request / active_turn_not_steerable
    #   401 — unauthorized
    #   402 — session_budget_exceeded / usage_limit_exceeded
    #   500 — internal_server_error / sandbox_error / context_window_exceeded / other
    _CODEX_ERROR_HTTP_STATUS: dict[str, int] = {
        CodexErrorCode.BAD_REQUEST.value: 400,
        CodexErrorCode.ACTIVE_TURN_NOT_STEERABLE.value: 409,
        CodexErrorCode.UNAUTHORIZED.value: 401,
        CodexErrorCode.SESSION_BUDGET_EXCEEDED.value: 402,
        CodexErrorCode.USAGE_LIMIT_EXCEEDED.value: 402,
        CodexErrorCode.INTERNAL_SERVER_ERROR.value: 500,
        CodexErrorCode.SANDBOX_ERROR.value: 500,
        CodexErrorCode.CONTEXT_WINDOW_EXCEEDED.value: 500,
        CodexErrorCode.OTHER.value: 500,
    }

    @instance.exception_handler(AppServerError)
    async def _handle_app_server_error(
        _request, exc: AppServerError,
    ) -> JSONResponse:
        code = exc.error_code.value if exc.error_code else CodexErrorCode.OTHER.value
        status_code = _CODEX_ERROR_HTTP_STATUS.get(code, 500)
        logger.warning(
            "app_server_error",
            extra={
                "error_code": code,
                "status_code": status_code,
                "error_message": str(exc),
                "details": exc.additional_details,
            },
        )
        return JSONResponse(
            status_code=status_code,
            content={
                "code": code,
                "message": str(exc),
                "additional_details": exc.additional_details,
            },
        )

    return instance


app = make_app()


WORKSPACE = api_state.get_workspace()


def _start_background_workers() -> None:
    global DOWNLOAD_RETRY_WORKER
    global COMPACTION_WORKER
    config = load_config()
    if config.download_retry.background_worker_enabled:
        DOWNLOAD_RETRY_WORKER = DownloadRetryWorker(
            download_task_store_from_config(config),
            make_download_retry_handler(config),
            interval_seconds=config.download_retry.interval_seconds,
            batch_size=config.download_retry.batch_size,
        )
        DOWNLOAD_RETRY_WORKER.start()
        logger.info("download_retry_worker_started")
    if config.compaction.background_worker_enabled:
        from littrace.codex_runtime.compaction import CompactionWorker
        from littrace.codex_runtime.runtime import shared_runtime_manager

        # Reuse the existing runtime manager cache. The worker just
        # calls ``get_client`` / ``release_client`` so the App
        # Server process is started exactly once across both
        # background workers.
        manager = shared_runtime_manager(
            (
                ("codex", "app-server"),
                config.agent_runtime.startup_timeout_seconds,
                config.agent_runtime.request_timeout_seconds,
                tuple(),
            ),
            ("codex", "app-server"),
            client_options={},
        )
        COMPACTION_WORKER = CompactionWorker(
            state_store_from_config(config),
            interval_seconds=config.compaction.interval_seconds,
            batch_size=config.compaction.batch_size,
            threshold_turns=config.compaction.threshold_turns,
            threshold_tokens=config.compaction.threshold_tokens,
        )
        # Stash the manager on the worker so run_pending_compaction
        # can pick it up at the CLI / async path. (Daemon threads
        # only do the cheap scan; the actual RPC runs via
        # run_pending_compaction on the asyncio loop.)
        COMPACTION_WORKER._runtime_manager = manager  # type: ignore[attr-defined]
        COMPACTION_WORKER.start()
        logger.info("compaction_worker_started")


def _stop_background_workers() -> None:
    global DOWNLOAD_RETRY_WORKER
    global COMPACTION_WORKER
    if DOWNLOAD_RETRY_WORKER is not None:
        DOWNLOAD_RETRY_WORKER.stop(timeout=5.0)
        DOWNLOAD_RETRY_WORKER = None
        logger.info("download_retry_worker_stopped")
    if COMPACTION_WORKER is not None:
        COMPACTION_WORKER.stop(timeout=5.0)
        COMPACTION_WORKER = None
        logger.info("compaction_worker_stopped")


def load_config(path: str = "config.yaml"):
    return _load_config(path)


def append_trace(config, event: str, payload: dict[str, object]) -> None:
    _append_trace(config, event, payload)


def _set_workspace(workspace: LiteratureWorkspace) -> LiteratureWorkspace:
    global WORKSPACE
    WORKSPACE = api_state.set_workspace(workspace)
    return WORKSPACE
