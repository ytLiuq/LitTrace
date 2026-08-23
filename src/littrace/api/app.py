from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from littrace.api import state as api_state
from littrace.api.routes.agents import router as agents_router
from littrace.api.routes.artifacts import router as artifacts_router
from littrace.api.routes.context import router as context_router
from littrace.api.routes.downloads import router as downloads_router
from littrace.api.routes.eval import router as eval_router
from littrace.api.routes.publishers import router as publishers_router
from littrace.api.routes.research import router as research_router
from littrace.api.routes.sessions import router as sessions_router
from littrace.api.routes.system import router as system_router
from littrace.codex_runtime.runtime import shutdown_runtime_managers
from littrace.config import load_config as _load_config
from littrace.download_tasks import DownloadRetryWorker, download_task_store_from_config
from littrace.downloads import make_download_retry_handler
from littrace.log import get_logger, metrics
from littrace.models import LiteratureWorkspace
from littrace.tracing import append_trace as _append_trace

logger = get_logger("api")

DOWNLOAD_RETRY_WORKER: DownloadRetryWorker | None = None


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _start_background_workers()
    try:
        yield
    finally:
        _stop_background_workers()
        await asyncio.to_thread(shutdown_runtime_managers)


app = FastAPI(title="LitTrace API", version="0.1.0", lifespan=_lifespan)
app.include_router(system_router)
app.include_router(agents_router)
app.include_router(context_router)
app.include_router(downloads_router)
app.include_router(publishers_router)
app.include_router(eval_router)
app.include_router(research_router)
app.include_router(sessions_router)
app.include_router(artifacts_router)


@app.middleware("http")
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


WORKSPACE = api_state.get_workspace()


def _start_background_workers() -> None:
    global DOWNLOAD_RETRY_WORKER
    config = load_config()
    if not config.download_retry.background_worker_enabled:
        return
    DOWNLOAD_RETRY_WORKER = DownloadRetryWorker(
        download_task_store_from_config(config),
        make_download_retry_handler(config),
        interval_seconds=config.download_retry.interval_seconds,
        batch_size=config.download_retry.batch_size,
    )
    DOWNLOAD_RETRY_WORKER.start()
    logger.info("download_retry_worker_started")


def _stop_background_workers() -> None:
    global DOWNLOAD_RETRY_WORKER
    if DOWNLOAD_RETRY_WORKER is None:
        return
    DOWNLOAD_RETRY_WORKER.stop(timeout=5.0)
    DOWNLOAD_RETRY_WORKER = None
    logger.info("download_retry_worker_stopped")


def load_config(path: str = "config.yaml"):
    return _load_config(path)


def append_trace(config, event: str, payload: dict[str, object]) -> None:
    _append_trace(config, event, payload)


def _set_workspace(workspace: LiteratureWorkspace) -> LiteratureWorkspace:
    global WORKSPACE
    WORKSPACE = api_state.set_workspace(workspace)
    return WORKSPACE
