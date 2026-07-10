"""Unified retry abstraction for LitTrace.

Replaces the 4 scattered hand-written retry loops in llm.py, search.py,
full_text.py, and login_flow.py with a single configurable abstraction.

Usage::

    from littrace.retry import retry_async, RetryConfig

    config = RetryConfig(max_attempts=5, backoff_strategy="exponential")

    @retry_async(config, retry_on=(httpx.HTTPStatusError, httpx.TimeoutException))
    async def fetch(url: str) -> httpx.Response:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import random
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, TypeVar

from littrace.log import get_logger

logger = get_logger("retry")

T = TypeVar("T")

DEFAULT_RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class BackoffStrategy(StrEnum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIXED = "fixed"
    JITTERED = "jittered"


@dataclass
class RetryConfig:
    """Configuration for retry behaviour.

    Attributes:
        max_attempts: Maximum number of attempts (including the first call).
        backoff_strategy: Strategy for computing delay between attempts.
        base_delay_seconds: Base delay for backoff calculation.
        max_delay_seconds: Maximum delay between retries (caps exponential growth).
        retry_status_codes: HTTP status codes that should trigger a retry.
        retry_on: Exception types that should trigger a retry.
    """

    max_attempts: int = 3
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    base_delay_seconds: float = 0.8
    max_delay_seconds: float = 30.0
    retry_status_codes: frozenset[int] = DEFAULT_RETRY_STATUS_CODES
    retry_on: tuple[type[Exception], ...] = (
        Exception,  # Broad by default; callers narrow via retry_on=
    )


@dataclass
class RetryAttempt:
    """Record of a single retry attempt for observability."""

    attempt: int
    error: str
    delay_seconds: float
    will_retry: bool


@dataclass
class RetryTrace:
    """Accumulated trace of all retry attempts for a single logical call.

    Stored by the RetryTracker so harness checks can inspect retry health.
    """

    operation: str
    attempts: list[RetryAttempt] = field(default_factory=list)
    succeeded: bool = False
    total_delay_seconds: float = 0.0

    @property
    def retry_count(self) -> int:
        """Number of retries (not counting the first attempt)."""
        if not self.attempts:
            return 0
        return max(len(self.attempts) - 1, 0)

    @property
    def failed(self) -> bool:
        return not self.succeeded and len(self.attempts) >= 1


class RetryTracker:
    """Thread-safe in-memory tracker for retry attempts across the system.

    Harness checks query this tracker to assess retry health.

    Supports optional disk persistence: when ``persist_path`` is set,
    traces are saved after each ``record()`` call and loaded on startup.
    Uses the same max_entries rotation pattern as CostTracker.
    """

    def __init__(
        self,
        persist_path: str | None = None,
        *,
        max_entries: int = 0,
    ) -> None:
        self._traces: list[RetryTrace] = []
        self._lock = asyncio.Lock()
        self._persist_path: Path | None = Path(persist_path) if persist_path else None
        self._max_entries: int = max_entries  # 0 = unlimited
        # Auto-load from disk if persist_path is set and file exists
        if self._persist_path and self._persist_path.exists():
            self.load_from_disk(self._persist_path)

    def set_persist_path(self, path: str | Path) -> None:
        """Set or update the persistence path; auto-loads existing data if file present."""
        self._persist_path = Path(path)
        if self._persist_path.exists():
            self.load_from_disk(self._persist_path)

    def set_max_entries(self, max_entries: int) -> None:
        """Set the maximum number of traces to retain. 0 = unlimited."""
        self._max_entries = max_entries
        if max_entries > 0:
            if len(self._traces) > max_entries:
                self._traces = self._traces[-max_entries:]

    async def record(self, trace: RetryTrace) -> None:
        async with self._lock:
            self._traces.append(trace)
            if self._max_entries > 0 and len(self._traces) > self._max_entries:
                self._traces = self._traces[-self._max_entries :]
        if self._persist_path:
            try:
                self.save_to_disk(self._persist_path)
            except OSError:
                pass  # best-effort persistence

    def record_sync(self, trace: RetryTrace) -> None:
        self._traces.append(trace)
        if self._max_entries > 0 and len(self._traces) > self._max_entries:
            self._traces = self._traces[-self._max_entries :]
        if self._persist_path:
            try:
                self.save_to_disk(self._persist_path)
            except OSError:
                pass

    def snapshot(self) -> list[RetryTrace]:
        return list(self._traces)

    def reset(self) -> None:
        self._traces.clear()

    @property
    def total_calls(self) -> int:
        return len(self._traces)

    @property
    def total_retries(self) -> int:
        return sum(t.retry_count for t in self._traces)

    @property
    def failed_calls(self) -> int:
        return sum(1 for t in self._traces if t.failed)

    def operations_summary(self) -> dict[str, dict[str, Any]]:
        """Per-operation retry statistics."""
        summary: dict[str, list[RetryTrace]] = {}
        for t in self._traces:
            summary.setdefault(t.operation, []).append(t)
        result: dict[str, dict[str, Any]] = {}
        for op, traces in summary.items():
            total = len(traces)
            retries = sum(t.retry_count for t in traces)
            failures = sum(1 for t in traces if t.failed)
            result[op] = {
                "calls": total,
                "retries": retries,
                "failures": failures,
                "retry_rate": round(retries / total, 3) if total else 0.0,
                "failure_rate": round(failures / total, 3) if total else 0.0,
            }
        return result

    def save_to_disk(self, path: str | Path) -> None:
        """Persist retry traces to a JSON file.

        Called automatically after each record() if a persist_path is set.
        Implements file rotation: previous file is moved to ``<path>.1``.
        """
        import json as _json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        traces_to_save = self._traces
        if self._max_entries > 0 and len(traces_to_save) > self._max_entries:
            traces_to_save = traces_to_save[-self._max_entries :]
        data = {
            "traces": [
                {
                    "operation": t.operation,
                    "succeeded": t.succeeded,
                    "total_delay_seconds": t.total_delay_seconds,
                    "attempts": [
                        {
                            "attempt": a.attempt,
                            "error": a.error,
                            "delay_seconds": a.delay_seconds,
                            "will_retry": a.will_retry,
                        }
                        for a in t.attempts
                    ],
                }
                for t in traces_to_save
            ],
        }
        # Rotate: move current file to .1 before writing
        backup = path.with_suffix(path.suffix + ".1")
        if path.exists():
            try:
                if backup.exists():
                    backup.unlink()
                path.rename(backup)
            except OSError:
                pass
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False)

    def load_from_disk(self, path: str | Path) -> None:
        """Load retry traces from a JSON file written by save_to_disk."""
        import json as _json

        path = Path(path)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except (_json.JSONDecodeError, OSError):
            return
        for raw in data.get("traces", []):
            trace = RetryTrace(operation=raw.get("operation", "unknown"))
            trace.succeeded = raw.get("succeeded", False)
            trace.total_delay_seconds = raw.get("total_delay_seconds", 0.0)
            for a in raw.get("attempts", []):
                trace.attempts.append(
                    RetryAttempt(
                        attempt=a.get("attempt", 0),
                        error=a.get("error", ""),
                        delay_seconds=a.get("delay_seconds", 0.0),
                        will_retry=a.get("will_retry", False),
                    )
                )
            self._traces.append(trace)


# Singleton tracker — importable from anywhere
retry_tracker = RetryTracker()


def compute_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Compute the delay for a given attempt number (1-indexed).

    The first attempt (attempt=1) has no delay; subsequent attempts use
    the configured backoff strategy.
    """
    if attempt <= 1:
        return 0.0
    n = attempt - 1  # n=1 for first retry, n=2 for second, etc.
    strategy = config.backoff_strategy
    base = config.base_delay_seconds

    if strategy == BackoffStrategy.FIXED:
        delay = base
    elif strategy == BackoffStrategy.LINEAR:
        delay = base * n
    elif strategy == BackoffStrategy.EXPONENTIAL:
        delay = base * (2 ** (n - 1))
    elif strategy == BackoffStrategy.JITTERED:
        delay = base * (2 ** (n - 1))
        delay = delay / 2 + random.uniform(0, delay / 2)
    else:
        delay = base * n  # fallback to linear

    return min(delay, config.max_delay_seconds)


def _should_retry(
    exc: Exception,
    config: RetryConfig,
    response_status: int | None = None,
) -> bool:
    """Determine if an exception should trigger a retry."""
    # Check status code first (for HTTP responses)
    if response_status is not None and response_status in config.retry_status_codes:
        return True
    # Check exception types
    for exc_type in config.retry_on:
        if isinstance(exc, exc_type):
            return True
    return False


def retry_async(
    config: RetryConfig | None = None,
    *,
    operation: str = "",
    retry_on: tuple[type[Exception], ...] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Callable:
    """Decorator that wraps an async function with retry logic.

    Args:
        config: Retry configuration. Defaults to RetryConfig().
        operation: Human-readable name for observability (defaults to func.__qualname__).
        retry_on: Override exception types to retry on (overrides config.retry_on).
        on_retry: Optional callback invoked before each retry sleep.
            Receives (attempt, exception, delay_seconds). Use for diagnostics
            recording (e.g. appending to diagnostics.errors).
    """
    cfg = config or RetryConfig()
    if retry_on is not None:
        cfg.retry_on = retry_on

    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__qualname__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = RetryTrace(operation=op_name)
            last_exc: Exception | None = None

            for attempt in range(1, cfg.max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    trace.succeeded = True
                    await retry_tracker.record(trace)
                    return result
                except Exception as exc:
                    last_exc = exc
                    should_retry = _should_retry(exc, cfg)
                    will_retry = should_retry and attempt < cfg.max_attempts
                    delay = compute_delay(attempt + 1, cfg) if will_retry else 0.0

                    trace.attempts.append(
                        RetryAttempt(
                            attempt=attempt,
                            error=f"{exc.__class__.__name__}: {exc}",
                            delay_seconds=delay,
                            will_retry=will_retry,
                        )
                    )
                    trace.total_delay_seconds += delay

                    if not should_retry:
                        # Non-retryable exception — raise immediately
                        await retry_tracker.record(trace)
                        raise
                    if attempt < cfg.max_attempts:
                        if on_retry:
                            on_retry(attempt, exc, delay)
                        logger.warning(
                            "retry_attempt",
                            extra={
                                "operation": op_name,
                                "attempt": attempt,
                                "max_attempts": cfg.max_attempts,
                                "delay_seconds": delay,
                                "error": f"{exc.__class__.__name__}: {exc}",
                            },
                        )
                        if delay > 0:
                            await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "retry_exhausted",
                            extra={
                                "operation": op_name,
                                "attempts": attempt,
                                "max_attempts": cfg.max_attempts,
                                "error": f"{exc.__class__.__name__}: {exc}",
                            },
                        )

            await retry_tracker.record(trace)
            assert last_exc is not None
            raise last_exc

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = RetryTrace(operation=op_name)
            last_exc: Exception | None = None

            for attempt in range(1, cfg.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    trace.succeeded = True
                    retry_tracker.record_sync(trace)
                    return result
                except Exception as exc:
                    last_exc = exc
                    should_retry = _should_retry(exc, cfg)
                    will_retry = should_retry and attempt < cfg.max_attempts
                    delay = compute_delay(attempt + 1, cfg) if will_retry else 0.0

                    trace.attempts.append(
                        RetryAttempt(
                            attempt=attempt,
                            error=f"{exc.__class__.__name__}: {exc}",
                            delay_seconds=delay,
                            will_retry=will_retry,
                        )
                    )
                    trace.total_delay_seconds += delay

                    if not should_retry:
                        retry_tracker.record_sync(trace)
                        raise
                    if attempt < cfg.max_attempts:
                        if on_retry:
                            on_retry(attempt, exc, delay)
                        logger.warning(
                            "retry_attempt",
                            extra={
                                "operation": op_name,
                                "attempt": attempt,
                                "max_attempts": cfg.max_attempts,
                                "delay_seconds": delay,
                                "error": f"{exc.__class__.__name__}: {exc}",
                            },
                        )
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        logger.error(
                            "retry_exhausted",
                            extra={
                                "operation": op_name,
                                "attempts": attempt,
                                "max_attempts": cfg.max_attempts,
                                "error": f"{exc.__class__.__name__}: {exc}",
                            },
                        )

            retry_tracker.record_sync(trace)
            assert last_exc is not None
            raise last_exc

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
