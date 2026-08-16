"""Structured logging and performance metrics for LitTrace core paths.

Usage:
    from littrace.log import get_logger, timed, metrics

    logger = get_logger(__name__)
    logger.info("search_completed", extra={"paper_count": 15, "duration_ms": 320})

    with timed("search_papers"):
        ...

    metrics.record("llm_tokens", 1024, labels={"model": "deepseek-chat"})
"""

from __future__ import annotations

import functools
import json
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


# ── JSON Formatter ──────────────────────────────────────────────


class _JsonFormatter(logging.Formatter):
    """Single-line JSON log records for machine-parseable output."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge structured extras (everything not in standard attrs)
        standard = set(
            logging.LogRecord(
                name="",
                level=0,
                pathname="",
                lineno=0,
                msg="",
                args=None,
                exc_info=None,
            ).__dict__
        ) | {"message", "asctime", "msg"}
        for key, value in record.__dict__.items():
            if key not in standard and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_CONFIGURED = False


def _ensure_configured() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger("littrace")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Prevent double-output through root logger
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``littrace`` namespace."""
    _ensure_configured()
    if not name.startswith("littrace"):
        name = f"littrace.{name}"
    return logging.getLogger(name)


def quiet_call(
    func,
    *args,
    logger=None,
    level: str = "debug",
    op: str = "",
    default=None,
    **kwargs,
):
    """Call ``func(*args, **kwargs)`` swallowing ``Exception``.

    Centralises the 24 bare ``except Exception: pass`` swallows scattered
    across the codebase. At least one log line is emitted so a silent
    failure leaves a trail. Returns ``default`` (None) on error.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - intentional swallow
        if logger is not None:
            getattr(logger, level)(
                "quiet_call_failed",
                extra={"op": op, "error": f"{exc.__class__.__name__}: {exc}"},
            )
        return default


# ── Timed context manager ───────────────────────────────────────


@dataclass
class TimingResult:
    name: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@contextmanager
def timed(name: str, **metadata: Any):
    """Context manager that logs duration and records a metric.

    Usage::

        with timed("search_papers", source="openalex"):
            ...

    Yields a ``TimingResult`` that is populated with ``duration_ms``
    after the block exits.  The result is also logged at INFO level
    and recorded in the in-memory metrics store.
    """
    logger = get_logger("perf")
    start = time.perf_counter()
    result = TimingResult(name=name, duration_ms=0.0, metadata=dict(metadata))
    try:
        yield result
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        logger.error(
            "timed_failed",
            extra={"step": name, "duration_ms": round(elapsed, 2), **metadata},
        )
        raise
    else:
        elapsed = (time.perf_counter() - start) * 1000
        result.duration_ms = round(elapsed, 2)
        metrics.record(f"duration_ms:{name}", result.duration_ms, labels=metadata)
        logger.info(
            "timed_ok",
            extra={"step": name, "duration_ms": result.duration_ms, **metadata},
        )


def log_duration(func: Callable) -> Callable:
    """Decorator that logs function duration and records a metric."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        step = f"{func.__module__}.{func.__name__}"
        with timed(step):
            return await func(*args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        step = f"{func.__module__}.{func.__name__}"
        with timed(step):
            return func(*args, **kwargs)

    import inspect

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# ── In-memory metrics collector ────────────────────────────────


@dataclass
class _MetricEntry:
    count: int = 0
    total: float = 0.0
    min_val: float = float("inf")
    max_val: float = float("-inf")
    labels: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Thread-safe in-memory metrics store with snapshot support.

    Not a full Prometheus — intentionally lightweight, no external deps.
    Use ``snapshot()`` to get a dict suitable for JSON serialization or
    a ``/metrics`` endpoint.
    """

    def __init__(self) -> None:
        self._entries: dict[str, list[_MetricEntry]] = defaultdict(list)
        self._lock = Lock()

    def record(
        self,
        name: str,
        value: float,
        labels: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            entry = _MetricEntry(
                count=1,
                total=value,
                min_val=value,
                max_val=value,
                labels=labels or {},
            )
            self._entries[name].append(entry)

    def snapshot(self) -> dict[str, Any]:
        """Return a summary dict of all collected metrics."""
        with self._lock:
            result: dict[str, Any] = {}
            for name, entries in self._entries.items():
                if not entries:
                    continue
                count = sum(e.count for e in entries)
                total = sum(e.total for e in entries)
                result[name] = {
                    "count": count,
                    "total": round(total, 2),
                    "avg": round(total / count, 2) if count else 0.0,
                    "min": round(min(e.min_val for e in entries), 2),
                    "max": round(max(e.max_val for e in entries), 2),
                }
            return result

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()


# Singleton
metrics = MetricsCollector()


# ── Cost Tracker ────────────────────────────────────────────────


@dataclass
class CostEntry:
    """A single LLM cost record."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """Thread-safe cost tracker for LLM token usage.

    Records token usage per model and estimates cost using a price table.
    Harness checks query this tracker for budget enforcement.

    Persistence with rotation: when ``max_entries`` is set, old entries are
    pruned from memory before saving, and the previous file is rotated to
    ``<path>.1`` before writing (keeping one backup).

    Usage::

        from littrace.log import cost_tracker
        cost_tracker.record("deepseek-chat", prompt_tokens=1000, completion_tokens=500)
        total = cost_tracker.total_tokens
        cost = cost_tracker.total_cost_usd
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        *,
        max_entries: int = 0,
    ) -> None:
        self._entries: list[CostEntry] = []
        self._lock = Lock()
        self._persist_path: Path | None = Path(persist_path) if persist_path else None
        self._max_entries: int = max_entries  # 0 = unlimited
        # Price table: model -> (input_per_1k, output_per_1k) in USD
        self._price_table: dict[str, tuple[float, float]] = {
            "deepseek-chat": (0.001, 0.002),
            "deepseek-reasoner": (0.004, 0.016),
        }
        # Auto-load from disk if persist_path is set and file exists
        if self._persist_path and self._persist_path.exists():
            self.load_from_disk(self._persist_path)

    def set_persist_path(self, path: str | Path) -> None:
        """Set or update the persistence path; auto-loads existing data if file present."""
        self._persist_path = Path(path)
        if self._persist_path.exists():
            self.load_from_disk(self._persist_path)

    def set_price(self, model: str, input_per_1k: float, output_per_1k: float) -> None:
        with self._lock:
            self._price_table[model] = (input_per_1k, output_per_1k)

    def set_max_entries(self, max_entries: int) -> None:
        """Set the maximum number of entries to retain in memory and on disk.

        When the limit is exceeded, the oldest entries are pruned.
        Set to 0 for unlimited.
        """
        self._max_entries = max_entries
        if max_entries > 0:
            with self._lock:
                if len(self._entries) > max_entries:
                    self._entries = self._entries[-max_entries:]

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> CostEntry:
        """Record token usage and estimate cost."""
        input_price, output_price = self._price_table.get(model, (0.001, 0.002))
        cost = (prompt_tokens / 1000.0 * input_price) + (completion_tokens / 1000.0 * output_price)
        entry = CostEntry(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=round(cost, 6),
        )
        with self._lock:
            self._entries.append(entry)
        # Auto-persist if a persist_path is configured
        if self._persist_path:
            try:
                self.save_to_disk(self._persist_path)
            except OSError:
                pass  # best-effort persistence, don't crash on disk errors
        return entry

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(e.prompt_tokens + e.completion_tokens for e in self._entries)

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return round(sum(e.estimated_cost_usd for e in self._entries), 6)

    @property
    def prompt_tokens(self) -> int:
        with self._lock:
            return sum(e.prompt_tokens for e in self._entries)

    @property
    def completion_tokens(self) -> int:
        with self._lock:
            return sum(e.completion_tokens for e in self._entries)

    def by_model(self) -> dict[str, dict[str, Any]]:
        """Per-model cost breakdown."""
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for entry in self._entries:
                if entry.model not in result:
                    result[entry.model] = {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                    }
                r = result[entry.model]
                r["calls"] += 1
                r["prompt_tokens"] += entry.prompt_tokens
                r["completion_tokens"] += entry.completion_tokens
                r["total_tokens"] += entry.prompt_tokens + entry.completion_tokens
                r["cost_usd"] = round(r["cost_usd"] + entry.estimated_cost_usd, 6)
            return result

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "by_model": self.by_model(),
        }

    def save_to_disk(self, path: str | Path) -> None:
        """Persist cost entries to a JSON file.

        Called automatically after each record() if a persist_path is set,
        or manually by the application on shutdown.

        Implements file rotation: the previous file is moved to ``<path>.1``
        before writing, keeping one backup of the last state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # Prune old entries if max_entries is set
            entries_to_save = self._entries
            if self._max_entries > 0 and len(entries_to_save) > self._max_entries:
                entries_to_save = entries_to_save[-self._max_entries :]
            data = {
                "entries": [
                    {
                        "model": e.model,
                        "prompt_tokens": e.prompt_tokens,
                        "completion_tokens": e.completion_tokens,
                        "estimated_cost_usd": e.estimated_cost_usd,
                        "timestamp": e.timestamp,
                    }
                    for e in entries_to_save
                ],
                "price_table": {model: list(prices) for model, prices in self._price_table.items()},
            }
        # Rotate: move current file to .1 before writing new one
        backup = path.with_suffix(path.suffix + ".1")
        if path.exists():
            try:
                if backup.exists():
                    backup.unlink()
                path.rename(backup)
            except OSError:
                pass  # best-effort rotation
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load_from_disk(self, path: str | Path) -> None:
        """Load cost entries from a JSON file written by save_to_disk.

        Called automatically at startup if the file exists.
        Merges with existing in-memory entries.
        """
        path = Path(path)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        with self._lock:
            for raw in data.get("entries", []):
                self._entries.append(
                    CostEntry(
                        model=raw["model"],
                        prompt_tokens=raw["prompt_tokens"],
                        completion_tokens=raw["completion_tokens"],
                        estimated_cost_usd=raw["estimated_cost_usd"],
                        timestamp=raw.get("timestamp", time.time()),
                    )
                )
            for model, prices in data.get("price_table", {}).items():
                if isinstance(prices, list) and len(prices) == 2:
                    self._price_table[model] = (prices[0], prices[1])


# Singleton
cost_tracker = CostTracker()
