"""LLM rate limiter for LitTrace.

Implements two layers of rate limiting:
1. **Concurrency limit**: max simultaneous in-flight LLM requests (asyncio.Semaphore)
2. **Rate limit**: max requests per minute (token bucket with sliding window)

Usage::

    from littrace.rate_limit import rate_limiter
    async with rate_limiter.acquire():
        response = await llm_call(...)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from collections import deque

from littrace.log import get_logger

logger = get_logger("rate_limit")


@dataclass
class RateLimitConfig:
    """Configuration for the LLM rate limiter.

    Attributes:
        max_concurrent: Max simultaneous in-flight requests. 0 = unlimited.
        max_requests_per_minute: Max requests in any 60-second window. 0 = unlimited.
    """

    max_concurrent: int = 0
    max_requests_per_minute: int = 0


class RateLimiter:
    """Async rate limiter combining concurrency control and sliding-window rate limiting.

    When ``max_concurrent`` is set, an ``asyncio.Semaphore`` limits parallel calls.
    When ``max_requests_per_minute`` is set, a sliding-window token deque enforces
    a per-minute rate cap. Requests that would exceed the rate wait until the
    oldest request in the window expires.

    Both limits are optional — set to 0 to disable.
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        cfg = config or RateLimitConfig()
        self._max_concurrent = cfg.max_concurrent
        self._max_rpm = cfg.max_requests_per_minute
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(cfg.max_concurrent) if cfg.max_concurrent > 0 else None
        )
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._total_acquired = 0
        self._total_waited = 0

    def configure(self, config: RateLimitConfig) -> None:
        """Reconfigure the limiter at runtime."""
        self._max_concurrent = config.max_concurrent
        self._max_rpm = config.max_requests_per_minute
        self._semaphore = (
            asyncio.Semaphore(config.max_concurrent) if config.max_concurrent > 0 else None
        )

    async def acquire(self) -> None:
        """Acquire a rate-limit slot. Blocks if rate limit would be exceeded."""
        # Layer 1: Concurrency control
        if self._semaphore is not None:
            await self._semaphore.acquire()

        # Layer 2: Rate control (sliding window)
        if self._max_rpm > 0:
            async with self._lock:
                now = time.monotonic()
                # Evict timestamps older than 60 seconds
                while self._timestamps and self._timestamps[0] < now - 60.0:
                    self._timestamps.popleft()

                if len(self._timestamps) >= self._max_rpm:
                    # Need to wait for the oldest request to expire
                    wait_until = self._timestamps[0] + 60.0
                    wait_seconds = max(wait_until - now, 0.01)
                    self._total_waited += 1
                    logger.info(
                        "rate_limit_wait",
                        extra={
                            "wait_seconds": round(wait_seconds, 2),
                            "current_window": len(self._timestamps),
                            "max_rpm": self._max_rpm,
                        },
                    )
                    await asyncio.sleep(wait_seconds)
                    # Evict again after sleeping
                    now = time.monotonic()
                    while self._timestamps and self._timestamps[0] < now - 60.0:
                        self._timestamps.popleft()

                self._timestamps.append(time.monotonic())

        self._total_acquired += 1

    def release(self) -> None:
        """Release a concurrency slot (if concurrency limiting is active)."""
        if self._semaphore is not None:
            self._semaphore.release()

    @property
    def total_acquired(self) -> int:
        return self._total_acquired

    @property
    def total_waited(self) -> int:
        return self._total_waited

    def snapshot(self) -> dict[str, int | str]:
        """Return current limiter state for observability."""
        return {
            "max_concurrent": self._max_concurrent,
            "max_rpm": self._max_rpm,
            "current_window": len(self._timestamps),
            "total_acquired": self._total_acquired,
            "total_waited": self._total_waited,
        }


# Singleton — importable from anywhere
rate_limiter = RateLimiter()


class RateLimitSlot:
    """Async context manager for rate-limited code blocks.

    Usage::

        async with RateLimitSlot(rate_limiter):
            response = await llm_call(...)
    """

    def __init__(self, limiter: RateLimiter) -> None:
        self._limiter = limiter

    async def __aenter__(self) -> None:
        await self._limiter.acquire()

    async def __aexit__(self, *args: object) -> None:
        self._limiter.release()
