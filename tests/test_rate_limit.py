"""Tests for LLM rate limiter (rate_limit.py) and rate limit harness check."""

from __future__ import annotations

import asyncio
import time

import pytest

from littrace.rate_limit import (
    RateLimiter,
    RateLimitConfig,
    RateLimitSlot,
    rate_limiter,
)
from littrace.evaluation.harnesses import (
    HarnessConfig,
    RateLimitHealthItem,
    check_rate_limit,
)


# ---------------------------------------------------------------------------
# RateLimiter config tests
# ---------------------------------------------------------------------------


class TestRateLimiterConfig:
    def test_defaults_disabled(self):
        rl = RateLimiter()
        assert rl._max_concurrent == 0
        assert rl._max_rpm == 0
        assert rl._semaphore is None

    def test_configure_concurrency_only(self):
        rl = RateLimiter()
        rl.configure(RateLimitConfig(max_concurrent=3))
        assert rl._max_concurrent == 3
        assert rl._semaphore is not None

    def test_configure_rpm_only(self):
        rl = RateLimiter()
        rl.configure(RateLimitConfig(max_requests_per_minute=60))
        assert rl._max_rpm == 60
        assert rl._semaphore is None

    def test_configure_both(self):
        rl = RateLimiter(RateLimitConfig(max_concurrent=5, max_requests_per_minute=100))
        assert rl._max_concurrent == 5
        assert rl._max_rpm == 100
        assert rl._semaphore is not None

    def test_reconfigure(self):
        rl = RateLimiter(RateLimitConfig(max_concurrent=2))
        assert rl._max_concurrent == 2
        rl.configure(RateLimitConfig(max_concurrent=10))
        assert rl._max_concurrent == 10

    def test_config_from_config_module(self):
        from littrace.config import RateLimitConfig as CfgRL

        cfg = CfgRL(max_concurrent=4, max_requests_per_minute=30)
        assert cfg.max_concurrent == 4
        assert cfg.max_requests_per_minute == 30

    def test_rate_limiter_singleton_importable(self):
        assert rate_limiter is not None
        assert isinstance(rate_limiter, RateLimiter)


# ---------------------------------------------------------------------------
# RateLimiter acquire/release tests (via asyncio.run)
# ---------------------------------------------------------------------------


def test_acquire_no_limits_is_noop():
    rl = RateLimiter()

    async def _run():
        await rl.acquire()
        rl.release()
        return rl.total_acquired

    assert asyncio.run(_run()) == 1


def test_acquire_with_concurrency():
    rl = RateLimiter(RateLimitConfig(max_concurrent=2))

    async def _run():
        await rl.acquire()
        await rl.acquire()
        try:
            await asyncio.wait_for(rl.acquire(), timeout=0.05)
            return False
        except asyncio.TimeoutError:
            return True

    assert asyncio.run(_run())


def test_release_restores_concurrency():
    rl = RateLimiter(RateLimitConfig(max_concurrent=1))

    async def _run():
        await rl.acquire()
        try:
            await asyncio.wait_for(rl.acquire(), timeout=0.05)
            return False
        except asyncio.TimeoutError:
            pass
        rl.release()
        await asyncio.wait_for(rl.acquire(), timeout=0.5)
        rl.release()
        return True

    assert asyncio.run(_run())


def test_rpm_sliding_window_blocks():
    rl = RateLimiter(RateLimitConfig(max_requests_per_minute=3))

    async def _run():
        for _ in range(3):
            await rl.acquire()
            rl.release()
        try:
            await asyncio.wait_for(rl.acquire(), timeout=0.05)
            return False
        except asyncio.TimeoutError:
            return True

    assert asyncio.run(_run())


def test_rpm_window_expires():
    """After 60 seconds, old timestamps should be evicted."""
    rl = RateLimiter(RateLimitConfig(max_requests_per_minute=2))

    async def _run():
        await rl.acquire()
        rl.release()
        await rl.acquire()
        rl.release()
        # Manually age timestamps
        rl._timestamps.clear()
        rl._timestamps.extend([time.monotonic() - 61, time.monotonic() - 60.5])
        await asyncio.wait_for(rl.acquire(), timeout=0.5)
        rl.release()
        return True

    assert asyncio.run(_run())


# ---------------------------------------------------------------------------
# RateLimitSlot tests
# ---------------------------------------------------------------------------


def test_slot_acquires_and_releases():
    rl = RateLimiter(RateLimitConfig(max_concurrent=1))

    async def _run():
        async with RateLimitSlot(rl):
            try:
                await asyncio.wait_for(rl.acquire(), timeout=0.05)
                return False
            except asyncio.TimeoutError:
                return True

    assert asyncio.run(_run())


def test_slot_no_limits():
    rl = RateLimiter()

    async def _run():
        async with RateLimitSlot(rl):
            pass
        return True

    assert asyncio.run(_run())


def test_slot_exception_still_releases():
    rl = RateLimiter(RateLimitConfig(max_concurrent=1))

    async def _run():
        with pytest.raises(ValueError, match="test"):
            async with RateLimitSlot(rl):
                raise ValueError("test")
        await asyncio.wait_for(rl.acquire(), timeout=0.5)
        rl.release()
        return True

    assert asyncio.run(_run())


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


def test_snapshot_initial():
    rl = RateLimiter()
    snap = rl.snapshot()
    assert snap["max_concurrent"] == 0
    assert snap["max_rpm"] == 0
    assert snap["current_window"] == 0
    assert snap["total_acquired"] == 0
    assert snap["total_waited"] == 0


def test_snapshot_after_acquires():
    rl = RateLimiter(RateLimitConfig(max_requests_per_minute=10))

    async def _run():
        await rl.acquire()
        rl.release()
        await rl.acquire()
        rl.release()
        snap = rl.snapshot()
        assert snap["total_acquired"] == 2
        assert snap["current_window"] == 2

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Concurrency test
# ---------------------------------------------------------------------------


def test_parallel_acquires_respect_concurrency():
    rl = RateLimiter(RateLimitConfig(max_concurrent=2))
    in_flight = 0
    max_in_flight = 0

    async def worker():
        nonlocal in_flight, max_in_flight
        async with RateLimitSlot(rl):
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1

    async def _run():
        await asyncio.gather(*[worker() for _ in range(6)])

    asyncio.run(_run())
    assert max_in_flight <= 2


def test_concurrency_all_succeed():
    """All tasks eventually complete even with tight concurrency."""
    rl = RateLimiter(RateLimitConfig(max_concurrent=1))
    results: list[int] = []

    async def worker(i: int):
        async with RateLimitSlot(rl):
            await asyncio.sleep(0.01)
            results.append(i)

    async def _run():
        await asyncio.gather(*[worker(i) for i in range(5)])

    asyncio.run(_run())
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Harness check_rate_limit tests
# ---------------------------------------------------------------------------


class TestCheckRateLimit:
    def test_not_configured_warning(self):
        item = RateLimitHealthItem(max_concurrent=0, max_rpm=0)
        report = check_rate_limit([item])
        assert any(f.severity == "warning" for f in report.findings)

    def test_configured_healthy(self):
        item = RateLimitHealthItem(
            max_concurrent=5,
            max_rpm=60,
            current_window=3,
            total_acquired=100,
            total_waited=2,
        )
        report = check_rate_limit([item])
        assert report.passed
        assert len(report.findings) == 0

    def test_high_wait_ratio_warning(self):
        item = RateLimitHealthItem(
            max_concurrent=2,
            max_rpm=10,
            total_acquired=100,
            total_waited=60,
        )
        report = check_rate_limit([item])
        assert any(f.severity == "warning" for f in report.findings)

    def test_configured_but_unused_info(self):
        item = RateLimitHealthItem(
            max_concurrent=5,
            max_rpm=60,
            total_acquired=0,
            configured_but_unused=True,
        )
        report = check_rate_limit([item])
        assert any(f.severity == "info" for f in report.findings)

    def test_empty_items_list(self):
        report = check_rate_limit([])
        assert report.passed
        assert report.item_count == 0

    def test_multiple_items_mixed(self):
        items = [
            RateLimitHealthItem(max_concurrent=0, max_rpm=0),
            RateLimitHealthItem(max_concurrent=5, max_rpm=60, total_acquired=10, total_waited=1),
            RateLimitHealthItem(max_concurrent=1, max_rpm=5, total_acquired=50, total_waited=40),
        ]
        report = check_rate_limit(items)
        assert report.item_count == 3
        warnings = [f for f in report.findings if f.severity == "warning"]
        assert len(warnings) >= 2

    def test_score_calculation(self):
        item = RateLimitHealthItem(
            max_concurrent=5,
            max_rpm=60,
            total_acquired=50,
            total_waited=5,
        )
        report = check_rate_limit([item])
        assert report.score == 1.0

    def test_check_name(self):
        item = RateLimitHealthItem()
        report = check_rate_limit([item])
        assert report.check_name == "check_rate_limit"

    def test_with_custom_config(self):
        config = HarnessConfig()
        item = RateLimitHealthItem(max_concurrent=0, max_rpm=0)
        report = check_rate_limit([item], config=config)
        assert report is not None

    def test_remediation_hint_present(self):
        item = RateLimitHealthItem(max_concurrent=0, max_rpm=0)
        report = check_rate_limit([item])
        for f in report.findings:
            if f.severity == "warning":
                assert f.remediation_hint is not None
                assert len(f.remediation_hint) > 10

    def test_registered_in_registry(self):
        from littrace.evaluation.harnesses import registry

        names = list(registry.all_checks().keys())
        assert "check_rate_limit" in names

    def test_low_wait_ratio_no_warning(self):
        item = RateLimitHealthItem(
            max_concurrent=5,
            max_rpm=60,
            total_acquired=100,
            total_waited=5,
        )
        report = check_rate_limit([item])
        assert report.passed
        assert len(report.findings) == 0

    def test_zero_acquired_no_wait_ratio_check(self):
        item = RateLimitHealthItem(
            max_concurrent=5,
            max_rpm=60,
            total_acquired=0,
            total_waited=0,
        )
        report = check_rate_limit([item])
        assert report.passed
