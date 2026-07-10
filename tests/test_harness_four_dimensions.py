"""Tests for the four-dimension harness: retry, hallucination, schema, cost."""

from __future__ import annotations

import asyncio
import pytest

from littrace.harnesses import (
    HarnessConfig,
    HarnessReport,
    Severity,
    registry,
    check_retry_health,
    check_hallucination_grounding,
    check_schema_compliance,
    check_cost_budget,
    RetryHealthItem,
    HallucinationCheckItem,
    SchemaCheckItem,
    CostCheckItem,
)
from littrace.retry import (
    RetryConfig,
    BackoffStrategy,
    RetryTracker,
    RetryTrace,
    RetryAttempt,
    compute_delay,
    retry_async,
    retry_tracker,
)
from littrace.log import cost_tracker, CostTracker


# =========================================================================
# Dimension 1: Retry & Fallback Health
# =========================================================================


class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert cfg.base_delay_seconds == 0.8
        assert 429 in cfg.retry_status_codes
        assert 200 not in cfg.retry_status_codes

    def test_custom_config(self):
        cfg = RetryConfig(
            max_attempts=5,
            backoff_strategy=BackoffStrategy.LINEAR,
            base_delay_seconds=0.5,
            max_delay_seconds=10.0,
        )
        assert cfg.max_attempts == 5
        assert cfg.backoff_strategy == BackoffStrategy.LINEAR
        assert cfg.base_delay_seconds == 0.5


class TestComputeDelay:
    def test_first_attempt_no_delay(self):
        cfg = RetryConfig()
        assert compute_delay(1, cfg) == 0.0

    def test_exponential(self):
        cfg = RetryConfig(
            base_delay_seconds=1.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        )
        assert compute_delay(2, cfg) == 1.0  # 2^0 * 1.0
        assert compute_delay(3, cfg) == 2.0  # 2^1 * 1.0
        assert compute_delay(4, cfg) == 4.0  # 2^2 * 1.0

    def test_linear(self):
        cfg = RetryConfig(
            base_delay_seconds=1.0,
            backoff_strategy=BackoffStrategy.LINEAR,
        )
        assert compute_delay(2, cfg) == 1.0
        assert compute_delay(3, cfg) == 2.0
        assert compute_delay(4, cfg) == 3.0

    def test_fixed(self):
        cfg = RetryConfig(
            base_delay_seconds=2.0,
            backoff_strategy=BackoffStrategy.FIXED,
        )
        assert compute_delay(2, cfg) == 2.0
        assert compute_delay(5, cfg) == 2.0

    def test_max_delay_cap(self):
        cfg = RetryConfig(
            base_delay_seconds=1.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            max_delay_seconds=5.0,
        )
        assert compute_delay(10, cfg) == 5.0  # would be 2^8=256, capped to 5

    def test_jittered_within_range(self):
        cfg = RetryConfig(
            base_delay_seconds=2.0,
            backoff_strategy=BackoffStrategy.JITTERED,
        )
        delay = compute_delay(3, cfg)  # base=2, n=2, exp=2*2=4, jitter=[2,4]
        assert 1.0 <= delay <= 4.0


class TestRetryAsync:
    def test_sync_success_no_retry(self):
        calls = [0]

        @retry_async(RetryConfig(max_attempts=3), operation="test_sync")
        def func():
            calls[0] += 1
            return "ok"

        result = func()
        assert result == "ok"
        assert calls[0] == 1

    def test_sync_retries_on_failure_then_succeeds(self):
        calls = [0]

        @retry_async(
            RetryConfig(
                max_attempts=3, backoff_strategy=BackoffStrategy.FIXED, base_delay_seconds=0.01
            ),
            retry_on=(ValueError,),
            operation="test_retry_then_ok",
        )
        def func():
            calls[0] += 1
            if calls[0] < 2:
                raise ValueError("transient")
            return "recovered"

        result = func()
        assert result == "recovered"
        assert calls[0] == 2

    def test_sync_exhausts_retries(self):
        calls = [0]

        @retry_async(
            RetryConfig(max_attempts=3, base_delay_seconds=0.01),
            retry_on=(ValueError,),
            operation="test_exhaust",
        )
        def func():
            calls[0] += 1
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            func()
        assert calls[0] == 3

    def test_sync_non_retryable_exception_not_retried(self):
        calls = [0]

        @retry_async(
            RetryConfig(max_attempts=3),
            retry_on=(ValueError,),
            operation="test_no_retry",
        )
        def func():
            calls[0] += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            func()
        assert calls[0] == 1

    def test_async_success(self):
        @retry_async(
            RetryConfig(max_attempts=3, base_delay_seconds=0.01),
            retry_on=(ConnectionError,),
            operation="test_async_ok",
        )
        async def func():
            return "async_ok"

        result = asyncio.run(func())
        assert result == "async_ok"

    def test_async_retries(self):
        calls = [0]

        @retry_async(
            RetryConfig(
                max_attempts=3, backoff_strategy=BackoffStrategy.FIXED, base_delay_seconds=0.01
            ),
            retry_on=(ConnectionError,),
            operation="test_async_retry",
        )
        async def func():
            calls[0] += 1
            if calls[0] < 2:
                raise ConnectionError("transient")
            return "recovered"

        result = asyncio.run(func())
        assert result == "recovered"
        assert calls[0] == 2


class TestRetryTracker:
    def test_record_and_snapshot(self):
        tracker = RetryTracker()
        trace = RetryTrace(operation="test_op")
        trace.attempts.append(
            RetryAttempt(attempt=1, error="err", delay_seconds=0, will_retry=True)
        )
        trace.attempts.append(
            RetryAttempt(attempt=2, error="err2", delay_seconds=0.5, will_retry=False)
        )
        trace.succeeded = False
        tracker.record_sync(trace)
        assert tracker.total_calls == 1
        assert tracker.total_retries == 1
        assert tracker.failed_calls == 1

    def test_operations_summary(self):
        tracker = RetryTracker()
        t1 = RetryTrace(operation="op_a", succeeded=True)
        t1.attempts.append(RetryAttempt(attempt=1, error="", delay_seconds=0, will_retry=False))
        tracker.record_sync(t1)

        t2 = RetryTrace(operation="op_a", succeeded=False)
        t2.attempts.append(RetryAttempt(attempt=1, error="err", delay_seconds=0, will_retry=True))
        t2.attempts.append(
            RetryAttempt(attempt=2, error="err2", delay_seconds=0.5, will_retry=False)
        )
        tracker.record_sync(t2)

        summary = tracker.operations_summary()
        assert "op_a" in summary
        assert summary["op_a"]["calls"] == 2
        assert summary["op_a"]["retries"] == 1
        assert summary["op_a"]["failures"] == 1

    def test_reset(self):
        tracker = RetryTracker()
        tracker.record_sync(RetryTrace(operation="x", succeeded=True))
        assert tracker.total_calls == 1
        tracker.reset()
        assert tracker.total_calls == 0


class TestCheckRetryHealth:
    def test_all_healthy(self):
        items = [
            RetryHealthItem(
                operation="llm_call",
                total_calls=10,
                total_retries=2,
                failed_calls=0,
                retry_rate=0.2,
                failure_rate=0.0,
            ),
        ]
        report = check_retry_health(items)
        assert report.passed
        assert len(report.errors) == 0

    def test_high_failure_rate(self):
        items = [
            RetryHealthItem(
                operation="search",
                total_calls=10,
                total_retries=5,
                failed_calls=3,
                retry_rate=0.5,
                failure_rate=0.3,
            ),
        ]
        report = check_retry_health(items)
        assert not report.passed
        assert any("failure rate" in e for e in report.errors)

    def test_high_retry_rate_warning(self):
        items = [
            RetryHealthItem(
                operation="download",
                total_calls=10,
                total_retries=7,
                failed_calls=0,
                retry_rate=0.7,
                failure_rate=0.0,
            ),
        ]
        report = check_retry_health(items)
        assert report.passed  # warning only, not error
        assert any("retry rate" in w for w in report.warnings)

    def test_failures_without_retries(self):
        items = [
            RetryHealthItem(
                operation="fetch",
                total_calls=5,
                total_retries=0,
                failed_calls=2,
                retry_rate=0.0,
                failure_rate=0.4,
            ),
        ]
        report = check_retry_health(items)
        # failure_rate 0.4 > 0.2 threshold -> error
        assert not report.passed
        # Also has the "failures without retries" warning
        assert any("0 retries" in w for w in report.warnings)

    def test_empty_items(self):
        report = check_retry_health([])
        assert report.passed
        assert report.item_count == 0


# =========================================================================
# Dimension 2: Hallucination / Grounding
# =========================================================================


class TestCheckHallucinationGrounding:
    def test_all_grounded(self):
        items = [
            HallucinationCheckItem(
                text="论文 p1 表明 sensitivity 提升至 0.9",
                checked_sentence_count=1,
                unsupported_sentence_count=0,
                unsupported_sentences=[],
                source="chat",
            ),
        ]
        report = check_hallucination_grounding(items)
        assert report.passed
        assert len(report.findings) == 0

    def test_few_unsupported_warning(self):
        items = [
            HallucinationCheckItem(
                text="some text",
                checked_sentence_count=5,
                unsupported_sentence_count=1,
                unsupported_sentences=["unsupported claim here"],
                source="chat",
            ),
        ]
        report = check_hallucination_grounding(items)
        assert report.passed  # only 1 unsupported -> WARNING, not ERROR
        assert any("1 unsupported" in w for w in report.warnings)

    def test_many_unsupported_error(self):
        items = [
            HallucinationCheckItem(
                text="some text",
                checked_sentence_count=5,
                unsupported_sentence_count=4,
                unsupported_sentences=["s1", "s2", "s3", "s4"],
                source="chat",
            ),
        ]
        report = check_hallucination_grounding(items)
        assert not report.passed  # > 2 unsupported -> ERROR
        assert any("4 unsupported" in e for e in report.errors)

    def test_no_checked_sentences_warning(self):
        items = [
            HallucinationCheckItem(
                text="text with content",
                checked_sentence_count=0,
                unsupported_sentence_count=0,
                unsupported_sentences=[],
                source="chat",
            ),
        ]
        report = check_hallucination_grounding(items)
        assert report.passed
        assert any("no claim-bearing sentences" in w for w in report.warnings)

    def test_empty_text_no_warning(self):
        items = [
            HallucinationCheckItem(
                text="",
                checked_sentence_count=0,
                unsupported_sentence_count=0,
                unsupported_sentences=[],
                source="chat",
            ),
        ]
        report = check_hallucination_grounding(items)
        assert report.passed
        assert len(report.warnings) == 0


# =========================================================================
# Dimension 3: Schema Compliance
# =========================================================================


class TestCheckSchemaCompliance:
    def test_all_valid(self):
        items = [
            SchemaCheckItem(
                source="tables.py",
                total_items=10,
                valid_items=10,
                invalid_items=[],
                schema_name="PerformanceCell",
            ),
        ]
        report = check_schema_compliance(items)
        assert report.passed
        assert len(report.findings) == 0

    def test_some_invalid_strict(self):
        items = [
            SchemaCheckItem(
                source="tables.py",
                total_items=10,
                valid_items=7,
                invalid_items=["Item 0 (value): Input should be a valid number"],
                schema_name="PerformanceCell",
            ),
        ]
        report = check_schema_compliance(items)
        assert not report.passed  # strict mode -> ERROR
        assert any("Schema violation" in e for e in report.errors)

    def test_some_invalid_non_strict(self):
        config = HarnessConfig(schema_strict=False)
        items = [
            SchemaCheckItem(
                source="tables.py",
                total_items=10,
                valid_items=7,
                invalid_items=["Item 0 (value): error"],
                schema_name="PerformanceCell",
            ),
        ]
        report = check_schema_compliance(items, config)
        assert report.passed  # non-strict -> WARNING only
        assert any("Schema violation" in w for w in report.warnings)

    def test_all_invalid(self):
        items = [
            SchemaCheckItem(
                source="tables.py",
                total_items=5,
                valid_items=0,
                invalid_items=["err1", "err2", "err3", "err4", "err5"],
                schema_name="PerformanceCell",
            ),
        ]
        report = check_schema_compliance(items)
        assert not report.passed
        assert any("All 5 items failed" in e for e in report.errors)

    def test_many_errors_capped(self):
        items = [
            SchemaCheckItem(
                source="tables.py",
                total_items=20,
                valid_items=5,
                invalid_items=[f"err_{i}" for i in range(15)],
                schema_name="PerformanceCell",
            ),
        ]
        report = check_schema_compliance(items)
        # Should cap at 10 findings + 1 "additional" warning
        error_findings = [f for f in report.findings if f.severity == Severity.ERROR]
        assert len(error_findings) <= 11  # 10 + the "all failed" error
        assert any("additional schema violations suppressed" in w for w in report.warnings)

    def test_empty_items(self):
        report = check_schema_compliance([])
        assert report.passed


# =========================================================================
# Dimension 4: Cost Budget
# =========================================================================


class TestCheckCostBudget:
    def test_no_budget_set(self):
        items = [
            CostCheckItem(
                total_tokens=10000,
                total_cost_usd=0.05,
                max_total_tokens=0,  # unlimited
            ),
        ]
        report = check_cost_budget(items)
        assert report.passed
        assert len(report.findings) == 0

    def test_budget_exceeded(self):
        items = [
            CostCheckItem(
                total_tokens=10000,
                total_cost_usd=0.05,
                max_total_tokens=8000,
            ),
        ]
        report = check_cost_budget(items)
        assert not report.passed
        assert any("budget exhausted" in e for e in report.errors)

    def test_budget_warning_threshold(self):
        items = [
            CostCheckItem(
                total_tokens=8000,
                total_cost_usd=0.04,
                max_total_tokens=10000,
                budget_warning_threshold=0.8,
            ),
        ]
        report = check_cost_budget(items)
        assert report.passed  # warning only
        assert any("approaching limit" in w for w in report.warnings)

    def test_under_threshold_no_findings(self):
        items = [
            CostCheckItem(
                total_tokens=1000,
                total_cost_usd=0.005,
                max_total_tokens=10000,
                budget_warning_threshold=0.8,
            ),
        ]
        report = check_cost_budget(items)
        assert report.passed
        assert len(report.findings) == 0

    def test_high_avg_tokens_per_call(self):
        items = [
            CostCheckItem(
                total_tokens=200000,
                total_cost_usd=0.5,
                max_total_tokens=0,  # no budget limit
                by_model={
                    "deepseek-chat": {
                        "calls": 1,
                        "total_tokens": 200000,
                    }
                },
            ),
        ]
        report = check_cost_budget(items)
        assert report.passed
        assert any("high avg tokens/call" in w for w in report.warnings)


class TestCostTracker:
    def test_record_and_totals(self):
        tracker = CostTracker()
        tracker.record("deepseek-chat", prompt_tokens=1000, completion_tokens=500)
        tracker.record("deepseek-chat", prompt_tokens=2000, completion_tokens=1000)
        assert tracker.total_tokens == 4500
        assert tracker.prompt_tokens == 3000
        assert tracker.completion_tokens == 1500
        assert tracker.total_cost_usd > 0

    def test_by_model(self):
        tracker = CostTracker()
        tracker.record("deepseek-chat", prompt_tokens=1000, completion_tokens=500)
        tracker.record("deepseek-reasoner", prompt_tokens=500, completion_tokens=200)
        by_model = tracker.by_model()
        assert "deepseek-chat" in by_model
        assert "deepseek-reasoner" in by_model
        assert by_model["deepseek-chat"]["calls"] == 1
        assert by_model["deepseek-chat"]["total_tokens"] == 1500

    def test_reset(self):
        tracker = CostTracker()
        tracker.record("deepseek-chat", prompt_tokens=100, completion_tokens=50)
        assert tracker.total_tokens == 150
        tracker.reset()
        assert tracker.total_tokens == 0

    def test_snapshot(self):
        tracker = CostTracker()
        tracker.record("deepseek-chat", prompt_tokens=1000, completion_tokens=500)
        snap = tracker.snapshot()
        assert snap["total_tokens"] == 1500
        assert "by_model" in snap
        assert snap["by_model"]["deepseek-chat"]["calls"] == 1

    def test_custom_price(self):
        tracker = CostTracker()
        tracker.set_price("custom-model", input_per_1k=0.01, output_per_1k=0.03)
        tracker.record("custom-model", prompt_tokens=1000, completion_tokens=1000)
        # cost = 1000/1000 * 0.01 + 1000/1000 * 0.03 = 0.04
        assert abs(tracker.total_cost_usd - 0.04) < 1e-6


# =========================================================================
# Integration: HarnessConfig from_littrace_config
# =========================================================================


class TestHarnessConfigFromLitTraceConfig:
    def test_reads_all_dimensions(self):
        from littrace.config import (
            LitTraceConfig,
            HarnessThresholdConfig,
            RetryPolicyConfig,
            CostBudgetConfig,
            SchemaValidationConfig,
        )

        config = LitTraceConfig(
            harness=HarnessThresholdConfig(
                performance_confidence=0.8,
                artifact_confidence=0.7,
                storyline_confidence=0.75,
                storyline_chain_min_evidence=5,
            ),
            retry=RetryPolicyConfig(
                max_retry_rate=0.3,
                max_failure_rate=0.1,
            ),
            cost_budget=CostBudgetConfig(
                budget_warning_threshold=0.9,
            ),
            schema_validation=SchemaValidationConfig(
                strict=False,
                enabled=False,
            ),
        )
        harness_config = HarnessConfig.from_littrace_config(config)
        assert harness_config.performance_confidence_threshold == 0.8
        assert harness_config.max_retry_rate == 0.3
        assert harness_config.max_failure_rate == 0.1
        assert harness_config.budget_warning_threshold == 0.9
        assert harness_config.schema_strict is False
        assert harness_config.schema_enabled is False

    def test_defaults_when_no_config_sections(self):
        harness_config = HarnessConfig.from_littrace_config(object())
        assert harness_config.max_retry_rate == 0.5
        assert harness_config.max_failure_rate == 0.2
        assert harness_config.schema_strict is True


# =========================================================================
# Registry: all 8 checks registered
# =========================================================================


class TestRegistryCompleteness:
    def test_all_eight_checks_registered(self):
        names = registry.list_checks()
        expected = {
            "check_citations",
            "check_performance_cells",
            "check_structured_artifacts",
            "check_storyline_claims",
            "check_retry_health",
            "check_hallucination_grounding",
            "check_schema_compliance",
            "check_cost_budget",
        }
        assert expected.issubset(set(names))

    def test_four_dimension_groups(self):
        assert "check_retry_health" in registry.checks_in_group("reliability")
        assert "check_hallucination_grounding" in registry.checks_in_group("grounding")
        assert "check_schema_compliance" in registry.checks_in_group("schema")
        assert "check_cost_budget" in registry.checks_in_group("cost")


# =========================================================================
# Schema validation in tables.py
# =========================================================================


class TestTablesSchemaValidation:
    def test_parse_llm_cells_valid(self):
        from littrace.tables import _parse_llm_cells

        raw_cells = [
            {
                "metric": "sensitivity",
                "value": 0.95,
                "unit": "%",
                "section": "results",
                "snippet": "sensitivity was 0.95",
            },
            {
                "metric": "lod",
                "value": 0.1,
                "unit": "ppm",
                "section": "results",
                "snippet": "LOD of 0.1 ppm",
            },
        ]
        cells, errors = _parse_llm_cells("p1", raw_cells, {"sections": []})
        assert len(cells) == 2
        assert len(errors) == 0

    def test_parse_llm_cells_invalid_value(self):
        from littrace.tables import _parse_llm_cells

        raw_cells = [
            {"metric": "sensitivity", "value": "not_a_number", "unit": "%", "section": "results"},
        ]
        cells, errors = _parse_llm_cells("p1", raw_cells, {"sections": []})
        assert len(cells) == 0
        assert len(errors) == 1
        assert "value" in errors[0].lower()

    def test_parse_llm_cells_missing_metric(self):
        from littrace.tables import _parse_llm_cells

        raw_cells = [
            {"value": 0.9, "unit": "%"},  # no metric
        ]
        cells, errors = _parse_llm_cells("p1", raw_cells, {"sections": []})
        assert len(cells) == 0
        assert len(errors) == 1

    def test_parse_llm_cells_mixed_valid_invalid(self):
        from littrace.tables import _parse_llm_cells

        raw_cells = [
            {
                "metric": "sensitivity",
                "value": 0.95,
                "unit": "%",
                "section": "results",
                "snippet": "ok",
            },
            {"metric": "lod", "value": "bad", "section": "results"},
            {
                "metric": "accuracy",
                "value": 0.88,
                "unit": "%",
                "section": "results",
                "snippet": "acc=0.88",
            },
        ]
        cells, errors = _parse_llm_cells("p1", raw_cells, {"sections": []})
        assert len(cells) == 2
        assert len(errors) == 1

    def test_parse_llm_cells_non_dict_item(self):
        from littrace.tables import _parse_llm_cells

        raw_cells = [
            "not a dict",
            {"metric": "sensitivity", "value": 0.9, "section": "results"},
        ]
        cells, errors = _parse_llm_cells("p1", raw_cells, {"sections": []})
        assert len(cells) == 1
        assert len(errors) == 1
        assert "expected object" in errors[0]


# =========================================================================
# LLM cost budget pre-check
# =========================================================================


class TestLLMBudgetCheck:
    def test_budget_exceeded_blocks_llm(self):
        from littrace.llm import _check_budget
        from littrace.config import LitTraceConfig, CostBudgetConfig

        # Set up a budget that's already exceeded
        cost_tracker.reset()
        cost_tracker.record("deepseek-chat", prompt_tokens=5000, completion_tokens=5000)

        config = LitTraceConfig(
            cost_budget=CostBudgetConfig(max_total_tokens=8000),
        )
        error = _check_budget(config)
        assert error is not None
        assert "budget exhausted" in error

    def test_no_budget_allows_llm(self):
        from littrace.llm import _check_budget
        from littrace.config import LitTraceConfig

        cost_tracker.reset()
        config = LitTraceConfig()  # default: max_total_tokens=0 (unlimited)
        error = _check_budget(config)
        assert error is None

    def test_budget_not_yet_exceeded(self):
        from littrace.llm import _check_budget
        from littrace.config import LitTraceConfig, CostBudgetConfig

        cost_tracker.reset()
        cost_tracker.record("deepseek-chat", prompt_tokens=1000, completion_tokens=500)

        config = LitTraceConfig(
            cost_budget=CostBudgetConfig(max_total_tokens=10000),
        )
        error = _check_budget(config)
        assert error is None
