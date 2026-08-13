"""Tests for the Harness Engine — registry, config, reports, and orchestration."""

import pytest

from littrace.config import HarnessThresholdConfig, LitTraceConfig
from littrace.evaluation.harnesses import (
    HarnessCheck,
    HarnessConfig,
    HarnessEngine,
    HarnessFinding,
    HarnessRegistry,
    HarnessReport,
    Severity,
    check_citations,
    check_performance_cells,
    check_storyline_claims,
    check_structured_artifacts,
    registry,
)
from littrace.models import (
    CitationRecord,
    EvidenceSpan,
    LinkStatus,
    PerformanceCell,
    StorylineClaim,
    StructuredArtifact,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cell(
    paper_id="p1",
    metric="sensitivity",
    value=2.3,
    confidence=0.9,
    page=1,
    snippet="test snippet",
    higher_is_better=True,
):
    return PerformanceCell(
        paper_id=paper_id,
        metric=metric,
        value=value,
        higher_is_better=higher_is_better,
        evidence=EvidenceSpan(
            paper_id=paper_id,
            snippet=snippet,
            page=page,
            confidence=confidence,
        ),
    )


def _make_citation(paper_id="p1", link_status=LinkStatus.VERIFIED_200, text="Author et al."):
    return CitationRecord(
        paper_id=paper_id,
        citation_text=text,
        access_url="https://example.com/paper.pdf",
        link_status=link_status,
    )


def _make_artifact(
    paper_id="p1",
    artifact_type="table",
    text="Table 1: Performance data",
    confidence=0.8,
    page=1,
):
    return StructuredArtifact(
        paper_id=paper_id,
        artifact_type=artifact_type,
        text=text,
        confidence=confidence,
        evidence=EvidenceSpan(paper_id=paper_id, page=page, snippet=text),
    )


def _make_claim(
    claim_type="prior_solution",
    claim="Paper A solved X",
    confidence=0.85,
    evidence_count=2,
):
    return StorylineClaim(
        claim_type=claim_type,
        claim=claim,
        confidence=confidence,
        evidence=[
            EvidenceSpan(paper_id=f"p{i}", snippet=f"evidence {i}", confidence=0.8)
            for i in range(evidence_count)
        ],
    )


# ---------------------------------------------------------------------------
# check_citations
# ---------------------------------------------------------------------------


class TestCheckCitations:
    def test_passes_with_verified_links(self):
        records = [_make_citation(link_status=LinkStatus.VERIFIED_200)]
        report = check_citations(records)
        assert report.passed
        assert report.score == 1.0
        assert report.check_name == "check_citations"
        assert report.item_count == 1

    def test_fails_with_unchecked_link(self):
        records = [_make_citation(link_status=LinkStatus.UNCHECKED)]
        report = check_citations(records)
        assert not report.passed
        assert "not verified" in report.errors[0]

    def test_fails_with_failed_link(self):
        records = [_make_citation(link_status=LinkStatus.FAILED)]
        report = check_citations(records)
        assert not report.passed

    def test_fails_with_empty_citation_text(self):
        records = [_make_citation(text="")]
        report = check_citations(records)
        assert not report.passed
        assert "missing citation text" in report.errors[0]

    def test_empty_list_passes(self):
        report = check_citations([])
        assert report.passed
        assert report.score == 1.0

    def test_findings_have_remediation_hints(self):
        records = [_make_citation(link_status=LinkStatus.FAILED)]
        report = check_citations(records)
        assert report.findings[0].remediation_hint is not None

# ---------------------------------------------------------------------------
# check_performance_cells
# ---------------------------------------------------------------------------


class TestCheckPerformanceCells:
    def test_passes_with_good_cells(self):
        cells = [_make_cell()]
        report = check_performance_cells(cells)
        assert report.passed
        assert report.score == 1.0

    def test_fails_without_evidence(self):
        cell = PerformanceCell(
            paper_id="p1",
            metric="sensitivity",
            value=2.3,
            higher_is_better=True,
            evidence=EvidenceSpan(paper_id="p1", snippet=None, page=None, confidence=0.9),
        )
        report = check_performance_cells([cell])
        assert not report.passed
        assert "lacks traceable evidence" in report.errors[0]

    def test_warns_on_low_confidence(self):
        cell = _make_cell(confidence=0.4)
        report = check_performance_cells([cell])
        assert report.passed  # warning, not error
        assert any("low extraction confidence" in w for w in report.warnings)

    def test_warns_on_missing_metric_direction(self):
        cell = _make_cell(higher_is_better=None)
        report = check_performance_cells([cell])
        assert report.passed
        assert any("metric direction missing" in w for w in report.warnings)

    def test_custom_confidence_threshold(self):
        cell = _make_cell(confidence=0.7)
        config = HarnessConfig(performance_confidence_threshold=0.8)
        report = check_performance_cells([cell], config)
        assert any("low extraction confidence" in w for w in report.warnings)

    def test_empty_list_passes(self):
        report = check_performance_cells([])
        assert report.passed


# ---------------------------------------------------------------------------
# check_structured_artifacts
# ---------------------------------------------------------------------------


class TestCheckStructuredArtifacts:
    def test_passes_with_valid_artifacts(self):
        artifacts = [_make_artifact()]
        report = check_structured_artifacts(artifacts)
        assert report.passed

    def test_fails_with_unsupported_type(self):
        artifact = _make_artifact(artifact_type="unknown_type")
        report = check_structured_artifacts([artifact])
        assert not report.passed
        assert "unsupported artifact type" in report.errors[0]

    def test_fails_with_empty_text(self):
        artifact = _make_artifact(text="")
        report = check_structured_artifacts([artifact])
        assert not report.passed
        assert "empty" in report.errors[0]

    def test_warns_on_low_confidence(self):
        artifact = _make_artifact(confidence=0.3)
        report = check_structured_artifacts([artifact])
        assert report.passed
        assert any("low-confidence" in w for w in report.warnings)

    def test_custom_artifact_confidence_threshold(self):
        artifact = _make_artifact(confidence=0.55)
        config = HarnessConfig(artifact_confidence_threshold=0.7)
        report = check_structured_artifacts([artifact], config)
        assert any("low-confidence" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# check_storyline_claims
# ---------------------------------------------------------------------------


class TestCheckStorylineClaims:
    def test_passes_with_valid_claims(self):
        claims = [_make_claim(evidence_count=3)]
        report = check_storyline_claims(claims)
        assert report.passed

    def test_fails_with_unsupported_type(self):
        claim = _make_claim(claim_type="unknown_type")
        report = check_storyline_claims([claim])
        assert not report.passed
        assert "Unsupported storyline claim type" in report.errors[0]

    def test_fails_with_ungrounded_claim(self):
        claim = StorylineClaim(
            claim_type="prior_solution",
            claim="Ungrounded",
            confidence=0.9,
            evidence=[],
        )
        report = check_storyline_claims([claim])
        assert not report.passed
        assert "Ungrounded" in report.errors[0]

    def test_chain_requires_min_evidence(self):
        claim = _make_claim(
            claim_type="solution_limit_response_chain",
            evidence_count=2,
        )
        report = check_storyline_claims([claim])
        assert not report.passed
        assert "chain" in report.errors[0].lower()

    def test_warns_on_single_paper_for_trend(self):
        claim = _make_claim(
            claim_type="trend_by_year_and_method",
            evidence_count=1,
        )
        report = check_storyline_claims([claim])
        assert report.passed  # warning only
        assert any("two supporting papers" in w for w in report.warnings)

    def test_warns_on_low_confidence(self):
        claim = _make_claim(confidence=0.5)
        report = check_storyline_claims([claim])
        assert any("Low-confidence" in w for w in report.warnings)

    def test_custom_chain_min_evidence(self):
        claim = _make_claim(
            claim_type="solution_limit_response_chain",
            evidence_count=3,
        )
        config = HarnessConfig(storyline_chain_min_evidence=5)
        report = check_storyline_claims([claim], config)
        assert not report.passed


# ---------------------------------------------------------------------------
# HarnessRegistry
# ---------------------------------------------------------------------------


class TestHarnessRegistry:
    def test_builtin_checks_registered(self):
        names = registry.list_checks()
        assert "check_citations" in names
        assert "check_performance_cells" in names
        assert "check_structured_artifacts" in names
        assert "check_storyline_claims" in names

    def test_checks_in_group(self):
        tables_checks = registry.checks_in_group("tables")
        assert "check_performance_cells" in tables_checks
        assert "check_structured_artifacts" in tables_checks

    def test_citations_group(self):
        assert "check_citations" in registry.checks_in_group("citations")

    def test_storyline_group(self):
        assert "check_storyline_claims" in registry.checks_in_group("storyline")

    def test_get_returns_check(self):
        check = registry.get("check_citations")
        assert check is not None
        assert check.name == "check_citations"

    def test_get_returns_none_for_unknown(self):
        assert registry.get("nonexistent") is None

    def test_register_custom_check(self):
        custom_registry = HarnessRegistry()

        @custom_registry.register(name="my_check", group="custom")
        def my_check(items, config=None):
            return HarnessReport(
                check_name="my_check",
                passed=True,
                score=1.0,
                item_count=len(items),
            )

        assert "my_check" in custom_registry.list_checks()
        assert "my_check" in custom_registry.checks_in_group("custom")

    def test_register_instance(self):
        custom_registry = HarnessRegistry()

        class MyCheck(HarnessCheck):
            name = "instance_check"
            group = "custom"

            def run(self, items, config=None):
                return HarnessReport(
                    check_name=self.name, passed=True, score=1.0, item_count=len(items)
                )

        custom_registry.register_instance(MyCheck())
        assert "instance_check" in custom_registry.list_checks()

    def test_register_instance_requires_name(self):
        custom_registry = HarnessRegistry()

        class NamelessCheck(HarnessCheck):
            name = ""

        with pytest.raises(ValueError, match="non-empty name"):
            custom_registry.register_instance(NamelessCheck())


# ---------------------------------------------------------------------------
# HarnessEngine
# ---------------------------------------------------------------------------


class TestHarnessEngine:
    def test_run_single_check(self):
        engine = HarnessEngine(registry)
        report = engine.run("check_citations", [_make_citation()])
        assert report.check_name == "check_citations"
        assert report.passed

    def test_run_unknown_check_raises(self):
        engine = HarnessEngine(registry)
        with pytest.raises(KeyError, match="not registered"):
            engine.run("nonexistent", [])

    def test_run_group(self):
        engine = HarnessEngine(registry)
        reports = engine.run_group(
            "tables",
            {
                "check_performance_cells": [_make_cell()],
                "check_structured_artifacts": [_make_artifact()],
            },
        )
        assert "check_performance_cells" in reports
        assert "check_structured_artifacts" in reports
        assert reports["check_performance_cells"].passed
        assert reports["check_structured_artifacts"].passed

    def test_combine_reports(self):
        engine = HarnessEngine(registry)
        r1 = HarnessReport(check_name="a", passed=True, score=0.8, findings=[])
        r2 = HarnessReport(check_name="b", passed=False, score=0.4, findings=[])
        combined = engine.combine([r1, r2])
        assert not combined.passed
        assert abs(combined.score - 0.6) < 1e-9

    def test_combine_empty_reports(self):
        engine = HarnessEngine(registry)
        combined = engine.combine([])
        assert combined.passed
        assert combined.score == 1.0

    def test_combine_aggregates_findings(self):
        engine = HarnessEngine(registry)
        r1 = HarnessReport(
            check_name="a",
            passed=True,
            score=1.0,
            findings=[HarnessFinding(severity=Severity.ERROR, message="e1")],
        )
        r2 = HarnessReport(
            check_name="b",
            passed=True,
            score=1.0,
            findings=[HarnessFinding(severity=Severity.WARNING, message="w1")],
        )
        combined = engine.combine([r1, r2])
        assert len(combined.findings) == 2
        assert combined.errors == ["e1"]
        assert combined.warnings == ["w1"]

    def test_run_with_deps_no_dependencies(self):
        engine = HarnessEngine(registry)
        reports = engine.run_with_deps(
            "check_citations",
            {"check_citations": [_make_citation()]},
        )
        assert "check_citations" in reports

    def test_run_with_deps_resolves_chain(self):
        """Verify dependency ordering with a custom registry."""
        custom_registry = HarnessRegistry()
        execution_order = []

        @custom_registry.register(name="dep_a", group="g")
        def dep_a(items, config=None):
            execution_order.append("dep_a")
            return HarnessReport(check_name="dep_a", passed=True, score=1.0)

        @custom_registry.register(name="dep_b", group="g", depends_on=["dep_a"])
        def dep_b(items, config=None):
            execution_order.append("dep_b")
            return HarnessReport(check_name="dep_b", passed=True, score=1.0)

        engine = HarnessEngine(custom_registry)
        reports = engine.run_with_deps("dep_b", {"dep_a": [], "dep_b": []})
        assert execution_order == ["dep_a", "dep_b"]
        assert "dep_a" in reports
        assert "dep_b" in reports

    def test_engine_uses_custom_config(self):
        engine = HarnessEngine(
            registry,
            HarnessConfig(performance_confidence_threshold=0.99),
        )
        cell = _make_cell(confidence=0.7)
        report = engine.run("check_performance_cells", [cell])
        assert any("low extraction confidence" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# HarnessConfig
# ---------------------------------------------------------------------------


class TestHarnessConfig:
    def test_defaults_match_original_values(self):
        config = HarnessConfig()
        assert config.performance_confidence_threshold == 0.65
        assert config.artifact_confidence_threshold == 0.6
        assert config.storyline_confidence_threshold == 0.7
        assert config.storyline_chain_min_evidence == 3

    def test_from_littrace_config_with_harness_field(self):
        lt_config = LitTraceConfig(
            harness=HarnessThresholdConfig(
                performance_confidence=0.8,
                artifact_confidence=0.75,
                storyline_confidence=0.85,
                storyline_chain_min_evidence=5,
            )
        )
        harness_config = HarnessConfig.from_littrace_config(lt_config)
        assert harness_config.performance_confidence_threshold == 0.8
        assert harness_config.artifact_confidence_threshold == 0.75
        assert harness_config.storyline_confidence_threshold == 0.85
        assert harness_config.storyline_chain_min_evidence == 5

    def test_from_littrace_config_without_harness_field(self):
        """Should return defaults when config has no harness field."""
        harness_config = HarnessConfig.from_littrace_config(object())
        assert harness_config.performance_confidence_threshold == 0.65

    def test_allowed_types_are_sets(self):
        config = HarnessConfig()
        assert isinstance(config.allowed_artifact_types, set)
        assert isinstance(config.allowed_storyline_types, set)
        assert isinstance(config.storyline_multi_paper_types, set)
