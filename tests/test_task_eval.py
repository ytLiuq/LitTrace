from littrace.models import (
    ClaimStatus,
    EvidenceSpan,
    LiteratureWorkspace,
    PaperMetadata,
    PerformanceCell,
    VerificationReport,
)
from littrace.task_eval import (
    ExpectedPerformanceCell,
    evaluate_performance_cells,
    evaluate_task_runs,
    evaluate_workspace_task_case,
)


def _grounded_workspace(status: ClaimStatus = ClaimStatus.VERIFIED) -> LiteratureWorkspace:
    evidence = EvidenceSpan(
        paper_id="p1",
        page=6,
        table_id="T2",
        snippet="The improved response time was 45 ms.",
        parser="docling",
        observed_value=45.0,
        observed_unit="ms",
    )
    workspace = LiteratureWorkspace(
        papers={
            "p1": PaperMetadata(
                paper_id="p1",
                title="Sensor paper",
                doi="10.1000/sensor",
            )
        },
        performance_cells=[
            PerformanceCell(
                paper_id="p1",
                metric="response time",
                value=45.0,
                unit="ms",
                evidence=evidence,
            )
        ],
        claim_verification_reports=[
            VerificationReport(
                claim="The improved response time was 45 ms.",
                status=status,
                evidence=[evidence],
                semantic_supported=status == ClaimStatus.VERIFIED,
                critical=True,
            )
        ],
    )
    workspace.context.active_papers = ["p1"]
    return workspace


def test_exact_performance_cell_scoring_normalizes_units():
    workspace = _grounded_workspace()
    expected = [
        ExpectedPerformanceCell(
            doi="10.1000/sensor",
            metric="response time",
            value=0.045,
            unit="s",
            page=6,
            table_id="T2",
        )
    ]

    report = evaluate_performance_cells(expected, workspace)

    assert report.matched_count == 1
    assert report.recall == 1.0
    assert report.precision == 1.0
    assert report.exact_match == 1.0


def test_evidence_grounded_task_requires_publishable_claims_and_exact_cells():
    case = {
        "case_id": "sensor-task",
        "expected_dois": ["10.1000/sensor"],
        "required_claims": ["improved response time"],
        "expected_performance_cells": [
            {
                "doi": "10.1000/sensor",
                "metric": "response time",
                "value": 45.0,
                "unit": "ms",
                "page": 6,
                "table_id": "T2",
            }
        ],
    }

    result = evaluate_workspace_task_case(case, _grounded_workspace())

    assert result.passed
    assert result.doi_recall == 1.0
    assert result.table_cell_recall == 1.0
    assert result.critical_claim_precision == 1.0
    assert result.citation_recall == 1.0


def test_evidence_grounded_task_fails_for_draft_only_critical_claim():
    case = {
        "case_id": "sensor-task",
        "expected_dois": ["10.1000/sensor"],
        "required_claims": ["improved response time"],
    }

    result = evaluate_workspace_task_case(
        case,
        _grounded_workspace(status=ClaimStatus.SUPPORTED),
    )

    assert not result.passed
    assert result.critical_claim_precision == 0.0
    assert result.unsupported_critical_claim_rate == 1.0


def test_task_report_aggregates_success_rate_and_missing_runs():
    cases = [
        {"case_id": "pass", "expected_dois": ["10.1000/sensor"]},
        {"case_id": "missing", "expected_dois": ["10.1000/missing"]},
    ]

    report = evaluate_task_runs(cases, {"pass": _grounded_workspace()})

    assert report.case_count == 1
    assert report.evidence_grounded_task_success_rate == 1.0
    assert report.warnings == ["No workspace result was supplied for task case missing."]
