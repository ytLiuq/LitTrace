import asyncio
import json

from littrace.models import EvidenceSpan, LiteratureWorkspace, PaperMetadata, PerformanceCell
from littrace.evidence.tables import (
    build_comparison_matrices,
    decide_artifact_extraction_need,
    extract_performance_cells,
    extract_structured_artifacts,
)


def _mock_llm(monkeypatch, llm_response):
    """Patch chat_completion in tables module to return a canned response."""

    async def _fake_chat_completion(config, system_prompt, user_message, workspace=None, **kwargs):
        from littrace.llm import LLMReply

        return LLMReply(text=json.dumps(llm_response), used_llm=True)

    monkeypatch.setattr("littrace.evidence.tables.chat_completion", _fake_chat_completion)


def _run(coro):
    """Run async coroutine in sync test."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_extract_performance_cells_from_parsed_sections(monkeypatch):
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": {
                "parsed": True,
                "sections": [
                    {
                        "name": "Results",
                        "text": "The flexible sensor achieved sensitivity 2.3 kPa-1 and response time 45 ms.",
                        "evidence": {
                            "paper_id": "p1",
                            "section": "Results",
                            "page": 5,
                            "parser": "paddleocr",
                            "confidence": 0.82,
                        },
                    }
                ],
            }
        }
    )

    _mock_llm(
        monkeypatch,
        [
            {
                "metric": "sensitivity",
                "value": 2.3,
                "unit": "kPa-1",
                "section": "Results",
                "snippet": "sensitivity 2.3 kPa-1",
            },
            {
                "metric": "response time",
                "value": 45,
                "unit": "ms",
                "section": "Results",
                "snippet": "response time 45 ms",
            },
        ],
    )
    workspace, harness = _run(extract_performance_cells(workspace))

    assert harness.passed
    assert len(workspace.performance_cells) >= 2
    # LLM cells should have page 5 evidence
    llm_cells = [c for c in workspace.performance_cells if c.evidence.page == 5]
    assert len(llm_cells) >= 2
    metrics = {c.metric for c in llm_cells}
    assert "sensitivity" in metrics
    assert "response time" in metrics


def test_extract_performance_cells_from_parsed_tables(monkeypatch):
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": {
                "parsed": True,
                "tables": [
                    {
                        "table_id": "T1",
                        "caption": "Performance comparison",
                        "cells": [
                            {
                                "row": "Our method",
                                "column": "Gauge factor",
                                "value": "gauge factor 12.5",
                            }
                        ],
                        "evidence": {
                            "paper_id": "p1",
                            "table_id": "T1",
                            "parser": "docling",
                            "confidence": 0.8,
                        },
                    }
                ],
            }
        }
    )

    _mock_llm(
        monkeypatch,
        [
            {
                "metric": "gauge factor",
                "value": 12.5,
                "unit": None,
                "section": "section",
                "snippet": "gauge factor 12.5",
            },
        ],
    )
    workspace, harness = _run(extract_performance_cells(workspace))

    assert harness.passed
    assert workspace.performance_cells[0].metric == "gauge factor"


def test_extraction_uses_its_own_timeout_and_reports_schema_failures(monkeypatch):
    from littrace.config import LLMConfig, LitTraceConfig
    from littrace.llm import LLMReply

    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": {
                "parsed": True,
                "sections": [{"name": "Results", "text": "Sensitivity 2.3 kPa-1."}],
            }
        }
    )
    observed_timeouts: list[float] = []

    async def fake_completion(config, *args, **kwargs):
        observed_timeouts.append(config.llm.request_timeout_seconds)
        return LLMReply(
            text=json.dumps(
                [
                    {"metric": "sensitivity", "value": 2.3, "unit": "kPa-1", "snippet": "Sensitivity 2.3 kPa-1."},
                    {"metric": "young's modulus", "value": "not given", "unit": "kPa"},
                ]
            ),
            used_llm=True,
        )

    monkeypatch.setattr("littrace.evidence.tables.chat_completion", fake_completion)
    config = LitTraceConfig(
        llm=LLMConfig(request_timeout_seconds=3, metric_extraction_timeout_seconds=90)
    )
    _, harness = _run(extract_performance_cells(workspace, config))

    assert observed_timeouts == [90]
    assert not harness.passed
    assert harness.score < 1
    assert any("Schema violation" in error for error in harness.errors)


def test_build_comparison_matrices_groups_metrics_and_preserves_evidence(monkeypatch):
    workspace = LiteratureWorkspace(
        papers={
            "p1": PaperMetadata(paper_id="p1", title="Paper 1", year=2026),
            "p2": PaperMetadata(paper_id="p2", title="Paper 2", year=2025),
        },
        parsed_papers={
            "p1": {
                "parsed": True,
                "sections": [
                    {
                        "name": "Results",
                        "text": "The sensor achieved sensitivity 2.3 kPa-1.",
                        "evidence": {
                            "paper_id": "p1",
                            "page": 5,
                            "parser": "docling",
                            "confidence": 0.8,
                        },
                    }
                ],
            },
            "p2": {
                "parsed": True,
                "sections": [
                    {
                        "name": "Results",
                        "text": "The sensor achieved sensitivity 1.8 kPa-1.",
                        "evidence": {
                            "paper_id": "p2",
                            "page": 6,
                            "parser": "paddleocr",
                            "confidence": 0.75,
                        },
                    }
                ],
            },
        },
    )

    call_count = [0]

    async def _fake_chat(config, system_prompt, user_message, workspace=None, **kwargs):
        from littrace.llm import LLMReply

        call_count[0] += 1
        if call_count[0] == 1:
            return LLMReply(
                text=json.dumps(
                    [
                        {
                            "metric": "sensitivity",
                            "value": 2.3,
                            "unit": "kPa-1",
                            "section": "Results",
                            "snippet": "sensitivity 2.3 kPa-1",
                        }
                    ]
                ),
                used_llm=True,
            )
        else:
            return LLMReply(
                text=json.dumps(
                    [
                        {
                            "metric": "sensitivity",
                            "value": 1.8,
                            "unit": "kPa-1",
                            "section": "Results",
                            "snippet": "sensitivity 1.8 kPa-1",
                        }
                    ]
                ),
                used_llm=True,
            )

    monkeypatch.setattr("littrace.evidence.tables.chat_completion", _fake_chat)

    workspace, _ = _run(extract_performance_cells(workspace))

    report = build_comparison_matrices(workspace)

    assert len(report.matrices) == 1
    assert report.matrices[0].metric == "sensitivity"
    assert report.matrices[0].rows[0].paper_id == "p1"
    assert report.matrices[0].rows[0].evidence.page == 5
    assert report.matrices[0].warnings


def test_build_comparison_matrices_marks_mixed_units_not_comparable():
    workspace = LiteratureWorkspace(
        performance_cells=[
            {
                "paper_id": "p1",
                "metric": "response time",
                "value": 45.0,
                "unit": "ms",
                "higher_is_better": False,
                "evidence": {
                    "paper_id": "p1",
                    "snippet": "response time 45 ms",
                    "confidence": 0.8,
                },
            },
            {
                "paper_id": "p2",
                "metric": "response time",
                "value": 1.0,
                "unit": "s",
                "higher_is_better": False,
                "evidence": {
                    "paper_id": "p2",
                    "snippet": "response time 1 s",
                    "confidence": 0.8,
                },
            },
        ]
    )

    report = build_comparison_matrices(workspace)

    assert not any("Mixed units" in warning for warning in report.matrices[0].warnings)
    assert report.matrices[0].rows[0].unit == "ms"
    assert report.matrices[0].rows[1].value == 1000.0


def test_comparison_matrix_reports_missing_experimental_conditions():
    workspace = LiteratureWorkspace(
        performance_cells=[
            PerformanceCell(
                paper_id="p1",
                metric="sensitivity",
                value=12.5,
                unit="kPa-1",
                evidence=EvidenceSpan(
                    paper_id="p1", page=2, snippet="Sensitivity reached 12.5 kPa-1."
                ),
            )
        ]
    )

    report = build_comparison_matrices(workspace)

    assert any(
        "Experimental conditions are incomplete" in warning
        for warning in report.matrices[0].warnings
    )


def test_comparison_matrix_marks_mixed_conditions_and_tasks_not_comparable():
    workspace = LiteratureWorkspace(
        performance_cells=[
            PerformanceCell(
                paper_id="p1",
                metric="sensitivity",
                value=12.0,
                unit="kPa-1",
                task="pressure sensing",
                conditions={"test_protocol": "cyclic loading", "environment": "room temperature"},
                evidence=EvidenceSpan(paper_id="p1", page=2, snippet="Sensitivity was 12 kPa-1."),
            ),
            PerformanceCell(
                paper_id="p2",
                metric="sensitivity",
                value=20.0,
                unit="kPa-1",
                task="strain sensing",
                conditions={"test_protocol": "tensile test", "environment": "humidity-controlled"},
                evidence=EvidenceSpan(paper_id="p2", page=2, snippet="Sensitivity was 20 kPa-1."),
            ),
        ]
    )

    report = build_comparison_matrices(workspace)

    assert all(not row.comparable for row in report.matrices[0].rows)
    assert any("Mixed tasks" in warning for warning in report.matrices[0].warnings)
    assert any(
        "Experimental conditions differ" in warning for warning in report.matrices[0].warnings
    )


def test_extract_materials_chemistry_metrics(monkeypatch):
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": {
                "parsed": True,
                "sections": [
                    {
                        "name": "Electrochemical results",
                        "text": (
                            "The electrode showed conductivity 120 S/m, specific capacitance "
                            "245 F/g, cycle retention 91 %, and tensile strength 18 MPa."
                        ),
                        "evidence": {
                            "paper_id": "p1",
                            "page": 4,
                            "parser": "docling",
                            "confidence": 0.84,
                        },
                    }
                ],
            }
        }
    )

    _mock_llm(
        monkeypatch,
        [
            {
                "metric": "conductivity",
                "value": 120,
                "unit": "S/m",
                "section": "Electrochemical results",
                "snippet": "conductivity 120 S/m",
            },
            {
                "metric": "specific capacitance",
                "value": 245,
                "unit": "F/g",
                "section": "Electrochemical results",
                "snippet": "specific capacitance 245 F/g",
            },
            {
                "metric": "cycle retention",
                "value": 91,
                "unit": "%",
                "section": "Electrochemical results",
                "snippet": "cycle retention 91 %",
            },
            {
                "metric": "tensile strength",
                "value": 18,
                "unit": "MPa",
                "section": "Electrochemical results",
                "snippet": "tensile strength 18 MPa",
            },
        ],
    )
    workspace, harness = _run(extract_performance_cells(workspace))

    metrics = {cell.metric for cell in workspace.performance_cells}
    assert harness.passed
    assert {
        "conductivity",
        "specific capacitance",
        "cycle retention",
        "tensile strength",
    } <= metrics


def test_conductivity_units_are_normalized_for_comparison():
    workspace = LiteratureWorkspace(
        performance_cells=[
            {
                "paper_id": "p1",
                "metric": "conductivity",
                "value": 1.0,
                "unit": "S/cm",
                "higher_is_better": True,
                "evidence": {"paper_id": "p1", "snippet": "conductivity 1 S/cm", "confidence": 0.8},
            },
            {
                "paper_id": "p2",
                "metric": "conductivity",
                "value": 80.0,
                "unit": "S/m",
                "higher_is_better": True,
                "evidence": {"paper_id": "p2", "snippet": "conductivity 80 S/m", "confidence": 0.8},
            },
        ]
    )

    report = build_comparison_matrices(workspace)

    assert not any("Mixed units" in warning for warning in report.matrices[0].warnings)
    assert report.matrices[0].rows[0].value == 100.0
    assert report.matrices[0].rows[0].unit == "S/m"


def test_extracts_uncertainty_and_range_values(monkeypatch):
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": {
                "parsed": True,
                "sections": [
                    {
                        "name": "Results",
                        "text": "The sensor reached sensitivity 2.3 ± 0.1 kPa-1 and retention 90-95 %.",
                        "evidence": {"page": 3, "confidence": 0.8},
                    }
                ],
            }
        }
    )

    _mock_llm(
        monkeypatch,
        [
            {
                "metric": "sensitivity",
                "value": 2.3,
                "uncertainty": 0.1,
                "unit": "kPa-1",
                "section": "Results",
                "snippet": "sensitivity 2.3 ± 0.1 kPa-1",
            },
            {
                "metric": "retention",
                "value": 92.5,
                "value_min": 90,
                "value_max": 95,
                "unit": "%",
                "section": "Results",
                "snippet": "retention 90-95 %",
            },
        ],
    )
    workspace, _ = _run(extract_performance_cells(workspace))

    sensitivity = next(cell for cell in workspace.performance_cells if cell.metric == "sensitivity")
    retention = next(cell for cell in workspace.performance_cells if cell.metric == "retention")
    assert sensitivity.uncertainty == 0.1
    assert retention.value_min == 90.0
    assert retention.value_max == 95.0


def test_extract_structured_artifacts_with_evidence():
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": {
                "sections": [
                    {
                        "name": "Results",
                        "text": (
                            "Figure 2. SEM image showing porous MXene network and crack bridging.\n\n"
                            "Table 1. Performance comparison of sensitivity and response time.\n\n"
                            "Equation (1): S = delta R / R0 / delta P"
                        ),
                        "evidence": {
                            "paper_id": "p1",
                            "section": "Results",
                            "page": 4,
                            "parser": "paddleocr",
                            "confidence": 0.82,
                        },
                    }
                ]
            }
        }
    )

    workspace, harness = extract_structured_artifacts(workspace)
    artifacts = workspace.context.filters.structured_artifacts

    assert harness.passed
    assert {artifact["artifact_type"] for artifact in artifacts} >= {"figure", "table", "equation"}
    assert all(artifact["evidence"]["page"] == 4 for artifact in artifacts)


def test_decide_artifact_extraction_need_does_not_block_when_text_metrics_exist(monkeypatch):
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": {
                "parsed": True,
                "sections": [
                    {
                        "name": "Results",
                        "text": "The sensitivity reached 10 kPa-1.",
                        "evidence": {"page": 1},
                    }
                ],
            }
        }
    )

    _mock_llm(
        monkeypatch,
        [
            {
                "metric": "sensitivity",
                "value": 10,
                "unit": "kPa-1",
                "section": "Results",
                "snippet": "sensitivity reached 10 kPa-1",
            },
        ],
    )
    workspace, _ = _run(extract_performance_cells(workspace))

    report = decide_artifact_extraction_need(workspace)

    assert not report.needs_artifact_extraction
    assert report.performance_cell_count >= 1
    assert report.recommended_parse_strategy == "text_only"
    assert any(button["id"] == "text_only" for button in report.buttons)


def test_decide_artifact_extraction_need_recommends_paddleocr_without_text():
    report = decide_artifact_extraction_need(LiteratureWorkspace())

    assert report.needs_artifact_extraction
    assert "paddleocr" in report.recommended_tools
    assert report.recommended_parse_strategy == "ocr"
    assert any(
        button["id"] == "ocr" and button["recommended"] == "true" for button in report.buttons
    )
