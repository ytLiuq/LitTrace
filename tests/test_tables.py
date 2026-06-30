from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.tables import build_comparison_matrices, extract_performance_cells


def test_extract_performance_cells_from_parsed_sections():
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": {
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
                ]
            }
        }
    )

    workspace, harness = extract_performance_cells(workspace)

    assert harness.passed
    assert len(workspace.performance_cells) == 2
    assert workspace.performance_cells[0].evidence.page == 5


def test_extract_performance_cells_from_parsed_tables():
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": {
                "tables": [
                    {
                        "table_id": "T1",
                        "caption": "Performance comparison",
                        "cells": [{"row": "Our method", "column": "Gauge factor", "value": "gauge factor 12.5"}],
                        "evidence": {"paper_id": "p1", "table_id": "T1", "parser": "docling", "confidence": 0.8},
                    }
                ]
            }
        }
    )

    workspace, harness = extract_performance_cells(workspace)

    assert harness.passed
    assert workspace.performance_cells[0].metric == "gauge factor"
    assert workspace.performance_cells[0].evidence.table_id == "T1"


def test_build_comparison_matrices_groups_metrics_and_preserves_evidence():
    workspace = LiteratureWorkspace(
        papers={
            "p1": PaperMetadata(paper_id="p1", title="Paper 1", year=2026),
            "p2": PaperMetadata(paper_id="p2", title="Paper 2", year=2025),
        },
        parsed_papers={
            "p1": {
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
                ]
            },
            "p2": {
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
                ]
            },
        },
    )
    workspace, _ = extract_performance_cells(workspace)

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

    assert "Mixed units" in report.matrices[0].warnings[0]
    assert not report.matrices[0].rows[0].comparable
