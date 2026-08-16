"""Structured-artifact extraction test (figure/table/equation from real
section text)."""

from __future__ import annotations

import pytest

from littrace.evidence.tables import extract_structured_artifacts
from littrace.models import LiteratureWorkspace


pytestmark = pytest.mark.domain


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
