"""Workflow status recommendation test — pure recommendation engine, no I/O."""

from __future__ import annotations

import pytest

from littrace.models import LiteratureWorkspace
from littrace.workflow_status import build_workflow_status


pytestmark = pytest.mark.domain


def test_workflow_status_recommends_retrieval_when_empty():
    report = build_workflow_status(LiteratureWorkspace())

    assert report.transitions
    assert "search_papers skill" in report.recommended_next_steps
    assert report.blocked_count > 0
