"""Runtime status test.

The session-memory roundtrip is covered by ``test_session_state.py`` against
the real metadata store; supplementary/source-adapter tests were removed
because they only asserted trivial helpers.
"""

from __future__ import annotations

import pytest

from littrace.runtime_components import component_statuses


pytestmark = pytest.mark.unit


def test_runtime_components_expose_coordinator_reviewer_and_quality_gates():
    status_names = {status.name for status in component_statuses()}

    assert "LitTrace Coordinator" in status_names
    assert "Optional Reviewer" in status_names
    assert "Citation and Evidence Gates" in status_names
