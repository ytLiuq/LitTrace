from littrace.runtime_components import component_statuses


def test_runtime_components_expose_coordinator_reviewer_and_quality_gates():
    status_names = {status.name for status in component_statuses()}

    assert "LitTrace Coordinator" in status_names
    assert "Optional Reviewer" in status_names
    assert "Citation and Evidence Gates" in status_names
