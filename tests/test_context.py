import pytest

from littrace.context import _merge_filters
from littrace.models import WorkspaceFilters


def test_merge_filters_rejects_unknown_workspace_metadata():
    with pytest.raises(ValueError, match="typoed_filter"):
        _merge_filters(WorkspaceFilters(), {"typoed_filter": True})


def test_workspace_filters_reject_unknown_input_fields():
    with pytest.raises(ValueError, match="unknown_filter"):
        WorkspaceFilters.model_validate({"unknown_filter": True})
