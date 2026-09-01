"""Backward-compatible re-export shim.

The table-extraction / comparison-matrix implementation moved to
:mod:`littrace.tables` (its canonical home). Older imports that say
``from littrace.evidence.tables import ...`` still work via this
shim, but new code should import from ``littrace.tables`` directly
— that module is what ``tests/test_tables.py`` and the public skills
plugin API target.

A handful of private helpers (``_parse_llm_cells``, ``_LLMCellSchema``,
the regex patterns) are also re-exported so the four-dimensions eval
suite and any older internal callers that reached past the API
boundary keep working. New code should not depend on these — the
underscore prefix marks them as implementation details.
"""
from __future__ import annotations

from littrace.tables import (  # noqa: F401 — re-export
    FIGURE_PATTERN,
    METRIC_DIRECTIONS,
    METRIC_PATTERN,
    TABLE_PATTERN,
    ArtifactNeedReport,
    _LLMCellSchema,
    _build_extraction_payload,
    _caption_artifacts,
    _cells_from_sections,
    _cells_from_tables,
    _combine_harnesses,
    _condition_profile,
    _conditions_from_text,
    _equation_artifacts,
    _guess_dataset,
    _matrix_row,
    _normalize_metric,
    _normalized_cell,
    _parse_llm_cells,
    _row_sort_key,
    _store_structured_artifacts,
    _stored_structured_artifacts,
    _window,
    build_comparison_matrices,
    decide_artifact_extraction_need,
    extract_performance_cells,
    extract_structured_artifacts,
)

__all__ = [
    "ArtifactNeedReport",
    "build_comparison_matrices",
    "decide_artifact_extraction_need",
    "extract_performance_cells",
    "extract_structured_artifacts",
]