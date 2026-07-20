"""Retrieval, source routing, and full-text resolution boundary."""

from littrace.retrieval.full_text import (
    backfill_workspace_by_dois,
    resolve_full_text_for_paper,
    resolve_full_text_for_papers,
    resolve_workspace_full_text,
)
from littrace.retrieval.search import (
    LiveSearchClient,
    MockMaterialsSearchClient,
    SearchDiagnostics,
    build_query_variants,
)
from littrace.retrieval.source_router import SourceRoute, route_sources
from littrace.retrieval.adapters import (
    SourceAdapter,
    SourceFailureClass,
    SourceHealth,
    SourceResult,
    classify_source_exception,
    record_search_provenance,
)

__all__ = [
    "LiveSearchClient",
    "MockMaterialsSearchClient",
    "SearchDiagnostics",
    "SourceAdapter",
    "SourceFailureClass",
    "SourceHealth",
    "SourceResult",
    "SourceRoute",
    "backfill_workspace_by_dois",
    "build_query_variants",
    "classify_source_exception",
    "record_search_provenance",
    "resolve_full_text_for_paper",
    "resolve_full_text_for_papers",
    "resolve_workspace_full_text",
    "route_sources",
]
