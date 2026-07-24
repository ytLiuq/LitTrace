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
from littrace.retrieval.pgvector_store import PgvectorRagStore, RagChunkRecord, pgvector_setup_sql
from littrace.retrieval.rag_refresh import (
    RagRefreshReport,
    build_rag_chunk_drafts,
    refresh_session_rag_index,
)
from littrace.retrieval.rag_search import (
    RagSearchResult,
    rag_hits_to_evidence_spans,
    search_session_rag,
    search_workspace_rag,
)

__all__ = [
    "LiveSearchClient",
    "MockMaterialsSearchClient",
    "SearchDiagnostics",
    "PgvectorRagStore",
    "SourceAdapter",
    "SourceFailureClass",
    "SourceHealth",
    "SourceResult",
    "SourceRoute",
    "RagChunkRecord",
    "RagRefreshReport",
    "RagSearchResult",
    "backfill_workspace_by_dois",
    "build_query_variants",
    "build_rag_chunk_drafts",
    "classify_source_exception",
    "record_search_provenance",
    "pgvector_setup_sql",
    "rag_hits_to_evidence_spans",
    "refresh_session_rag_index",
    "search_session_rag",
    "search_workspace_rag",
    "resolve_full_text_for_paper",
    "resolve_full_text_for_papers",
    "resolve_workspace_full_text",
    "route_sources",
]
