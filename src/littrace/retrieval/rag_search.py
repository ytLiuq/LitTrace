from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from littrace.models import EvidenceSpan, EvidenceSourceKind

from littrace.config import LitTraceConfig
from littrace.retrieval.embeddings import embedding_client_from_config
from littrace.retrieval.pgvector_store import PgvectorRagStore, RagSearchHit
from littrace.retrieval.rag_profile import RagProfile, load_session_rag_profile


@dataclass(frozen=True)
class RagSearchResult:
    profile: RagProfile
    hits: list[RagSearchHit]


async def search_session_rag(
    config: LitTraceConfig,
    session: object,
    question: str,
    *,
    top_k: int | None = None,
) -> RagSearchResult | None:
    profile = load_session_rag_profile(session, config=config)
    if profile is None:
        return None
    if profile.backend != "pgvector" or not config.rag.enabled or config.rag.backend != "pgvector":
        return None
    if not question.strip():
        return RagSearchResult(profile=profile, hits=[])
    embedding_client = embedding_client_from_config(config, profile)
    embedding = await embedding_client.embed_texts([question.strip()])
    store = PgvectorRagStore(config, profile)
    hits = store.query_chunks(embedding[0], top_k=top_k or profile.top_k)
    return RagSearchResult(profile=profile, hits=hits)


async def search_workspace_rag(
    config: LitTraceConfig,
    workspace: object,
    question: str,
    *,
    top_k: int | None = None,
) -> RagSearchResult | None:
    profile_data = getattr(getattr(workspace, "context", None), "filters", None)
    profile_raw = getattr(profile_data, "rag_profile", None) if profile_data is not None else None
    if not isinstance(profile_raw, dict):
        return None
    try:
        profile = RagProfile.model_validate(profile_raw)
    except Exception:
        return None
    session_root = Path(config.storage.sessions_dir) / profile.session_id
    if not session_root.exists():
        return None
    session = SimpleNamespace(
        session_id=profile.session_id,
        workspace_dir=session_root / "workspace",
    )
    return await search_session_rag(config, session, question, top_k=top_k or profile.top_k)


def rag_hits_to_evidence_spans(
    profile: RagProfile,
    hits: list[RagSearchHit],
    *,
    query: str,
) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for hit in hits:
        evidence_id = f"rag:{profile.profile_id}:{hit.chunk_id}"
        spans.append(
            EvidenceSpan(
                paper_id=hit.paper_id,
                evidence_id=evidence_id,
                source_record_id=f"rag:{profile.profile_id}:{hit.chunk_id}",
                section=hit.section or "rag_chunk",
                page=hit.page,
                table_id=hit.table_id,
                snippet=hit.text[:700],
                parser="rag",
                parser_version=f"rag:{profile.embedding_model}",
                content_hash=hit.chunk_hash,
                source_kind=EvidenceSourceKind.PRIMARY_DOCUMENT,
                confidence=hit.score,
                captured_at=None,
            )
        )
    return spans
