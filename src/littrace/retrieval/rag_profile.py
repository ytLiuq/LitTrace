from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.retrieval.source_router import route_sources


class RagProfile(BaseModel):
    schema_version: str = "littrace.rag_profile.v1"
    profile_id: str
    user_id: str
    session_id: str
    namespace: str
    topic: str | None = None
    query_variants: list[str] = Field(default_factory=list)
    source_routes: list[str] = Field(default_factory=list)
    source_policy: str = "session_fixed"
    backend: str = "pgvector"
    postgres_schema: str = "littrace_rag"
    collection_name: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    top_k: int
    refresh_frequency: str
    auto_refresh_enabled: bool
    auto_download_open_access: bool
    login_required_policy: str
    last_refreshed_at: str | None = None
    last_refresh_status: str | None = None
    last_refresh_error: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def session_rag_dir(session: object) -> Path:
    return Path(getattr(session, "workspace_dir")) / "rag"


def session_rag_profile_path(session: object) -> Path:
    return session_rag_dir(session) / "profile.json"


def build_session_rag_profile(
    config: LitTraceConfig,
    session: object,
    workspace: LiteratureWorkspace,
) -> RagProfile:
    user_id = str(getattr(session, "user_id", config.storage.default_user_id))
    session_id = str(getattr(session, "session_id"))
    topic = workspace.context.filters.research_background or workspace.context.filters.topic
    query_variants = _query_variants(workspace, topic)
    source_routes = _source_routes(workspace, topic)
    namespace = f"{_safe_segment(user_id)}.{_safe_segment(session_id)}"
    fingerprint = "\0".join([user_id, session_id])
    profile_id = f"rag:{sha256(fingerprint.encode()).hexdigest()[:16]}"
    collection_name = "_".join(
        [
            _safe_segment(config.rag.collection_prefix),
            _safe_segment(user_id),
            _safe_segment(session_id),
        ]
    )
    return RagProfile(
        profile_id=profile_id,
        user_id=user_id,
        session_id=session_id,
        namespace=namespace,
        topic=topic,
        query_variants=query_variants,
        source_routes=source_routes,
        backend=config.rag.backend,
        postgres_schema=config.rag.schema_name,
        collection_name=collection_name,
        embedding_provider=config.rag.embedding_provider,
        embedding_model=config.rag.embedding_model,
        embedding_dimension=config.rag.embedding_dimension,
        chunk_target_tokens=config.rag.chunk_target_tokens,
        chunk_overlap_tokens=config.rag.chunk_overlap_tokens,
        top_k=config.rag.top_k,
        refresh_frequency=config.rag.refresh_frequency,
        auto_refresh_enabled=config.rag.auto_refresh_enabled,
        auto_download_open_access=config.rag.auto_download_open_access,
        login_required_policy=config.rag.login_required_policy,
    )


def save_session_rag_profile(
    config: LitTraceConfig,
    session: object,
    workspace: LiteratureWorkspace,
) -> RagProfile:
    profile = build_session_rag_profile(config, session, workspace)
    existing = load_session_rag_profile(session)
    if existing is not None and existing.profile_id == profile.profile_id:
        profile = profile.model_copy(
            update={
                "created_at": existing.created_at,
                "source_routes": existing.source_routes or profile.source_routes,
                "last_refreshed_at": existing.last_refreshed_at,
                "last_refresh_status": existing.last_refresh_status,
                "last_refresh_error": existing.last_refresh_error,
            }
        )
    target = session_rag_profile_path(session)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return profile


def persist_session_rag_profile(session: object, profile: RagProfile) -> RagProfile:
    target = session_rag_profile_path(session)
    target.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.model_copy(update={"updated_at": datetime.now(UTC).isoformat()})
    target.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return profile


def load_session_rag_profile(session: object) -> RagProfile | None:
    path = session_rag_profile_path(session)
    if not path.exists():
        return None
    try:
        return RagProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _query_variants(workspace: LiteratureWorkspace, topic: str | None) -> list[str]:
    diagnostics = workspace.context.filters.search_diagnostics or {}
    raw_variants = diagnostics.get("query_variants") if isinstance(diagnostics, dict) else None
    variants = raw_variants if isinstance(raw_variants, list) else []
    cleaned = [str(item).strip() for item in variants if str(item).strip()]
    if topic and topic not in cleaned:
        cleaned.insert(0, topic)
    return cleaned


def _source_routes(workspace: LiteratureWorkspace, topic: str | None) -> list[str]:
    filters = workspace.context.filters
    routes = [str(route).strip() for route in getattr(filters, "source_routes", []) if str(route).strip()]
    if routes:
        return routes
    discipline = str(getattr(filters, "discipline", "") or "materials chemistry")
    wants_recent = bool(topic)
    return [route.name for route in route_sources(discipline, wants_recent=wants_recent)]


def _safe_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    cleaned = cleaned.strip("_").lower()
    return cleaned or "default"
