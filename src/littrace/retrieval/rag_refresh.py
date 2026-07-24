from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace, ParsedPaper, coerce_parsed
from littrace.retrieval.embeddings import embedding_client_from_config
from littrace.retrieval.pgvector_store import PgvectorRagStore, RagChunkRecord
from littrace.retrieval.rag_profile import (
    RagProfile,
    persist_session_rag_profile,
    save_session_rag_profile,
    session_rag_dir,
)


class RagRefreshReport(BaseModel):
    schema_version: str = "littrace.rag_refresh_report.v1"
    profile_id: str
    user_id: str
    session_id: str
    collection_name: str
    backend: str
    refreshed_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    paper_count: int = 0
    source_count: int = 0
    chunk_count: int = 0
    upserted_count: int = 0
    stale_chunk_count: int = 0
    skipped: bool = False
    skip_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class RagChunkDraft:
    chunk_id: str
    paper_id: str
    text: str
    chunk_hash: str
    source_record_id: str | None = None
    section: str | None = None
    page: int | None = None
    table_id: str | None = None
    metadata: dict[str, object] | None = None


async def refresh_session_rag_index(
    config: LitTraceConfig,
    session: object,
    workspace: LiteratureWorkspace | None = None,
) -> tuple[RagProfile | None, RagRefreshReport]:
    if not config.rag.enabled or config.rag.backend != "pgvector":
        profile = _build_profile_if_possible(config, session, workspace)
        if profile is None:
            return None, _skipped_report(config, session, "rag_disabled_or_unconfigured")
        report = _skipped_report(config, session, "rag_disabled_or_unconfigured", profile=profile)
        if workspace is not None:
            workspace.context.filters.rag_profile = profile.model_dump(mode="json")
            workspace.context.filters.rag_enabled = config.rag.enabled
            workspace.context.filters.rag_backend = config.rag.backend
            workspace.context.filters.rag_source_routes = list(profile.source_routes)
            workspace.context.filters.rag_refresh_report = report.model_dump(mode="json")
            _persist_refresh_report(session, report)
            persist_session_rag_profile(
                session,
                profile.model_copy(
                    update={
                        "last_refresh_status": "skipped",
                        "last_refresh_error": report.skip_reason,
                    }
                ),
            )
        return profile, report

    if workspace is None:
        from littrace.session import load_workspace

        workspace = load_workspace(session)

    profile = save_session_rag_profile(config, session, workspace)
    drafts = build_rag_chunk_drafts(workspace, profile)
    warnings: list[str] = []
    if not drafts:
        report = RagRefreshReport(
            profile_id=profile.profile_id,
            user_id=profile.user_id,
            session_id=profile.session_id,
            collection_name=profile.collection_name,
            backend=profile.backend,
            paper_count=len(workspace.parsed_papers),
            source_count=0,
            chunk_count=0,
            upserted_count=0,
            skipped=True,
            skip_reason="no_chunks",
            warnings=["No parsed sections or tables were available for RAG indexing."],
        )
        _persist_refresh_report(session, report)
        workspace.context.filters.rag_profile = profile.model_dump(mode="json")
        workspace.context.filters.rag_enabled = config.rag.enabled
        workspace.context.filters.rag_backend = config.rag.backend
        workspace.context.filters.rag_source_routes = list(profile.source_routes)
        workspace.context.filters.rag_last_refreshed_at = report.refreshed_at
        workspace.context.filters.rag_chunk_count = 0
        workspace.context.filters.rag_paper_count = len(workspace.parsed_papers)
        workspace.context.filters.rag_refresh_report = report.model_dump(mode="json")
        persist_session_rag_profile(
            session,
            profile.model_copy(
                update={
                    "last_refreshed_at": report.refreshed_at,
                    "last_refresh_status": "skipped",
                    "last_refresh_error": report.skip_reason,
                }
            ),
        )
        return profile, report

    store = PgvectorRagStore(config, profile)
    embedding_client = embedding_client_from_config(config, profile)
    embeddings = await embedding_client.embed_texts([draft.text for draft in drafts])
    upserted = store.upsert_chunks(
        [
            RagChunkRecord(
                chunk_id=draft.chunk_id,
                paper_id=draft.paper_id,
                text=draft.text,
                embedding=embedding,
                chunk_hash=draft.chunk_hash,
                source_record_id=draft.source_record_id,
                section=draft.section,
                page=draft.page,
                table_id=draft.table_id,
                metadata=draft.metadata or {},
            )
            for draft, embedding in zip(drafts, embeddings, strict=True)
        ]
    )
    stale_deleted = store.delete_missing_chunks(draft.chunk_id for draft in drafts)
    report = RagRefreshReport(
        profile_id=profile.profile_id,
        user_id=profile.user_id,
        session_id=profile.session_id,
        collection_name=profile.collection_name,
        backend=profile.backend,
        paper_count=len(workspace.parsed_papers),
        source_count=len({draft.paper_id for draft in drafts}),
        chunk_count=len(drafts),
        upserted_count=upserted,
        stale_chunk_count=stale_deleted,
        skipped=False,
        warnings=warnings,
    )
    workspace.context.filters.rag_profile = profile.model_dump(mode="json")
    workspace.context.filters.rag_enabled = config.rag.enabled
    workspace.context.filters.rag_backend = config.rag.backend
    workspace.context.filters.rag_source_routes = list(profile.source_routes)
    workspace.context.filters.rag_last_refreshed_at = report.refreshed_at
    workspace.context.filters.rag_chunk_count = len(drafts)
    workspace.context.filters.rag_stale_chunk_count = stale_deleted
    workspace.context.filters.rag_paper_count = len(workspace.parsed_papers)
    workspace.context.filters.rag_refresh_report = report.model_dump(mode="json")
    _persist_refresh_report(session, report)
    persist_session_rag_profile(
        session,
        profile.model_copy(
            update={
                "last_refreshed_at": report.refreshed_at,
                "last_refresh_status": "completed",
                "last_refresh_error": None,
            }
        ),
    )
    return profile, report


def build_rag_chunk_drafts(
    workspace: LiteratureWorkspace,
    profile: RagProfile,
) -> list[RagChunkDraft]:
    drafts: list[RagChunkDraft] = []
    for paper_id, parsed_value in workspace.parsed_papers.items():
        parsed = coerce_parsed(parsed_value)
        if not parsed.parsed:
            continue
        source_index = 0
        for section in parsed.sections:
            if not isinstance(section, dict):
                continue
            text = str(section.get("text") or "").strip()
            if not text:
                continue
            name = str(section.get("name") or section.get("title") or "section")
            provenance = section.get("evidence") if isinstance(section.get("evidence"), dict) else {}
            page = provenance.get("page") if isinstance(provenance, dict) else None
            for chunk_index, chunk_text in enumerate(
                _chunk_text(
                    text,
                    profile.chunk_target_tokens,
                    profile.chunk_overlap_tokens,
                )
            ):
                drafts.append(
                    _build_chunk_draft(
                        profile,
                        paper_id=paper_id,
                        source_key=f"section:{source_index}:{chunk_index}",
                        text=chunk_text,
                        section=name,
                        page=page if isinstance(page, int) else None,
                        metadata={
                            "source": "section",
                            "section_name": name,
                            "parser": provenance.get("parser") if isinstance(provenance, dict) else None,
                        },
                    )
                )
            source_index += 1

        markdown = _structured_markdown(parsed)
        if markdown:
            for chunk_index, chunk_text in enumerate(
                _chunk_text(markdown, profile.chunk_target_tokens, profile.chunk_overlap_tokens)
            ):
                drafts.append(
                    _build_chunk_draft(
                        profile,
                        paper_id=paper_id,
                        source_key=f"markdown:{chunk_index}",
                        text=chunk_text,
                        section="structured_document",
                        metadata={"source": "structured_document"},
                    )
                )

        for table_index, table in enumerate(parsed.tables):
            table_text = _table_text(table)
            if not table_text:
                continue
            drafts.append(
                _build_chunk_draft(
                    profile,
                    paper_id=paper_id,
                    source_key=f"table:{table_index}",
                    text=table_text,
                    section="table",
                    table_id=getattr(table, "table_id", None),
                    metadata={"source": "table", "table_index": table_index},
                )
            )

    return drafts


def _build_chunk_draft(
    profile: RagProfile,
    *,
    paper_id: str,
    source_key: str,
    text: str,
    section: str | None = None,
    page: int | None = None,
    table_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> RagChunkDraft:
    cleaned = " ".join(text.split()).strip()
    digest = sha256(
        "\0".join([profile.profile_id, paper_id, source_key, cleaned]).encode("utf-8")
    ).hexdigest()[:16]
    chunk_id = f"rag:{digest}"
    return RagChunkDraft(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=cleaned,
        chunk_hash=sha256(cleaned.encode("utf-8")).hexdigest(),
        section=section,
        page=page,
        table_id=table_id,
        metadata={
            "profile_id": profile.profile_id,
            "user_id": profile.user_id,
            "session_id": profile.session_id,
            **(metadata or {}),
        },
    )


def _structured_markdown(parsed: ParsedPaper) -> str:
    markdown = parsed.structured_document.get("markdown")
    if isinstance(markdown, str) and markdown.strip():
        return markdown
    pieces: list[str] = []
    if parsed.title:
        pieces.append(parsed.title)
    if parsed.abstract:
        pieces.append(parsed.abstract)
    return "\n\n".join(pieces)


def _table_text(table: object) -> str:
    cells = getattr(table, "cells", None)
    parts: list[str] = []
    caption = getattr(table, "caption", None)
    if isinstance(caption, str) and caption.strip():
        parts.append(caption.strip())
    if isinstance(cells, list):
        for cell in cells[:100]:
            if not isinstance(cell, dict):
                continue
            value = cell.get("text") or cell.get("value") or cell.get("content")
            if value:
                parts.append(str(value))
    return "\n".join(parts).strip()


def _chunk_text(text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return []
    if len(tokens) <= target_tokens:
        return [" ".join(tokens)]
    chunks: list[str] = []
    step = max(target_tokens - overlap_tokens, 1)
    for start in range(0, len(tokens), step):
        window = tokens[start : start + target_tokens]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + target_tokens >= len(tokens):
            break
    return chunks


def _persist_refresh_report(session: object, report: RagRefreshReport) -> None:
    rag_root = session_rag_dir(session)
    refresh_runs = rag_root / "refresh_runs"
    refresh_runs.mkdir(parents=True, exist_ok=True)
    target = refresh_runs / f"{report.refreshed_at.replace(':', '').replace('-', '')}.json"
    target.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _build_profile_if_possible(
    config: LitTraceConfig, session: object, workspace: LiteratureWorkspace | None
) -> RagProfile | None:
    if workspace is None:
        return None
    return save_session_rag_profile(config, session, workspace)


def _skipped_report(
    config: LitTraceConfig,
    session: object,
    reason: str,
    *,
    profile: RagProfile | None = None,
) -> RagRefreshReport:
    if profile is None:
        profile = RagProfile(
            profile_id=f"rag:{getattr(session, 'session_id', 'session')}",
            user_id=str(getattr(session, "user_id", config.storage.default_user_id)),
            session_id=str(getattr(session, "session_id", "session")),
            namespace=f"{getattr(session, 'user_id', config.storage.default_user_id)}.{getattr(session, 'session_id', 'session')}",
            topic=None,
            query_variants=[],
            backend=config.rag.backend,
            postgres_schema=config.rag.schema_name,
            collection_name=f"{config.rag.collection_prefix}_{getattr(session, 'user_id', config.storage.default_user_id)}_{getattr(session, 'session_id', 'session')}",
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
    return RagRefreshReport(
        profile_id=profile.profile_id,
        user_id=profile.user_id,
        session_id=profile.session_id,
        collection_name=profile.collection_name,
        backend=profile.backend,
        skipped=True,
        skip_reason=reason,
        warnings=[],
    )
