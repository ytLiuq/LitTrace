from __future__ import annotations

from fastapi import APIRouter

from littrace.api.backend import api_app, current_session_id
from littrace.citations import citation_records_for_papers
from littrace.retrieval.full_text import backfill_workspace_by_dois
from littrace.models import (
    CitationAudit,
    CitationRecord,
    ContextUpdate,
    DOIBackfillRequest,
    LiteratureWorkspace,
)
from littrace.skill_runner import audit_citation_links_skill, resolve_workspace_full_text_skill
from littrace.session import load_or_create_session, save_workspace



router = APIRouter(tags=["context"])


@router.get("/context", response_model=LiteratureWorkspace)
def get_context() -> LiteratureWorkspace:
    return api_app.WORKSPACE


@router.patch("/context", response_model=LiteratureWorkspace)
def update_context(update: ContextUpdate) -> LiteratureWorkspace:
    from littrace.context import apply_context_update

    workspace = apply_context_update(api_app.WORKSPACE, update)
    if current_session_id():
        session = load_or_create_session(api_app.load_config(), current_session_id())
        save_workspace(session, workspace, config=api_app.load_config())
    return api_app._set_workspace(workspace)


@router.post("/full-text/resolve", response_model=dict[str, object])
async def full_text_resolve() -> dict[str, object]:
    config = api_app.load_config()
    workspace = await resolve_workspace_full_text_skill(api_app.WORKSPACE, config)
    if current_session_id():
        save_workspace(load_or_create_session(config, current_session_id()), workspace, config=config)
    return api_app._set_workspace(workspace).full_text_reports


@router.post("/papers/backfill-dois", response_model=LiteratureWorkspace)
async def papers_backfill_dois(request: DOIBackfillRequest) -> LiteratureWorkspace:
    config = api_app.load_config()
    workspace = await backfill_workspace_by_dois(api_app.WORKSPACE, request.dois, config)
    if current_session_id():
        save_workspace(load_or_create_session(config, current_session_id()), workspace, config=config)
    return api_app._set_workspace(workspace)


@router.get("/citations/context", response_model=list[CitationRecord])
def context_citations() -> list[CitationRecord]:
    papers = [
        api_app.WORKSPACE.papers[paper_id] for paper_id in api_app.WORKSPACE.context.active_papers
    ]
    return citation_records_for_papers(papers)


@router.post("/citations/audit", response_model=CitationAudit)
async def audit_context_citations() -> CitationAudit:
    config = api_app.load_config()
    papers = [
        api_app.WORKSPACE.papers[paper_id] for paper_id in api_app.WORKSPACE.context.active_papers
    ]
    return await audit_citation_links_skill(papers, config)
