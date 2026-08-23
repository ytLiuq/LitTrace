from __future__ import annotations

from fastapi import APIRouter

from littrace.api.backend import api_app
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



router = APIRouter()


@router.get("/context", response_model=LiteratureWorkspace)
def get_context() -> LiteratureWorkspace:
    return api_app.WORKSPACE


@router.patch("/context", response_model=LiteratureWorkspace)
def update_context(update: ContextUpdate) -> LiteratureWorkspace:
    from littrace.context import apply_context_update

    api_app._set_workspace(apply_context_update(api_app.WORKSPACE, update))
    return api_app.WORKSPACE


@router.post("/full-text/resolve", response_model=dict[str, object])
async def full_text_resolve() -> dict[str, object]:
    api_app._set_workspace(
        await resolve_workspace_full_text_skill(api_app.WORKSPACE, api_app.load_config())
    )
    return api_app.WORKSPACE.full_text_reports


@router.post("/papers/backfill-dois", response_model=LiteratureWorkspace)
async def papers_backfill_dois(request: DOIBackfillRequest) -> LiteratureWorkspace:
    api_app._set_workspace(
        await backfill_workspace_by_dois(api_app.WORKSPACE, request.dois, api_app.load_config())
    )
    return api_app.WORKSPACE


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
