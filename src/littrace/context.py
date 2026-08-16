from __future__ import annotations

from littrace.models import (
    ContextUpdate,
    LiteratureContext,
    LiteratureWorkspace,
    PaperMetadata,
    WorkspaceFilters,
)
from littrace.identity import register_paper_identity
from littrace.retrieval.search import select_context_papers
from littrace.models import PaperSearchRequest


def _merge_filters(filters: WorkspaceFilters, updates: dict | WorkspaceFilters) -> None:
    """Merge a dict or WorkspaceFilters into an existing WorkspaceFilters in-place."""
    if isinstance(updates, WorkspaceFilters):
        updates = updates.model_dump(exclude_none=True)
    unknown = sorted(set(updates) - set(WorkspaceFilters.model_fields))
    if unknown:
        raise ValueError(f"Unknown workspace filter keys: {', '.join(unknown)}")
    validated = WorkspaceFilters.model_validate({**filters.model_dump(), **updates})
    for key in updates:
        setattr(filters, key, getattr(validated, key))


def add_papers(workspace: LiteratureWorkspace, papers: list[PaperMetadata]) -> LiteratureWorkspace:
    for paper in papers:
        workspace.papers[paper.paper_id] = paper
        register_paper_identity(workspace, paper)
        if (
            paper.paper_id not in workspace.context.active_papers
            and paper.paper_id not in workspace.context.excluded_papers
        ):
            workspace.context.active_papers.append(paper.paper_id)
    return workspace


def add_ranked_candidate_papers(
    workspace: LiteratureWorkspace,
    papers: list[PaperMetadata],
    request: PaperSearchRequest,
    active_limit: int = 15,
) -> LiteratureWorkspace:
    for paper in papers:
        workspace.papers[paper.paper_id] = paper
        register_paper_identity(workspace, paper)

    ranked_ids = [paper.paper_id for paper in papers]
    active = [
        paper.paper_id
        for paper in select_context_papers(papers, request, limit=active_limit)
        if paper.paper_id not in workspace.context.excluded_papers
    ]
    workspace.context.active_papers = active
    _merge_filters(
        workspace.context.filters,
        {
            "candidate_pool_ids": ranked_ids,
            "candidate_pool_count": len(ranked_ids),
            "active_context_limit": active_limit,
            "ranking_policy": "wide_recall_rank_then_top_context",
        },
    )
    return workspace


def apply_context_update(
    workspace: LiteratureWorkspace,
    update: ContextUpdate,
) -> LiteratureWorkspace:
    context = workspace.context

    if update.visible_to_user is not None:
        context.visible_to_user = update.visible_to_user
    if update.filters is not None:
        _merge_filters(context.filters, update.filters)

    for paper_id in update.include_paper_ids:
        _remove(context.excluded_papers, paper_id)
        _append_unique(context.active_papers, paper_id)

    for paper_id in update.exclude_paper_ids:
        _remove(context.active_papers, paper_id)
        _remove(context.pinned_papers, paper_id)
        _remove(context.selected_for_download, paper_id)
        _append_unique(context.excluded_papers, paper_id)

    for paper_id in update.pin_paper_ids:
        if paper_id in context.active_papers:
            _append_unique(context.pinned_papers, paper_id)

    for paper_id in update.unpin_paper_ids:
        _remove(context.pinned_papers, paper_id)

    for paper_id in update.select_for_download:
        if paper_id in workspace.papers and paper_id not in context.excluded_papers:
            _append_unique(context.selected_for_download, paper_id)

    for paper_id in update.deselect_for_download:
        _remove(context.selected_for_download, paper_id)

    return workspace


def visible_context(workspace: LiteratureWorkspace) -> LiteratureWorkspace | LiteratureContext:
    if workspace.context.visible_to_user:
        return workspace
    return workspace.context


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _remove(items: list[str], item: str) -> None:
    if item in items:
        items.remove(item)
