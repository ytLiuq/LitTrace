from __future__ import annotations

import re
from dataclasses import dataclass

from littrace.config import LitTraceConfig
from littrace.context import _merge_filters
from littrace.retrieval.full_text import resolve_full_text_for_papers
from littrace.models import (
    AccessType,
    DownloadExecutionRequest,
    LiteratureWorkspace,
    PaperMetadata,
    PaperSearchRequest,
    ParsedPaper,
    coerce_parsed,
)
from littrace.evidence.parsing import local_pdf_path
from littrace.retrieval.search import build_query_variants, rank_papers
from littrace.skill_runner import execute_downloads_skill, parse_workspace_skill
from littrace.tool_contracts import ToolCallContext, ToolExecutionLedger, ToolExecutionPolicy


@dataclass
class FullTextContextResult:
    workspace: LiteratureWorkspace
    candidate_count: int
    valid_candidate_count: int
    resolved_count: int
    downloaded_count: int
    parsed_count: int
    active_count: int
    rejected_count: int
    warnings: list[str]


async def build_full_text_context(
    workspace: LiteratureWorkspace,
    request: PaperSearchRequest,
    config: LitTraceConfig,
    *,
    context: ToolCallContext | None = None,
    ledger: ToolExecutionLedger | None = None,
    policy: ToolExecutionPolicy | None = None,
) -> FullTextContextResult:
    candidates = _candidate_papers(workspace)
    valid = _valid_candidate_papers(candidates)
    warnings: list[str] = []
    rejected_count = len(candidates) - len(valid)
    if rejected_count:
        warnings.append(
            f"Rejected {rejected_count} empty or invalid candidate papers before full-text retrieval."
        )

    reports = await resolve_full_text_for_papers(valid, config)
    for report in reports:
        workspace.full_text_reports[report.paper_id] = report
        paper = workspace.papers[report.paper_id]
        if report.best_pdf_url:
            paper.pdf_url = report.best_pdf_url
            paper.access_type = AccessType.OPEN_ACCESS
        elif report.login_required_candidate_count:
            paper.access_type = AccessType.REQUIRES_LOGIN

    downloadable = [
        paper for paper in valid if paper.access_type == AccessType.OPEN_ACCESS and paper.pdf_url
    ]
    if downloadable:
        tool_kwargs = {
            key: value
            for key, value in {
                "context": context,
                "ledger": ledger,
                "policy": policy,
            }.items()
            if value is not None
        }
        download_result = await execute_downloads_skill(
            config,
            workspace,
            DownloadExecutionRequest(
                paper_ids=[paper.paper_id for paper in downloadable],
                dry_run=False,
            ),
            **tool_kwargs,
        )
        for item in download_result.items:
            if item.error:
                warnings.append(f"download:{item.paper_id}: {item.error}")

    downloaded_ids = [paper.paper_id for paper in valid if local_pdf_path(config, paper).exists()]
    original_active = list(workspace.context.active_papers)
    workspace.context.active_papers = downloaded_ids
    if downloaded_ids:
        tool_kwargs = {
            key: value
            for key, value in {
                "context": context,
                "ledger": ledger,
                "policy": policy,
            }.items()
            if value is not None
        }
        workspace, parse_report = await parse_workspace_skill(
            workspace,
            config,
            **tool_kwargs,
        )
    else:
        parse_report = {"parsed_count": 0}

    ranked = _rank_full_text_papers(
        [workspace.papers[paper_id] for paper_id in downloaded_ids],
        request,
        workspace,
        config,
    )
    active_limit = config.literature_context.active_context_limit
    workspace.context.active_papers = [paper.paper_id for paper in ranked[:active_limit]]
    _merge_filters(
        workspace.context.filters,
        {
            "full_text_context_policy": "validate_retrieve_parse_rank",
            "candidate_pool_count": len(candidates),
            "valid_candidate_count": len(valid),
            "full_text_resolved_count": sum(1 for report in reports if report.best_pdf_url),
            "downloaded_full_text_count": len(downloaded_ids),
            "parsed_full_text_count": int(parse_report.get("parsed_count") or 0),
            "active_context_count": len(workspace.context.active_papers),
            "active_context_source": "downloaded_full_text_pdfs",
            "pre_full_text_active_papers": original_active,
            "requires_login_candidate_ids": [
                paper.paper_id for paper in valid if paper.access_type == AccessType.REQUIRES_LOGIN
            ],
        },
    )
    if not workspace.context.active_papers:
        warnings.append("No downloaded full-text PDFs are available for the current context.")
    return FullTextContextResult(
        workspace=workspace,
        candidate_count=len(candidates),
        valid_candidate_count=len(valid),
        resolved_count=sum(1 for report in reports if report.best_pdf_url),
        downloaded_count=len(downloaded_ids),
        parsed_count=int(parse_report.get("parsed_count") or 0),
        active_count=len(workspace.context.active_papers),
        rejected_count=rejected_count,
        warnings=warnings,
    )


def _candidate_papers(workspace: LiteratureWorkspace) -> list[PaperMetadata]:
    ids = getattr(workspace.context.filters, "candidate_pool_ids", None)
    if not isinstance(ids, list):
        ids = list(workspace.papers)
    return [
        workspace.papers[paper_id]
        for paper_id in ids
        if isinstance(paper_id, str) and paper_id in workspace.papers
    ]


def _valid_candidate_papers(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    valid: list[PaperMetadata] = []
    seen: set[str] = set()
    for paper in papers:
        if not paper.title.strip():
            continue
        if not paper.doi and not paper.source_urls and not paper.pdf_url:
            continue
        key = (paper.doi or paper.title).lower()
        if key in seen:
            continue
        seen.add(key)
        valid.append(paper)
    return valid


def _rank_full_text_papers(
    papers: list[PaperMetadata],
    request: PaperSearchRequest,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> list[PaperMetadata]:
    ranked = rank_papers(list(papers), request)
    for paper in ranked:
        _raw = workspace.parsed_papers.get(paper.paper_id)
        _parsed = coerce_parsed(_raw) if _raw is not None else None
        parsed_score = (
            _parsed_content_relevance(request.topic, _parsed) if _parsed is not None else 0.0
        )
        full_text_bonus = 0.18 if local_pdf_path(config, paper).exists() else 0.0
        paper.relevance_score = min(
            1.0,
            0.74 * (paper.relevance_score or 0.0) + 0.22 * parsed_score + full_text_bonus,
        )
    return sorted(
        ranked,
        key=lambda paper: (
            paper.relevance_score or 0.0,
            paper.year or 0,
            paper.citation_count or 0,
        ),
        reverse=True,
    )


def _parsed_content_relevance(topic: str, parsed: ParsedPaper) -> float:
    sections = parsed.sections if hasattr(parsed, "sections") else None
    if not isinstance(sections, list) or not sections:
        return 0.0
    query_tokens = _tokens(" ".join(build_query_variants(topic)))
    if not query_tokens:
        return 0.0
    text = " ".join(
        str(section.get("text") or "") for section in sections if isinstance(section, dict)
    )
    candidate_tokens = _tokens(text[:12000])
    if not candidate_tokens:
        return 0.0
    return min(1.0, len(query_tokens & candidate_tokens) / max(len(query_tokens), 1))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", text.lower())
        if len(token) > 2
        and token
        not in {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "paper",
            "study",
            "research",
        }
    }
