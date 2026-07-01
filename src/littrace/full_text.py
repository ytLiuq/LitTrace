from __future__ import annotations

import httpx

from littrace.config import LitTraceConfig
from littrace.citations import best_access_url
from littrace.models import (
    AccessType,
    FullTextCandidate,
    FullTextResolutionReport,
    LiteratureWorkspace,
    PaperMetadata,
)


async def resolve_workspace_full_text(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> LiteratureWorkspace:
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    reports = await resolve_full_text_for_papers(papers, config)
    for report in reports:
        workspace.full_text_reports[report.paper_id] = report
        paper = workspace.papers[report.paper_id]
        if report.best_pdf_url and not paper.pdf_url:
            paper.pdf_url = report.best_pdf_url
            paper.access_type = AccessType.OPEN_ACCESS
    return workspace


async def resolve_full_text_for_papers(
    papers: list[PaperMetadata],
    config: LitTraceConfig,
) -> list[FullTextResolutionReport]:
    timeout = httpx.Timeout(config.api.request_timeout_seconds)
    headers = {"User-Agent": config.api.user_agent}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        reports = []
        for paper in papers:
            reports.append(await resolve_full_text_for_paper(client, paper, config))
    return reports


async def resolve_full_text_for_paper(
    client: httpx.AsyncClient,
    paper: PaperMetadata,
    config: LitTraceConfig,
) -> FullTextResolutionReport:
    candidates = _seed_candidates(paper)
    warnings: list[str] = []
    if paper.doi and ".mock" not in paper.doi:
        crossref_candidates, crossref_warnings = await _crossref_full_text_candidates(client, paper)
        candidates.extend(crossref_candidates)
        warnings.extend(crossref_warnings)
        if config.api.unpaywall_email:
            unpaywall_candidates, unpaywall_warnings = await _unpaywall_candidates(
                client, paper, config.api.unpaywall_email
            )
            candidates.extend(unpaywall_candidates)
            warnings.extend(unpaywall_warnings)
    candidates = _dedupe_candidates(candidates)
    return _build_report(paper, candidates, warnings)


def _seed_candidates(paper: PaperMetadata) -> list[FullTextCandidate]:
    candidates: list[FullTextCandidate] = []
    if paper.pdf_url:
        candidates.append(
            FullTextCandidate(
                paper_id=paper.paper_id,
                url=paper.pdf_url,
                source="paper.pdf_url",
                content_type="pdf",
                access_type=AccessType.OPEN_ACCESS,
                is_pdf=True,
                confidence=0.9,
            )
        )
    if paper.doi:
        candidates.append(
            FullTextCandidate(
                paper_id=paper.paper_id,
                url=f"https://doi.org/{paper.doi}",
                source="doi",
                content_type="landing_page",
                access_type=paper.access_type,
                requires_login=paper.access_type == AccessType.REQUIRES_LOGIN,
                confidence=0.65,
            )
        )
    for url in paper.source_urls:
        candidates.append(
            FullTextCandidate(
                paper_id=paper.paper_id,
                url=url,
                source="paper.source_url",
                content_type=_content_type_from_url(str(url)),
                access_type=paper.access_type,
                requires_login=paper.access_type == AccessType.REQUIRES_LOGIN,
                is_pdf=str(url).lower().endswith(".pdf"),
                confidence=0.55,
            )
        )
    if not candidates:
        candidates.append(
            FullTextCandidate(
                paper_id=paper.paper_id,
                url=best_access_url(paper),
                source="best_access_url",
                access_type=paper.access_type,
                requires_login=paper.access_type == AccessType.REQUIRES_LOGIN,
                confidence=0.3,
            )
        )
    return candidates


async def _crossref_full_text_candidates(
    client: httpx.AsyncClient,
    paper: PaperMetadata,
) -> tuple[list[FullTextCandidate], list[str]]:
    warnings: list[str] = []
    if not paper.doi:
        return [], warnings
    try:
        response = await client.get(f"https://api.crossref.org/works/{paper.doi}")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return [], [f"crossref_full_text: {exc.__class__.__name__}: {exc}"]
    item = response.json().get("message", {})
    candidates: list[FullTextCandidate] = []
    resource_url = (item.get("resource") or {}).get("primary", {}).get("URL")
    if resource_url:
        candidates.append(
            FullTextCandidate(
                paper_id=paper.paper_id,
                url=resource_url,
                source="crossref.resource",
                content_type="landing_page",
                access_type=paper.access_type,
                requires_login=_looks_gated(resource_url, paper),
                confidence=0.7,
            )
        )
    for link in item.get("link") or []:
        url = link.get("URL")
        if not url:
            continue
        content_type = str(link.get("content-type") or _content_type_from_url(url))
        candidates.append(
            FullTextCandidate(
                paper_id=paper.paper_id,
                url=url,
                source="crossref.link",
                content_type=content_type,
                access_type=AccessType.REQUIRES_LOGIN
                if _looks_gated(url, paper)
                else AccessType.OPEN_ACCESS,
                requires_login=_looks_gated(url, paper),
                is_pdf="pdf" in content_type.lower() or url.lower().endswith(".pdf"),
                is_xml="xml" in content_type.lower() or url.lower().endswith(".xml"),
                confidence=0.78,
            )
        )
    return candidates, warnings


async def _unpaywall_candidates(
    client: httpx.AsyncClient,
    paper: PaperMetadata,
    email: str,
) -> tuple[list[FullTextCandidate], list[str]]:
    warnings: list[str] = []
    if not paper.doi:
        return [], warnings
    try:
        response = await client.get(
            f"https://api.unpaywall.org/v2/{paper.doi}",
            params={"email": email},
        )
        if response.status_code == 404:
            return [], warnings
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return [], [f"unpaywall: {exc.__class__.__name__}: {exc}"]
    data = response.json()
    locations = []
    if data.get("best_oa_location"):
        locations.append(("unpaywall.best_oa_location", data["best_oa_location"]))
    for location in data.get("oa_locations") or []:
        locations.append(("unpaywall.oa_location", location))

    candidates: list[FullTextCandidate] = []
    for source, location in locations:
        for key in ("url_for_pdf", "url"):
            url = location.get(key)
            if not url:
                continue
            candidates.append(
                FullTextCandidate(
                    paper_id=paper.paper_id,
                    url=url,
                    source=source,
                    content_type="pdf" if key == "url_for_pdf" else "landing_page",
                    access_type=AccessType.OPEN_ACCESS,
                    requires_login=False,
                    is_pdf=key == "url_for_pdf" or url.lower().endswith(".pdf"),
                    confidence=0.92 if key == "url_for_pdf" else 0.82,
                )
            )
    return candidates, warnings


def _dedupe_candidates(candidates: list[FullTextCandidate]) -> list[FullTextCandidate]:
    by_url: dict[str, FullTextCandidate] = {}
    for candidate in candidates:
        key = str(candidate.url)
        existing = by_url.get(key)
        if existing is None or candidate.confidence > existing.confidence:
            by_url[key] = candidate
    return sorted(by_url.values(), key=lambda item: item.confidence, reverse=True)


def _build_report(
    paper: PaperMetadata,
    candidates: list[FullTextCandidate],
    warnings: list[str],
) -> FullTextResolutionReport:
    pdfs = [
        candidate
        for candidate in candidates
        if candidate.is_pdf and candidate.access_type == AccessType.OPEN_ACCESS
    ]
    landings = [candidate for candidate in candidates if not candidate.is_pdf]
    return FullTextResolutionReport(
        paper_id=paper.paper_id,
        doi=paper.doi,
        candidates=candidates,
        best_pdf_url=pdfs[0].url if pdfs else None,
        best_landing_url=landings[0].url if landings else None,
        open_access_candidate_count=sum(
            candidate.access_type == AccessType.OPEN_ACCESS for candidate in candidates
        ),
        login_required_candidate_count=sum(candidate.requires_login for candidate in candidates),
        warnings=warnings,
    )


def _content_type_from_url(url: str) -> str:
    lowered = url.lower()
    if lowered.endswith(".pdf") or "/pdf/" in lowered:
        return "pdf"
    if lowered.endswith(".xml") or "full-xml" in lowered:
        return "xml"
    return "landing_page"


def _looks_gated(url: str, paper: PaperMetadata) -> bool:
    lowered = f"{url} {paper.publisher or ''} {paper.journal or ''}".lower()
    gated_markers = [
        "pubs.acs.org",
        "onlinelibrary.wiley.com",
        "advanced.onlinelibrary.wiley.com",
        "nature.com",
        "sciencedirect.com",
        "springer",
    ]
    return any(marker in lowered for marker in gated_markers)
