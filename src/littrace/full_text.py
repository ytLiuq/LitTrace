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
from littrace.search import (
    _crossref_authors,
    _crossref_year,
    _first,
    _normalize_doi,
    _paper_id,
    _strip_crossref_abstract,
)


def full_text_config_warnings(config: LitTraceConfig) -> list[str]:
    warnings = []
    if not config.api.unpaywall_email:
        warnings.append("Set api.unpaywall_email to improve OA full-text PDF discovery.")
    if not config.api.crossref_mailto:
        warnings.append("Set api.crossref_mailto for more reliable Crossref polite-pool access.")
    if "mailto:" not in config.api.user_agent.lower():
        warnings.append("Add mailto:your-email@example.com to api.user_agent for API etiquette.")
    return warnings


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
    candidates, verify_warnings = await verify_full_text_candidates(client, candidates)
    warnings.extend(verify_warnings)
    return _build_report(paper, candidates, warnings)


async def backfill_workspace_by_dois(
    workspace: LiteratureWorkspace,
    dois: list[str],
    config: LitTraceConfig,
) -> LiteratureWorkspace:
    existing = {paper.doi.lower() for paper in workspace.papers.values() if paper.doi}
    missing = [doi for doi in dois if doi.lower() not in existing]
    timeout = httpx.Timeout(config.api.request_timeout_seconds)
    headers = {"User-Agent": config.api.user_agent}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for doi in missing:
            paper = await fetch_crossref_paper_by_doi(client, doi)
            if paper is None:
                continue
            workspace.papers[paper.paper_id] = paper
            if paper.paper_id not in workspace.context.active_papers:
                workspace.context.active_papers.append(paper.paper_id)
    return workspace


async def fetch_crossref_paper_by_doi(
    client: httpx.AsyncClient,
    doi: str,
) -> PaperMetadata | None:
    try:
        response = await client.get(f"https://api.crossref.org/works/{doi}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    item = response.json().get("message", {})
    title = _first(item.get("title"))
    if not title:
        return None
    normalized_doi = _normalize_doi(item.get("DOI") or doi)
    source_urls = [item.get("URL")] if item.get("URL") else [f"https://doi.org/{normalized_doi}"]
    probe = PaperMetadata(
        paper_id="probe",
        title=title,
        publisher=item.get("publisher"),
        journal=_first(item.get("container-title")),
    )
    return PaperMetadata(
        paper_id=_paper_id(normalized_doi, title),
        title=title,
        authors=_crossref_authors(item),
        year=_crossref_year(item),
        journal=_first(item.get("container-title")),
        publisher=item.get("publisher"),
        doi=normalized_doi,
        abstract=_strip_crossref_abstract(item.get("abstract")),
        citation_count=item.get("is-referenced-by-count"),
        source_urls=source_urls,
        access_type=AccessType.REQUIRES_LOGIN
        if _looks_gated(str(source_urls[0]), probe)
        else AccessType.METADATA_ONLY,
    )


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


async def verify_full_text_candidates(
    client: httpx.AsyncClient,
    candidates: list[FullTextCandidate],
    max_candidates: int = 8,
) -> tuple[list[FullTextCandidate], list[str]]:
    verified: list[FullTextCandidate] = []
    warnings: list[str] = []
    for index, candidate in enumerate(candidates):
        if index >= max_candidates:
            verified.append(candidate)
            continue
        try:
            verified.append(await _verify_candidate(client, candidate))
        except httpx.HTTPError as exc:
            update = candidate.model_dump()
            update["note"] = f"verify_failed: {exc.__class__.__name__}"
            verified.append(FullTextCandidate.model_validate(update))
            warnings.append(f"verify_candidate: {exc.__class__.__name__}: {candidate.url}")
    return verified, warnings


async def _verify_candidate(
    client: httpx.AsyncClient,
    candidate: FullTextCandidate,
) -> FullTextCandidate:
    response = await client.head(str(candidate.url), follow_redirects=True)
    if response.status_code == 405:
        response = await client.get(str(candidate.url), follow_redirects=True)
    content_type = response.headers.get("content-type", "")
    update = candidate.model_dump()
    update["status_code"] = response.status_code
    update["checked_content_type"] = content_type
    update["verified"] = 200 <= response.status_code < 400
    if "pdf" in content_type.lower():
        update["is_pdf"] = True
        update["content_type"] = content_type
    if "xml" in content_type.lower():
        update["is_xml"] = True
        update["content_type"] = content_type
    if response.status_code in {401, 402, 403}:
        if _has_open_access_evidence(candidate):
            update["requires_login"] = False
            update["access_type"] = AccessType.OPEN_ACCESS
            update["note"] = "oa_evidence_but_http_forbidden"
        elif _is_doi_resolver_candidate(candidate):
            update["requires_login"] = False
            update["note"] = "doi_resolver_http_forbidden"
        else:
            update["requires_login"] = True
            update["access_type"] = AccessType.REQUIRES_LOGIN
    return FullTextCandidate.model_validate(update)


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
        verified_candidate_count=sum(candidate.verified for candidate in candidates),
        warnings=warnings,
    )


def _content_type_from_url(url: str) -> str:
    lowered = url.lower()
    if lowered.endswith(".pdf") or "/pdf/" in lowered:
        return "pdf"
    if lowered.endswith(".xml") or "full-xml" in lowered:
        return "xml"
    return "landing_page"


def _has_open_access_evidence(candidate: FullTextCandidate) -> bool:
    if candidate.access_type == AccessType.OPEN_ACCESS and candidate.source.startswith("unpaywall"):
        return True
    if candidate.source.startswith("unpaywall"):
        return True
    lowered = str(candidate.url).lower()
    oa_hosts = [
        "mdpi.com",
        "pubs.rsc.org",
        "link.springer.com/content/pdf",
        "frontiersin.org",
        "plos.org",
        "arxiv.org/pdf",
    ]
    return candidate.access_type == AccessType.OPEN_ACCESS and any(host in lowered for host in oa_hosts)


def _is_doi_resolver_candidate(candidate: FullTextCandidate) -> bool:
    return candidate.source == "doi" and "doi.org/" in str(candidate.url).lower()


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
