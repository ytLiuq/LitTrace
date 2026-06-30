from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from littrace.config import LitTraceConfig
from littrace.models import AccessType, PaperMetadata, PaperSearchRequest, PaperSearchResult


class PaperSearchClient(Protocol):
    name: str

    async def search(self, request: PaperSearchRequest) -> PaperSearchResult:
        """Search one source and return normalized metadata."""


@dataclass
class SearchDiagnostics:
    live_attempted: bool = False
    used_fallback: bool = False
    source_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class MockMaterialsSearchClient:
    name = "mock_materials_search"

    async def search(self, request: PaperSearchRequest) -> PaperSearchResult:
        seed = _slug(request.topic)
        papers = [
            PaperMetadata(
                paper_id=f"{seed}-wiley-2026",
                title=f"{request.topic}: stability-focused flexible materials study",
                authors=["Example Author", "Second Author"],
                year=2026,
                journal="Advanced Functional Materials",
                publisher="Wiley",
                doi="10.1002/adfm.mock2026001",
                abstract="Mock metadata for workflow development.",
                source_urls=["https://doi.org/10.1002/adfm.mock2026001"],
                access_type=AccessType.REQUIRES_LOGIN,
                relevance_score=0.92,
                recency_score=1.0,
            ),
            PaperMetadata(
                paper_id=f"{seed}-acs-2025",
                title=f"{request.topic}: nanoscale mechanism and performance comparison",
                authors=["ACS Example"],
                year=2025,
                journal="ACS Nano",
                publisher="American Chemical Society",
                doi="10.1021/acsnano.mock2025001",
                abstract="Mock ACS-style metadata for table extraction planning.",
                source_urls=["https://doi.org/10.1021/acsnano.mock2025001"],
                access_type=AccessType.REQUIRES_LOGIN,
                relevance_score=0.88,
                recency_score=0.86,
            ),
            PaperMetadata(
                paper_id=f"{seed}-mdpi-2024",
                title=f"Open-access review of {request.topic}",
                authors=["Open Access Author"],
                year=2024,
                journal="Nanomaterials",
                publisher="MDPI",
                doi="10.3390/nano.mock2024001",
                abstract="Mock open-access paper for download planning.",
                source_urls=["https://doi.org/10.3390/nano.mock2024001"],
                pdf_url="https://example.org/mock-paper.pdf",
                access_type=AccessType.OPEN_ACCESS,
                relevance_score=0.81,
                recency_score=0.72,
            ),
        ]
        if request.year_min is not None:
            papers = [paper for paper in papers if paper.year is None or paper.year >= request.year_min]
        return PaperSearchResult(request=request, papers=papers[: request.limit])


class LiveSearchClient:
    name = "live_search"

    def __init__(self, config: LitTraceConfig):
        self.config = config
        self.diagnostics = SearchDiagnostics(live_attempted=True)

    async def search(self, request: PaperSearchRequest) -> PaperSearchResult:
        timeout = httpx.Timeout(self.config.api.request_timeout_seconds)
        headers = {"User-Agent": self.config.api.user_agent}
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            openalex, crossref = await _gather_named(
                {
                    "openalex": self._search_openalex(client, request),
                    "crossref": self._search_crossref(client, request),
                },
                self.diagnostics,
            )
            self.diagnostics.source_counts["openalex"] = len(openalex)
            self.diagnostics.source_counts["crossref"] = len(crossref)

            merged = merge_papers([*openalex, *crossref])
            if self.config.api.unpaywall_email:
                merged = await self._enrich_unpaywall(client, merged)
                self.diagnostics.source_counts["unpaywall_enriched"] = len(
                    [paper for paper in merged if paper.access_type == AccessType.OPEN_ACCESS]
                )
            merged = rank_papers(merged, request)
            return PaperSearchResult(request=request, papers=merged[: request.limit])

    async def _search_openalex(
        self,
        client: httpx.AsyncClient,
        request: PaperSearchRequest,
    ) -> list[PaperMetadata]:
        params: dict[str, str | int] = {
            "search": request.topic,
            "per-page": min(max(request.limit, 10), 50),
            "sort": "publication_date:desc" if request.wants_recent else "relevance_score:desc",
        }
        filters = []
        if request.year_min is not None:
            filters.append(f"from_publication_date:{request.year_min}-01-01")
        if filters:
            params["filter"] = ",".join(filters)
        if self.config.api.openalex_api_key:
            params["api_key"] = self.config.api.openalex_api_key

        response = await client.get("https://api.openalex.org/works", params=params)
        response.raise_for_status()
        data = response.json()
        papers = []
        for item in data.get("results", []):
            title = item.get("title")
            if not title:
                continue
            doi = _normalize_doi(item.get("doi"))
            best_oa = (item.get("open_access") or {}).get("oa_url")
            source = item.get("primary_location") or {}
            source_info = source.get("source") or {}
            authorships = item.get("authorships") or []
            authors = [
                (authorship.get("author") or {}).get("display_name")
                for authorship in authorships
            ]
            source_urls = [url for url in [item.get("id"), item.get("doi")] if url]
            paper = PaperMetadata(
                paper_id=_paper_id(doi, title),
                title=title,
                authors=[author for author in authors if author],
                year=item.get("publication_year"),
                journal=source_info.get("display_name"),
                publisher=source_info.get("host_organization_name"),
                doi=doi,
                abstract=_openalex_abstract(item.get("abstract_inverted_index")),
                citation_count=item.get("cited_by_count"),
                source_urls=source_urls,
                pdf_url=best_oa,
                access_type=AccessType.OPEN_ACCESS if best_oa else AccessType.METADATA_ONLY,
                relevance_score=_bounded(item.get("relevance_score")),
            )
            papers.append(paper)
        return papers

    async def _search_crossref(
        self,
        client: httpx.AsyncClient,
        request: PaperSearchRequest,
    ) -> list[PaperMetadata]:
        params: dict[str, str | int] = {
            "query.bibliographic": request.topic,
            "rows": min(max(request.limit, 10), 50),
            "sort": "published" if request.wants_recent else "relevance",
            "order": "desc",
        }
        filters = []
        if request.year_min is not None:
            filters.append(f"from-pub-date:{request.year_min}-01-01")
        if filters:
            params["filter"] = ",".join(filters)
        if self.config.api.crossref_mailto:
            params["mailto"] = self.config.api.crossref_mailto

        response = await client.get("https://api.crossref.org/works", params=params)
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        papers = []
        for item in items:
            title = _first(item.get("title"))
            if not title:
                continue
            doi = _normalize_doi(item.get("DOI"))
            year = _crossref_year(item)
            source_urls = [item.get("URL")] if item.get("URL") else []
            papers.append(
                PaperMetadata(
                    paper_id=_paper_id(doi, title),
                    title=title,
                    authors=_crossref_authors(item),
                    year=year,
                    journal=_first(item.get("container-title")),
                    publisher=item.get("publisher"),
                    doi=doi,
                    abstract=_strip_crossref_abstract(item.get("abstract")),
                    citation_count=item.get("is-referenced-by-count"),
                    source_urls=source_urls,
                    access_type=AccessType.METADATA_ONLY,
                )
            )
        return papers

    async def _enrich_unpaywall(
        self,
        client: httpx.AsyncClient,
        papers: list[PaperMetadata],
    ) -> list[PaperMetadata]:
        for paper in papers:
            if not paper.doi:
                continue
            response = await client.get(
                f"https://api.unpaywall.org/v2/{paper.doi}",
                params={"email": self.config.api.unpaywall_email},
            )
            if response.status_code == 404:
                continue
            response.raise_for_status()
            data = response.json()
            best = data.get("best_oa_location") or {}
            pdf_url = best.get("url_for_pdf")
            landing_url = best.get("url")
            if data.get("is_oa") and (pdf_url or landing_url):
                paper.pdf_url = pdf_url or landing_url
                paper.access_type = AccessType.OPEN_ACCESS
                if landing_url and landing_url not in [str(url) for url in paper.source_urls]:
                    paper.source_urls.append(landing_url)
            elif paper.access_type == AccessType.METADATA_ONLY and _publisher_requires_login(paper):
                paper.access_type = AccessType.REQUIRES_LOGIN
        return papers


async def _gather_named(
    coroutines: dict[str, object],
    diagnostics: SearchDiagnostics,
) -> list[list[PaperMetadata]]:
    results: list[list[PaperMetadata]] = []
    for name, coroutine in coroutines.items():
        try:
            results.append(await coroutine)
        except httpx.HTTPError as exc:
            diagnostics.errors.append(f"{name}: {exc.__class__.__name__}: {exc}")
            results.append([])
    return results


def merge_papers(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    merged: dict[str, PaperMetadata] = {}
    for paper in papers:
        key = paper.doi.lower() if paper.doi else _title_key(paper.title)
        if paper.doi is None:
            fuzzy_key = _find_fuzzy_title_key(merged, paper)
            if fuzzy_key:
                key = fuzzy_key
        existing = merged.get(key)
        if existing is None:
            merged[key] = paper
            continue
        merged[key] = _merge_paper(existing, paper)
    return list(merged.values())


def _find_fuzzy_title_key(merged: dict[str, PaperMetadata], paper: PaperMetadata) -> str | None:
    candidate = _title_tokens(paper.title)
    if not candidate:
        return None
    for key, existing in merged.items():
        if paper.year and existing.year and abs(paper.year - existing.year) > 1:
            continue
        existing_tokens = _title_tokens(existing.title)
        overlap = len(candidate & existing_tokens) / max(len(candidate | existing_tokens), 1)
        same_first_author = (
            bool(paper.authors)
            and bool(existing.authors)
            and paper.authors[0].split()[-1].lower() == existing.authors[0].split()[-1].lower()
        )
        if overlap >= 0.82 or (overlap >= 0.68 and same_first_author):
            return key
    return None


def rank_papers(papers: list[PaperMetadata], request: PaperSearchRequest) -> list[PaperMetadata]:
    current_year = 2026
    for paper in papers:
        paper.recency_score = _recency_score(paper.year, current_year)
        citation_score = math.log1p(paper.citation_count or 0) / 10
        relevance = paper.relevance_score or 0.5
        access = 1.0 if paper.access_type == AccessType.OPEN_ACCESS else 0.4
        paper.relevance_score = min(
            1.0,
            0.35 * relevance + 0.35 * paper.recency_score + 0.2 * citation_score + 0.1 * access,
        )
    return sorted(
        papers,
        key=lambda paper: (
            paper.relevance_score or 0,
            paper.year or 0,
            paper.citation_count or 0,
        ),
        reverse=True,
    )


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in slug.split("-") if part)[:48] or "paper"


def _paper_id(doi: str | None, title: str) -> str:
    if doi:
        return doi.replace("/", "_").replace(":", "_").lower()
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    return f"{_slug(title)[:40]}-{digest}"


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = value.strip()
    doi = doi.removeprefix("https://doi.org/")
    doi = doi.removeprefix("http://doi.org/")
    return doi.lower()


def _first(value: list[str] | str | None) -> str | None:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _openalex_abstract(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        words.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(words))


def _crossref_year(item: dict) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        parts = (item.get(key) or {}).get("date-parts") or []
        if parts and parts[0]:
            return parts[0][0]
    return None


def _crossref_authors(item: dict) -> list[str]:
    authors = []
    for author in item.get("author") or []:
        given = author.get("given")
        family = author.get("family")
        name = " ".join(part for part in [given, family] if part)
        if name:
            authors.append(name)
    return authors


def _strip_crossref_abstract(value: str | None) -> str | None:
    if not value:
        return None
    return value.replace("<jats:p>", "").replace("</jats:p>", "").strip()


def _bounded(value: float | int | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(float(value), 1.0))


def _title_key(title: str) -> str:
    return _slug(title).lower()


def _title_tokens(title: str) -> set[str]:
    stopwords = {"the", "a", "an", "of", "and", "for", "with", "by", "on", "in", "to"}
    return {
        token
        for token in re.split(r"[^a-z0-9]+", title.lower())
        if len(token) > 2 and token not in stopwords
    }


def _merge_paper(left: PaperMetadata, right: PaperMetadata) -> PaperMetadata:
    data = left.model_dump()
    for attr_name in (
        "authors",
        "year",
        "journal",
        "publisher",
        "doi",
        "abstract",
        "citation_count",
        "pdf_url",
        "relevance_score",
        "recency_score",
    ):
        if data.get(attr_name) in (None, [], "") and getattr(right, attr_name):
            data[attr_name] = getattr(right, attr_name)
    data["source_urls"] = list({str(url) for url in [*left.source_urls, *right.source_urls]})
    if right.access_type == AccessType.OPEN_ACCESS:
        data["access_type"] = AccessType.OPEN_ACCESS
    elif left.access_type != AccessType.OPEN_ACCESS and right.access_type == AccessType.REQUIRES_LOGIN:
        data["access_type"] = AccessType.REQUIRES_LOGIN
    return PaperMetadata.model_validate(data)


def _recency_score(year: int | None, current_year: int) -> float:
    if year is None:
        return 0.2
    age = max(current_year - year, 0)
    half_life = 3
    return math.exp(-age / half_life)


def _publisher_requires_login(paper: PaperMetadata) -> bool:
    publisher = (paper.publisher or "").lower()
    journal = (paper.journal or "").lower()
    gated_markers = [
        "wiley",
        "american chemical society",
        "acs",
        "elsevier",
        "springer",
        "nature",
        "royal society of chemistry",
    ]
    return any(marker in publisher or marker in journal for marker in gated_markers)
