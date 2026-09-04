from __future__ import annotations

import hashlib
import asyncio
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Protocol

import httpx

from littrace.config import LitTraceConfig
from littrace.models import (
    AccessType,
    PaperMetadata,
    PaperSearchRequest,
    PaperSearchResult,
    TopicRetrievalPolicy,
)
from littrace.retrieval.adapters import SourceHealth, classify_source_exception
from littrace.retry import retry_async, RetryConfig, BackoffStrategy


class PaperSearchClient(Protocol):
    name: str

    async def search(self, request: PaperSearchRequest) -> PaperSearchResult:
        return await asyncio.wait_for(self._search_impl(request), timeout=180.0)

    async def _search_impl(self, request: PaperSearchRequest) -> PaperSearchResult:
        """Search one source and return normalized metadata."""


@dataclass
class SearchDiagnostics:
    live_attempted: bool = False
    used_fallback: bool = False
    source_counts: dict[str, int] = field(default_factory=dict)
    filtered_counts: dict[str, int] = field(default_factory=dict)
    ranking_counts: dict[str, int] = field(default_factory=dict)
    query_variants: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    source_health: dict[str, SourceHealth] = field(default_factory=dict)


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
            PaperMetadata(
                paper_id=f"{seed}-nature-2024",
                title=f"{request.topic}: interface design and device reliability",
                authors=["Nature Example"],
                year=2024,
                journal="Nature Communications",
                publisher="Springer Nature",
                doi="10.1038/s41467.mock2024001",
                abstract="Mock Nature-style metadata for mechanism and reliability discussion.",
                source_urls=["https://doi.org/10.1038/s41467.mock2024001"],
                access_type=AccessType.OPEN_ACCESS,
                relevance_score=0.79,
                recency_score=0.7,
            ),
            PaperMetadata(
                paper_id=f"{seed}-rsc-2023",
                title=f"{request.topic}: materials chemistry route for scalable sensors",
                authors=["RSC Example"],
                year=2023,
                journal="Journal of Materials Chemistry C",
                publisher="Royal Society of Chemistry",
                doi="10.1039/d3tc.mock001",
                abstract="Mock RSC metadata for materials chemistry source routing.",
                source_urls=["https://doi.org/10.1039/d3tc.mock001"],
                access_type=AccessType.REQUIRES_LOGIN,
                relevance_score=0.76,
                recency_score=0.62,
            ),
        ]
        if request.year_min is not None:
            papers = [
                paper for paper in papers if paper.year is None or paper.year >= request.year_min
            ]
        # Round 17: honour ``year_max`` too. ``None`` means "no
        # upper bound" (backward compat for pre-Round-17 callers).
        if request.year_max is not None:
            papers = [
                paper for paper in papers if paper.year is None or paper.year <= request.year_max
            ]
        return PaperSearchResult(request=request, papers=papers[: request.limit])

    async def fetch(self, request: PaperSearchRequest) -> PaperSearchResult:
        return await self.search(request)


class LiveSearchClient:
    name = "live_search"

    def __init__(
        self,
        config: LitTraceConfig,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ):
        self.config = config
        self.diagnostics = SearchDiagnostics(live_attempted=True)
        self.progress_callback = progress_callback

    def _progress(self, **event: object) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event)
        except Exception as exc:
            self.diagnostics.errors.append(
                f"progress_callback:{exc.__class__.__name__}: {exc}"
            )

    async def search(self, request: PaperSearchRequest) -> PaperSearchResult:
        timeout = httpx.Timeout(self.config.api.request_timeout_seconds)
        headers = {"User-Agent": self.config.api.user_agent}
        requests = _variant_requests(request)
        self.diagnostics.query_variants = [variant.topic for variant in requests]
        self._progress(stage="search_started", status="running", variant_count=len(requests))
        openalex_results: list[PaperMetadata] = []
        crossref_results: list[PaperMetadata] = []
        async with httpx.AsyncClient(
            timeout=timeout, headers=headers, follow_redirects=True
        ) as client:
            for index, variant_request in enumerate(requests, start=1):
                self._progress(
                    stage="search_variant_started", status="running",
                    variant=index, query=variant_request.topic,
                )
                sources = {
                    f"openalex_variant_{index}": self._search_openalex(client, variant_request),
                    f"crossref_variant_{index}": self._search_crossref(client, variant_request),
                }
                if self.config.api.enable_europe_pmc:
                    sources[f"europe_pmc_variant_{index}"] = self._search_europe_pmc(client, variant_request)
                if self.config.api.enable_arxiv:
                    sources[f"arxiv_variant_{index}"] = self._search_arxiv(client, variant_request)
                if self.config.api.core_api_key:
                    sources[f"core_variant_{index}"] = self._search_core(client, variant_request)
                source_results = await asyncio.wait_for(
                    _gather_named(sources, self.diagnostics), timeout=60.0
                )
                openalex_results.extend(source_results[0])
                crossref_results.extend(source_results[1])
                extra_results = [paper for source in source_results[2:] for paper in source]
                openalex_results.extend(extra_results)
                for source_name, source_papers in zip(sources, source_results):
                    self.diagnostics.source_counts[source_name] = len(source_papers)
                    self._progress(
                        stage="search_source_finished", status="finished",
                        variant=index, source=source_name, count=len(source_papers),
                    )
                all_results = [*openalex_results, *crossref_results]
                relevant_so_far = rank_papers(
                    filter_search_results(
                        merge_papers(all_results),
                        request,
                    ),
                    request,
                )
                context_ready_so_far = [
                    paper
                    for paper in relevant_so_far
                    if _context_relevance_score(request, paper) >= 0.45
                ]
                self.diagnostics.ranking_counts[f"context_ready_after_variant_{index}"] = len(
                    context_ready_so_far
                )
                self.diagnostics.filtered_counts[f"after_variant_{index}"] = len(all_results) - len(relevant_so_far)
                self.diagnostics.source_counts[f"relevant_after_variant_{index}"] = len(
                    relevant_so_far
                )
                if _has_enough_relevant_results(context_ready_so_far, request):
                    self._progress(
                        stage="search_variant_finished", status="finished",
                        variant=index, count=len(context_ready_so_far), early_stop=True,
                    )
                    break
                self._progress(
                    stage="search_variant_finished", status="finished",
                    variant=index, count=len(context_ready_so_far), early_stop=False,
                )

            self.diagnostics.source_counts["openalex"] = len(openalex_results)
            self.diagnostics.source_counts["crossref"] = len(crossref_results)

            merged_raw = merge_papers([*openalex_results, *crossref_results])
            merged = filter_search_results(merged_raw, request)
            self.diagnostics.filtered_counts["merged"] = len(merged_raw) - len(merged)
            self.diagnostics.filtered_counts["basic_candidate_pool"] = len(merged)
            if self.config.api.unpaywall_email:
                try:
                    merged = await asyncio.wait_for(
                        self._enrich_unpaywall(client, merged), timeout=45.0
                    )
                except asyncio.TimeoutError:
                    self.diagnostics.errors.append("unpaywall: timeout")
                    self.diagnostics.source_counts["unpaywall_enriched"] = len(
                        [paper for paper in merged if paper.access_type == AccessType.OPEN_ACCESS]
                    )
                    self._progress(
                        stage="search_unpaywall_finished", status="finished",
                        count=self.diagnostics.source_counts["unpaywall_enriched"],
                    )
            merged = rank_papers(merged, request)
            self.diagnostics.ranking_counts["ranked_candidate_pool"] = len(merged)
            self.diagnostics.ranking_counts["context_ready"] = len(
                [paper for paper in merged if _context_relevance_score(request, paper) >= 0.45]
            )
            result = PaperSearchResult(request=request, papers=merged[: request.limit])
            self._progress(
                stage="search_finished", status="finished", count=len(result.papers),
                source_counts=self.diagnostics.source_counts,
            )
            return result

    async def _search_europe_pmc(
        self, client: httpx.AsyncClient, request: PaperSearchRequest
    ) -> list[PaperMetadata]:
        target = min(max(request.limit, 25), 200)
        papers: list[PaperMetadata] = []
        cursor: str | None = "*"
        while cursor and len(papers) < target:
            response = await _get_with_retries(
                client,
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={
                    "query": request.topic,
                    "format": "json",
                    "pageSize": min(100, target),
                    "cursorMark": cursor,
                    "resultType": "core",
                },
                source="europe_pmc",
                diagnostics=self.diagnostics,
            )
            payload = response.json()
            results = payload.get("resultList", {}).get("result", [])
            if not results:
                break
            for item in results:
                try:
                    title = item.get("title")
                    if not title:
                        continue
                    doi = _normalize_doi(item.get("doi"))
                    full_text_payload = item.get("fullTextUrlList") or {}
                    full_text = (
                        full_text_payload.get("fullTextUrl", [])
                        if isinstance(full_text_payload, dict)
                        else []
                    )
                    if not isinstance(full_text, list):
                        full_text = []
                    pdf_url = next(
                        (
                            entry.get("url")
                            for entry in full_text
                            if isinstance(entry, dict)
                            and entry.get("documentStyle") == "pdf"
                        ),
                        None,
                    )
                    raw_id = item.get("id") or item.get("pmid")
                    source_urls = []
                    if raw_id:
                        raw_id_text = str(raw_id)
                        source_urls = [
                            raw_id_text
                            if raw_id_text.startswith(("http://", "https://"))
                            else f"https://europepmc.org/article/MED/{raw_id_text}"
                        ]
                    author_string = item.get("authorString")
                    papers.append(PaperMetadata(
                        paper_id=_paper_id(doi, str(title)), title=str(title),
                        authors=[author_string] if isinstance(author_string, str) else [],
                        year=_safe_year(item.get("pubYear")), journal=item.get("journalTitle"),
                        publisher="Europe PMC", doi=doi, abstract=item.get("abstractText"),
                        source_urls=source_urls, pdf_url=pdf_url,
                        access_type=AccessType.OPEN_ACCESS if pdf_url else AccessType.UNAVAILABLE,
                    ))
                except Exception as exc:  # noqa: BLE001 - isolate malformed source records
                    self.diagnostics.errors.append(
                        f"europe_pmc_record: {exc.__class__.__name__}: {exc}"
                    )
                    continue
            self._progress(
                stage="search_source_page", status="finished", source="europe_pmc",
                page=max(1, (len(papers) + min(100, target) - 1) // min(100, target)),
                count=len(results), total=len(papers),
            )
            cursor = payload.get("nextCursorMark")
        return papers[:target]

    async def _search_arxiv(
        self, client: httpx.AsyncClient, request: PaperSearchRequest
    ) -> list[PaperMetadata]:
        """Search arXiv's Atom API and normalize records into PaperMetadata.

        arXiv does not expose DOI for most preprints, so the stable arXiv
        identifier is used as the paper id and source URL.  Parsing is kept
        defensive: malformed entries are skipped individually so one record
        cannot make the whole source unavailable.
        """
        target = min(max(request.limit, 25), 200)
        response = await _get_with_retries(
            client,
            "https://export.arxiv.org/api/query",
            params={
                "search_query": _arxiv_search_query(request.topic),
                "start": 0,
                "max_results": target,
                "sortBy": "submittedDate" if request.wants_recent else "relevance",
                "sortOrder": "descending",
            },
            source="arxiv",
            diagnostics=self.diagnostics,
        )
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise ValueError(f"arXiv Atom parse failed: {exc}") from exc
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: list[PaperMetadata] = []
        for entry in root.findall("atom:entry", ns):
            try:
                title = " ".join((entry.findtext("atom:title", default="", namespaces=ns)).split())
                entry_id = (entry.findtext("atom:id", default="", namespaces=ns)).strip()
                if not title or not entry_id:
                    continue
                arxiv_id = entry_id.rstrip("/").rsplit("/", 1)[-1]
                pdf_url = None
                for link in entry.findall("atom:link", ns):
                    if link.attrib.get("title") == "pdf":
                        pdf_url = link.attrib.get("href")
                        break
                if not pdf_url:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
                published = entry.findtext("atom:published", default="", namespaces=ns)
                authors = [
                    " ".join((author.findtext("atom:name", default="", namespaces=ns)).split())
                    for author in entry.findall("atom:author", ns)
                ]
                papers.append(PaperMetadata(
                    paper_id=f"arxiv:{arxiv_id}",
                    title=title,
                    authors=[author for author in authors if author],
                    year=_safe_year(published),
                    journal="arXiv",
                    publisher="arXiv",
                    abstract=" ".join((entry.findtext("atom:summary", default="", namespaces=ns)).split()) or None,
                    source_urls=[entry_id],
                    pdf_url=pdf_url,
                    access_type=AccessType.OPEN_ACCESS,
                ))
            except Exception as exc:  # noqa: BLE001
                self.diagnostics.errors.append(
                    f"arxiv_record: {exc.__class__.__name__}: {exc}"
                )
        self._progress(
            stage="search_source_page", status="finished", source="arxiv",
            page=1, count=len(papers), total=len(papers),
        )
        return papers[:target]

    async def _search_core(
        self, client: httpx.AsyncClient, request: PaperSearchRequest
    ) -> list[PaperMetadata]:
        target = min(max(request.limit, 25), 200)
        response = await client.get(
            "https://api.core.ac.uk/v3/search/works",
            params={"q": request.topic, "limit": target, "offset": 0},
            headers={"Authorization": f"Bearer {self.config.api.core_api_key}"},
        )
        response.raise_for_status()
        papers: list[PaperMetadata] = []
        for item in response.json().get("results", []):
            title = item.get("title")
            if not title:
                continue
            doi = _normalize_doi(item.get("doi"))
            pdf_url = item.get("downloadUrl")
            papers.append(PaperMetadata(
                paper_id=_paper_id(doi, title), title=title,
                authors=[author.get("name") for author in item.get("authors", []) if author.get("name")],
                year=_safe_year(item.get("yearPublished")), publisher="CORE", doi=doi,
                abstract=item.get("abstract"), source_urls=[item.get("sourceFulltextUrls", [None])[0]] if item.get("sourceFulltextUrls") else [],
                pdf_url=pdf_url, access_type=AccessType.OPEN_ACCESS if pdf_url else AccessType.UNAVAILABLE,
            ))
        return papers

    async def fetch(self, request: PaperSearchRequest) -> PaperSearchResult:
        return await self.search(request)

    async def _search_openalex(
        self,
        client: httpx.AsyncClient,
        request: PaperSearchRequest,
    ) -> list[PaperMetadata]:
        per_page = 50
        target = min(max(request.limit, 50), 200)
        params: dict[str, str | int] = {
            "search": request.topic,
            "per-page": per_page,
            "sort": "publication_date:desc" if request.wants_recent else "relevance_score:desc",
        }
        filters = []
        if request.year_min is not None:
            filters.append(f"from_publication_date:{request.year_min}-01-01")
        # Round 17: surface the user-selected "至" year as an
        # OpenAlex ``to_publication_date`` filter so the upstream
        # API doesn't return papers newer than the user asked for.
        # Without this, the year range was half-applied (lower
        # bound only) and the watchlist's "year upper bound" was a
        # silent no-op.
        if request.year_max is not None:
            filters.append(f"to_publication_date:{request.year_max}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        if self.config.api.openalex_api_key:
            params["api_key"] = self.config.api.openalex_api_key

        papers = []
        for page in range(1, (target + per_page - 1) // per_page + 1):
            page_params = {**params, "page": page}
            response = await _get_with_retries(
                client, "https://api.openalex.org/works", params=page_params,
                source="openalex", diagnostics=self.diagnostics,
            )
            items = response.json().get("results", [])
            if not items:
                break
            papers.extend(self._openalex_papers(items))
            self._progress(
                stage="search_source_page", status="finished", source="openalex",
                page=page, count=len(items), total=len(papers),
            )
            if len(items) < per_page:
                break
        return papers[:target]

    def _openalex_papers(self, items: list[dict]) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        for item in items:
            title = item.get("title")
            if not title:
                continue
            doi = _normalize_doi(item.get("doi"))
            best_oa = (item.get("open_access") or {}).get("oa_url")
            source = item.get("primary_location") or {}
            source_info = source.get("source") or {}
            authorships = item.get("authorships") or []
            authors = [
                (authorship.get("author") or {}).get("display_name") for authorship in authorships
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
                access_type=AccessType.OPEN_ACCESS if best_oa else AccessType.UNAVAILABLE,
                relevance_score=_bounded(item.get("relevance_score")),
            )
            papers.append(paper)
        return papers

    async def _search_crossref(
        self,
        client: httpx.AsyncClient,
        request: PaperSearchRequest,
    ) -> list[PaperMetadata]:
        attempts = _crossref_attempt_params(request)
        for params in attempts:
            if request.year_min is not None:
                params["filter"] = f"from-pub-date:{request.year_min}-01-01"
            # Round 17: same as the openalex path — push the upper
            # bound through to crossref so the user's "至" year
            # actually reaches the upstream API. The crossref
            # ``until-pub-date`` filter is the documented counterpart
            # of ``from-pub-date``.
            if request.year_max is not None:
                existing = params.get("filter")
                until = f"until-pub-date:{request.year_max}-12-31"
                params["filter"] = (
                    f"{existing},{until}" if existing else until
                )
            if self.config.api.crossref_mailto:
                params["mailto"] = self.config.api.crossref_mailto
        items: list[dict] = []
        for index, params in enumerate(attempts, start=1):
            try:
                items = await _crossref_items_paginated(
                    client, params, request.limit, progress_callback=self._progress
                )
            except httpx.HTTPError as exc:
                self.diagnostics.errors.append(
                    f"crossref_attempt_{index}: {exc.__class__.__name__}: {exc}"
                )
                continue
            if items:
                if index > 1:
                    self.diagnostics.errors.append(f"crossref_retry_succeeded:{index}")
                break
        papers = []
        for item in items:
            title = _first(item.get("title"))
            if not title:
                continue
            doi = _normalize_doi(item.get("DOI"))
            if _is_crossref_noise_record(title, doi, item):
                continue
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
                    access_type=AccessType.UNAVAILABLE,
                )
            )
        return papers

    async def _enrich_unpaywall(
        self,
        client: httpx.AsyncClient,
        papers: list[PaperMetadata],
    ) -> list[PaperMetadata]:
        semaphore = asyncio.Semaphore(10)

        async def enrich_one(paper: PaperMetadata) -> None:
            async with semaphore:
                await self._enrich_unpaywall_paper(client, paper)

        await asyncio.gather(*(enrich_one(paper) for paper in papers if paper.doi))
        return papers

    async def _enrich_unpaywall_paper(
        self, client: httpx.AsyncClient, paper: PaperMetadata
    ) -> None:
        if not paper.doi:
            return
        response = await client.get(
            f"https://api.unpaywall.org/v2/{paper.doi}",
            params={"email": self.config.api.unpaywall_email},
        )
        if response.status_code == 404:
            return
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
        elif paper.access_type == AccessType.UNAVAILABLE and _publisher_requires_login(paper):
            paper.access_type = AccessType.REQUIRES_LOGIN


async def _gather_named(
    coroutines: dict[str, object],
    diagnostics: SearchDiagnostics,
) -> list[list[PaperMetadata]]:
    async def run_one(name: str, coroutine: object) -> list[PaperMetadata]:
        try:
            response = await asyncio.wait_for(coroutine, timeout=30.0)
            diagnostics.source_health[name] = SourceHealth(
                source_name=name, healthy=True, request_count=1
            )
            return response
        except Exception as exc:
            diagnostics.errors.append(f"{name}: {exc.__class__.__name__}: {exc}")
            diagnostics.source_health[name] = SourceHealth(
                source_name=name,
                healthy=False,
                request_count=1,
                failure_count=1,
                last_failure_class=classify_source_exception(exc),
            )
            return []

    return list(await asyncio.gather(*(
        run_one(name, coroutine) for name, coroutine in coroutines.items()
    )))


def _crossref_attempt_params(request: PaperSearchRequest) -> list[dict[str, str | int]]:
    compact_topic = _compact_crossref_topic(request.topic)
    attempts = [
        _crossref_params("query.title", request.topic, request),
    ]
    if compact_topic != request.topic:
        attempts.append(_crossref_params("query.title", compact_topic, request, rows=12))
    attempts.append(_crossref_params("query.bibliographic", compact_topic, request, rows=12))
    return attempts


def build_query_variants(topic: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", topic).strip()
    # Session research backgrounds supply LLM-derived variants. A bare topic must
    # never acquire material or mechanism assumptions from a global heuristic.
    return [normalized] if normalized else []


def filter_papers_by_retrieval_policy(
    papers: list[PaperMetadata],
    policy: TopicRetrievalPolicy | None,
) -> tuple[list[PaperMetadata], dict[str, str]]:
    """Apply session-specific hard constraints before any automated download."""
    if policy is None:
        return papers, {}
    accepted: list[PaperMetadata] = []
    rejected: dict[str, str] = {}
    required_groups = [
        [term.strip().lower() for term in group if term.strip()]
        for group in policy.required_concept_groups
    ]
    excluded = [term.strip().lower() for term in policy.excluded_concepts if term.strip()]
    for paper in papers:
        text = _paper_search_text(paper)
        missing = [
            group for group in required_groups if group and not any(_contains_concept(text, term) for term in group)
        ]
        matched_excluded = [term for term in excluded if _contains_concept(text, term)]
        if missing:
            rejected[paper.paper_id] = "missing_required:" + "|".join(missing[0])
        elif matched_excluded:
            rejected[paper.paper_id] = "excluded_concept:" + matched_excluded[0]
        else:
            accepted.append(paper)
    return accepted, rejected


def _contains_concept(text: str, term: str) -> bool:
    normalized_text = re.sub(r"[-_/]+", " ", text.lower())
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    normalized_term = re.sub(r"[-_/]+", " ", term.lower()).strip()
    return normalized_term in normalized_text


def _variant_requests(request: PaperSearchRequest) -> list[PaperSearchRequest]:
    variants = request.query_variants or build_query_variants(request.topic)
    if request.topic not in variants:
        variants.insert(0, request.topic)
    return [
        request.model_copy(update={"topic": variant, "query_variants": []})
        for variant in list(dict.fromkeys(variants))[:10]
    ]


def _has_enough_relevant_results(
    papers: list[PaperMetadata],
    request: PaperSearchRequest,
) -> bool:
    if len(papers) < request.min_relevant_results:
        return False
    if len(papers) >= request.limit:
        return True
    if len(papers) >= min(request.limit, max(request.min_relevant_results * 4, 20)):
        return True
    return False


def _crossref_params(
    query_key: str,
    query: str,
    request: PaperSearchRequest,
    rows: int | None = None,
) -> dict[str, str | int]:
    return {
        query_key: query,
        "rows": rows or min(max(request.limit, 10), 25),
        "sort": "relevance",
        "order": "desc",
    }


def _compact_crossref_topic(topic: str) -> str:
    tokens = [
        token
        for token in re.split(r"[^A-Za-z0-9]+", topic)
        if token
        and token.lower()
        not in {
            "review",
            "materials",
            "material",
            "recent",
            "study",
            "paper",
            "papers",
            "wearable",
            "flexible",
        }
    ]
    return " ".join(tokens[:7]) or topic


def _safe_year(value: object) -> int | None:
    try:
        year = int(str(value)[:4])
    except (TypeError, ValueError):
        return None
    return year if 1000 <= year <= 3000 else None


def _arxiv_search_query(topic: str) -> str:
    """Build an arXiv query that requires every meaningful topic term.

    Passing ``all:<topic>`` to arXiv treats the whitespace-separated terms as
    a broad expression and can return unrelated newest submissions.  The API
    supports boolean ``AND`` between field clauses, which gives the live
    source useful precision while preserving a single-term/CJK query.
    """
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*|[\u4e00-\u9fff]+", topic or "")
    terms = list(dict.fromkeys(term for term in terms if term.strip()))
    if not terms:
        return f"all:{topic.strip()}"
    if len(terms) == 1:
        return f"all:{terms[0]}"
    return " AND ".join(f"all:{term}" for term in terms[:8])


async def _crossref_items(client: httpx.AsyncClient, params: dict[str, str | int]) -> list[dict]:
    response = await client.get("https://api.crossref.org/works", params=params)
    response.raise_for_status()
    return response.json().get("message", {}).get("items", [])


async def _crossref_items_paginated(
    client: httpx.AsyncClient,
    params: dict[str, str | int],
    limit: int,
    progress_callback: Callable[..., None] | None = None,
) -> list[dict]:
    target = min(max(limit, 25), 200)
    rows = min(int(params.get("rows", 25)), 50)
    items: list[dict] = []
    for offset in range(0, target, rows):
        page = await _crossref_items(client, {**params, "rows": rows, "offset": offset})
        items.extend(page)
        if progress_callback is not None:
            progress_callback(
                stage="search_source_page", status="finished", source="crossref",
                page=offset // rows + 1, count=len(page), total=len(items),
            )
        if len(page) < rows:
            break
    return items[:target]


async def _get_with_retries(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, str | int],
    source: str,
    diagnostics: SearchDiagnostics,
    attempts: int = 3,
) -> httpx.Response:
    """Fetch a URL with unified retry via @retry_async.

    The inner _single_get does one HTTP GET; retries are handled by the
    decorator. Retry traces are recorded in retry_tracker for harness checks.
    """
    retry_config = RetryConfig(
        max_attempts=attempts,
        backoff_strategy=BackoffStrategy.LINEAR,
        base_delay_seconds=0.4,
        retry_status_codes=frozenset({429, 500, 502, 503, 504}),
        retry_on=(httpx.HTTPError,),
    )

    def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
        status = ""
        if isinstance(exc, httpx.HTTPStatusError):
            status = f" HTTP {exc.response.status_code}"
        diagnostics.errors.append(f"{source}_retry_{attempt}:{status}")

    @retry_async(
        retry_config,
        operation=f"search_get:{source}",
        retry_on=(httpx.HTTPError,),
        on_retry=_on_retry,
    )
    async def _single_get() -> httpx.Response:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response

    try:
        return await _single_get()
    except httpx.HTTPError as exc:
        diagnostics.errors.append(f"{source}: {exc.__class__.__name__}: {exc}")
        raise


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
        lexical_score = _lexical_relevance_score(request.topic, paper)
        title_score = _title_relevance_score(request.topic, paper)
        phrase_score = _key_phrase_score(request.topic, paper)
        venue_score = _materials_venue_score(paper)
        relevance = paper.relevance_score or lexical_score
        access = 1.0 if paper.access_type == AccessType.OPEN_ACCESS else 0.4
        paper.relevance_score = min(
            1.0,
            0.08 * relevance
            + 0.25 * lexical_score
            + 0.33 * title_score
            + 0.14 * phrase_score
            + 0.10 * paper.recency_score
            + 0.06 * venue_score
            + 0.02 * citation_score
            + 0.02 * access,
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


def filter_search_results(
    papers: list[PaperMetadata],
    request: PaperSearchRequest,
    current_year: int = 2026,
) -> list[PaperMetadata]:
    """Keep a broad candidate pool and remove only obvious noise.

    Relevance-specific concept checks belong in ranking, not destructive recall filtering.
    """
    filtered: list[PaperMetadata] = []
    for paper in papers:
        if _is_noise_paper(paper):
            continue
        if _is_materials_sensor_domain_noise(request.topic, paper):
            continue
        if paper.year is not None:
            if paper.year > current_year:
                continue
            if request.year_min is not None and paper.year < request.year_min:
                continue
        lexical = _lexical_relevance_score(request.topic, paper)
        venue = _materials_venue_score(paper)
        if lexical < 0.08 and venue < 0.4:
            continue
        filtered.append(paper)
    return filtered


def select_context_papers(
    papers: list[PaperMetadata],
    request: PaperSearchRequest,
    limit: int = 15,
) -> list[PaperMetadata]:
    ranked = rank_papers(list(papers), request)
    strong = [paper for paper in ranked if _context_relevance_score(request, paper) >= 0.45]
    if len(strong) >= min(request.min_relevant_results, limit):
        return strong[:limit]
    return ranked[:limit]


def _context_relevance_score(request: PaperSearchRequest, paper: PaperMetadata) -> float:
    lexical = _lexical_relevance_score(request.topic, paper)
    title = _title_relevance_score(request.topic, paper)
    phrase = _key_phrase_score(request.topic, paper)
    ranked = paper.relevance_score or 0.0
    return min(
        1.0,
        0.42 * ranked
        + 0.27 * lexical
        + 0.23 * title
        + 0.08 * phrase,
    )


def _is_noise_paper(paper: PaperMetadata) -> bool:
    title = paper.title.lower()
    doi = (paper.doi or "").lower()
    if title.startswith("review for "):
        return True
    return any(marker in doi for marker in ["/review", "/decision", "/response"])


def _is_materials_sensor_domain_noise(topic: str, paper: PaperMetadata) -> bool:
    lowered_topic = topic.lower()
    if not any(
        marker in topic or marker in lowered_topic
        for marker in ["sensor", "传感", "压力", "压敏", "柔性", "pdms", "mxene", "hydrogel"]
    ):
        return False
    text = _paper_search_text(paper)
    title = paper.title.lower()
    material_markers = [
        "pdms",
        "polydimethylsiloxane",
        "mxene",
        "hydrogel",
        "carbon",
        "graphene",
        "nanotube",
        "cnt",
        "polymer",
        "composite",
        "film",
        "thin-film",
        "elastomer",
    ]
    if any(marker in text for marker in material_markers):
        return False
    cross_domain_markers = [
        "wireless sensor network",
        "wireless pressure monitoring network",
        "sensor networks",
        "routing protocol",
        "state estimator",
        "sensor faults",
        "fault diagnosis",
        "infrastructures",
    ]
    return any(marker in title or marker in text for marker in cross_domain_markers)


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


def _is_crossref_noise_record(title: str, doi: str | None, item: dict) -> bool:
    lowered_title = title.lower()
    lowered_doi = (doi or "").lower()
    if lowered_title.startswith("review for "):
        return True
    if any(marker in lowered_doi for marker in ["/review", "/decision", "/response"]):
        return True
    return item.get("type") in {"peer-review", "posted-content"}


def _bounded(value: float | int | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(float(value), 1.0))


def _title_key(title: str) -> str:
    return _slug(title).lower()


def _title_tokens(title: str) -> set[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "and",
        "for",
        "with",
        "by",
        "on",
        "in",
        "to",
        "based",
        "using",
        "study",
        "sensor",
        "sensors",
        "flexible",
        "wearable",
    }
    return {
        token
        for token in re.split(r"[^a-z0-9]+", title.lower())
        if len(token) > 2 and token not in stopwords
    }


def _query_tokens(topic: str) -> set[str]:
    tokens = _title_tokens(topic)
    lowered = topic.lower()
    if any(token in topic for token in ["碳基", "碳", "炭"]) or "carbon" in lowered:
        tokens.update({"carbon", "graphene", "cnt", "nanotube", "black"})
    if "pdms" in lowered or "聚二甲基硅氧烷" in topic:
        tokens.update({"pdms", "polydimethylsiloxane"})
    if any(token in topic for token in ["柔性", "薄膜", "压敏", "压力", "传感"]):
        tokens.update({"flexible", "film", "thin", "pressure", "sensor"})
    if any(token in topic for token in ["长时间", "长期", "受压", "漂移", "稳定"]):
        tokens.update({"long", "term", "drift", "stability", "hysteresis", "compression"})
    if "mxene" in topic.lower():
        tokens.add("mxene")
    if "hydrogel" in topic.lower():
        tokens.add("hydrogel")
    return tokens


def _lexical_relevance_score(topic: str, paper: PaperMetadata) -> float:
    query = _query_tokens(topic)
    if not query:
        return 0.0
    text = " ".join(
        value
        for value in [
            paper.title,
            paper.abstract or "",
            paper.journal or "",
            paper.publisher or "",
        ]
        if value
    )
    candidate = _title_tokens(text)
    if "mxene" in text.lower():
        candidate.add("mxene")
    if "hydrogel" in text.lower():
        candidate.add("hydrogel")
    overlap = len(query & candidate) / max(len(query), 1)
    phrase_bonus = 0.0
    topic_lowered = topic.lower()
    lowered = text.lower()
    if "mxene" in topic_lowered and "mxene" in lowered:
        phrase_bonus += 0.2
    if "hydrogel" in topic_lowered and "hydrogel" in lowered:
        phrase_bonus += 0.2
    if (
        "pressure" in topic_lowered or "压力" in topic or "压敏" in topic
    ) and "pressure" in lowered:
        phrase_bonus += 0.1
    if "strain" in topic_lowered and "strain" in lowered:
        phrase_bonus += 0.1
    if ("pdms" in topic_lowered or "聚二甲基硅氧烷" in topic) and "pdms" in lowered:
        phrase_bonus += 0.16
    if ("漂移" in topic or "drift" in topic_lowered) and "drift" in lowered:
        phrase_bonus += 0.16
    if ("稳定" in topic or "stability" in topic_lowered) and any(
        marker in lowered for marker in ["stable", "stability"]
    ):
        phrase_bonus += 0.12
    if ("碳" in topic or "carbon" in topic_lowered) and any(
        marker in lowered for marker in ["carbon", "graphene", "nanotube", "cnt"]
    ):
        phrase_bonus += 0.16
    return min(1.0, overlap + phrase_bonus)


def _title_relevance_score(topic: str, paper: PaperMetadata) -> float:
    query = _query_tokens(topic)
    if not query:
        return 0.0
    title = paper.title.lower()
    title_tokens = _title_tokens(title)
    if "mxene" in title:
        title_tokens.add("mxene")
    if "hydrogel" in title:
        title_tokens.add("hydrogel")
    overlap = len(query & title_tokens) / max(len(query), 1)
    return min(1.0, overlap + _title_phrase_bonus(topic, title))


def _title_phrase_bonus(topic: str, title: str) -> float:
    topic_lowered = topic.lower()
    bonus = 0.0
    pairs = [
        ("low temperature", ["low-temperature", "low temperature"]),
        ("bending fatigue", ["bending fatigue"]),
        ("pressure sensor", ["pressure sensor"]),
        ("strain sensor", ["strain sensor"]),
        ("flexible pressure", ["flexible pressure"]),
        ("conductive hydrogel", ["conductive hydrogel"]),
    ]
    for query_phrase, title_phrases in pairs:
        if query_phrase in topic_lowered and any(phrase in title for phrase in title_phrases):
            bonus += 0.12
    if "review" in topic_lowered and any(marker in title for marker in ["review", "progress"]):
        bonus += 0.18
    if "fatigue" in topic_lowered and "fatigue" in title:
        bonus += 0.12
    if "temperature" in topic_lowered and any(
        marker in title for marker in ["temperature", "low-temperature"]
    ):
        bonus += 0.08
    return min(0.45, bonus)


def _key_phrase_score(topic: str, paper: PaperMetadata) -> float:
    topic_lowered = topic.lower()
    text = _paper_search_text(paper)
    phrases = []
    if "low temperature" in topic_lowered or "low-temperature" in topic_lowered:
        phrases.append({"low-temperature", "low temperature"})
    if "bending" in topic_lowered:
        phrases.append({"bending"})
    if "fatigue" in topic_lowered:
        phrases.append({"fatigue"})
    if "review" in topic_lowered:
        phrases.append({"review", "progress"})
    if "temperature" in topic_lowered:
        phrases.append({"temperature"})
    if not phrases:
        return 1.0
    matched = sum(any(marker in text for marker in group) for group in phrases)
    return matched / len(phrases)


def _paper_search_text(paper: PaperMetadata) -> str:
    return " ".join(
        value
        for value in [
            paper.title,
            paper.abstract or "",
            paper.journal or "",
            paper.publisher or "",
        ]
        if value
    ).lower()


def _materials_venue_score(paper: PaperMetadata) -> float:
    venue = f"{paper.journal or ''} {paper.publisher or ''}".lower()
    markers = [
        "advanced materials",
        "advanced functional materials",
        "advanced materials interfaces",
        "acs nano",
        "acs applied materials",
        "american chemical society",
        "nano letters",
        "nanomaterials",
        "nature",
        "wiley",
        "mdpi",
        "materials",
        "rsc",
        "royal society of chemistry",
        "journal of materials",
        "chemical",
        "chemistry",
        "infomat",
        "diamond and related materials",
    ]
    return 1.0 if any(marker in venue for marker in markers) else 0.0


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
    elif (
        left.access_type != AccessType.OPEN_ACCESS
        and right.access_type == AccessType.REQUIRES_LOGIN
    ):
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
