from __future__ import annotations

import asyncio
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from littrace.config import LitTraceConfig
from littrace.models import AccessType, PaperMetadata, PaperSearchRequest, PaperSearchResult
from littrace.retry import retry_async, RetryConfig, BackoffStrategy


class PaperSearchClient(Protocol):
    name: str

    async def search(self, request: PaperSearchRequest) -> PaperSearchResult:
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
        return PaperSearchResult(request=request, papers=papers[: request.limit])


class LiveSearchClient:
    name = "live_search"

    def __init__(self, config: LitTraceConfig):
        self.config = config
        self.diagnostics = SearchDiagnostics(live_attempted=True)

    async def search(self, request: PaperSearchRequest) -> PaperSearchResult:
        timeout = httpx.Timeout(self.config.api.request_timeout_seconds)
        headers = {"User-Agent": self.config.api.user_agent}
        requests = _variant_requests(request)
        self.diagnostics.query_variants = [variant.topic for variant in requests]
        openalex_results: list[PaperMetadata] = []
        crossref_results: list[PaperMetadata] = []
        async with httpx.AsyncClient(
            timeout=timeout, headers=headers, follow_redirects=True
        ) as client:
            for index, variant_request in enumerate(requests, start=1):
                openalex, crossref = await _gather_named(
                    {
                        f"openalex_variant_{index}": self._search_openalex(client, variant_request),
                        f"crossref_variant_{index}": self._search_crossref(client, variant_request),
                    },
                    self.diagnostics,
                )
                self.diagnostics.source_counts[f"openalex_variant_{index}"] = len(openalex)
                self.diagnostics.source_counts[f"crossref_variant_{index}"] = len(crossref)
                openalex_results.extend(openalex)
                crossref_results.extend(crossref)
                relevant_so_far = rank_papers(
                    filter_search_results(
                        merge_papers([*openalex_results, *crossref_results]),
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
                self.diagnostics.filtered_counts[f"after_variant_{index}"] = (
                    len(openalex_results) + len(crossref_results) - len(relevant_so_far)
                )
                self.diagnostics.source_counts[f"relevant_after_variant_{index}"] = len(
                    relevant_so_far
                )
                if _has_enough_relevant_results(context_ready_so_far, request):
                    break

            self.diagnostics.source_counts["openalex"] = len(openalex_results)
            self.diagnostics.source_counts["crossref"] = len(crossref_results)

            merged_raw = merge_papers([*openalex_results, *crossref_results])
            merged = filter_search_results(merged_raw, request)
            self.diagnostics.filtered_counts["merged"] = len(merged_raw) - len(merged)
            self.diagnostics.filtered_counts["basic_candidate_pool"] = len(merged)
            if self.config.api.unpaywall_email:
                merged = await self._enrich_unpaywall(client, merged)
                self.diagnostics.source_counts["unpaywall_enriched"] = len(
                    [paper for paper in merged if paper.access_type == AccessType.OPEN_ACCESS]
                )
            merged = rank_papers(merged, request)
            self.diagnostics.ranking_counts["ranked_candidate_pool"] = len(merged)
            self.diagnostics.ranking_counts["context_ready"] = len(
                [paper for paper in merged if _context_relevance_score(request, paper) >= 0.45]
            )
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

        response = await _get_with_retries(
            client,
            "https://api.openalex.org/works",
            params=params,
            source="openalex",
            diagnostics=self.diagnostics,
        )
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
            if self.config.api.crossref_mailto:
                params["mailto"] = self.config.api.crossref_mailto
        items: list[dict] = []
        for index, params in enumerate(attempts, start=1):
            try:
                items = await _crossref_items(client, params)
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
            elif paper.access_type == AccessType.UNAVAILABLE and _publisher_requires_login(paper):
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
    variants = [normalized] if normalized else []
    lowered = normalized.lower()
    expanded: list[str] = []

    if any(token in normalized for token in ["碳基", "碳", "炭"]) or "carbon" in lowered:
        expanded.extend(["carbon-based", "carbon black", "graphene", "carbon nanotube", "CNT"])
    if "pdms" in lowered or "聚二甲基硅氧烷" in normalized:
        expanded.extend(["PDMS", "polydimethylsiloxane"])
    if any(token in normalized for token in ["柔性", "薄膜", "压敏", "压力", "传感"]):
        expanded.extend(["flexible thin-film pressure sensor", "flexible pressure sensor"])
    if any(token in normalized for token in ["长时间", "长期", "受压", "漂移", "稳定"]):
        expanded.extend(
            [
                "long-term pressure drift",
                "compression drift",
                "signal drift",
                "hysteresis",
                "stability",
            ]
        )
    if any(token in normalized for token in ["材料", "化学"]) or expanded:
        expanded.extend(["materials chemistry", "wearable sensor"])

    if expanded:
        variants.append(" ".join(dict.fromkeys(expanded)))
        variants.append("carbon PDMS flexible pressure sensor long-term stability drift hysteresis")
        variants.append("carbon black PDMS pressure sensor stability drift compression")
        variants.append("graphene PDMS flexible pressure sensor long-term stability")
        variants.append("CNT PDMS flexible pressure sensor drift hysteresis")
        variants.append("PDMS flexible pressure sensor long-term stability")
        variants.append("flexible piezoresistive pressure sensor drift stability")
        variants.append("conductive polymer composite flexible pressure sensor hysteresis")
        variants.append("carbon composite flexible pressure sensor durability")

    return list(dict.fromkeys(variant for variant in variants if variant))[:10]


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


async def _crossref_items(client: httpx.AsyncClient, params: dict[str, str | int]) -> list[dict]:
    response = await client.get("https://api.crossref.org/works", params=params)
    response.raise_for_status()
    return response.json().get("message", {}).get("items", [])


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
        # Preserve backward-compatible diagnostics format: "openalex_retry_1: HTTP 503"
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
        topical_score = _topical_specificity_score(request.topic, paper)
        concept_score = _concept_group_score(request.topic, paper)
        venue_score = _materials_venue_score(paper)
        mandatory = 1.0 if _matches_mandatory_concepts(request.topic, paper) else 0.35
        relevance = paper.relevance_score or lexical_score
        access = 1.0 if paper.access_type == AccessType.OPEN_ACCESS else 0.4
        paper.relevance_score = min(
            1.0,
            0.10 * relevance
            + 0.19 * lexical_score
            + 0.24 * title_score
            + 0.18 * concept_score
            + 0.10 * phrase_score
            + 0.07 * topical_score
            + 0.07 * paper.recency_score
            + 0.05 * venue_score
            + 0.025 * citation_score
            + 0.01 * access
            + 0.005 * mandatory,
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
        concept_score = _concept_group_score(request.topic, paper)
        if lexical < 0.08 and concept_score < 0.25 and venue < 0.4:
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
    concept = _concept_group_score(request.topic, paper)
    phrase = _key_phrase_score(request.topic, paper)
    mandatory = 1.0 if _matches_mandatory_concepts(request.topic, paper) else 0.0
    ranked = paper.relevance_score or 0.0
    return min(
        1.0,
        0.24 * ranked
        + 0.24 * lexical
        + 0.22 * title
        + 0.18 * concept
        + 0.08 * phrase
        + 0.04 * mandatory,
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


def _topical_specificity_score(topic: str, paper: PaperMetadata) -> float:
    topic_lowered = topic.lower()
    text = _paper_search_text(paper)
    score = 0.0
    if "hydrogel" in topic_lowered and "hydrogel" in text:
        score += 0.25
    if "conductive" in topic_lowered and any(
        marker in text for marker in ["conductive", "conductivity"]
    ):
        score += 0.2
    if "strain" in topic_lowered and "strain" in text:
        score += 0.2
    if (
        "temperature" in text
        and "strain" in text
        and ("hydrogel" in topic_lowered or "strain" in topic_lowered)
    ):
        score += 0.16
    if "wearable" in topic_lowered and any(
        marker in text for marker in ["wearable", "human motion", "motion monitoring", "flexible"]
    ):
        score += 0.12
    if any(marker in text for marker in ["meeting abstracts", "conference", "figshare"]):
        score -= 0.18
    return max(0.0, min(1.0, score))


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


def _concept_group_score(topic: str, paper: PaperMetadata) -> float:
    groups = _required_concept_groups(topic)
    if not groups:
        return 1.0
    text = _paper_search_text(paper)
    matched = sum(any(marker in text for marker in markers) for markers in groups)
    return matched / len(groups)


def _matches_mandatory_concepts(topic: str, paper: PaperMetadata) -> bool:
    lowered = topic.lower()
    text = _paper_search_text(paper)
    if ("pdms" in lowered or "聚二甲基硅氧烷" in topic) and not any(
        marker in text for marker in ["pdms", "polydimethylsiloxane"]
    ):
        return False
    if "mxene" in lowered and "mxene" not in text:
        return False
    if "hydrogel" in lowered and "hydrogel" not in text:
        return False
    return True


def _required_concept_score(topic: str) -> float:
    groups = _required_concept_groups(topic)
    if not groups:
        return 0.0
    if len(groups) <= 2:
        return 1.0
    return 0.75


def _required_concept_groups(topic: str) -> list[set[str]]:
    lowered = topic.lower()
    groups: list[set[str]] = []
    if "pdms" in lowered or "聚二甲基硅氧烷" in topic:
        groups.append({"pdms", "polydimethylsiloxane"})
    if any(token in topic for token in ["碳基", "碳", "炭"]) or "carbon" in lowered:
        groups.append({"carbon", "graphene", "nanotube", "cnt", "carbon black"})
    if any(token in topic for token in ["压力", "压敏", "受压"]) or "pressure" in lowered:
        groups.append({"pressure", "piezoresistive", "tactile"})
    if any(token in topic for token in ["传感", "传感器"]) or "sensor" in lowered:
        groups.append({"sensor", "sensing"})
    if "mxene" in lowered:
        groups.append({"mxene"})
    if "hydrogel" in lowered:
        groups.append({"hydrogel"})
    return groups


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
