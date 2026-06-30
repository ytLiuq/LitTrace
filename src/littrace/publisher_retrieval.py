from __future__ import annotations

import re
from html import unescape

import httpx
from pydantic import BaseModel, Field, HttpUrl

from littrace.config import LitTraceConfig
from littrace.models import AccessType, PaperMetadata
from littrace.publisher_connectors import PublisherSearchPlan


class PublisherRetrievalResult(BaseModel):
    publisher_family: str
    query_url: HttpUrl
    papers: list[PaperMetadata] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BrowserRetrievalPlan(BaseModel):
    publisher_family: str
    query_url: HttpUrl
    steps: list[str]
    extract_selectors: list[str]
    requires_user_login: bool = False


class PublisherEnrichment(BaseModel):
    doi: str | None = None
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    article_type: str | None = None
    supplementary_links: list[HttpUrl] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def build_browser_retrieval_plan(plan: PublisherSearchPlan) -> BrowserRetrievalPlan:
    return BrowserRetrievalPlan(
        publisher_family=plan.publisher_family,
        query_url=plan.query_url,
        requires_user_login=plan.requires_browser,
        steps=[
            "Open the publisher search URL in an authenticated browser if needed.",
            "Wait for the search results to render.",
            "Scroll until result count stops changing or the desired limit is reached.",
            "Extract titles, DOI links, years, journal/source labels, and access badges.",
            "Do not bypass paywalls or license checks.",
        ],
        extract_selectors=[
            "a[href*='10.']",
            "a[href*='/doi/']",
            "h2, h3, .issue-item__title, .c-card__title, .app-card-open__heading",
            "[class*='abstract'], [id*='abstract']",
            "a[href*='supplement'], a[href*='supporting'], a[href*='suppl']",
        ],
    )


async def fetch_publisher_search_results(
    config: LitTraceConfig,
    plan: PublisherSearchPlan,
) -> PublisherRetrievalResult:
    timeout = httpx.Timeout(config.api.request_timeout_seconds)
    headers = {"User-Agent": config.api.user_agent}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            response = await client.get(str(plan.query_url))
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return PublisherRetrievalResult(
            publisher_family=plan.publisher_family,
            query_url=plan.query_url,
            warnings=[f"{exc.__class__.__name__}: {exc}"],
        )

    return parse_publisher_search_html(plan, response.text)


def parse_publisher_search_html(
    plan: PublisherSearchPlan,
    html: str,
) -> PublisherRetrievalResult:
    papers: list[PaperMetadata] = []
    for index, doi in enumerate(_extract_dois(html), start=1):
        title = _title_near_doi(html, doi) or f"{plan.publisher_family} result {index}"
        paper_id = f"{plan.publisher_family}-{_slug(doi)}"
        papers.append(
            PaperMetadata(
                paper_id=paper_id,
                title=title,
                publisher=plan.publisher_family,
                doi=doi,
                source_urls=[f"https://doi.org/{doi}"],
                access_type=AccessType.REQUIRES_LOGIN if plan.requires_browser else AccessType.METADATA_ONLY,
            )
        )
    warnings = []
    if not papers:
        warnings.append("No DOI-like records found in publisher search HTML.")
    return PublisherRetrievalResult(
        publisher_family=plan.publisher_family,
        query_url=plan.query_url,
        papers=papers,
        warnings=warnings,
    )


def parse_publisher_article_html(html: str) -> PublisherEnrichment:
    dois = _extract_dois(html)
    abstract = _extract_abstract(html)
    keywords = _extract_keywords(html)
    article_type = _extract_article_type(html)
    supplementary_links = _extract_supplementary_links(html)
    warnings = []
    if not abstract:
        warnings.append("No abstract found in publisher HTML.")
    return PublisherEnrichment(
        doi=dois[0] if dois else None,
        abstract=abstract,
        keywords=keywords,
        article_type=article_type,
        supplementary_links=supplementary_links,
        warnings=warnings,
    )


def _extract_dois(html: str) -> list[str]:
    matches = re.findall(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", html, flags=re.IGNORECASE)
    cleaned = [match.rstrip(".,;\"'<> )]") for match in matches]
    return list(dict.fromkeys(cleaned))


def _title_near_doi(html: str, doi: str) -> str | None:
    position = html.lower().find(doi.lower())
    if position < 0:
        return None
    window = html[max(0, position - 600) : min(len(html), position + 600)]
    candidates = re.findall(r"<(?:h2|h3|h4|a|span)[^>]*>(.*?)</(?:h2|h3|h4|a|span)>", window, re.I | re.S)
    cleaned = [_strip_tags(candidate) for candidate in candidates]
    cleaned = [candidate for candidate in cleaned if 12 <= len(candidate) <= 240]
    return cleaned[-1] if cleaned else None


def _strip_tags(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return unescape(value).strip()


def _extract_abstract(html: str) -> str | None:
    patterns = [
        r"<section[^>]*(?:abstract|Abs1)[^>]*>(.*?)</section>",
        r"<div[^>]*(?:abstract|Abs1)[^>]*>(.*?)</div>",
        r"<meta[^>]+name=[\"']description[\"'][^>]+content=[\"']([^\"']+)[\"']",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.I | re.S)
        if match:
            text = _strip_tags(match.group(1))
            if len(text) > 40:
                return text
    return None


def _extract_keywords(html: str) -> list[str]:
    match = re.search(r"<meta[^>]+name=[\"']keywords[\"'][^>]+content=[\"']([^\"']+)[\"']", html, re.I)
    if not match:
        return []
    return [keyword.strip() for keyword in match.group(1).split(",") if keyword.strip()]


def _extract_article_type(html: str) -> str | None:
    patterns = [
        r"<meta[^>]+name=[\"']citation_article_type[\"'][^>]+content=[\"']([^\"']+)[\"']",
        r"<meta[^>]+property=[\"']og:type[\"'][^>]+content=[\"']([^\"']+)[\"']",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.I)
        if match:
            return _strip_tags(match.group(1))
    return None


def _extract_supplementary_links(html: str) -> list[HttpUrl]:
    links = []
    for href in re.findall(r"href=[\"']([^\"']+)[\"']", html, re.I):
        lowered = href.lower()
        if any(token in lowered for token in ["supplement", "supporting", "suppl", "esi"]):
            if href.startswith("http://") or href.startswith("https://"):
                links.append(href)
    return list(dict.fromkeys(links))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
