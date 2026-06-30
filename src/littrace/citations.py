from __future__ import annotations

import re
from collections.abc import Iterable

import httpx

from littrace.cache import cache_key, read_text_cache, write_text_cache
from littrace.config import LitTraceConfig
from littrace.harnesses import check_citations
from littrace.models import CitationAudit, CitationRecord, LinkStatus, PaperMetadata


LOGIN_STATUS_CODES = {401, 402, 403}
SUCCESS_STATUS_CODES = set(range(200, 300))
REDIRECT_STATUS_CODES = set(range(300, 400))


def citation_for_paper(paper: PaperMetadata) -> CitationRecord:
    access_url = best_access_url(paper)
    return CitationRecord(
        paper_id=paper.paper_id,
        citation_text=format_apa_like_citation(paper),
        access_url=access_url,
        doi=paper.doi,
    )


def citation_records_for_papers(papers: Iterable[PaperMetadata]) -> list[CitationRecord]:
    return [citation_for_paper(paper) for paper in papers]


async def audit_citation_links(
    papers: Iterable[PaperMetadata],
    config: LitTraceConfig,
) -> CitationAudit:
    records = citation_records_for_papers(papers)
    timeout = httpx.Timeout(config.api.request_timeout_seconds)
    headers = {"User-Agent": config.api.user_agent}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=False) as client:
        checked_records = [await check_link(client, record, config) for record in records]
    result = check_citations(checked_records)
    return CitationAudit(
        records=checked_records,
        passed=result.passed,
        score=result.score,
        errors=result.errors,
        warnings=result.warnings,
    )


async def check_link(
    client: httpx.AsyncClient,
    record: CitationRecord,
    config: LitTraceConfig | None = None,
) -> CitationRecord:
    if config is not None:
        cached = read_text_cache(config, "citation_links", cache_key(str(record.access_url)))
        if cached:
            return CitationRecord.model_validate_json(cached)
    try:
        response = await client.head(str(record.access_url))
        if response.status_code == 405:
            response = await client.get(str(record.access_url))
    except httpx.HTTPError as exc:
        record.link_status = LinkStatus.FAILED
        record.error = f"{exc.__class__.__name__}: {exc}"
        if config is not None:
            write_text_cache(
                config,
                "citation_links",
                cache_key(str(record.access_url)),
                record.model_dump_json(),
            )
        return record

    record.status_code = response.status_code
    checked = _with_link_check_result(
        record,
        status_code=response.status_code,
        checked_url=response.headers.get("location") or str(record.access_url),
    )
    if config is not None:
        write_text_cache(
            config,
            "citation_links",
            cache_key(str(record.access_url)),
            checked.model_dump_json(),
        )
    return checked


def _with_link_check_result(
    record: CitationRecord,
    status_code: int,
    checked_url: str,
) -> CitationRecord:
    update = record.model_dump()
    update["status_code"] = status_code
    update["checked_url"] = checked_url
    update["requires_login"] = status_code in LOGIN_STATUS_CODES
    if status_code in SUCCESS_STATUS_CODES:
        update["link_status"] = LinkStatus.VERIFIED_200
    elif status_code in REDIRECT_STATUS_CODES:
        update["link_status"] = LinkStatus.VERIFIED_REDIRECT
    elif status_code in LOGIN_STATUS_CODES:
        update["link_status"] = LinkStatus.REQUIRES_LOGIN
    else:
        update["link_status"] = LinkStatus.FAILED
        update["error"] = f"Unexpected HTTP status {status_code}"
    return CitationRecord.model_validate(update)


def best_access_url(paper: PaperMetadata) -> str:
    if paper.pdf_url:
        return str(paper.pdf_url)
    if paper.doi:
        return f"https://doi.org/{paper.doi}"
    if paper.source_urls:
        return str(paper.source_urls[0])
    return "https://example.org/unavailable"


def format_apa_like_citation(paper: PaperMetadata) -> str:
    authors = _format_authors(paper.authors)
    year = f"({paper.year})." if paper.year else "(n.d.)."
    title = paper.title.rstrip(".")
    journal = f" {paper.journal}." if paper.journal else ""
    doi = f" https://doi.org/{paper.doi}" if paper.doi else ""
    return _squash_spaces(f"{authors} {year} {title}.{journal}{doi}")


def _format_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown author."
    if len(authors) == 1:
        return authors[0]
    if len(authors) <= 20:
        return ", ".join(authors[:-1]) + f", & {authors[-1]}"
    return ", ".join(authors[:19]) + f", ... {authors[-1]}"


def _squash_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
