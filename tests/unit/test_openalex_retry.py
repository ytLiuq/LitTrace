"""Unit tests for OpenAlex retry + query planning.

Round 4 P3 step 14 of 15.

The previous ``tests/live/test_external_services.py`` mixed real
network paths (CDP, publishers, daily-update) with three pure
unit tests for OpenAlex retry, query-variant generation, and
search-result counting. The unit tests did not need any live
service, but lived under ``tests/live/`` so they were skipped
whenever the suite was collected with the default marker filter.

Move the unit tests here so they run as part of the regular suite
and the live module is reserved for tests that genuinely need a
real service.
"""

from __future__ import annotations

import httpx
import pytest

from littrace.config import APIConfig, LitTraceConfig
from littrace.models import PaperMetadata, PaperSearchRequest
from littrace.retrieval.search import (
    LiveSearchClient,
    _has_enough_relevant_results,
    build_query_variants,
)


@pytest.mark.anyio
async def test_openalex_retries_transient_503() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "MXene flexible pressure sensor",
                        "doi": "https://doi.org/10.1000/example",
                        "publication_year": 2026,
                        "primary_location": {
                            "source": {
                                "display_name": "Journal of Materials Science",
                                "host_organization_name": "Springer",
                            }
                        },
                        "open_access": {"oa_url": "https://example.org/paper.pdf"},
                    }
                ]
            },
        )

    config = LitTraceConfig(api=APIConfig(openalex_api_key="key"))
    client = LiveSearchClient(config)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        papers = await client._search_openalex(
            http_client,
            PaperSearchRequest(topic="MXene flexible pressure sensor", year_min=2024),
        )

    assert calls == 2
    assert len(papers) == 1
    assert any("openalex_retry_1: HTTP 503" in error for error in client.diagnostics.errors)


def test_bare_topic_does_not_inject_material_or_mechanism_variants() -> None:
    variants = build_query_variants("碳基PDMS柔性薄膜传感器长时间受压漂移")

    assert variants == ["碳基PDMS柔性薄膜传感器长时间受压漂移"]


def test_live_search_continues_past_minimum_to_return_extra_results() -> None:
    request = PaperSearchRequest(
        topic="carbon PDMS pressure sensor", limit=40, min_relevant_results=5
    )
    papers = [PaperMetadata(paper_id=f"p{i}", title=f"Paper {i}") for i in range(5)]

    assert not _has_enough_relevant_results(papers, request)
    assert not _has_enough_relevant_results(
        [PaperMetadata(paper_id=f"p{i}", title=f"Paper {i}") for i in range(10)],
        request,
    )
    assert _has_enough_relevant_results(
        [PaperMetadata(paper_id=f"p{i}", title=f"Paper {i}") for i in range(20)],
        request,
    )
