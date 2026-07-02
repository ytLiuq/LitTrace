import httpx
import pytest

from littrace.config import APIConfig, LitTraceConfig
from littrace.models import PaperMetadata, PaperSearchRequest
from littrace.search import LiveSearchClient, _has_enough_relevant_results, build_query_variants


@pytest.mark.anyio
async def test_openalex_retries_transient_503():
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


def test_carbon_pdms_chinese_topic_builds_english_query_variants():
    variants = build_query_variants("碳基PDMS柔性薄膜传感器长时间受压漂移")

    joined = " ".join(variants).lower()
    assert variants[0] == "碳基PDMS柔性薄膜传感器长时间受压漂移"
    assert "carbon" in joined
    assert "pdms" in joined
    assert "drift" in joined
    assert "stability" in joined


def test_live_search_continues_past_minimum_to_return_extra_results():
    request = PaperSearchRequest(topic="carbon PDMS pressure sensor", limit=40, min_relevant_results=5)
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
