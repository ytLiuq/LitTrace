import httpx
import pytest

from littrace.config import APIConfig, LitTraceConfig
from littrace.models import PaperSearchRequest
from littrace.search import LiveSearchClient


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
