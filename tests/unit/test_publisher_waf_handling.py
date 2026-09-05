from types import SimpleNamespace

import httpx
import pytest

from littrace.access_layer.cdp_core import (
    CDPBrowser,
    PDF_URL_EXTRACTION_JS,
    discover_crossref_pdf_url,
    publisher_urls,
)
from littrace.config import LitTraceConfig
from littrace.downloads import _execute_one
from littrace.download_tasks import DownloadTask
from littrace.models import AccessType, PaperMetadata
from littrace.models import PaperSearchRequest
from littrace.retrieval.search import LiveSearchClient, _has_arxiv_terms
from littrace.state_db import sanitize_json_value


def test_sciengine_cloudwaf_page_is_treated_as_interactive_challenge():
    browser = CDPBrowser.__new__(CDPBrowser)
    browser.get_title = lambda: "访问被拦截！"
    browser.get_url = lambda: "https://www.sciengine.com/doi/10.1360/SSC-2023-0138"
    browser.get_body_text = lambda _limit=500: "CloudWAF block-event-id"
    assert browser.is_cloudflare_challenge()


def test_ieee_doi_has_direct_stamp_pdf_route():
    urls = publisher_urls("10.1109/nmdc57951.2023.10344200", "ieee")
    assert "10344200" in (urls["pdf"] or "")
    assert "stampPDF" in (urls["pdf"] or "")
    assert "pdf.hanspub.org" in PDF_URL_EXTRACTION_JS


def test_7507_journals_use_stable_landing_hosts():
    biomed = publisher_urls("10.7507/1001-5515.202404019", "unknown")
    rrsurg = publisher_urls("10.7507/1002-1892.202601061", "unknown")
    assert "biomedeng.cn/article/10.7507/1001-5515.202404019" in biomed["landing"]
    assert "rrsurg.com/article/10.7507/1002-1892.202601061" in rrsurg["landing"]


@pytest.mark.anyio
async def test_doi_without_verified_pdf_routes_to_cdp(monkeypatch, tmp_path):
    config = LitTraceConfig()
    monkeypatch.setattr("littrace.downloads._record_task_lifecycle", lambda *args: None)
    paper = PaperMetadata(
        paper_id="sciengine-paper",
        title="TFT process",
        doi="10.1360/ssc-2023-0138",
        access_type=AccessType.UNAVAILABLE,
    )
    task = DownloadTask.from_paper(config, paper, session_id="test-session")

    async def fake_cdp(*args, **kwargs):
        return SimpleNamespace(
            paper_id=paper.paper_id,
            status="requires_login",
            error="interactive browser challenge",
        ), task

    monkeypatch.setattr("littrace.downloads._execute_cdp_download_async", fake_cdp)
    item, _ = await _execute_one(
        SimpleNamespace(get=lambda _url: None), config, paper, False, task
    )
    assert item.status == "requires_login"
    assert "interactive" in item.error


def test_crossref_pdf_link_is_preserved_and_cjk_skips_arxiv():
    assert _has_arxiv_terms("flexible pressure sensor")
    assert not _has_arxiv_terms("电容式压力传感器")
    assert sanitize_json_value({"text": "a\x00b"}) == {"text": "ab"}


@pytest.mark.anyio
async def test_crossref_link_pdf_becomes_download_candidate():
    config = LitTraceConfig()
    client = LiveSearchClient(config)
    payload = {
        "message": {
            "items": [{
                "DOI": "10.3724/sp.j.1123.2025.06022",
                "title": ["Example paper"],
                "URL": "https://www.sciengine.com/doi/10.3724/SP.J.1123.2025.06022",
                "link": [{
                    "URL": "https://www.sciengine.com/doi/pdf/example",
                    "content-type": "application/pdf",
                }],
            }],
        }
    }

    async def handler(request):
        return httpx.Response(200, json=payload, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        papers = await client._search_crossref(
            http,
            PaperSearchRequest(topic="sensor", limit=1),
        )
    assert str(papers[0].pdf_url).endswith("/doi/pdf/example")
    assert papers[0].access_type == AccessType.OPEN_ACCESS


def test_crossref_pdf_discovery_reads_regional_publisher_link(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "link": [{
                        "URL": "https://www.sciengine.com/doi/pdf/abc",
                        "content-type": "application/pdf",
                    }]
                }
            }

    monkeypatch.setattr("littrace.access_layer.cdp_core.httpx.get", lambda *a, **k: Response())
    assert discover_crossref_pdf_url("10.3724/sp.j.1123.2025.06022").endswith("/doi/pdf/abc")
