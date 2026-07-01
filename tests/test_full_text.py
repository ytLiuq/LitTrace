import httpx
import pytest

from littrace.config import APIConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.full_text import (
    fetch_crossref_paper_by_doi,
    full_text_config_warnings,
    resolve_full_text_for_paper,
    resolve_workspace_full_text,
)
from littrace.eval_api import full_text_metrics_from_workspace
from littrace.models import AccessType, LiteratureWorkspace, PaperMetadata


@pytest.mark.anyio
async def test_full_text_resolver_extracts_crossref_pdf_link():
    async def handler(request: httpx.Request) -> httpx.Response:
        if "api.crossref.org" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "message": {
                        "resource": {"primary": {"URL": "https://doi.org/10.1000/example"}},
                        "link": [
                            {
                                "URL": "https://example.org/paper.pdf",
                                "content-type": "application/pdf",
                            }
                        ],
                    }
                },
            )
        if str(request.url) == "https://example.org/paper.pdf":
            return httpx.Response(200, headers={"content-type": "application/pdf"})
        if str(request.url) == "https://doi.org/10.1000/example":
            return httpx.Response(302, headers={"location": "https://example.org/article"})
        if str(request.url) == "https://example.org/article":
            return httpx.Response(200, headers={"content-type": "text/html"})
        raise AssertionError(str(request.url))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await resolve_full_text_for_paper(
            client,
            PaperMetadata(
                paper_id="p1",
                title="Open paper",
                doi="10.1000/example",
                access_type=AccessType.OPEN_ACCESS,
            ),
            LitTraceConfig(),
        )

    assert str(report.best_pdf_url) == "https://example.org/paper.pdf"
    assert report.open_access_candidate_count >= 1
    assert report.verified_candidate_count >= 1


@pytest.mark.anyio
async def test_workspace_full_text_resolution_updates_pdf_url_from_seed():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Seed OA paper",
                pdf_url="https://example.org/seed.pdf",
                access_type=AccessType.OPEN_ACCESS,
            )
        ],
    )

    workspace = await resolve_workspace_full_text(workspace, LitTraceConfig())

    report = workspace.full_text_reports["p1"]
    assert str(report.best_pdf_url) == "https://example.org/seed.pdf"
    assert workspace.papers["p1"].pdf_url


@pytest.mark.anyio
async def test_backfill_workspace_by_dois_adds_crossref_metadata():
    async def handler(request: httpx.Request) -> httpx.Response:
        if "api.crossref.org/works/10.1000/example" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "message": {
                        "DOI": "10.1000/example",
                        "title": ["Backfilled paper"],
                        "publisher": "Example Publisher",
                        "container-title": ["Example Journal"],
                        "issued": {"date-parts": [[2026]]},
                        "URL": "https://doi.org/10.1000/example",
                    }
                },
            )
        raise AssertionError(str(request.url))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        paper = await fetch_crossref_paper_by_doi(client, "10.1000/example")

    assert paper
    workspace = LiteratureWorkspace()
    workspace.papers[paper.paper_id] = paper
    workspace.context.active_papers.append(paper.paper_id)
    assert workspace.context.active_papers


def test_full_text_config_warnings_prompt_unpaywall_and_mailto():
    warnings = full_text_config_warnings(
        LitTraceConfig(api=APIConfig(user_agent="LitTrace/0.1", unpaywall_email=None))
    )

    assert any("unpaywall" in warning.lower() for warning in warnings)
    assert any("mailto" in warning.lower() for warning in warnings)


def test_full_text_metrics_only_count_successfully_parsed_papers():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper")],
    )
    workspace.parsed_papers["p1"] = {"parsed": False, "error": "parser unavailable"}

    assert full_text_metrics_from_workspace(workspace)["parsed_full_text_rate"] == 0.0

    workspace.parsed_papers["p1"] = {"parsed": True}

    assert full_text_metrics_from_workspace(workspace)["parsed_full_text_rate"] == 1.0


@pytest.mark.anyio
async def test_unpaywall_oa_candidate_keeps_oa_status_when_head_forbidden():
    async def handler(request: httpx.Request) -> httpx.Response:
        if "api.crossref.org" in str(request.url):
            return httpx.Response(200, json={"message": {}})
        if "api.unpaywall.org" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "is_oa": True,
                    "best_oa_location": {
                        "url_for_pdf": "https://www.mdpi.com/1996-1944/13/18/3947/pdf"
                    },
                },
            )
        if str(request.url) == "https://www.mdpi.com/1996-1944/13/18/3947/pdf":
            return httpx.Response(403, headers={"content-type": "text/html"})
        if str(request.url) == "https://doi.org/10.3390/ma13183947":
            return httpx.Response(403, headers={"content-type": "text/html"})
        raise AssertionError(str(request.url))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await resolve_full_text_for_paper(
            client,
            PaperMetadata(
                paper_id="p1",
                title="MDPI paper",
                doi="10.3390/ma13183947",
            ),
            LitTraceConfig(api=APIConfig(unpaywall_email="user@example.com")),
        )

    assert str(report.best_pdf_url) == "https://www.mdpi.com/1996-1944/13/18/3947/pdf"
    assert report.open_access_candidate_count >= 1
    assert report.login_required_candidate_count == 0
    assert report.candidates[0].note == "oa_evidence_but_http_forbidden"
