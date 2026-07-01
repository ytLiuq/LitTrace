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
