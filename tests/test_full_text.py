import httpx
import pytest

from littrace.config import LitTraceConfig
from littrace.context import add_papers
from littrace.full_text import resolve_full_text_for_paper, resolve_workspace_full_text
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
