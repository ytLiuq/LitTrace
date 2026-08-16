"""Real download flow tests — one parametrized test that exercises three
scenarios against a real httpx.AsyncClient-like fake and the real
Postgres artifact registry.

Other download tests were removed: most relied on monkeypatching the CDP
fallback to inject fake browser results, which only tested the mock
boundary rather than the real download code path.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

pytestmark = pytest.mark.adapters

from littrace.artifact_registry import artifact_registry_from_config
from littrace.config import ArtifactStorageConfig, LitTraceConfig, StorageConfig
from littrace.download_tasks import DownloadTask, DownloadTaskStatus
from littrace.downloads import _execute_one
from littrace.models import AccessType, PaperMetadata


# ---------------------------------------------------------------------------
# Real download flow (3 scenarios merged into one parametrized test)
# ---------------------------------------------------------------------------


def _recaptcha_html_response() -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request(
            "GET",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC13367112/pdf/ADVS-9999-e76572.pdf",
        ),
        headers={"content-type": "text/html; charset=utf-8"},
        content=(
            b"<!doctype html><title>Checking your browser - reCAPTCHA</title>"
            b"<body>Checking your browser before accessing pmc.ncbi.nlm.nih.gov</body>"
        ),
    )


def _pdf_response(url: str, content: bytes = b"%PDF-1.4\nexample") -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("GET", url),
        headers={"content-type": "application/pdf"},
        content=content,
    )


def _landing_page_response(landing_url: str, pdf_url: str) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("GET", landing_url),
        headers={"content-type": "text/html; charset=utf-8"},
        content=f'<html><body><a href="{pdf_url}">PDF</a></body></html>'.encode("utf-8"),
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "scenario",
    [
        "stores_downloaded_pdf",
        "recaptcha_auth_required",
        "open_access_landing_page",
    ],
)
async def test_execute_one_real_download_flows(tmp_path, scenario):
    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
        ),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    session_id = f"s-{uuid.uuid4().hex[:12]}"

    if scenario == "stores_downloaded_pdf":
        pdf_url = "https://example.org/paper.pdf"

        class FakeClient:
            async def get(self, _url):
                return _pdf_response(pdf_url)

        paper = PaperMetadata(
            paper_id="p1",
            title="OA paper",
            pdf_url=pdf_url,
            access_type=AccessType.OPEN_ACCESS,
        )

    elif scenario == "recaptcha_auth_required":
        pdf_url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC13367112/pdf/ADVS-9999-e76572.pdf"

        class FakeClient:
            async def get(self, _url):
                return _recaptcha_html_response()

        paper = PaperMetadata(
            paper_id="p1",
            title="OA paper",
            pdf_url=pdf_url,
            access_type=AccessType.OPEN_ACCESS,
        )

    else:  # open_access_landing_page
        landing_url = "https://example.org/article"
        pdf_url = "https://example.org/downloads/paper.pdf"

        class FakeClient:
            async def get(self, url):
                if str(url) == landing_url:
                    return _landing_page_response(landing_url, pdf_url)
                if str(url) == pdf_url:
                    return _pdf_response(pdf_url)
                raise AssertionError(str(url))

        paper = PaperMetadata(
            paper_id="p1",
            title="OA landing page paper",
            pdf_url=landing_url,
            access_type=AccessType.OPEN_ACCESS,
        )

    task = DownloadTask.from_paper(config, paper, session_id=session_id)
    item, updated = await _execute_one(FakeClient(), config, paper, False, task)
    registry_get = artifact_registry_from_config(config).get("paper_pdf:p1", session_id=session_id)

    if scenario == "stores_downloaded_pdf":
        assert item.status == "downloaded"
        assert item.storage_ref
        assert item.storage_ref["object_key"] == f"sessions/{session_id}/papers/p1/paper.pdf"
        assert updated.status == DownloadTaskStatus.VERIFIED
        assert updated.target_object_key == item.storage_ref["object_key"]
        assert (tmp_path / "objects" / item.storage_ref["object_key"]).exists()
        assert registry_get is not None
        assert registry_get.object_key == item.storage_ref["object_key"]
    elif scenario == "recaptcha_auth_required":
        assert item.status == "requires_login"
        assert item.storage_ref is None
        assert item.login_instructions
        assert updated.status == DownloadTaskStatus.AUTH_REQUIRED
        assert updated.requires_login
        assert not (tmp_path / "papers" / "p1.pdf").exists()
        assert registry_get is None
    else:  # open_access_landing_page
        assert item.status == "downloaded"
        assert item.storage_ref
        assert updated.status == DownloadTaskStatus.VERIFIED
        assert registry_get is not None
