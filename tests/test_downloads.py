import pytest
import httpx

from littrace.artifact_registry import artifact_registry_from_config
from littrace.config import LitTraceConfig, ArtifactStorageConfig, StorageConfig
from littrace.download_tasks import DownloadTask, DownloadTaskStatus
from littrace.downloads import _execute_one, execute_downloads
from littrace.models import AccessType, DownloadExecutionRequest, PaperMetadata


@pytest.mark.anyio
async def test_execute_downloads_plans_cdp_for_gated_papers_with_doi():
    result = await execute_downloads(
        LitTraceConfig(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Login paper",
                doi="10.1002/example",
                access_type=AccessType.REQUIRES_LOGIN,
                source_urls=["https://example.org/login-paper"],
            )
        ],
        DownloadExecutionRequest(dry_run=True),
    )

    assert result.requires_login_count == 1
    assert result.items[0].action == "cdp_publisher_download"
    assert result.items[0].status == "planned"
    assert result.items[0].target_path


@pytest.mark.anyio
async def test_execute_one_stores_downloaded_pdf_as_object_ref(tmp_path):
    class FakeClient:
        async def get(self, _url):
            return httpx.Response(
                200,
                request=httpx.Request("GET", "https://example.org/paper.pdf"),
                headers={"content-type": "application/pdf"},
                content=b"%PDF-1.4\nexample",
            )

    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
        ),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    paper = PaperMetadata(
        paper_id="p1",
        title="OA paper",
        pdf_url="https://example.org/paper.pdf",
        access_type=AccessType.OPEN_ACCESS,
    )
    task = DownloadTask.from_paper(config, paper, session_id="s1", user_id="u1")

    item, updated = await _execute_one(FakeClient(), config, paper, False, task)

    assert item.status == "downloaded"
    assert item.storage_ref
    assert item.storage_ref["object_key"] == "users/u1/sessions/s1/papers/p1/paper.pdf"
    assert updated.status == DownloadTaskStatus.VERIFIED
    assert updated.target_object_key == item.storage_ref["object_key"]
    assert (tmp_path / "objects" / item.storage_ref["object_key"]).exists()
    record = artifact_registry_from_config(config).get("paper_pdf:p1", user_id="u1", session_id="s1")
    assert record is not None
    assert record.object_key == item.storage_ref["object_key"]


@pytest.mark.anyio
async def test_execute_one_marks_recaptcha_html_pdf_url_as_auth_required(tmp_path):
    class FakeClient:
        async def get(self, _url):
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

    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
        ),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    paper = PaperMetadata(
        paper_id="p1",
        title="OA paper",
        pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13367112/pdf/ADVS-9999-e76572.pdf",
        access_type=AccessType.OPEN_ACCESS,
    )
    task = DownloadTask.from_paper(config, paper, session_id="s1", user_id="u1")

    item, updated = await _execute_one(FakeClient(), config, paper, False, task)

    assert item.status == "requires_login"
    assert item.storage_ref is None
    assert item.login_instructions
    assert updated.status == DownloadTaskStatus.AUTH_REQUIRED
    assert updated.requires_login
    assert not (tmp_path / "papers" / "p1.pdf").exists()
    assert artifact_registry_from_config(config).get("paper_pdf:p1", user_id="u1", session_id="s1") is None


@pytest.mark.anyio
async def test_execute_one_downloads_pdf_link_from_open_access_landing_page(tmp_path):
    class FakeClient:
        async def get(self, url):
            if str(url) == "https://example.org/article":
                return httpx.Response(
                    200,
                    request=httpx.Request("GET", "https://example.org/article"),
                    headers={"content-type": "text/html; charset=utf-8"},
                    content=b'<html><body><a href="/downloads/paper.pdf">PDF</a></body></html>',
                )
            if str(url) == "https://example.org/downloads/paper.pdf":
                return httpx.Response(
                    200,
                    request=httpx.Request("GET", "https://example.org/downloads/paper.pdf"),
                    headers={"content-type": "application/pdf"},
                    content=b"%PDF-1.4\nexample",
                )
            raise AssertionError(str(url))

    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
        ),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    paper = PaperMetadata(
        paper_id="p1",
        title="OA landing page paper",
        pdf_url="https://example.org/article",
        access_type=AccessType.OPEN_ACCESS,
    )
    task = DownloadTask.from_paper(config, paper, session_id="s1", user_id="u1")

    item, updated = await _execute_one(FakeClient(), config, paper, False, task)

    assert item.status == "downloaded"
    assert item.storage_ref
    assert updated.status == DownloadTaskStatus.VERIFIED
    assert artifact_registry_from_config(config).get("paper_pdf:p1", user_id="u1", session_id="s1") is not None


@pytest.mark.anyio
async def test_execute_one_falls_back_to_cdp_for_open_access_403(monkeypatch, tmp_path):
    class FakeClient:
        async def get(self, _url):
            raise httpx.HTTPStatusError(
                "forbidden",
                request=httpx.Request("GET", "https://example.org/paper.pdf"),
                response=httpx.Response(403, request=httpx.Request("GET", "https://example.org/paper.pdf")),
            )

    cdp_calls: dict[str, object] = {}

    def fake_download_paper_via_cdp(config, doi, target_path):
        cdp_calls["doi"] = doi
        cdp_calls["target_path"] = str(target_path)

        class Result:
            downloaded = False
            requires_user_action = True
            user_action = "请登录"
            error = "needs login"

        return Result()

    monkeypatch.setattr("littrace.downloads.download_paper_via_cdp", fake_download_paper_via_cdp)
    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
        ),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    paper = PaperMetadata(
        paper_id="p1",
        title="OA paper",
        doi="10.1002/example",
        pdf_url="https://example.org/paper.pdf",
        access_type=AccessType.OPEN_ACCESS,
    )
    task = DownloadTask.from_paper(config, paper, session_id="s1", user_id="u1")

    item, updated = await _execute_one(FakeClient(), config, paper, False, task)

    assert item.action == "cdp_publisher_download"
    assert item.status == "requires_login"
    assert cdp_calls["doi"] == "10.1002/example"
    assert updated.status == DownloadTaskStatus.AUTH_REQUIRED


@pytest.mark.anyio
async def test_execute_one_falls_back_to_cdp_for_open_access_timeout(monkeypatch, tmp_path):
    class FakeClient:
        async def get(self, _url):
            raise httpx.ConnectTimeout("timed out", request=httpx.Request("GET", "https://example.org/paper.pdf"))

    cdp_calls: dict[str, object] = {}

    def fake_download_paper_via_cdp(config, doi, target_path):
        cdp_calls["doi"] = doi
        cdp_calls["target_path"] = str(target_path)

        class Result:
            downloaded = False
            requires_user_action = True
            user_action = "请登录"
            error = "needs login"

        return Result()

    monkeypatch.setattr("littrace.downloads.download_paper_via_cdp", fake_download_paper_via_cdp)
    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
        ),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    paper = PaperMetadata(
        paper_id="p1",
        title="OA paper",
        doi="10.1002/example",
        pdf_url="https://example.org/paper.pdf",
        access_type=AccessType.OPEN_ACCESS,
    )
    task = DownloadTask.from_paper(config, paper, session_id="s1", user_id="u1")

    item, updated = await _execute_one(FakeClient(), config, paper, False, task)

    assert item.action == "cdp_publisher_download"
    assert item.status == "requires_login"
    assert cdp_calls["doi"] == "10.1002/example"
    assert updated.status == DownloadTaskStatus.AUTH_REQUIRED


@pytest.mark.anyio
async def test_execute_one_does_not_cdp_fallback_unknown_publisher_timeout(monkeypatch, tmp_path):
    class FakeClient:
        async def get(self, _url):
            raise httpx.ConnectTimeout("timed out", request=httpx.Request("GET", "https://example.org/paper.pdf"))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("unknown publisher timeout should not call CDP")

    monkeypatch.setattr("littrace.downloads.download_paper_via_cdp", fail_if_called)
    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
        ),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )
    paper = PaperMetadata(
        paper_id="p1",
        title="OA paper",
        doi="10.37904/example",
        pdf_url="https://example.org/paper.pdf",
        access_type=AccessType.OPEN_ACCESS,
    )
    task = DownloadTask.from_paper(config, paper, session_id="s1", user_id="u1")

    item, updated = await _execute_one(FakeClient(), config, paper, False, task)

    assert item.action == "download"
    assert item.status == "failed"
    assert "ConnectTimeout" in (item.error or "")
    assert updated.status == DownloadTaskStatus.FAILED


@pytest.mark.anyio
async def test_execute_downloads_uses_request_user_and_session_for_storage_refs(monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, _url):
            return httpx.Response(
                200,
                request=httpx.Request("GET", "https://example.org/paper.pdf"),
                headers={"content-type": "application/pdf"},
                content=b"%PDF-1.4\nexample",
            )

    monkeypatch.setattr("littrace.downloads.httpx.AsyncClient", FakeClient)
    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
        ),
        artifact_storage=ArtifactStorageConfig(local_root=tmp_path / "objects"),
    )

    result = await execute_downloads(
        config,
        [
            PaperMetadata(
                paper_id="p1",
                title="OA paper",
                pdf_url="https://example.org/paper.pdf",
                access_type=AccessType.OPEN_ACCESS,
            )
        ],
        DownloadExecutionRequest(user_id="u1", session_id="s1"),
    )

    assert result.items[0].storage_ref["object_key"] == "users/u1/sessions/s1/papers/p1/paper.pdf"
