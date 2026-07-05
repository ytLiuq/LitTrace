import base64

from littrace.authorized_pdf_archiver import archive_authorized_pdf_response
from littrace.config import LitTraceConfig, StorageConfig
from littrace.models import PaperMetadata


def test_archive_authorized_pdf_response_writes_base64_pdf(monkeypatch, tmp_path):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
    )
    calls = []

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = 0
            self.recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        calls.append(args)
        if "requests" in args:
            return Result(
                "# format: csv\n"
                "request_id,method,status,resource_type,mime_type,timestamp,url\n"
                "req1,GET,200,Document,application/pdf,1,"
                "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF\n"
            )
        return Result(
            "request_id=req1\n"
            "status=200\n"
            "mime_type=application/pdf\n"
            "response_headers:\n"
            "  content-type=application/pdf;charset=UTF-8\n"
            "  content-disposition=inline; filename=paper.pdf\n"
            "response_body_base64_encoded=True\n"
            "response_body=JVBERi0xLjQK"
        )

    monkeypatch.setattr("littrace.authorized_pdf_archiver.run_browser_act", fake_run)

    result = archive_authorized_pdf_response(
        config,
        paper,
        "littrace-acs-pdf",
        "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF",
    )

    assert result.archived
    assert result.filename == "paper.pdf"
    assert calls[0][2:4] == ["network", "requests"]
    assert (tmp_path / "papers" / "unknown-year" / "10.1021_acsomega.2c06548" / "paper.pdf").read_bytes().startswith(b"%PDF")


def test_archive_authorized_pdf_response_reports_viewer_shell(monkeypatch, tmp_path):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
    )

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = 0
            self.recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        if "requests" in args:
            return Result(
                "# format: csv\n"
                "request_id,method,status,resource_type,mime_type,timestamp,url\n"
                "req1,GET,200,Document,application/pdf,1,"
                "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF\n"
            )
        return Result(
            "request_id=req1\n"
            "status=200\n"
            "response_headers:\n"
            "  content-type=application/pdf;charset=UTF-8\n"
            "response_body_base64_encoded=False\n"
            "response_body=<!doctype html><embed type='application/pdf'>"
        )

    monkeypatch.setattr("littrace.authorized_pdf_archiver.run_browser_act", fake_run)

    result = archive_authorized_pdf_response(
        config,
        paper,
        "littrace-acs-pdf",
        "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF",
    )

    assert not result.archived
    assert result.warning == "needs_binary_body_export"


def test_archive_authorized_pdf_response_falls_back_to_browser_context_fetch(
    monkeypatch, tmp_path
):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
    )

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = 0
            self.recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        if "requests" in args:
            return Result(
                "# format: csv\n"
                "request_id,method,status,resource_type,mime_type,timestamp,url\n"
                "req1,GET,200,Document,application/pdf,1,"
                "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF\n"
            )
        if "request" in args:
            return Result(
                "request_id=req1\n"
                "status=200\n"
                "response_headers:\n"
                "  content-type=application/pdf;charset=UTF-8\n"
                "response_body_base64_encoded=False\n"
                "response_body=<!doctype html><embed type='application/pdf'>"
            )
        encoded_pdf = base64.b64encode(b"%PDF-1.7\nbrowser-context").decode()
        return Result(
            "{"
            '"status":200,'
            '"contentType":"application/pdf;charset=UTF-8",'
            '"contentDisposition":"inline; filename=context.pdf",'
            f'"bodyBase64":"{encoded_pdf}"'
            "}"
        )

    monkeypatch.setattr("littrace.authorized_pdf_archiver.run_browser_act", fake_run)

    result = archive_authorized_pdf_response(
        config,
        paper,
        "littrace-acs-pdf",
        "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF",
    )

    assert result.archived
    assert result.method == "browser_context_fetch"
    assert result.filename == "context.pdf"
    target = tmp_path / "papers" / "unknown-year" / "10.1021_acsomega.2c06548" / "paper.pdf"
    assert target.read_bytes().startswith(b"%PDF-1.7")


def test_archive_authorized_pdf_response_fetches_without_network_pdf_record(
    monkeypatch, tmp_path
):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
    )

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = 0
            self.recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        if "requests" in args:
            return Result("# format: csv\nrequest_id,method,status,resource_type,mime_type,timestamp,url\n")
        encoded_pdf = base64.b64encode(b"%PDF-1.7\nno-network-record").decode()
        return Result(
            "{"
            '"status":200,'
            '"contentType":"application/pdf;charset=UTF-8",'
            '"contentDisposition":"inline; filename=fetched.pdf",'
            f'"bodyBase64":"{encoded_pdf}"'
            "}"
        )

    monkeypatch.setattr("littrace.authorized_pdf_archiver.run_browser_act", fake_run)

    result = archive_authorized_pdf_response(
        config,
        paper,
        "littrace-acs-auth",
        "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548",
    )

    assert result.archived
    assert result.method == "browser_context_fetch"
    assert result.filename == "fetched.pdf"


def test_archive_authorized_pdf_response_cookie_http_after_missing_network_and_failed_fetch(
    monkeypatch, tmp_path
):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    paper = PaperMetadata(
        paper_id="mdpi",
        title="MDPI paper",
        doi="10.3390/s24010001",
    )

    class Result:
        def __init__(self, stdout, returncode=0, stderr=""):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode
            self.recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        if "requests" in args:
            return Result("# format: csv\nrequest_id,method,status,resource_type,mime_type,timestamp,url\n")
        if "fetch(" in args[-1]:
            return Result("", returncode=1, stderr="TypeError: Failed to fetch")
        if "cookies" in args:
            return Result(
                "# format: csv\n"
                "name,value,domain,path,secure,httpOnly,sameSite,expires\n"
                "ak_bmsc,token,.mdpi.com,/,true,true,None,\n"
            )
        if args[-1] == "navigator.userAgent || ''":
            return Result('"LitTrace Test UA"')
        return Result('{"cookie":"visible=ok","userAgent":"Ignored UA"}')

    class FakeResponse:
        status_code = 200
        content = b"%PDF-1.7\nmdpi-cookie-http"
        headers = {"content-type": "application/pdf"}

    class FakeClient:
        def __init__(self, *_, **__):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url):
            return FakeResponse()

    monkeypatch.setattr("littrace.authorized_pdf_archiver.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.authorized_pdf_archiver.httpx.Client", FakeClient)

    result = archive_authorized_pdf_response(
        config,
        paper,
        "littrace-mdpi-auth",
        "https://www.mdpi.com/1424-8220/24/1/1/pdf?version=1702959558",
    )

    assert result.archived
    assert result.method == "browser_cookie_http"


def test_archive_authorized_pdf_response_click_download_after_http_fallbacks_fail(
    monkeypatch, tmp_path
):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    paper = PaperMetadata(
        paper_id="mdpi",
        title="MDPI paper",
        doi="10.3390/s24010001",
    )
    downloaded = tmp_path / "papers" / "downloaded-by-browser.pdf"

    class Result:
        def __init__(self, stdout, returncode=0, stderr=""):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode
            self.recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        if "requests" in args:
            return Result("# format: csv\nrequest_id,method,status,resource_type,mime_type,timestamp,url\n")
        if "fetch(" in args[-1]:
            return Result("", returncode=1, stderr="TypeError: Failed to fetch")
        if "cookies" in args:
            return Result(
                "# format: csv\n"
                "name,value,domain,path,secure,httpOnly,sameSite,expires\n"
                "ak_bmsc,token,.mdpi.com,/,true,true,None,\n"
            )
        if args[-1] == "navigator.userAgent || ''":
            return Result('"LitTrace Test UA"')
        if "clickedHref" in args[-1]:
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            downloaded.write_bytes(b"%PDF-1.7\nbrowser-click")
            return Result(
                '{"clicked":true,"clickedHref":"https://www.mdpi.com/1424-8220/24/1/1/pdf?version=1"}'
            )
        return Result('{"cookie":"visible=ok","userAgent":"Ignored UA"}')

    class FakeResponse:
        status_code = 403
        content = b"<html>blocked</html>"
        headers = {"content-type": "text/html"}

    class FakeClient:
        def __init__(self, *_, **__):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url):
            return FakeResponse()

    monkeypatch.setattr("littrace.authorized_pdf_archiver.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.authorized_pdf_archiver.httpx.Client", FakeClient)

    result = archive_authorized_pdf_response(
        config,
        paper,
        "littrace-mdpi-auth",
        "https://www.mdpi.com/1424-8220/24/1/1/pdf?version=1702959558",
    )

    target = tmp_path / "papers" / "unknown-year" / "10.3390_s24010001" / "paper.pdf"
    assert result.archived
    assert result.method == "browser_click_download"
    assert target.read_bytes().startswith(b"%PDF")
    assert not downloaded.exists()


def test_archive_authorized_pdf_response_falls_back_to_cookie_http(
    monkeypatch, tmp_path
):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
    )
    seen_headers = {}

    class Result:
        def __init__(self, stdout, returncode=0):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode
            self.recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        if "requests" in args:
            return Result(
                "# format: csv\n"
                "request_id,method,status,resource_type,mime_type,timestamp,url\n"
                "req1,GET,200,Document,application/pdf,1,"
                "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF\n"
            )
        if "request" in args:
            return Result(
                "request_id=req1\n"
                "status=200\n"
                "response_headers:\n"
                "  content-type=application/pdf;charset=UTF-8\n"
                "response_body_base64_encoded=False\n"
                "response_body=<!doctype html><embed type='application/pdf'>"
            )
        if "fetch(" in args[-1]:
            return Result(
                '{"status":403,"contentType":"text/html","contentDisposition":"","bodyBase64":"PGh0bWw+"}'
            )
        if "cookies" in args:
            return Result(
                "# format: csv\n"
                "name,value,domain,path,secure,httpOnly,sameSite,expires\n"
                "session,abc,.pubs.acs.org,/,true,true,None,\n"
                "access,1,.pubs.acs.org,/,true,false,None,\n"
            )
        if "navigator.userAgent" in args[-1]:
            return Result('"LitTrace Test UA"')
        return Result('{"cookie":"session=abc; access=1","userAgent":"LitTrace Test UA"}')

    class FakeResponse:
        status_code = 200
        content = b"%PDF-1.7\ncookie-http"
        headers = {
            "content-type": "application/pdf",
            "content-disposition": "inline; filename=cookie.pdf",
        }

    class FakeClient:
        def __init__(self, *_, headers=None, **__):
            seen_headers.update(headers or {})

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url):
            assert url.endswith("article_openPDF")
            return FakeResponse()

    monkeypatch.setattr("littrace.authorized_pdf_archiver.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.authorized_pdf_archiver.httpx.Client", FakeClient)

    result = archive_authorized_pdf_response(
        config,
        paper,
        "littrace-acs-pdf",
        "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF",
    )

    assert result.archived
    assert result.method == "browser_cookie_http"
    assert result.filename == "cookie.pdf"
    assert seen_headers["Cookie"] == "session=abc; access=1"
    assert seen_headers["User-Agent"] == "LitTrace Test UA"


def test_archive_authorized_pdf_response_cookie_http_falls_back_to_document_cookie(
    monkeypatch, tmp_path
):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
    )
    seen_headers = {}

    class Result:
        def __init__(self, stdout, returncode=0, stderr=""):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode
            self.recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        if "requests" in args:
            return Result(
                "# format: csv\n"
                "request_id,method,status,resource_type,mime_type,timestamp,url\n"
                "req1,GET,200,Document,application/pdf,1,"
                "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF\n"
            )
        if "request" in args:
            return Result(
                "request_id=req1\n"
                "status=200\n"
                "response_headers:\n"
                "  content-type=application/pdf;charset=UTF-8\n"
                "response_body_base64_encoded=False\n"
                "response_body=<!doctype html><embed type='application/pdf'>"
            )
        if "fetch(" in args[-1]:
            return Result(
                '{"status":403,"contentType":"text/html","contentDisposition":"","bodyBase64":"PGh0bWw+"}'
            )
        if "cookies" in args:
            return Result("", returncode=1, stderr="cookies command unavailable")
        if args[-1] == "navigator.userAgent || ''":
            return Result('"Fallback UA"')
        return Result('{"cookie":"visible=ok","userAgent":"Ignored UA"}')

    class FakeResponse:
        status_code = 200
        content = b"%PDF-1.7\ncookie-http"
        headers = {"content-type": "application/pdf"}

    class FakeClient:
        def __init__(self, *_, headers=None, **__):
            seen_headers.update(headers or {})

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url):
            return FakeResponse()

    monkeypatch.setattr("littrace.authorized_pdf_archiver.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.authorized_pdf_archiver.httpx.Client", FakeClient)

    result = archive_authorized_pdf_response(
        config,
        paper,
        "littrace-acs-pdf",
        "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF",
    )

    assert result.archived
    assert result.method == "browser_cookie_http"
    assert seen_headers["Cookie"] == "visible=ok"
    assert seen_headers["User-Agent"] == "Fallback UA"


def test_archive_authorized_pdf_response_follows_wiley_pdfdirect(
    monkeypatch, tmp_path
):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    paper = PaperMetadata(
        paper_id="wiley",
        title="Wiley paper",
        doi="10.1002/adfm.202316712",
    )
    requested_urls = []

    class Result:
        def __init__(self, stdout, returncode=0):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode
            self.recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        if "requests" in args:
            return Result(
                "# format: csv\n"
                "request_id,method,status,resource_type,mime_type,timestamp,url\n"
                "req1,GET,200,Document,text/html,1,"
                "https://advanced.onlinelibrary.wiley.com/doi/pdf/10.1002/adfm.202316712\n"
            )
        if "request" in args:
            return Result(
                "request_id=req1\n"
                "status=200\n"
                "response_headers:\n"
                "  content-type=text/html;charset=UTF-8\n"
                "response_body_base64_encoded=False\n"
                "response_body=<html><iframe id='pdf-iframe'></iframe></html>"
            )
        if "fetch(" in args[-1]:
            return Result(
                '{"status":200,"contentType":"text/html;charset=UTF-8","contentDisposition":"","bodyBase64":"PGh0bWw+"}'
            )
        if "cookies" in args:
            return Result(
                "# format: csv\n"
                "name,value,domain,path,secure,httpOnly,sameSite,expires\n"
                "JSESSIONID,wiley-session,.onlinelibrary.wiley.com,/,true,true,None,\n"
            )
        if args[-1] == "navigator.userAgent || ''":
            return Result('"Wiley Test UA"')
        return Result('{"cookie":"JSESSIONID=wiley-session","userAgent":"Wiley Test UA"}')

    class FakeResponse:
        def __init__(self, url, content, content_type, headers=None):
            self.status_code = 200
            self.content = content
            self.headers = {"content-type": content_type, **(headers or {})}
            self.url = url

        @property
        def text(self):
            return self.content.decode("utf-8")

    class FakeClient:
        def __init__(self, *_, **__):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url):
            requested_urls.append(url)
            if "/doi/pdfdirect/" in url:
                return FakeResponse(
                    url,
                    b"%PDF-1.7\nwiley",
                    "application/pdf",
                    {"content-disposition": "inline; filename=wiley.pdf"},
                )
            return FakeResponse(
                url,
                b'<script>var src = "/doi/pdfdirect/10.1002/adfm.202316712";</script>',
                "text/html;charset=UTF-8",
            )

    monkeypatch.setattr("littrace.authorized_pdf_archiver.run_browser_act", fake_run)
    monkeypatch.setattr("littrace.authorized_pdf_archiver.httpx.Client", FakeClient)

    result = archive_authorized_pdf_response(
        config,
        paper,
        "littrace-wiley-pdf",
        "https://advanced.onlinelibrary.wiley.com/doi/pdf/10.1002/adfm.202316712",
    )

    assert result.archived
    assert result.method == "wiley_pdfdirect"
    assert result.filename == "wiley.pdf"
    assert requested_urls[-1].endswith("/doi/pdfdirect/10.1002/adfm.202316712")
    target = tmp_path / "papers" / "unknown-year" / "10.1002_adfm.202316712" / "paper.pdf"
    assert target.read_bytes().startswith(b"%PDF")


def test_archive_authorized_pdf_response_prefers_pdf_over_viewer_html(monkeypatch, tmp_path):
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    paper = PaperMetadata(
        paper_id="acs",
        title="ACS paper",
        doi="10.1021/acsomega.2c06548",
    )
    requested_ids = []

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = 0
            self.recoverable_window_closed = False

    def fake_run(_config, args, **_kwargs):
        if "requests" in args:
            return Result(
                "# format: csv\n"
                "request_id,method,status,resource_type,mime_type,timestamp,url\n"
                "pdf_req,GET,200,Document,application/pdf,1,"
                "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF\n"
                "html_req,GET,200,Document,text/html,2,"
                "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF\n"
            )
        requested_ids.append(args[-1])
        return Result(
            "request_id=pdf_req\n"
            "status=200\n"
            "response_headers:\n"
            "  content-type=application/pdf;charset=UTF-8\n"
            "response_body_base64_encoded=True\n"
            "response_body=JVBERi0xLjQK"
        )

    monkeypatch.setattr("littrace.authorized_pdf_archiver.run_browser_act", fake_run)

    result = archive_authorized_pdf_response(
        config,
        paper,
        "littrace-acs-pdf",
        "https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548?ref=article_openPDF",
    )

    assert result.archived
    assert requested_ids == ["pdf_req"]
