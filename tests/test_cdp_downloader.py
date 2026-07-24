
from littrace.cdp_core import CDPBrowser, STEALTH_JS, discover_elsevier_pdf_candidates
from littrace.cdp_downloader import (
    _same_origin_relative_url,
    check_cdp_status,
    identify_publisher,
    publisher_urls,
)
from littrace.config import CDPDownloaderConfig, LitTraceConfig


def test_identify_publisher_from_doi_prefix():
    assert identify_publisher("10.1002/adfm.202316712") == "wiley"
    assert identify_publisher("10.1007/s10853-025-11937-9") == "springer_nature"
    assert identify_publisher("10.1021/acsnano.6c02465") == "acs"
    assert identify_publisher("10.1039/d5nr04405g") == "rsc"
    assert identify_publisher("10.9999/example") == "unknown"


def test_wiley_uses_pdfdirect_download_url():
    urls = publisher_urls("10.1002/adfm.202316712", "wiley")

    assert urls["landing"] == "https://advanced.onlinelibrary.wiley.com/doi/10.1002/adfm.202316712"
    assert urls["pdf"] == (
        "https://advanced.onlinelibrary.wiley.com/doi/pdfdirect/"
        "10.1002/adfm.202316712?download=true"
    )


def test_springer_1007_uses_link_springer_pdf_url():
    urls = publisher_urls("10.1007/s10853-025-11937-9", "springer_nature")

    assert urls["landing"] == "https://link.springer.com/article/10.1007/s10853-025-11937-9"
    assert urls["pdf"] == (
        "https://link.springer.com/content/pdf/10.1007/s10853-025-11937-9.pdf"
    )


def test_same_origin_relative_url_avoids_cors():
    assert _same_origin_relative_url(
        "https://pubs.acs.org/doi/10.1021/example",
        "https://pubs.acs.org/doi/pdf/10.1021/example",
    ) == "/doi/pdf/10.1021/example"
    assert _same_origin_relative_url(
        "https://pubs.acs.org/doi/10.1021/example",
        "https://example.org/paper.pdf",
    ) == "https://example.org/paper.pdf"


def test_check_cdp_status_reports_unavailable(monkeypatch):
    class Boom:
        def raise_for_status(self):
            raise RuntimeError("no chrome")

    monkeypatch.setattr("littrace.cdp_downloader.httpx.get", lambda *_args, **_kwargs: Boom())
    config = LitTraceConfig(cdp_downloader=CDPDownloaderConfig(cdp_url="http://127.0.0.1:65535"))

    status = check_cdp_status(config)

    assert not status.available
    assert "RuntimeError" in (status.error or "")


def test_stealth_js_contains_latest_skill_enhancements():
    assert "hardwareConcurrency" in STEALTH_JS
    assert "userAgentData" in STEALTH_JS
    assert "HeadlessChrome" in STEALTH_JS
    assert "Chrome PDF Viewer" in STEALTH_JS


def test_prepare_stealth_context_normalizes_headless_http_ua(monkeypatch):
    sent: list[tuple[str, dict | None]] = []
    browser = CDPBrowser("http://127.0.0.1:19222")

    def fake_send(method, params=None):
        sent.append((method, params))
        return {}

    monkeypatch.setattr(browser, "send", fake_send)
    monkeypatch.setattr(
        browser,
        "eval",
        lambda _expr: (
            "Mozilla/5.0 AppleWebKit/537.36 "
            "HeadlessChrome/144.0.7559.60 Safari/537.36"
        ),
    )

    notes = browser.prepare_stealth_context()

    assert ("Page.enable", None) in sent
    assert ("Network.enable", None) in sent
    assert any(method == "Network.setUserAgentOverride" for method, _params in sent)
    override = [params for method, params in sent if method == "Network.setUserAgentOverride"][0]
    assert "HeadlessChrome" not in override["userAgent"]
    assert "HTTP user-agent normalized" in notes[0]


def test_discover_elsevier_pdf_candidates_prefers_sciencedirectassets(monkeypatch):
    browser = CDPBrowser("http://127.0.0.1:19222")
    payload = [
        "https://www.sciencedirect.com/science/article/pii/S0008622325011558/pdfft?md5=x",
        "https://pdf.sciencedirectassets.com/271535/1-s2.0-S0008622325011558-main.pdf?X-Amz-Security-Token=abc",
    ]
    monkeypatch.setattr(browser, "eval", lambda _expr: __import__("json").dumps(payload))

    candidates = discover_elsevier_pdf_candidates(browser)

    assert candidates[0].startswith("https://pdf.sciencedirectassets.com/")
    assert "/pdfft" in candidates[1]
