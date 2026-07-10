from pathlib import Path

from littrace.cdp_downloader import (
    _same_origin_relative_url,
    check_cdp_status,
    identify_publisher,
    publisher_urls,
)
from littrace.config import CDPDownloaderConfig, LitTraceConfig


def test_identify_publisher_from_doi_prefix():
    assert identify_publisher("10.1002/adfm.202316712") == "wiley"
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
