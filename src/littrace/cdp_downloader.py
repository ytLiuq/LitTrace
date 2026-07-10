"""CDP-based paper PDF downloader — project-integrated version.

This module wraps :mod:`littrace.cdp_core` (the shared CDP constants and
browser automation primitives) and adds LitTrace-specific concerns:

- ``LitTraceConfig`` integration (CDP URL, timeouts, email)
- Pydantic result models (``CDPDownloadResult``, ``CDPStatus``)
- Unpaywall OA lookup
- curl fallback download
- Publisher-specific download orchestration (Wiley, RSC, IEEE, etc.)

The standalone ``universal_paper_downloader.py`` SKILL script shares the
same core logic via :mod:`littrace.cdp_core`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from littrace.cdp_core import (
    CDPBrowser,
    DOI_PREFIX_MAP,
    STEALTH_JS,
    find_recent_pdf,
    identify_publisher,
    is_pdf_file,
    looks_like_institutional_login_needed,
    move_pdf,
    normalize_doi,
    prepare_ieee_pdf_url,
    prepare_rsc_pdf_url,
    publisher_urls,
    same_origin_relative_url,
    extract_pdf_url_from_page,
)

# Backward-compatible aliases — these functions used to have underscore prefixes
# in cdp_downloader.py; they now live in cdp_core without the prefix.
# Kept so existing ``from littrace.cdp_downloader import _normalize_doi`` etc. work.
_same_origin_relative_url = same_origin_relative_url
_normalize_doi = normalize_doi
_is_pdf_file = is_pdf_file
_find_recent_pdf = find_recent_pdf
_move_pdf = move_pdf
_identify_publisher = identify_publisher
_publisher_urls = publisher_urls
_looks_like_institutional_login_needed = looks_like_institutional_login_needed
_extract_pdf_url_from_page = extract_pdf_url_from_page
_prepare_rsc_pdf_url = prepare_rsc_pdf_url
_prepare_ieee_pdf_url = prepare_ieee_pdf_url
from littrace.config import LitTraceConfig


# ---------------------------------------------------------------------------
# Result models (project-specific)
# ---------------------------------------------------------------------------


class CDPDownloadResult(BaseModel):
    doi: str
    publisher: str = "unknown"
    target_path: str
    downloaded: bool = False
    method: str | None = None
    source_url: str | None = None
    file_size: int | None = None
    requires_user_action: bool = False
    user_action: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)


class CDPStatus(BaseModel):
    available: bool
    cdp_url: str
    browser: str | None = None
    web_socket_debugger_url: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# CDP status check
# ---------------------------------------------------------------------------


def check_cdp_status(config: LitTraceConfig) -> CDPStatus:
    """Check if a CDP browser is reachable at the configured URL."""
    cdp_url = config.cdp_downloader.cdp_url
    try:
        response = httpx.get(f"{cdp_url.rstrip('/')}/json/version", timeout=5.0)
        response.raise_for_status()
        payload = response.json()
        return CDPStatus(
            available=True,
            cdp_url=cdp_url,
            browser=payload.get("Browser"),
            web_socket_debugger_url=payload.get("webSocketDebuggerUrl"),
        )
    except Exception as exc:
        return CDPStatus(available=False, cdp_url=cdp_url, error=f"{exc.__class__.__name__}: {exc}")


# ---------------------------------------------------------------------------
# Main download orchestration
# ---------------------------------------------------------------------------


def download_paper_via_cdp(
    config: LitTraceConfig,
    doi: str,
    target_path: Path,
    email: str | None = None,
) -> CDPDownloadResult:
    """Download a paper PDF via the three-step pipeline:

    1. Unpaywall → curl (if OA mirror exists)
    2. CDP browser + stealth + fetch+blob
    3. Fallback: ``<a download>`` trigger
    """
    normalized_doi = normalize_doi(doi)
    publisher = identify_publisher(normalized_doi)
    result = CDPDownloadResult(
        doi=normalized_doi,
        publisher=publisher,
        target_path=str(target_path),
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    email = email or config.api.unpaywall_email or config.api.crossref_mailto

    if email:
        result.steps.append("unpaywall")
        oa_downloaded = _try_unpaywall_download(normalized_doi, email, target_path)
        if oa_downloaded[0]:
            size = target_path.stat().st_size
            result.downloaded = True
            result.method = oa_downloaded[1]
            result.file_size = size
            result.source_url = oa_downloaded[2]
            return result
        if oa_downloaded[1]:
            result.warnings.append(oa_downloaded[1])
    else:
        result.warnings.append("No Unpaywall email configured; skipped OA repository lookup.")

    status = check_cdp_status(config)
    if not status.available:
        result.error = (
            f"CDP browser is not available at {status.cdp_url}. "
            "Start Chrome with --remote-debugging-port=19222."
        )
        result.warnings.append(status.error or "CDP unavailable")
        return result

    browser = CDPBrowser(
        config.cdp_downloader.cdp_url,
        reconnect_attempts=config.cdp_downloader.websocket_reconnect_attempts,
    )
    try:
        result.steps.append("cdp_open")
        browser.connect_new_tab()
        browser.inject_stealth()
        browser.set_download_path(target_path.parent)
        urls = publisher_urls(normalized_doi, publisher)
        landing_url = urls["landing"]
        result.steps.append(f"landing:{landing_url}")
        browser.navigate(landing_url, wait_seconds=12.0)
        if browser.is_cloudflare_challenge():
            result.steps.append("cloudflare_wait")
            if not browser.wait_for_cloudflare(config.cdp_downloader.cloudflare_wait_seconds):
                result.requires_user_action = True
                result.user_action = "请在已打开的本地 Chrome 窗口中完成 Cloudflare 人机验证。"
                time.sleep(max(config.cdp_downloader.user_action_wait_seconds, 0.0))
                if browser.is_cloudflare_challenge():
                    result.error = (
                        "Cloudflare verification did not complete in the local CDP browser."
                    )
                    return result

        if looks_like_institutional_login_needed(browser.get_body_text(1200)):
            result.requires_user_action = True
            result.user_action = "请在已打开的本地 Chrome 窗口中完成机构登录。"
            time.sleep(max(config.cdp_downloader.user_action_wait_seconds, 0.0))

        pdf_url = urls.get("pdf") or extract_pdf_url_from_page(browser, normalized_doi, publisher)
        if not pdf_url:
            result.error = "Could not infer a publisher PDF URL from the CDP browser page."
            return result
        result.source_url = pdf_url

        if publisher == "wiley":
            result.steps.append("wiley_pdfdirect_goto")
            browser.navigate(pdf_url, wait_seconds=12.0)
            found = find_recent_pdf(target_path.parent, target_path)
            if found:
                move_pdf(found, target_path)
                result.downloaded = True
                result.method = "cdp_goto_download"
                result.file_size = target_path.stat().st_size
                return result

        if publisher == "rsc":
            pdf_url = prepare_rsc_pdf_url(browser, normalized_doi, pdf_url)
            result.source_url = pdf_url
        if publisher == "ieee":
            pdf_url = prepare_ieee_pdf_url(browser, pdf_url)
            result.source_url = pdf_url

        result.steps.append("fetch_blob")
        ok, info = browser.fetch_blob_to_file(pdf_url, target_path)
        if ok:
            result.downloaded = True
            result.method = "cdp_fetch_blob"
            result.file_size = int(info)
            return result

        result.warnings.append(f"fetch_blob failed: {info}")
        result.steps.append("anchor_download")
        browser.trigger_anchor_download(pdf_url, target_path.name)
        time.sleep(15.0)
        found = find_recent_pdf(target_path.parent, target_path)
        if found:
            move_pdf(found, target_path)
            result.downloaded = True
            result.method = "cdp_anchor_download"
            result.file_size = target_path.stat().st_size
            return result
        result.error = "All CDP publisher download methods failed."
        return result
    except Exception as exc:
        result.error = f"{exc.__class__.__name__}: {exc}"
        return result
    finally:
        browser.close()


# ---------------------------------------------------------------------------
# Unpaywall OA lookup + curl download (project-specific, uses httpx)
# ---------------------------------------------------------------------------


def _try_unpaywall_download(
    doi: str,
    email: str,
    target_path: Path,
) -> tuple[bool, str | None, str | None]:
    """Query Unpaywall for OA PDF mirrors and try curl download."""
    try:
        response = httpx.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email},
            timeout=20.0,
            follow_redirects=True,
        )
        if response.status_code == 404:
            return False, None, None
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return False, f"Unpaywall lookup failed: {exc.__class__.__name__}: {exc}", None
    locations = []
    if data.get("best_oa_location"):
        locations.append(data["best_oa_location"])
    locations.extend(data.get("oa_locations") or [])
    for location in locations:
        url = location.get("url_for_pdf") or location.get("url")
        if not url:
            continue
        host_type = location.get("host_type")
        if host_type != "repository" and not data.get("is_oa"):
            continue
        ok, info = _curl_pdf(url, target_path)
        if ok:
            method = (
                "unpaywall_repository_curl"
                if host_type == "repository"
                else "unpaywall_publisher_curl"
            )
            return True, method, url
    return False, None, None


def _curl_pdf(url: str, target_path: Path, timeout_seconds: float = 60.0) -> tuple[bool, str]:
    """Download a PDF via ``curl`` with temporary file and PDF validation."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        resolved_url = _resolve_repository_handle(url)
        completed = subprocess.run(
            [
                "curl",
                "-sL",
                "-o",
                str(tmp_path),
                "-w",
                "%{http_code}",
                "--max-time",
                str(int(timeout_seconds)),
                resolved_url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 10,
        )
        if tmp_path.exists() and tmp_path.stat().st_size > 1000 and is_pdf_file(tmp_path):
            move_pdf(tmp_path, target_path)
            return True, str(target_path.stat().st_size)
        if tmp_path.exists():
            tmp_path.unlink()
        return False, f"curl failed: HTTP {completed.stdout.strip() or '000'}"
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return False, f"{exc.__class__.__name__}: {exc}"


def _resolve_repository_handle(url: str) -> str:
    """Resolve DSpace/handle repository URLs to direct PDF links."""
    if "hdl.handle.net" not in url and "handle" not in url:
        return url
    try:
        response = httpx.get(url, timeout=15.0, follow_redirects=True)
        match = re.search(r'href="([^"]*bitstream[^"]*\.pdf[^"]*)"', response.text)
        if not match:
            return url
        return str(httpx.URL(response.url).join(match.group(1)))
    except Exception:
        return url
