"""CDP-based paper PDF downloader — project-integrated version.

This module wraps :mod:`littrace.access_layer.cdp_core` (the shared CDP constants and
browser automation primitives) and adds LitTrace-specific concerns:

- ``LitTraceConfig`` integration (CDP URL, timeouts, email)
- Pydantic result models (``CDPDownloadResult``, ``CDPStatus``)
- Unpaywall OA lookup
- curl fallback download
- Publisher-specific download orchestration (Wiley, RSC, IEEE, etc.)

The standalone ``universal_paper_downloader.py`` SKILL script shares the
same core logic via :mod:`littrace.access_layer.cdp_core`.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from littrace.access_layer.cdp_core import (
    CDPBrowser,
    click_sciencedirect_institution_login,
    find_recent_pdf,
    identify_publisher,
    is_pdf_file,
    looks_like_institutional_login_needed,
    move_pdf,
    normalize_doi,
    extract_pdf_url_from_page,
    prepare_elsevier_pdf_url,
    prepare_ieee_pdf_url,
    prepare_rsc_pdf_url,
    publisher_urls,
    sciencedirect_access_status,
    wait_for_sciencedirect_authorization,
)
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
        oa_downloaded = _try_unpaywall_download(
            normalized_doi,
            email,
            target_path,
            timeout_seconds=config.cdp_downloader.repository_download_timeout_seconds,
        )
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
    if not status.available and config.cdp_downloader.auto_launch_chrome:
        # Import lazily: chrome_profiles uses the shared cdp module for status
        # models, while this module is itself imported by that module.
        from littrace.chrome_profiles import launch_chrome_for_cdp

        launch = launch_chrome_for_cdp(
            config,
            profile_name=config.cdp_downloader.chrome_profile_name,
        )
        if launch.cdp_status is not None:
            status = launch.cdp_status
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
        command_timeout_seconds=config.cdp_downloader.command_timeout_seconds,
    )
    try:
        result.steps.append("cdp_open")
        browser.connect_new_tab()
        result.steps.append("cdp_stealth_prepare")
        for note in browser.prepare_and_inject_stealth():
            result.steps.append(f"stealth:{note}")
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
                result.steps.append("user_action:cloudflare")
                _surface_browser(browser.get_url())
                time.sleep(max(config.cdp_downloader.user_action_wait_seconds, 0.0))
                if browser.is_cloudflare_challenge():
                    result.error = (
                        "Cloudflare verification did not complete in the local CDP browser."
                    )
                    return result

        if looks_like_institutional_login_needed(browser.get_body_text(1200)):
            result.requires_user_action = True
            result.user_action = "请在已打开的本地 Chrome 窗口中完成机构登录。"
            result.steps.append("user_action:institutional_login")
            _surface_browser(browser.get_url())
            time.sleep(max(config.cdp_downloader.user_action_wait_seconds, 0.0))

        if publisher == "elsevier":
            access_status = sciencedirect_access_status(browser)
            result.steps.append(f"elsevier_access:{access_status}")
            if access_status == "no_access":
                clicked = click_sciencedirect_institution_login(browser)
                if clicked:
                    result.steps.append(f"elsevier_login_clicked:{clicked[:80]}")
                    time.sleep(3.0)
                result.requires_user_action = True
                result.user_action = (
                    "ScienceDirect 当前会话没有 PDF 机构授权。"
                    "请在已打开的本地 Chrome 窗口中完成机构登录/CARSI/Shibboleth 授权。"
                )
                result.steps.append("user_action:elsevier_institution_login")
                _surface_browser(browser.get_url())
                access_status = wait_for_sciencedirect_authorization(
                    browser,
                    landing_url,
                    timeout_seconds=max(config.cdp_downloader.user_action_wait_seconds, 0.0),
                )
                result.steps.append(f"elsevier_access_after_login:{access_status}")
                if access_status == "no_access":
                    result.error = "ScienceDirect PDF authorization is still unavailable after user login wait."
                    return result

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
            result.steps.append("rsc_prepare_cdn_pdf")
            pdf_url = prepare_rsc_pdf_url(browser, normalized_doi, pdf_url)
            result.source_url = pdf_url
        if publisher == "ieee":
            result.steps.append("ieee_prepare_stamp_pdf")
            pdf_url = prepare_ieee_pdf_url(browser, pdf_url)
            result.source_url = pdf_url

        if publisher == "elsevier" and "pdfft" in (pdf_url or ""):
            result.steps.append("elsevier_pdfft_cdn")
            prepared_url = prepare_elsevier_pdf_url(
                browser,
                pdf_url,
                max_wait_seconds=min(config.cdp_downloader.command_timeout_seconds, 45.0),
            )
            if prepared_url != pdf_url:
                pdf_url = prepared_url
                result.source_url = prepared_url
            if "sciencedirectassets" in prepared_url or "els-cdn" in prepared_url:
                result.steps.append("elsevier_cdn_ready")
            else:
                result.steps.append(f"elsevier_cdn_not_reached:{prepared_url[:120]}")

        if publisher == "acs":
            result.steps.append("acs_same_origin_pdf_fetch")

        result.steps.append("fetch_blob")
        ok, info = browser.fetch_blob_to_file(pdf_url, target_path)
        if not ok and "HTTP 403" in str(info):
            result.requires_user_action = True
            result.user_action = (
                "PDF 请求返回 403。请在已打开的本地 Chrome 窗口中完成账号/机构授权。"
            )
            result.steps.append("user_action:pdf_403_retry")
            _surface_browser(browser.get_url())
            time.sleep(max(config.cdp_downloader.user_action_wait_seconds, 0.0))
            result.steps.append("fetch_blob_retry_after_user_action")
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
        browser.close_tab()


# ---------------------------------------------------------------------------
# Unpaywall OA lookup + curl download (project-specific, uses httpx)
# ---------------------------------------------------------------------------


def _try_unpaywall_download(
    doi: str,
    email: str,
    target_path: Path,
    *,
    timeout_seconds: float = 120.0,
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
        ok, info = _curl_pdf(url, target_path, timeout_seconds=timeout_seconds)
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
                "--http1.1",
                "-A",
                "Mozilla/5.0 AppleWebKit/537.36 Chrome/144.0.0.0 Safari/537.36",
                "-H",
                "Accept: application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
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


def _surface_browser(url: str) -> None:
    """Bring the current publisher page to the foreground for user action."""
    if not url:
        return
    if sys.platform == "darwin":
        command = ["open", "-a", "Google Chrome", url]
    else:
        command = ["python3", "-m", "webbrowser", url]
    subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
