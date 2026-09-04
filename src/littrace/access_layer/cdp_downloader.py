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
    *,
    browser: "CDPBrowser | None" = None,
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

    own_browser = browser is None
    if own_browser:
        browser = CDPBrowser(
            config.cdp_downloader.cdp_url,
            reconnect_attempts=config.cdp_downloader.websocket_reconnect_attempts,
            command_timeout_seconds=config.cdp_downloader.command_timeout_seconds,
        )
    try:
        result.steps.append("cdp_open")
        if own_browser:
            # Round 28: only open a fresh tab when we own the
            # browser. When the caller passes a shared browser
            # (e.g. across a batch of gated papers) we reuse the
            # existing tab and just ``browser.navigate(url)`` per
            # paper — otherwise 45 gated papers would create 45
            # ``about:blank`` phantom tabs the user has to close
            # by hand. ``download_blob_to_file`` below calls
            # ``browser.navigate`` on the same tab, so reuse is
            # free.
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
            # Round 27: when CF blocks us we now (1) write a flag
            # file the GUI watches so it can pop a modal asking the
            # user to clear the challenge in their Chrome, and
            # (2) poll for a user-acknowledgement file before
            # giving up. The previous "wait 60s + sleep 30s +
            # silently fail" path made the user wonder why 45 gated
            # papers silently dropped. The new path stays alive up
            # to ``user_action_wait_seconds`` (now 5 min default)
            # and only fails if the user doesn't acknowledge.
            _write_cf_wait_flag(
                config,
                doi=normalized_doi,
                url=landing_url,
                publisher=publisher,
            )
            ack = _wait_for_user_acknowledgement(
                config,
                max_wait_seconds=config.cdp_downloader.user_action_wait_seconds,
                browser=browser,
            )
            _clear_cf_wait_flag(config)
            if not ack:
                # User never clicked "I handled it" — surface the
                # failure so sentinel can record a download warning
                # instead of just dropping the paper.
                result.requires_user_action = True
                result.user_action = (
                    "sentinel waited for you to clear the Cloudflare "
                    "challenge in the local Chrome window and timed out. "
                    "Click retry on the next run — the cookie should now "
                    "be live in the shared profile."
                )
                result.steps.append("user_action:cloudflare_timeout")
                result.error = (
                    "Cloudflare verification did not complete in the local "
                    "CDP browser (user did not acknowledge within "
                    f"{config.cdp_downloader.user_action_wait_seconds:.0f}s)."
                )
                return result
            result.steps.append("user_action:cloudflare_acknowledged")
        # Round 27: clear the institutional-login wait flag too
        # before the body-text check so a successful CF pass doesn't
        # leave stale "waiting" markers in the data dir.
        if looks_like_institutional_login_needed(browser.get_body_text(1200)):
            result.requires_user_action = True
            result.user_action = "请在已打开的本地 Chrome 窗口中完成机构登录。"
            result.steps.append("user_action:institutional_login")
            _surface_browser(browser.get_url())
            time.sleep(max(config.cdp_downloader.user_action_wait_seconds, 0.0))

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
        # Round 29: only tear down the tab when we own the
        # browser. ``downloads.execute_downloads`` builds a single
        # shared ``CDPBrowser`` for the whole batch and threads it
        # through every gated paper; closing the tab here would
        # force the next paper to open a fresh ``about:blank`` and
        # 45 gated papers → 45 phantom tabs the user has to close
        # by hand. When we own the browser (``own_browser`` was
        # true above) the per-tab close is part of the per-paper
        # lifecycle, but a follow-up Round 30 will move that
        # cleanup into the batch teardown instead of leaking the
        # tab open at the end of a per-paper call.
        if own_browser:
            try:
                browser.close_tab()
            except Exception:
                pass


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
    """Bring the current publisher page to the foreground for user action.

    Round 22: the previous behaviour (``open -a "Google Chrome" <url>``)
    spawned a *new* tab in the system Chrome, which the user had to
    re-authorize against Cloudflare / SSO every single time — even
    when QtCDPBrowser already had a logged-in Chrome open on the
    same ``--user-data-dir``. The cookie jar of the QtCDPBrowser
    Chrome and the system Chrome are *different* (different
    profile dir, different Keychain entry on macOS), so the
    ``cf_clearance`` cookie that sentinel relied on for ACS / Wiley
    / Springer lived in one and was missing from the other. The
    user-visible symptom: every gated PDF the user tried to fetch
    prompted a fresh login prompt.

    Round 22 fix: don't open anything. Tell QtCDPBrowser's live
    Chrome window to come to the front instead. Sentinel already
    navigates the QtCDPBrowser Chrome to the publisher URL via CDP
    (``browser.navigate`` above) — the only thing missing is
    focus. On macOS we use ``osascript`` to activate the running
    Chrome process; on Linux / Windows we fall back to ``wmctrl`` /
    PowerShell + a CDP ``Page.bringToFront``. None of these
    requires opening a new tab.
    """
    if not url:
        return
    # Round 28: try multiple "raise the Chrome window" strategies
    # per platform so the user is never left without a visible
    # browser. Each strategy is a best-effort no-op if it fails
    # (the sentinel will still time out and the user can
    # manually switch to Chrome). We log nothing — failures here
    # are not actionable for the sentinel run.
    def _try(cmd: list[str]) -> None:
        try:
            subprocess.run(
                cmd, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    if sys.platform == "darwin":
        # osascript: activate any Google Chrome process. Most
        # reliable on macOS — no extra deps required.
        _try(["osascript", "-e",
              'tell application "Google Chrome" to activate'])
    elif sys.platform.startswith("linux"):
        # wmctrl is the standard tool but not always installed.
        # xdotool is more common on GNOME. Try in order, ignore
        # all failures — the user can still click on Chrome
        # manually if every strategy fails.
        if subprocess.run(
            ["which", "wmctrl"], check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0:
            _try(["wmctrl", "-a", "Google Chrome"])
        if subprocess.run(
            ["which", "xdotool"], check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0:
            _try(["xdotool", "search", "--name", "Google Chrome",
                  "windowactivate"])
    else:  # Windows
        _try(["powershell", "-NoProfile", "-Command",
              "(New-Object -ComObject Shell.Application)."
              "AppActivate('Google Chrome')"])
    # Note: any per-strategy failure is swallowed inside _try —
    # if every strategy failed, the user can still switch to
    # Chrome manually. The sentinel will time out and report
    # requires_user_action; this surface is best-effort.


# ---------------------------------------------------------------------------
# Round 27: cross-process "user is clearing the CF challenge" signal
# ---------------------------------------------------------------------------
#
# The sentinel subprocess writes ``sentinel_cf_wait.json`` under the
# project's ``data/`` dir the moment it hits a Cloudflare challenge.
# The littrace-qt main process polls for the same file every 2s and
# pops a modal pointing the user at the Chrome window. When the user
# clicks "我处理好了", the GUI deletes the wait file and writes
# ``sentinel_cf_ack.json``. The sentinel subprocess polls the ack
# file in ``_wait_for_user_acknowledgement`` and continues once the
# ack is present.
#
# File-based signalling (instead of DB rows or websockets) keeps this
# self-contained, cross-platform, and zero-permission. The data dir
# already exists (sentinel and GUI both write to it).



def _cf_data_dir() -> "Path":
    """Absolute path under the LitTrace project root. Using an
    absolute path removes any cwd-dependence — both the sentinel
    subprocess and the GUI process compute the same value
    regardless of which directory they were launched from."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data"


def _cf_wait_file(config: "LitTraceConfig") -> "Path":
    return _cf_data_dir() / "sentinel_cf_wait.json"


def _cf_ack_file(config: "LitTraceConfig") -> "Path":
    return _cf_data_dir() / "sentinel_cf_ack.json"


def _write_cf_wait_flag(
    config: "LitTraceConfig",
    *,
    doi: str,
    url: str,
    publisher: str,
) -> None:
    """Round 28: append the gated paper to the wait-flag list rather
    than overwriting it. Sentinel processes gated papers serially
    within a single run, and 45 papers can each trigger a CF
    challenge — without accumulation the GUI's modal would
    ``close()`` the previous entry's dialog and pop a new one for
    every paper, hiding the earlier DOIs. With accumulation the
    dialog shows the full list of papers blocked on CF.

    Older flag files (single-DOI schema) are upgraded in place so
    the GUI keeps working across upgrades.
    """
    import json
    path = _cf_wait_file(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_entry = {
        "doi": doi,
        "url": url,
        "publisher": publisher,
        "added_at": time.time(),
    }
    # Best-effort merge. If anything goes wrong we silently
    # leave the existing flag alone — sentinel will still time
    # out and report the user-action error to the GUI.
    try:
        existing_entries: list = []
        oldest_added_at: float | None = None
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    # New schema: {"dois": [...], "oldest_added_at": ...}
                    if "dois" in payload and isinstance(
                        payload["dois"], list
                    ):
                        existing_entries = list(payload["dois"])
                        oldest_added_at = payload.get("oldest_added_at")
                    # Old schema: {"doi": ..., "url": ..., "started_at": ...}
                    elif "doi" in payload:
                        existing_entries = [payload]
                        oldest_added_at = payload.get("started_at")
            except Exception:
                existing_entries = []
        # De-dup: if the same DOI is already queued, just refresh
        # the URL / publisher / added_at rather than re-adding.
        existing_entries = [
            e for e in existing_entries
            if isinstance(e, dict) and e.get("doi") != doi
        ]
        existing_entries.append(new_entry)
        payload = {
            "dois": existing_entries,
            "oldest_added_at": (
                oldest_added_at
                if oldest_added_at is not None
                else new_entry["added_at"]
            ),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # Best effort. The sentinel will still time out and
        # surface the error to the GUI; we just lose the live
        # "I am waiting" indicator.
        pass


def _clear_cf_wait_flag(config: "LitTraceConfig") -> None:
    """Remove the wait file once the user has handled the challenge
    OR the wait timed out. The GUI also clears this file when the
    user clicks "我处理好了" — whichever side wins, the other side
    is idempotent (``Path.unlink(missing_ok=True)``-style)."""
    path = _cf_wait_file(config)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _wait_for_user_acknowledgement(
    config: "LitTraceConfig",
    *,
    max_wait_seconds: float,
    browser,
) -> bool:
    """Block (poll every 2s, ~no CPU) until either the GUI writes
    ``sentinel_cf_ack.json`` (the user clicked "我处理好了") **or**
    the cf_clearance cookie actually appears in the live Chrome
    session — whichever comes first. We poll the cookie too because
    in the happy path the user clears the challenge but forgets to
    click OK; we shouldn't make them do both.

    Returns True on success, False on timeout.
    """
    ack_path = _cf_ack_file(config)
    deadline = time.monotonic() + max(max_wait_seconds, 5.0)
    while time.monotonic() < deadline:
        # 1) User clicked OK in the GUI
        if ack_path.exists():
            try:
                ack_path.unlink()
            except Exception:
                pass
            return True
        # 2) Cookies cleared the challenge on their own
        try:
            if not browser.is_cloudflare_challenge():
                return True
        except Exception:
            # Chrome went away — give up gracefully so the sentinel
            # subprocess can report a clean error instead of hanging
            # forever.
            return False
        time.sleep(2.0)
    return False

