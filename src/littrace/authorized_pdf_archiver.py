from __future__ import annotations

import base64
import csv
import json
import re
import shutil
import time
from io import StringIO
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx
from pydantic import BaseModel

from littrace.access import target_pdf_path
from littrace.browser import run_browser_act
from littrace.config import LitTraceConfig
from littrace.models import PaperMetadata


class AuthorizedPdfArchiveResult(BaseModel):
    paper_id: str
    pdf_url: str
    target_path: str
    archived: bool = False
    status_code: int | None = None
    mime_type: str | None = None
    request_id: str | None = None
    filename: str | None = None
    method: str | None = None
    error: str | None = None
    warning: str | None = None


def archive_authorized_pdf_response(
    config: LitTraceConfig,
    paper: PaperMetadata,
    session_name: str,
    pdf_url: str,
    timeout_seconds: float = 20.0,
) -> AuthorizedPdfArchiveResult:
    target = target_pdf_path(config, paper)
    requests = run_browser_act(
        config,
        [
            "--session",
            session_name,
            "network",
            "requests",
            "--filter",
            _request_filter(pdf_url, paper),
        ],
        timeout_seconds=timeout_seconds,
    )
    candidate = _select_pdf_request(requests.stdout, pdf_url)
    if candidate is None:
        fallback = _download_pdf_in_browser_context(
            config,
            paper,
            session_name,
            pdf_url,
            target,
            timeout_seconds=timeout_seconds,
        )
        if fallback.archived:
            return fallback
        cookie_fallback = _download_pdf_with_browser_cookies(
            config,
            paper,
            session_name,
            pdf_url,
            target,
            timeout_seconds=timeout_seconds,
        )
        if cookie_fallback.archived:
            return cookie_fallback
        click_fallback = _download_pdf_by_browser_click(
            config,
            paper,
            session_name,
            pdf_url,
            target,
            timeout_seconds=timeout_seconds,
        )
        if click_fallback.archived:
            return click_fallback
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            method="browser_context_fetch",
            error=(
                "No authorized application/pdf network response was found. "
                f"Browser-context fetch: {fallback.error}; "
                f"cookie HTTP fallback: {cookie_fallback.error}; "
                f"browser click fallback: {click_fallback.error}"
            ),
        )

    detail = run_browser_act(
        config,
        ["--session", session_name, "network", "request", candidate.request_id],
        timeout_seconds=timeout_seconds,
    )
    body = _extract_response_body(detail.stdout)
    encoded = _extract_bool_field(detail.stdout, "response_body_base64_encoded")
    mime_type = _extract_header(detail.stdout, "content-type") or candidate.mime_type
    filename = _filename_from_content_disposition(
        _extract_header(detail.stdout, "content-disposition")
    )
    if body is None:
        fallback = _download_pdf_in_browser_context(
            config,
            paper,
            session_name,
            pdf_url,
            target,
            timeout_seconds=timeout_seconds,
        )
        if fallback.archived:
            return fallback
        fallback = _download_pdf_with_browser_cookies(
            config,
            paper,
            session_name,
            pdf_url,
            target,
            timeout_seconds=timeout_seconds,
        )
        if fallback.archived:
            return fallback
        click_fallback = _download_pdf_by_browser_click(
            config,
            paper,
            session_name,
            pdf_url,
            target,
            timeout_seconds=timeout_seconds,
        )
        if click_fallback.archived:
            return click_fallback
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            status_code=candidate.status_code,
            mime_type=mime_type,
            request_id=candidate.request_id,
            filename=filename,
            method="browser_network",
            error=(
                "PDF response body was not available from browser-act. "
                f"Cookie fallback: {fallback.error}; "
                f"browser click fallback: {click_fallback.error}"
            ),
        )

    data = base64.b64decode(body) if encoded else body.encode("latin1", errors="ignore")
    if not data.startswith(b"%PDF"):
        fallback = _download_pdf_in_browser_context(
            config,
            paper,
            session_name,
            pdf_url,
            target,
            timeout_seconds=timeout_seconds,
        )
        if fallback.archived:
            return fallback
        fallback = _download_pdf_with_browser_cookies(
            config,
            paper,
            session_name,
            pdf_url,
            target,
            timeout_seconds=timeout_seconds,
        )
        if fallback.archived:
            return fallback
        click_fallback = _download_pdf_by_browser_click(
            config,
            paper,
            session_name,
            pdf_url,
            target,
            timeout_seconds=timeout_seconds,
        )
        if click_fallback.archived:
            return click_fallback
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            status_code=candidate.status_code,
            mime_type=mime_type,
            request_id=candidate.request_id,
            filename=filename,
            method="browser_network",
            error=(
                "Authorized response was application/pdf, but browser-act exposed a PDF viewer shell instead of PDF bytes. "
                f"Browser click fallback: {click_fallback.error}"
            ),
            warning="needs_binary_body_export",
        )

    _write_pdf(target, data)
    return AuthorizedPdfArchiveResult(
        paper_id=paper.paper_id,
        pdf_url=pdf_url,
        target_path=str(target),
        archived=True,
        status_code=candidate.status_code,
        mime_type=mime_type,
        request_id=candidate.request_id,
        filename=filename,
        method="browser_network",
    )


def _download_pdf_by_browser_click(
    config: LitTraceConfig,
    paper: PaperMetadata,
    session_name: str,
    pdf_url: str,
    target: Path,
    timeout_seconds: float,
) -> AuthorizedPdfArchiveResult:
    watch_dirs = _browser_download_watch_dirs(config, target)
    before = _snapshot_pdf_files(watch_dirs)
    script = _browser_click_download_script(pdf_url)
    click = run_browser_act(
        config,
        ["--session", session_name, "eval", script],
        timeout_seconds=min(timeout_seconds, 15.0),
    )
    if click.returncode != 0:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            method="browser_click_download",
            error=click.stderr or click.stdout or "Browser download click failed.",
        )

    payload_text = _last_nonempty_line(click.stdout)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict) and payload.get("requiresUserConfirmation"):
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            method="browser_click_download",
            error="Browser page requires user confirmation before PDF download can continue.",
            warning="requires_user_confirmation",
        )
    if isinstance(payload, dict) and payload.get("clicked") is False:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            method="browser_click_download",
            error=str(payload.get("error") or "No PDF download control was found on the browser page."),
        )

    navigated_pdf = _wait_for_browser_pdf_navigation(
        config,
        session_name,
        pdf_url,
        timeout_seconds=min(timeout_seconds, 10.0),
    )
    if navigated_pdf:
        navigation_result = _archive_from_current_pdf_page(
            config,
            paper,
            session_name,
            navigated_pdf,
            target,
            timeout_seconds=timeout_seconds,
        )
        if navigation_result.archived:
            return navigation_result

    direct_navigation_error = None
    direct_navigation = _navigate_browser_to_pdf_url(
        config,
        session_name,
        pdf_url,
        timeout_seconds=min(timeout_seconds, 10.0),
    )
    if direct_navigation.returncode != 0:
        direct_navigation_error = direct_navigation.stderr or direct_navigation.stdout
    else:
        navigated_pdf = _wait_for_browser_pdf_navigation(
            config,
            session_name,
            pdf_url,
            timeout_seconds=min(timeout_seconds, 10.0),
        )
        if navigated_pdf:
            navigation_result = _archive_from_current_pdf_page(
                config,
                paper,
                session_name,
                navigated_pdf,
                target,
                timeout_seconds=timeout_seconds,
            )
            if navigation_result.archived:
                return navigation_result

    downloaded = _wait_for_new_pdf_download(
        watch_dirs,
        before,
        timeout_seconds=max(timeout_seconds, 5.0),
    )
    if downloaded is None:
        clicked = payload.get("clickedHref") if isinstance(payload, dict) else None
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=str(navigated_pdf or clicked or pdf_url),
            target_path=str(target),
            method="browser_click_download",
            error=(
                "Browser click did not produce a completed PDF download in watched directories: "
                + ", ".join(str(path) for path in watch_dirs)
                + (
                    f"; direct PDF navigation failed: {direct_navigation_error}"
                    if direct_navigation_error
                    else ""
                )
            ),
        )
    if not downloaded.read_bytes().startswith(b"%PDF"):
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            filename=downloaded.name,
            method="browser_click_download",
            error=f"Downloaded file was not a PDF: {downloaded}",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if downloaded.resolve() != target.resolve():
        shutil.move(str(downloaded), str(target))
    return AuthorizedPdfArchiveResult(
        paper_id=paper.paper_id,
        pdf_url=pdf_url,
        target_path=str(target),
        archived=True,
        filename=downloaded.name,
        method="browser_click_download",
    )


def _browser_click_download_script(pdf_url: str) -> str:
    return rf"""
(() => {{
  const normalize = (href) => {{
    try {{ return new URL(href, location.href).href; }} catch (_) {{ return null; }}
  }};
  const currentText = [
    document.title || '',
    location.href || '',
    document.body ? document.body.innerText.slice(0, 3000) : ''
  ].join('\\n').toLowerCase();
  if (
    currentText.includes('cloudflare') ||
    currentText.includes('just a moment') ||
    currentText.includes('verify you are human') ||
    currentText.includes('captcha') ||
    location.href.includes('__cf_chl')
  ) {{
    return JSON.stringify({{ clicked: false, requiresUserConfirmation: true }});
  }}
  const expected = {json.dumps(pdf_url)};
  const expectedBase = expected.split('?')[0];
  const elements = Array.from(document.querySelectorAll('a[href], button, [role="button"]'));
  const scored = elements.map((element) => {{
    const href = normalize(element.getAttribute('href') || element.href || '');
    const text = [
      element.textContent || '',
      element.getAttribute('title') || '',
      element.getAttribute('aria-label') || '',
      element.getAttribute('class') || '',
      href || ''
    ].join(' ').toLowerCase();
    let score = 0;
    if (href && href === expected) score += 30;
    if (href && href.split('?')[0] === expectedBase) score += 24;
    if (href && href.includes('/doi/pdf/')) score += 16;
    if (href && href.includes('/pdf?')) score += 18;
    if (href && href.includes('/pdf#')) score += 14;
    if (href && href.endsWith('/pdf')) score += 14;
    if (href && /\/pdf(?:$|[?#/])/.test(href)) score += 12;
    if (href && href.includes('/content/articlepdf/')) score += 16;
    if (href && href.includes('/science/article/pii/') && href.includes('/pdfft')) score += 16;
    if (href && href.includes('/pdfft')) score += 14;
    if (text.includes('download pdf')) score += 12;
    if (text.includes('view pdf')) score += 8;
    if (text.includes('pdf')) score += 5;
    return {{ element, href, text, score }};
  }}).filter((item) => item.score > 0);
  scored.sort((left, right) => right.score - left.score);
  if (!scored.length) {{
    return JSON.stringify({{ clicked: false, error: 'no_pdf_download_control' }});
  }}
  const best = scored[0];
  best.element.scrollIntoView({{ block: 'center', inline: 'center' }});
  if (best.href) {{
    location.href = best.href;
  }} else {{
    best.element.click();
  }}
  return JSON.stringify({{
    clicked: true,
    clickedHref: best.href,
    clickedText: best.text.slice(0, 160),
    score: best.score
  }});
}})()
""".strip()


def _navigate_browser_to_pdf_url(
    config: LitTraceConfig,
    session_name: str,
    pdf_url: str,
    timeout_seconds: float,
):
    script = f"location.assign({json.dumps(pdf_url)}); JSON.stringify({{navigatingTo:{json.dumps(pdf_url)}}})"
    return run_browser_act(
        config,
        ["--session", session_name, "eval", script],
        timeout_seconds=timeout_seconds,
    )


def _wait_for_browser_pdf_navigation(
    config: LitTraceConfig,
    session_name: str,
    expected_pdf_url: str,
    timeout_seconds: float,
) -> str | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() <= deadline:
        result = run_browser_act(
            config,
            ["--session", session_name, "eval", "location.href || ''"],
            timeout_seconds=5.0,
        )
        current = _last_nonempty_line(result.stdout).strip('"').strip("'")
        if _looks_like_same_pdf_url(current, expected_pdf_url):
            return current
        time.sleep(0.5)
    return None


def _archive_from_current_pdf_page(
    config: LitTraceConfig,
    paper: PaperMetadata,
    session_name: str,
    pdf_url: str,
    target: Path,
    timeout_seconds: float,
) -> AuthorizedPdfArchiveResult:
    browser_fetch = _download_pdf_in_browser_context(
        config,
        paper,
        session_name,
        pdf_url,
        target,
        timeout_seconds=timeout_seconds,
    )
    if browser_fetch.archived:
        browser_fetch.method = "browser_click_navigation_fetch"
        return browser_fetch
    cookie_fetch = _download_pdf_with_browser_cookies(
        config,
        paper,
        session_name,
        pdf_url,
        target,
        timeout_seconds=timeout_seconds,
    )
    if cookie_fetch.archived:
        cookie_fetch.method = "browser_click_navigation_cookie_http"
        return cookie_fetch
    return AuthorizedPdfArchiveResult(
        paper_id=paper.paper_id,
        pdf_url=pdf_url,
        target_path=str(target),
        method="browser_click_navigation",
        error=f"PDF navigation happened, but archiving failed: {browser_fetch.error}; {cookie_fetch.error}",
    )


def _looks_like_same_pdf_url(current_url: str, expected_pdf_url: str) -> bool:
    if not current_url:
        return False
    current = current_url.split("#", 1)[0]
    expected = expected_pdf_url.split("#", 1)[0]
    current_base = current.split("?", 1)[0]
    expected_base = expected.split("?", 1)[0]
    return current == expected or current_base == expected_base or _looks_like_pdf_url(current)


def _looks_like_pdf_url(url: str) -> bool:
    lowered = url.lower()
    return (
        lowered.endswith(".pdf")
        or ".pdf?" in lowered
        or ".pdf#" in lowered
        or "/doi/pdf/" in lowered
        or "/content/articlepdf/" in lowered
        or "/pdfft" in lowered
        or lowered.endswith("/pdf")
        or "/pdf?" in lowered
        or "/pdf#" in lowered
    )


def _browser_download_watch_dirs(config: LitTraceConfig, target: Path) -> list[Path]:
    candidates = [
        target.parent,
        Path(config.storage.paper_library_dir),
        Path.home() / "Downloads",
        Path.home() / "downloads",
    ]
    result: list[Path] = []
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded not in result:
            result.append(expanded)
    return result


def _snapshot_pdf_files(directories: list[Path]) -> dict[Path, float]:
    snapshot: dict[Path, float] = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.glob("*.pdf"):
            try:
                snapshot[path.resolve()] = path.stat().st_mtime
            except OSError:
                continue
    return snapshot


def _wait_for_new_pdf_download(
    directories: list[Path],
    before: dict[Path, float],
    timeout_seconds: float,
) -> Path | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() <= deadline:
        candidates: list[Path] = []
        for directory in directories:
            if not directory.exists():
                continue
            if list(directory.glob("*.crdownload")) or list(directory.glob("*.download")):
                continue
            for path in directory.glob("*.pdf"):
                try:
                    resolved = path.resolve()
                    stat = path.stat()
                except OSError:
                    continue
                if resolved not in before or stat.st_mtime > before[resolved]:
                    if stat.st_size > 0:
                        candidates.append(path)
        if candidates:
            return max(candidates, key=lambda item: item.stat().st_mtime)
        time.sleep(0.5)
    return None


def _download_pdf_in_browser_context(
    config: LitTraceConfig,
    paper: PaperMetadata,
    session_name: str,
    pdf_url: str,
    target: Path,
    timeout_seconds: float,
) -> AuthorizedPdfArchiveResult:
    script = f"""
(async () => {{
  const response = await fetch({json.dumps(pdf_url)}, {{
    credentials: 'include',
    headers: {{ Accept: 'application/pdf,application/octet-stream;q=0.9,*/*;q=0.8' }}
  }});
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {{
    binary += String.fromCharCode.apply(null, bytes.subarray(index, index + chunkSize));
  }}
  return JSON.stringify({{
    status: response.status,
    contentType: response.headers.get('content-type') || '',
    contentDisposition: response.headers.get('content-disposition') || '',
    bodyBase64: btoa(binary)
  }});
}})()
""".strip()
    result = run_browser_act(
        config,
        ["--session", session_name, "eval", script],
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            method="browser_context_fetch",
            error=result.stderr or result.stdout or "browser-context PDF fetch failed",
        )
    payload_text = _last_nonempty_line(result.stdout)
    try:
        payload = json.loads(payload_text)
    except Exception as exc:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            method="browser_context_fetch",
            error=f"Could not parse browser-context PDF fetch result: {exc}",
        )
    status_code = _int_or_none(payload.get("status"))
    content_type = payload.get("contentType") or None
    filename = _filename_from_content_disposition(payload.get("contentDisposition"))
    body_base64 = payload.get("bodyBase64")
    if status_code is not None and status_code >= 400:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            status_code=status_code,
            mime_type=content_type,
            filename=filename,
            method="browser_context_fetch",
            error=f"Browser-context PDF fetch returned status {status_code}.",
        )
    if not isinstance(body_base64, str) or not body_base64:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            status_code=status_code,
            mime_type=content_type,
            filename=filename,
            method="browser_context_fetch",
            error="Browser-context PDF fetch did not return a response body.",
        )
    try:
        data = base64.b64decode(body_base64)
    except Exception as exc:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            status_code=status_code,
            mime_type=content_type,
            filename=filename,
            method="browser_context_fetch",
            error=f"Could not decode browser-context PDF body: {exc}",
        )
    if not data.startswith(b"%PDF"):
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            status_code=status_code,
            mime_type=content_type,
            filename=filename,
            method="browser_context_fetch",
            error=f"Browser-context response was not a PDF: {content_type}",
        )
    _write_pdf(target, data)
    return AuthorizedPdfArchiveResult(
        paper_id=paper.paper_id,
        pdf_url=pdf_url,
        target_path=str(target),
        archived=True,
        status_code=status_code,
        mime_type=content_type,
        filename=filename,
        method="browser_context_fetch",
    )


def _download_pdf_with_browser_cookies(
    config: LitTraceConfig,
    paper: PaperMetadata,
    session_name: str,
    pdf_url: str,
    target: Path,
    timeout_seconds: float,
) -> AuthorizedPdfArchiveResult:
    browser_state = _read_browser_cookie_state(
        config, session_name, pdf_url, timeout_seconds
    )
    if browser_state.error:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            method="browser_cookie_http",
            error=browser_state.error,
        )
    headers = {
        "User-Agent": browser_state.user_agent or config.api.user_agent,
        "Referer": _referer_for_pdf(pdf_url),
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    }
    if browser_state.cookie:
        headers["Cookie"] = browser_state.cookie
    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers=headers,
            follow_redirects=True,
        ) as client:
            response = client.get(pdf_url)
    except httpx.HTTPError as exc:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            method="browser_cookie_http",
            error=f"{exc.__class__.__name__}: {exc}",
        )
    content_type = response.headers.get("content-type")
    filename = _filename_from_content_disposition(response.headers.get("content-disposition"))
    if response.status_code >= 400:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            status_code=response.status_code,
            mime_type=content_type,
            filename=filename,
            method="browser_cookie_http",
            error=f"Authorized cookie HTTP download returned status {response.status_code}.",
        )
    if not response.content.startswith(b"%PDF"):
        wiley_result = _download_wiley_pdfdirect_if_available(
            config,
            paper,
            session_name,
            pdf_url,
            target,
            response,
            headers,
            timeout_seconds,
        )
        if wiley_result.archived:
            return wiley_result
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            status_code=response.status_code,
            mime_type=content_type,
            filename=filename,
            method="browser_cookie_http",
            error=f"Authorized cookie HTTP response was not a PDF: {content_type}",
        )
    _write_pdf(target, response.content)
    return AuthorizedPdfArchiveResult(
        paper_id=paper.paper_id,
        pdf_url=pdf_url,
        target_path=str(target),
        archived=True,
        status_code=response.status_code,
        mime_type=content_type,
        filename=filename,
        method="browser_cookie_http",
    )


def _download_wiley_pdfdirect_if_available(
    config: LitTraceConfig,
    paper: PaperMetadata,
    session_name: str,
    pdf_url: str,
    target: Path,
    response: httpx.Response,
    headers: dict[str, str],
    timeout_seconds: float,
) -> AuthorizedPdfArchiveResult:
    content_type = response.headers.get("content-type")
    if "text/html" not in (content_type or "").lower():
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            method="wiley_pdfdirect",
            error=f"Wiley pdfdirect fallback skipped for content type: {content_type}",
        )
    pdfdirect_url = _extract_wiley_pdfdirect_url(response.text, str(response.url))
    if not pdfdirect_url:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdf_url,
            target_path=str(target),
            method="wiley_pdfdirect",
            error="No Wiley pdfdirect URL was found in the PDF HTML shell.",
        )
    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={**headers, "Referer": str(response.url)},
            follow_redirects=True,
        ) as client:
            pdfdirect_response = client.get(pdfdirect_url)
    except httpx.HTTPError as exc:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdfdirect_url,
            target_path=str(target),
            method="wiley_pdfdirect",
            error=f"{exc.__class__.__name__}: {exc}",
        )
    direct_content_type = pdfdirect_response.headers.get("content-type")
    filename = _filename_from_content_disposition(
        pdfdirect_response.headers.get("content-disposition")
    )
    if pdfdirect_response.status_code >= 400:
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdfdirect_url,
            target_path=str(target),
            status_code=pdfdirect_response.status_code,
            mime_type=direct_content_type,
            filename=filename,
            method="wiley_pdfdirect",
            error=f"Wiley pdfdirect returned status {pdfdirect_response.status_code}.",
        )
    if not pdfdirect_response.content.startswith(b"%PDF"):
        return AuthorizedPdfArchiveResult(
            paper_id=paper.paper_id,
            pdf_url=pdfdirect_url,
            target_path=str(target),
            status_code=pdfdirect_response.status_code,
            mime_type=direct_content_type,
            filename=filename,
            method="wiley_pdfdirect",
            error=f"Wiley pdfdirect response was not a PDF: {direct_content_type}",
        )
    _write_pdf(target, pdfdirect_response.content)
    return AuthorizedPdfArchiveResult(
        paper_id=paper.paper_id,
        pdf_url=pdfdirect_url,
        target_path=str(target),
        archived=True,
        status_code=pdfdirect_response.status_code,
        mime_type=direct_content_type,
        filename=filename,
        method="wiley_pdfdirect",
    )


def _extract_wiley_pdfdirect_url(html: str, base_url: str) -> str | None:
    patterns = [
        r'["\'](?P<url>[^"\']*/doi/pdfdirect/[^"\']+)["\']',
        r'[?&]file=(?P<url>[^"\'&<>]+/doi/pdfdirect/[^"\'&<>]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if not match:
            continue
        candidate = unquote(match.group("url"))
        return urljoin(base_url, candidate)
    return None


class _BrowserCookieState(BaseModel):
    cookie: str | None = None
    user_agent: str | None = None
    error: str | None = None


def _read_browser_cookie_state(
    config: LitTraceConfig,
    session_name: str,
    pdf_url: str,
    timeout_seconds: float,
) -> _BrowserCookieState:
    cookie = _read_browser_act_cookies(config, session_name, pdf_url, timeout_seconds)
    user_agent = _read_browser_user_agent(config, session_name, timeout_seconds)
    if cookie:
        return _BrowserCookieState(cookie=cookie, user_agent=user_agent)

    script = "JSON.stringify({cookie: document.cookie || '', userAgent: navigator.userAgent || ''})"
    result = run_browser_act(
        config,
        ["--session", session_name, "eval", script],
        timeout_seconds=min(timeout_seconds, 10.0),
    )
    if result.returncode != 0:
        return _BrowserCookieState(error=result.stderr or result.stdout or "browser cookie eval failed")
    payload_text = _last_nonempty_line(result.stdout)
    try:
        import json

        payload = json.loads(payload_text)
    except Exception as exc:
        return _BrowserCookieState(error=f"Could not parse browser cookie state: {exc}")
    if not isinstance(payload, dict):
        return _BrowserCookieState(error="Browser cookie state was not a JSON object.")
    return _BrowserCookieState(
        cookie=payload.get("cookie") or None,
        user_agent=user_agent or payload.get("userAgent") or None,
    )


def _read_browser_act_cookies(
    config: LitTraceConfig,
    session_name: str,
    pdf_url: str,
    timeout_seconds: float,
) -> str | None:
    result = run_browser_act(
        config,
        ["--session", session_name, "cookies", "get", "--url", _origin_for_url(pdf_url)],
        timeout_seconds=min(timeout_seconds, 10.0),
    )
    if result.returncode != 0:
        return None
    return _cookie_header_from_browser_act_output(result.stdout)


def _read_browser_user_agent(
    config: LitTraceConfig,
    session_name: str,
    timeout_seconds: float,
) -> str | None:
    result = run_browser_act(
        config,
        ["--session", session_name, "eval", "navigator.userAgent || ''"],
        timeout_seconds=min(timeout_seconds, 10.0),
    )
    if result.returncode != 0:
        return None
    return _last_nonempty_line(result.stdout).strip('"').strip("'") or None


def _cookie_header_from_browser_act_output(output: str) -> str | None:
    text = output.strip()
    if not text:
        return None
    parsed = _parse_json_fragment(text)
    if parsed is not None:
        cookies = _cookies_from_json_value(parsed)
        if cookies:
            return "; ".join(cookies)

    csv_cookies = _cookies_from_browser_act_csv(output)
    if csv_cookies:
        return "; ".join(csv_cookies)

    cookies: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.search(r"(?:^|\s)([A-Za-z0-9_.%-]+)=([^;\s]+)", line)
        if match and not line.lower().startswith(("expires=", "path=", "domain=")):
            cookies.append(f"{match.group(1)}={match.group(2)}")
    return "; ".join(dict.fromkeys(cookies)) or None


def _cookies_from_browser_act_csv(output: str) -> list[str]:
    lines = [
        line for line in output.splitlines() if line.strip() and not line.startswith("#")
    ]
    if not lines:
        return []
    reader = csv.DictReader(StringIO("\n".join(lines)))
    if not reader.fieldnames or "name" not in reader.fieldnames or "value" not in reader.fieldnames:
        return []
    cookies: list[str] = []
    for row in reader:
        name = (row.get("name") or "").strip()
        value = (row.get("value") or "").strip()
        if name and value:
            cookies.append(f"{name}={value}")
    return list(dict.fromkeys(cookies))


def _parse_json_fragment(text: str) -> object | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _cookies_from_json_value(value: object) -> list[str]:
    if isinstance(value, dict):
        if "cookies" in value:
            return _cookies_from_json_value(value["cookies"])
        name = value.get("name")
        cookie_value = value.get("value")
        if isinstance(name, str) and isinstance(cookie_value, str):
            return [f"{name}={cookie_value}"]
        return []
    if isinstance(value, list):
        cookies: list[str] = []
        for item in value:
            cookies.extend(_cookies_from_json_value(item))
        return cookies
    return []


def _origin_for_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    return f"{parsed.scheme}://{parsed.netloc}"


def _referer_for_pdf(pdf_url: str) -> str:
    return pdf_url.replace("/doi/pdf/", "/doi/").split("?", 1)[0]


def _last_nonempty_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _NetworkRequest(BaseModel):
    request_id: str
    status_code: int | None = None
    mime_type: str | None = None
    url: str


def _request_filter(pdf_url: str, paper: PaperMetadata) -> str:
    if paper.doi:
        return paper.doi
    return pdf_url.split("?", 1)[0].rsplit("/", 1)[-1]


def _select_pdf_request(output: str, pdf_url: str) -> _NetworkRequest | None:
    lines = [line for line in output.splitlines() if line and not line.startswith("#")]
    if lines and lines[0].startswith("request_id,"):
        lines = lines[1:]
    candidates: list[_NetworkRequest] = []
    for line in lines:
        parts = _split_csv_line(line)
        if len(parts) < 7:
            continue
        request_id, _method, status, _resource_type, mime_type, _timestamp, url = parts[:7]
        if "application/pdf" not in mime_type.lower() and "/doi/pdf/" not in url:
            continue
        if pdf_url.split("?", 1)[0] not in url:
            continue
        try:
            status_code = int(status)
        except ValueError:
            status_code = None
        candidates.append(
            _NetworkRequest(
                request_id=request_id,
                status_code=status_code,
                mime_type=mime_type,
                url=url,
            )
        )
    pdf_candidates = [
        candidate
        for candidate in candidates
        if candidate.mime_type and "application/pdf" in candidate.mime_type.lower()
    ]
    if pdf_candidates:
        return pdf_candidates[-1]
    return candidates[-1] if candidates else None


def _split_csv_line(line: str) -> list[str]:
    parts: list[str] = []
    current = []
    quoted = False
    for char in line:
        if char == '"':
            quoted = not quoted
            continue
        if char == "," and not quoted:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def _extract_response_body(output: str) -> str | None:
    marker = "response_body="
    index = output.find(marker)
    if index < 0:
        return None
    return output[index + len(marker) :]


def _extract_bool_field(output: str, name: str) -> bool:
    match = re.search(rf"^{re.escape(name)}=(true|false)$", output, re.MULTILINE | re.IGNORECASE)
    return bool(match and match.group(1).lower() == "true")


def _extract_header(output: str, name: str) -> str | None:
    match = re.search(rf"^\s+{re.escape(name)}=(.+)$", output, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _filename_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"filename\*=UTF-8''([^;]+)", value)
    if match:
        return match.group(1).strip().strip('"')
    match = re.search(r"filename=([^;]+)", value)
    if match:
        return match.group(1).strip().strip('"')
    return None


def _write_pdf(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
