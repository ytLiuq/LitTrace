"""Shared CDP constants, helpers, and browser automation core.

This module is the **single source of truth** for:

- ``DOI_PREFIX_MAP`` — DOI prefix → publisher mapping (7 publishers)
- ``STEALTH_JS`` — 8-item anti-detection override script
- ``CDPBrowser`` — Chrome DevTools Protocol WebSocket wrapper
- ``identify_publisher`` — DOI → publisher slug
- ``publisher_urls`` — publisher-specific landing/PDF URL templates
- ``CLOUDFLARE_MARKERS`` — strings indicating a Cloudflare challenge page
- ``INSTITUTIONAL_LOGIN_MARKERS`` — strings indicating CARSI/Shibboleth auth
- Helper functions: ``_is_pdf_file``, ``_same_origin_relative_url``,
  ``_normalize_doi``, ``_find_recent_pdf``, ``_move_pdf``

Both ``littrace.cdp_downloader`` (the project-integrated version) and the
standalone ``universal_paper_downloader.py`` SKILL script import from this
module, eliminating the previous code duplication.

When the SKILL script runs outside the littrace package, it falls back to
its own inline copies with a deprecation warning (see
``scripts/universal_paper_downloader.py`` header).
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: DOI prefix → publisher slug.  Add new publishers here.
DOI_PREFIX_MAP: dict[str, str] = {
    "10.1002": "wiley",
    "10.1038": "springer_nature",
    "10.3390": "mdpi",
    "10.1109": "ieee",
    "10.1021": "acs",
    "10.1016": "elsevier",
    "10.1039": "rsc",
}

#: Publisher display names (human-readable).
PUBLISHER_NAMES: dict[str, str] = {
    "wiley": "Wiley",
    "springer_nature": "Springer Nature",
    "mdpi": "MDPI",
    "ieee": "IEEE",
    "acs": "ACS",
    "elsevier": "Elsevier",
    "rsc": "RSC",
    "unknown": "Unknown",
}

#: 8-item stealth override script.
#:
#: 1. navigator.webdriver → undefined
#: 2. window.chrome forge
#: 3. permissions.query override
#: 4. navigator.plugins
#: 5. navigator.languages
#: 6. navigator.connection
#: 7. WebGL vendor override
#: 8. cdc_ variable cleanup
STEALTH_JS: str = r"""
(function() {
  Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
  window.chrome = window.chrome || { runtime: {} };
  const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
  if (originalQuery) {
    window.navigator.permissions.query = (parameters) => (
      parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : originalQuery(parameters)
    );
  }
  Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
  Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
  if (navigator.connection === undefined) {
    Object.defineProperty(navigator, 'connection', {
      get: () => ({ effectiveType: '4g', rtt: 50, downlink: 10 })
    });
  }
  const getParameter = WebGLRenderingContext.prototype.getParameter;
  WebGLRenderingContext.prototype.getParameter = function(parameter) {
    if (parameter === 37445) return 'Intel Inc.';
    if (parameter === 37446) return 'Intel Iris OpenGL Engine';
    return getParameter.apply(this, arguments);
  };
  for (const key of Object.keys(window)) {
    if (key.startsWith('cdc_') || key.startsWith('$cdc_')) delete window[key];
  }
})();
""".strip()

#: Strings that indicate a Cloudflare challenge / verification page.
CLOUDFLARE_MARKERS: list[str] = [
    "just a moment",
    "verify you are human",
    "are you a robot",
    "cloudflare",
    "__cf_chl",
    "captcha",
    "安全验证",
    "请验证",
    "请稍候",
]

#: Strings that indicate an institutional login (CARSI / Shibboleth) is required.
INSTITUTIONAL_LOGIN_MARKERS: list[str] = [
    "carsi",
    "shibboleth",
    "institutional login",
    "sign in through your institution",
    "选择您的机构",
]


# ---------------------------------------------------------------------------
# Publisher helpers
# ---------------------------------------------------------------------------


def identify_publisher(doi: str) -> str:
    """Identify the publisher from a DOI prefix.

    >>> identify_publisher("10.1039/d5nr04405g")
    'rsc'
    >>> identify_publisher("10.1002/adfm.202414678")
    'wiley'
    >>> identify_publisher("10.9999/unknown")
    'unknown'
    """
    for prefix, publisher in DOI_PREFIX_MAP.items():
        if doi.startswith(prefix):
            return publisher
    return "unknown"


def publisher_urls(doi: str, publisher: str) -> dict[str, str | None]:
    """Construct landing-page and direct-PDF URLs for a given publisher.

    Returns a dict with keys ``"landing"`` and ``"pdf"`` (pdf may be None,
    meaning it must be extracted from the page after navigation).
    """
    if publisher == "wiley":
        return {
            "landing": f"https://advanced.onlinelibrary.wiley.com/doi/{doi}",
            "pdf": f"https://advanced.onlinelibrary.wiley.com/doi/pdfdirect/{doi}?download=true",
        }
    if publisher == "acs":
        return {
            "landing": f"https://pubs.acs.org/doi/{doi}",
            "pdf": f"https://pubs.acs.org/doi/pdf/{doi}",
        }
    # For springer_nature, mdpi, ieee, elsevier, rsc, unknown:
    # doi.org redirects to the correct page; PDF URL is extracted later.
    return {"landing": f"https://doi.org/{doi}", "pdf": None}


def normalize_doi(doi: str) -> str:
    """Normalise a DOI string: strip whitespace, remove ``https://doi.org/`` prefix, lowercase."""
    return doi.strip().removeprefix("https://doi.org/").removeprefix("doi:").lower()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def is_pdf_file(path: Path) -> bool:
    """Check if a file starts with the ``%PDF-`` magic bytes."""
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def find_recent_pdf(
    directory: Path,
    preferred_path: Path,
    max_age_seconds: float = 90.0,
) -> Path | None:
    """Find the most recently created PDF in *directory* within *max_age_seconds*.

    If *preferred_path* exists and is a valid PDF, return it directly.
    Otherwise scan the directory for recently-written PDFs.
    """
    if (
        preferred_path.exists()
        and preferred_path.stat().st_size > 1000
        and is_pdf_file(preferred_path)
    ):
        return preferred_path
    now = time.time()
    if not directory.exists():
        return None
    candidates: list[Path] = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        if path.stat().st_size <= 1000:
            continue
        if now - path.stat().st_mtime > max_age_seconds:
            continue
        if is_pdf_file(path):
            candidates.append(path)
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def move_pdf(source: Path, target: Path) -> None:
    """Move *source* to *target*, falling back to copy if rename fails."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    try:
        if target.exists():
            target.unlink()
        source.rename(target)
    except OSError:
        shutil.copy2(source, target)


def same_origin_relative_url(current_url: str, pdf_url: str) -> str:
    """If *pdf_url* is same-origin as *current_url*, return only the path+query.

    This avoids CORS issues when fetch() is called from the browser context.
    If origins differ, the full *pdf_url* is returned unchanged.
    """
    if not current_url or not pdf_url.startswith("http"):
        return pdf_url
    current = urlparse(current_url)
    target = urlparse(pdf_url)
    if current.scheme == target.scheme and current.netloc == target.netloc:
        suffix = target.path
        if target.query:
            suffix += f"?{target.query}"
        return suffix
    return pdf_url


# ---------------------------------------------------------------------------
# CDPBrowser
# ---------------------------------------------------------------------------


class CDPBrowser:
    """Chrome DevTools Protocol browser automation wrapper.

    Provides a high-level API for:

    - Creating/connecting CDP tabs via WebSocket
    - Injecting the stealth script (anti-detection)
    - Setting download paths (for ``setDownloadBehavior``)
    - Navigating to URLs
    - Evaluating JavaScript expressions
    - Detecting Cloudflare challenges
    - Fetching binary blobs via in-page ``fetch()`` + base64
    - Triggering downloads via ``<a download>``

    Automatic reconnection on WebSocket failure is built-in.
    """

    def __init__(self, cdp_url: str, reconnect_attempts: int = 3) -> None:
        self.cdp_url = cdp_url.rstrip("/")
        self.reconnect_attempts = max(reconnect_attempts, 1)
        self.ws = None
        self.msg_id = 1
        self.tab_id: str | None = None
        self._download_path: str | None = None

    def connect_new_tab(self) -> None:
        """Create a new browser tab and connect to its WebSocket."""
        import websocket

        response = httpx.put(f"{self.cdp_url}/json/new?about:blank", timeout=10.0)
        response.raise_for_status()
        tab = response.json()
        self.tab_id = tab["id"]
        self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"])

    def send(self, method: str, params: dict | None = None) -> dict:
        """Send a CDP command and wait for its response.

        Reconnects automatically on failure, up to ``reconnect_attempts`` times.
        """
        last_error: Exception | None = None
        for attempt in range(self.reconnect_attempts):
            try:
                if self.ws is None:
                    self.connect_new_tab()
                    self._reinit_after_reconnect()
                message: dict = {"id": self.msg_id, "method": method}
                if params is not None:
                    message["params"] = params
                self.ws.send(json.dumps(message))
                current_id = self.msg_id
                self.msg_id += 1
                while True:
                    response = json.loads(self.ws.recv())
                    if response.get("id") == current_id:
                        return response
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.reconnect_attempts:
                    break
                self._reconnect()
                time.sleep(0.5)
        raise RuntimeError(f"CDP command failed: {last_error}")

    def inject_stealth(self) -> None:
        """Inject the stealth script into all future page loads."""
        self.send("Page.addScriptToEvaluateOnNewDocument", {"source": STEALTH_JS})

    def set_download_path(self, path: Path) -> None:
        """Configure the browser's download directory."""
        path.mkdir(parents=True, exist_ok=True)
        self._download_path = str(path)
        self.send("Page.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(path)})

    def navigate(self, url: str, wait_seconds: float = 8.0) -> None:
        """Navigate to *url* and wait *wait_seconds* for the page to load."""
        self.send("Page.navigate", {"url": url})
        time.sleep(max(wait_seconds, 0.0))

    def eval(self, expression: str, await_promise: bool = False) -> object:
        """Evaluate a JavaScript expression and return the result value."""
        response = self.send(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": await_promise},
        )
        result = response.get("result", {}).get("result", {})
        return result.get("value")

    def get_url(self) -> str:
        """Get the current page URL."""
        value = self.eval("window.location.href")
        return str(value or "")

    def get_title(self) -> str:
        """Get the current page title."""
        value = self.eval("document.title")
        return str(value or "")

    def get_body_text(self, max_chars: int = 1000) -> str:
        """Get the first *max_chars* characters of the page body text."""
        expression = f"document.body ? document.body.innerText.substring(0, {max_chars}) : ''"
        value = self.eval(expression)
        return str(value or "")

    def is_cloudflare_challenge(self) -> bool:
        """Check if the current page is a Cloudflare verification page."""
        haystack = f"{self.get_title()}\n{self.get_url()}\n{self.get_body_text(500)}".lower()
        return any(marker in haystack for marker in CLOUDFLARE_MARKERS)

    def wait_for_cloudflare(self, max_wait_seconds: float, interval: float = 5.0) -> bool:
        """Poll until the Cloudflare challenge clears or *max_wait_seconds* elapses."""
        deadline = time.monotonic() + max(max_wait_seconds, 0.0)
        while time.monotonic() <= deadline:
            if not self.is_cloudflare_challenge():
                return True
            time.sleep(max(interval, 0.5))
        return not self.is_cloudflare_challenge()

    def fetch_blob_to_file(self, pdf_url: str, target_path: Path) -> tuple[bool, str | int]:
        """Download a PDF via in-page ``fetch()`` + base64 blob transfer.

        Returns ``(True, size_bytes)`` on success or ``(False, error_message)`` on failure.
        """
        current_url = self.get_url()
        fetch_url = same_origin_relative_url(current_url, pdf_url)
        expression = f"""
        (async function() {{
          try {{
            const resp = await fetch({json.dumps(fetch_url)}, {{ credentials: 'same-origin' }});
            if (!resp.ok) return JSON.stringify({{error: 'HTTP ' + resp.status}});
            const blob = await resp.blob();
            const arrayBuffer = await blob.arrayBuffer();
            const bytes = new Uint8Array(arrayBuffer);
            let binary = '';
            const chunkSize = 8192;
            for (let i = 0; i < bytes.length; i += chunkSize) {{
              binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
            }}
            return JSON.stringify({{data: btoa(binary), size: blob.size}});
          }} catch (e) {{
            return JSON.stringify({{error: e.message}});
          }}
        }})()
        """
        raw = self.eval(expression, await_promise=True)
        try:
            payload = json.loads(str(raw or ""))
        except json.JSONDecodeError:
            return False, "Could not parse browser fetch payload."
        if payload.get("error"):
            return False, str(payload["error"])
        content = base64.b64decode(payload.get("data") or "")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(content)
        if not is_pdf_file(target_path):
            return False, "Browser fetch wrote a non-PDF response."
        return True, int(payload.get("size") or len(content))

    def trigger_anchor_download(self, pdf_url: str, filename: str) -> None:
        """Trigger a download by creating and clicking an ``<a download>`` element."""
        expression = f"""
        (function() {{
          const a = document.createElement('a');
          a.href = {json.dumps(pdf_url)};
          a.download = {json.dumps(filename)};
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        }})()
        """
        self.eval(expression)

    def close(self) -> None:
        """Close the WebSocket connection."""
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def _reconnect(self) -> None:
        """Attempt to reconnect to an existing tab, or create a new one."""
        self.close()
        if self.tab_id:
            try:
                import websocket

                tabs = httpx.get(f"{self.cdp_url}/json", timeout=10.0).json()
                for tab in tabs:
                    if tab.get("id") == self.tab_id and tab.get("webSocketDebuggerUrl"):
                        self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"])
                        self._reinit_after_reconnect()
                        return
            except Exception:
                pass
        self.connect_new_tab()
        self._reinit_after_reconnect()

    def _reinit_after_reconnect(self) -> None:
        """Re-inject stealth and download path after a reconnection."""
        try:
            self.inject_stealth()
            if self._download_path:
                self.send(
                    "Page.setDownloadBehavior",
                    {"behavior": "allow", "downloadPath": self._download_path},
                )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# PDF URL extraction helpers (used by both cdp_downloader and SKILL script)
# ---------------------------------------------------------------------------

#: JavaScript expression to extract a PDF URL from a publisher's page.
#: Tries multiple CSS selectors and falls back to regex on the HTML.
PDF_URL_EXTRACTION_JS: str = r"""
(function() {
  const selectors = [
    'a[href*="pdf"]',
    'a[href*="pdfft"]',
    'a[href*="pdfdirect"]',
    'a[href*="stampPDF"]',
    'a[aria-label*="PDF" i]',
    'a[title*="PDF" i]',
    'iframe[src*="pdf"]',
    'iframe[src*="stampPDF"]',
    'meta[name="citation_pdf_url"]'
  ];
  for (const selector of selectors) {
    for (const node of document.querySelectorAll(selector)) {
      const raw = node.href || node.src || node.content || node.getAttribute('href');
      if (raw) return new URL(raw, location.href).href;
    }
  }
  const html = document.documentElement.outerHTML;
  const pdfft = html.match(/["']([^"']*pdfft[^"']*)["']/);
  if (pdfft) return new URL(pdfft[1], location.href).href;
  return '';
})()
"""

#: JavaScript expression to extract the IEEE article number (arnumber).
IEEE_ARNUMBER_JS: str = r"""
(function() {
  const m = location.href.match(/document\/(\d+)/);
  if (m) return m[1];
  const html = document.documentElement.outerHTML;
  const m2 = html.match(/arnumber[=:]?(\d+)/);
  if (m2) return m2[1];
  return '';
})()
"""


def extract_pdf_url_from_page(browser: CDPBrowser, doi: str, publisher: str) -> str | None:
    """Try to extract a PDF URL from the current browser page.

    Uses CSS selectors, meta tags, and publisher-specific heuristics.
    """
    value = browser.eval(PDF_URL_EXTRACTION_JS)
    pdf_url = str(value or "")
    if pdf_url:
        return pdf_url
    current = browser.get_url()
    if any(marker in current.lower() for marker in ["pdf", "silverchair"]):
        return current
    if publisher == "rsc":
        article_code = doi.split("/")[-1]
        match = re.match(r"[a-z](\d)([a-z]+)", article_code)
        if match:
            year = 2020 + int(match.group(1))
            journal = match.group(2)
            return f"https://pubs.rsc.org/en/content/articlepdf/{year}/{journal}/{article_code}"
    return None


def prepare_rsc_pdf_url(browser: CDPBrowser, doi: str, pdf_url: str) -> str:
    """Navigate to the RSC PDF URL and return the CDN-redirected URL."""
    browser.navigate(pdf_url, wait_seconds=10.0)
    current = browser.get_url()
    return current if current else pdf_url


def prepare_ieee_pdf_url(browser: CDPBrowser, pdf_url: str) -> str:
    """Extract the IEEE arnumber and construct the stampPDF URL."""
    arnumber = str(browser.eval(IEEE_ARNUMBER_JS) or "")
    if arnumber:
        return f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arnumber}"
    return pdf_url


def looks_like_institutional_login_needed(text: str) -> bool:
    """Check if the page text indicates a CARSI/Shibboleth login is required."""
    lowered = text.lower()
    return any(marker in lowered for marker in INSTITUTIONAL_LOGIN_MARKERS)
