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
    "10.1007": "springer_nature",
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

#: Enhanced stealth override script.
#:
#: 1. navigator.webdriver → undefined
#: 2. window.chrome forge
#: 3. permissions.query override
#: 4. navigator.plugins
#: 5. navigator.languages
#: 6. navigator.connection
#: 7. WebGL vendor override
#: 8. cdc_ variable cleanup
#: 9. navigator.hardwareConcurrency
#: 10. navigator.deviceMemory
#: 11. navigator.platform
#: 12. navigator.userAgentData
#: 13. JS-layer HeadlessChrome UA cleanup
STEALTH_JS: str = r"""
(function() {
  Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
  window.chrome = {
    runtime: {
      OnInstalledReason: { CHROME_UPDATE: 'chrome_update', INSTALL: 'install', UPDATE: 'update' },
      PlatformArch: { ARM: 'arm', X86_32: 'x86-32', X86_64: 'x86-64' },
      PlatformOs: { ANDROID: 'android', CROS: 'cros', LINUX: 'linux', MAC: 'mac', WIN: 'win' }
    },
    loadTimes: function() { return {}; },
    csi: function() { return {}; }
  };
  const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
  if (originalQuery) {
    window.navigator.permissions.query = (parameters) => (
      parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : originalQuery(parameters)
    );
  }
  Object.defineProperty(navigator, 'plugins', {
    get: () => {
      const makePlugin = (name, filename, description) => {
        const p = { name, filename, description, length: 1 };
        p[0] = { type: 'application/pdf', suffixes: 'pdf', description: '' };
        return p;
      };
      const arr = [
        makePlugin('PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
        makePlugin('Chrome PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
        makePlugin('Chromium PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
        makePlugin('Microsoft Edge PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
        makePlugin('WebKit built-in PDF', 'internal-pdf-viewer', 'Portable Document Format')
      ];
      arr.namedItem = function(name) { return this.find(p => p.name === name) || null; };
      arr.refresh = function() {};
      return arr;
    }
  });
  Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
  if (navigator.connection === undefined) {
    Object.defineProperty(navigator, 'connection', {
      get: () => ({ effectiveType: '4g', rtt: 50, downlink: 10 })
    });
  }
  Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
  Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
  Object.defineProperty(navigator, 'platform', { get: () => 'MacIntel' });
  if (navigator.userAgentData === undefined) {
    Object.defineProperty(navigator, 'userAgentData', {
      get: () => ({
        brands: [
          { brand: 'Google Chrome', version: '144' },
          { brand: 'Chromium', version: '144' },
          { brand: 'Not?A_Brand', version: '99' }
        ],
        mobile: false,
        platform: 'macOS'
      })
    });
  }
  const originalUA = navigator.userAgent;
  if (originalUA && originalUA.includes('HeadlessChrome')) {
    Object.defineProperty(navigator, 'userAgent', {
      get: () => originalUA.replace('HeadlessChrome', 'Chrome')
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

SCIENCEDIRECT_NO_ACCESS_MARKERS: list[str] = [
    "get access",
    "purchase pdf",
    "check for this article elsewhere",
    "access through your institution",
    "sign in via your institution",
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
    if publisher == "springer_nature":
        if doi.startswith("10.1007/"):
            return {
                "landing": f"https://link.springer.com/article/{doi}",
                "pdf": f"https://link.springer.com/content/pdf/{doi}.pdf",
            }
        # nature.com articles: /articles/{article_id}.pdf
        # doi 10.1038/srep14751 → article_id = srep14751
        article_id = doi.split("/")[-1] if "/" in doi else doi
        return {
            "landing": f"https://doi.org/{doi}",
            "pdf": f"https://www.nature.com/articles/{article_id}.pdf",
        }
    # For mdpi, ieee, elsevier, rsc, unknown:
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

    def __init__(
        self,
        cdp_url: str,
        reconnect_attempts: int = 3,
        command_timeout_seconds: float = 60.0,
    ) -> None:
        self.cdp_url = cdp_url.rstrip("/")
        self.reconnect_attempts = max(reconnect_attempts, 1)
        self.command_timeout_seconds = max(command_timeout_seconds, 1.0)
        self.ws = None
        self.msg_id = 1
        self.tab_id: str | None = None
        self._download_path: str | None = None
        self.stealth_notes: list[str] = []

    def connect_new_tab(self) -> None:
        """Create a new browser tab and connect to its WebSocket."""
        import websocket

        response = httpx.put(f"{self.cdp_url}/json/new?about:blank", timeout=10.0)
        response.raise_for_status()
        tab = response.json()
        self.tab_id = tab["id"]
        self.ws = websocket.create_connection(
            tab["webSocketDebuggerUrl"],
            timeout=self.command_timeout_seconds,
        )
        try:
            self.ws.settimeout(self.command_timeout_seconds)
        except Exception:
            pass

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

    def prepare_stealth_context(self) -> list[str]:
        """Enable CDP domains and normalize browser fingerprint before navigation."""
        notes: list[str] = []
        for method in ("Page.enable", "Network.enable"):
            response = self.send(method)
            if response.get("error"):
                notes.append(f"{method} failed: {response['error']}")
        ua = str(self.eval("navigator.userAgent") or "")
        if "HeadlessChrome" in ua:
            fixed_ua = ua.replace("HeadlessChrome", "Chrome")
            response = self.send("Network.setUserAgentOverride", {"userAgent": fixed_ua})
            if response.get("error"):
                notes.append(f"Network.setUserAgentOverride failed: {response['error']}")
            else:
                notes.append("HTTP user-agent normalized: HeadlessChrome -> Chrome")
        self.stealth_notes = notes
        return notes

    def inject_stealth(self) -> None:
        """Inject the enhanced stealth script into all future page loads."""
        self.send("Page.addScriptToEvaluateOnNewDocument", {"source": STEALTH_JS})

    def prepare_and_inject_stealth(self) -> list[str]:
        """Prepare Page/Network domains, normalize UA, then inject stealth."""
        notes = self.prepare_stealth_context()
        self.inject_stealth()
        return notes

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

    def wait_for_url_markers(
        self,
        markers: list[str],
        max_wait_seconds: float,
        interval: float = 1.0,
    ) -> str:
        """Poll current URL until it contains one of *markers*, then return it."""
        lowered_markers = [marker.lower() for marker in markers]
        deadline = time.monotonic() + max(max_wait_seconds, 0.0)
        current = self.get_url()
        while time.monotonic() <= deadline:
            current = self.get_url()
            if any(marker in current.lower() for marker in lowered_markers):
                return current
            time.sleep(max(interval, 0.2))
        return current

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

    def close_tab(self) -> None:
        """Close the current Chrome tab and its WebSocket connection."""
        tab_id = self.tab_id
        self.close()
        if tab_id:
            try:
                httpx.get(f"{self.cdp_url}/json/close/{tab_id}", timeout=5.0)
            except Exception:
                pass
            self.tab_id = None

    def _reconnect(self) -> None:
        """Attempt to reconnect to an existing tab, or create a new one."""
        self.close()
        if self.tab_id:
            try:
                import websocket

                tabs = httpx.get(f"{self.cdp_url}/json", timeout=10.0).json()
                for tab in tabs:
                    if tab.get("id") == self.tab_id and tab.get("webSocketDebuggerUrl"):
                        self.ws = websocket.create_connection(
                            tab["webSocketDebuggerUrl"],
                            timeout=self.command_timeout_seconds,
                        )
                        try:
                            self.ws.settimeout(self.command_timeout_seconds)
                        except Exception:
                            pass
                        self._reinit_after_reconnect()
                        return
            except Exception:
                pass
        self.connect_new_tab()
        self._reinit_after_reconnect()

    def _reinit_after_reconnect(self) -> None:
        """Re-inject stealth and download path after a reconnection."""
        try:
            self.prepare_and_inject_stealth()
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
    if publisher == "elsevier":
        return infer_elsevier_pdfft_url(browser)
    if publisher == "rsc":
        article_code = doi.split("/")[-1]
        match = re.match(r"[a-z](\d)([a-z]+)", article_code)
        if match:
            year = 2020 + int(match.group(1))
            journal = match.group(2)
            return f"https://pubs.rsc.org/en/content/articlepdf/{year}/{journal}/{article_code}"
    return None


ELSEVIER_PII_JS: str = r"""
(function() {
  const haystack = [
    location.href || '',
    document.documentElement ? document.documentElement.outerHTML : ''
  ].join('\n');
  const patterns = [
    /\/science\/article\/pii\/([A-Z0-9]+)/i,
    /["']pii["']\s*:\s*["']([A-Z0-9]+)["']/i,
    /PII(?:%3D|=)([A-Z0-9]+)/i
  ];
  for (const pattern of patterns) {
    const match = haystack.match(pattern);
    if (match && match[1]) return match[1];
  }
  return '';
})()
"""

ELSEVIER_ACCESS_STATUS_JS: str = r"""
(function() {
  const body = document.body ? document.body.innerText.toLowerCase() : '';
  const href = location.href.toLowerCase();
  const pdfLink = document.querySelector(
    'a[href*="pdfft"], a[href*="/pdf"], a[aria-label*="PDF" i], a[title*="PDF" i]'
  );
  if (href.includes('pdf.sciencedirectassets.com') || href.includes('els-cdn')) {
    return 'has_access';
  }
  if (pdfLink || (body.includes('download') && body.includes('pdf'))) {
    return 'has_access';
  }
  if (
    body.includes('get access') ||
    body.includes('purchase pdf') ||
    body.includes('check for this article elsewhere') ||
    body.includes('access through your institution') ||
    body.includes('sign in via your institution') ||
    body.includes('sign in to view your account details') ||
    body.includes('not entitled') ||
    body.includes('no access')
  ) {
    return 'no_access';
  }
  const title = document.title || '';
  const pii = haystackPii();
  if (title === 'ScienceDirect' && pii && !pdfLink) {
    return 'no_access';
  }
  return 'unknown';

  function haystackPii() {
    const haystack = [location.href || '', document.documentElement ? document.documentElement.outerHTML : ''].join('\n');
    return /\/science\/article\/pii\/([A-Z0-9]+)/i.test(haystack);
  }
})()
"""

ELSEVIER_CLICK_INSTITUTION_LOGIN_JS: str = r"""
(function() {
  const candidates = Array.from(document.querySelectorAll('a, button, [role="button"]'));
  const needles = [
    'sign in via your institution',
    'access through your institution',
    'institution',
    'get access',
    'sign in'
  ];
  for (const node of candidates) {
    const text = [
      node.innerText || '',
      node.textContent || '',
      node.getAttribute('aria-label') || '',
      node.getAttribute('title') || '',
      node.getAttribute('href') || ''
    ].join(' ').toLowerCase();
    if (needles.some((needle) => text.includes(needle))) {
      node.scrollIntoView({block: 'center', inline: 'center'});
      node.click();
      return text.slice(0, 200);
    }
  }
  return '';
})()
"""


def infer_elsevier_pdfft_url(browser: CDPBrowser) -> str | None:
    """Infer a ScienceDirect pdfft URL from the current article page."""
    pii = str(browser.eval(ELSEVIER_PII_JS) or "").strip()
    if not pii:
        return None
    return f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft"


def sciencedirect_access_status(browser: CDPBrowser) -> str:
    """Return ``has_access``, ``no_access``, or ``unknown`` for current ScienceDirect page."""
    value = str(browser.eval(ELSEVIER_ACCESS_STATUS_JS) or "").strip()
    return value if value in {"has_access", "no_access", "unknown"} else "unknown"


def click_sciencedirect_institution_login(browser: CDPBrowser) -> str:
    """Click the most likely ScienceDirect institution-login/get-access control."""
    return str(browser.eval(ELSEVIER_CLICK_INSTITUTION_LOGIN_JS) or "")


def wait_for_sciencedirect_authorization(
    browser: CDPBrowser,
    landing_url: str,
    timeout_seconds: float,
    poll_seconds: float = 5.0,
) -> str:
    """Wait for a user-completed ScienceDirect institution login to expose PDF access.

    The browser may remain on an institution/CARSI page after the user logs in, so
    this periodically navigates back to the original article page and re-checks.
    """
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    next_revisit = time.monotonic() + max(poll_seconds * 2, 5.0)
    last_status = sciencedirect_access_status(browser)
    while time.monotonic() <= deadline:
        current_url = browser.get_url()
        last_status = sciencedirect_access_status(browser)
        if last_status == "has_access":
            return "has_access"
        if "sciencedirect.com/science/article" in current_url and last_status != "no_access":
            return last_status
        if time.monotonic() >= next_revisit:
            browser.navigate(landing_url, wait_seconds=8.0)
            last_status = sciencedirect_access_status(browser)
            if last_status == "has_access":
                return "has_access"
            next_revisit = time.monotonic() + max(poll_seconds * 3, 10.0)
        time.sleep(max(poll_seconds, 1.0))
    return last_status or "unknown"


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


ELSEVIER_PDF_CANDIDATES_JS: str = r"""
(function() {
  const values = new Set();
  const add = (raw) => {
    if (!raw) return;
    try { values.add(new URL(raw, location.href).href); } catch (_) {}
  };
  add(location.href);
  for (const entry of performance.getEntriesByType('resource')) {
    add(entry.name);
  }
  const selectors = [
    'a[href]', 'iframe[src]', 'embed[src]', 'object[data]',
    'meta[name="citation_pdf_url"]', 'link[href]'
  ];
  for (const selector of selectors) {
    for (const node of document.querySelectorAll(selector)) {
      add(node.href || node.src || node.data || node.content || node.getAttribute('href'));
    }
  }
  const html = document.documentElement ? document.documentElement.outerHTML : '';
  const patterns = [
    /https?:\\\/\\\/pdf\\.sciencedirectassets\\.com[^"'<>\\\s]+/g,
    /https?:\/\/pdf\.sciencedirectassets\.com[^"'<>\s]+/g,
    /https?:\/\/[^"'<>\s]*els-cdn[^"'<>\s]+/g,
    /["']([^"']*\/science\/article\/pii\/[^"']*\/pdfft[^"']*)["']/g
  ];
  for (const pattern of patterns) {
    for (const match of html.matchAll(pattern)) {
      add(match[1] || match[0]);
    }
  }
  return JSON.stringify(Array.from(values));
})()
"""


def discover_elsevier_pdf_candidates(browser: CDPBrowser) -> list[str]:
    """Return Elsevier PDF/CDN candidates visible in the current browser page."""
    raw = browser.eval(ELSEVIER_PDF_CANDIDATES_JS)
    try:
        candidates = json.loads(str(raw or "[]"))
    except json.JSONDecodeError:
        candidates = []
    if not isinstance(candidates, list):
        return []
    cleaned: list[str] = []
    for item in candidates:
        url = str(item or "").replace("\\/", "/")
        lowered = url.lower()
        if not url.startswith("http"):
            continue
        if (
            "pdf.sciencedirectassets.com" in lowered
            or "els-cdn" in lowered
            or "/science/article/pii/" in lowered
            and "/pdfft" in lowered
            or lowered.endswith(".pdf")
            or ".pdf?" in lowered
        ):
            if url not in cleaned:
                cleaned.append(url)
    return sorted(cleaned, key=_elsevier_candidate_rank, reverse=True)


def prepare_elsevier_pdf_url(
    browser: CDPBrowser,
    pdf_url: str,
    max_wait_seconds: float = 45.0,
) -> str:
    """Navigate through ScienceDirect pdfft and return the best CDN/PDF URL found."""
    best_seen = pdf_url
    attempts = _elsevier_pdf_url_variants(pdf_url)
    per_attempt_wait = max(min(max_wait_seconds / max(len(attempts), 1), 12.0), 4.0)
    for attempt_url in attempts:
        browser.navigate(attempt_url, wait_seconds=4.0)
        cdn_url = browser.wait_for_url_markers(
            ["sciencedirectassets", "els-cdn", ".pdf"],
            max_wait_seconds=per_attempt_wait,
        )
        best_seen = cdn_url or attempt_url
        if _looks_like_elsevier_cdn_pdf(best_seen):
            return best_seen
        candidates = discover_elsevier_pdf_candidates(browser)
        for candidate in candidates:
            if _looks_like_elsevier_cdn_pdf(candidate):
                return candidate
        if candidates:
            best_seen = candidates[0]
    return best_seen


def _elsevier_pdf_url_variants(pdf_url: str) -> list[str]:
    parsed = urlparse(pdf_url)
    base = pdf_url.split("?", 1)[0]
    variants = [pdf_url, base]
    if base.endswith("/pdfft"):
        variants.extend(
            [
                f"{base}?isDTMRedir=true",
                f"{base}?download=true",
                base.removesuffix("/pdfft") + "/pdf",
                base.removesuffix("/pdfft") + "/pdf?download=true",
            ]
        )
    elif base.endswith("/pdf"):
        variants.extend([f"{base}?download=true", base.removesuffix("/pdf") + "/pdfft"])
    elif "/science/article/pii/" in parsed.path:
        variants.extend([base.rstrip("/") + "/pdfft", base.rstrip("/") + "/pdf"])
    deduped: list[str] = []
    for variant in variants:
        if variant and variant not in deduped:
            deduped.append(variant)
    return deduped


def _looks_like_elsevier_cdn_pdf(url: str) -> bool:
    lowered = url.lower()
    return "pdf.sciencedirectassets.com" in lowered or "els-cdn" in lowered


def _elsevier_candidate_rank(url: str) -> int:
    lowered = url.lower()
    score = 0
    if "pdf.sciencedirectassets.com" in lowered:
        score += 100
    if "els-cdn" in lowered:
        score += 80
    if ".pdf" in lowered:
        score += 30
    if "/pdfft" in lowered:
        score += 10
    if "x-amz-security-token" in lowered:
        score += 10
    return score


def looks_like_institutional_login_needed(text: str) -> bool:
    """Check if the page text indicates a CARSI/Shibboleth login is required."""
    lowered = text.lower()
    return any(marker in lowered for marker in INSTITUTIONAL_LOGIN_MARKERS)
