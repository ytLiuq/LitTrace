#!/usr/bin/env python3
"""
通用学术论文 PDF 下载器
========================
一套方案，覆盖七家出版商：Wiley / Springer Nature / MDPI / IEEE / ACS / Elsevier / RSC

用法:
    python3 universal_paper_downloader.py <DOI> [--output PATH] [--cdp URL] [--email EMAIL]

示例:
    python3 universal_paper_downloader.py 10.1039/d5nr04405g
    python3 universal_paper_downloader.py 10.1002/adfm.202414678 --output ~/Desktop/paper.pdf

决策流程:
    1. 从 DOI 识别出版商
    2. 查 Unpaywall 获取 OA 状态和 PDF 链接
       ├─ 有 repository 镜像 → curl 直下（无需浏览器）
       └─ 无镜像或下载失败 → 进入浏览器方案
    3. stealth 注入 + CDP 浏览器导航
       ├─ Cloudflare 通过 → fetch+blob 下载 PDF
       └─ Cloudflare Captcha → 提示用户手动完成，然后继续 fetch+blob

核心常量（DOI_PREFIX_MAP、STEALTH_JS、CDPBrowser 等）同步自
littrace/src/littrace/cdp_core.py。当 littrace 包可导入时，
本脚本直接 import littrace.cdp_core，消除代码重复；否则 fallback 到
内置副本，保证独立运行。
"""

import sys
import os
import re
import json
import time
import argparse
import subprocess
import base64
from pathlib import Path

import requests

# ──────────────────────────────────────────────
# 尝试从 littrace.cdp_core 导入共享代码
# ──────────────────────────────────────────────

_SCRIPT_PATH = Path(__file__).resolve()
for _parent in _SCRIPT_PATH.parents:
    _src_dir = _parent / "src"
    if (_src_dir / "littrace" / "cdp_core.py").exists():
        sys.path.insert(0, str(_src_dir))
        break

try:
    from littrace.cdp_core import (
        DOI_PREFIX_MAP,
        STEALTH_JS,
        CLOUDFLARE_MARKERS,
        INSTITUTIONAL_LOGIN_MARKERS,
        PUBLISHER_NAMES,
        CDPBrowser,
        identify_publisher,
        publisher_urls,
        normalize_doi,
        is_pdf_file,
        find_recent_pdf,
        move_pdf,
        same_origin_relative_url,
        extract_pdf_url_from_page,
        prepare_elsevier_pdf_url,
        prepare_rsc_pdf_url,
        prepare_ieee_pdf_url,
        looks_like_institutional_login_needed,
        sciencedirect_access_status,
        click_sciencedirect_institution_login,
        wait_for_sciencedirect_authorization,
    )

    _USING_LITTRACE_CORE = True
except ImportError:
    _USING_LITTRACE_CORE = False

    # ── Fallback: 内置副本（同步自 littrace/src/littrace/cdp_core.py） ──

    DOI_PREFIX_MAP = {
        "10.1002": "wiley",
        "10.1038": "springer_nature",
        "10.3390": "mdpi",
        "10.1109": "ieee",
        "10.1021": "acs",
        "10.1016": "elsevier",
        "10.1039": "rsc",
    }

    PUBLISHER_NAMES = {
        "wiley": "Wiley",
        "springer_nature": "Springer Nature",
        "mdpi": "MDPI",
        "ieee": "IEEE",
        "acs": "ACS",
        "elsevier": "Elsevier",
        "rsc": "RSC",
        "unknown": "Unknown",
    }

    STEALTH_JS = r"""
(function() {
    // 1. navigator.webdriver → undefined
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    // 2. window.chrome 伪造
    window.chrome = { runtime: {} };
    // 3. permissions.query 覆盖
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
            Promise.resolve({ state: Notification.permission }) :
            originalQuery(parameters)
    );
    // 4. navigator.plugins
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
    // 5. navigator.languages
    Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
    // 6. navigator.connection
    if (navigator.connection === undefined) {
        Object.defineProperty(navigator, 'connection', {
            get: () => ({ effectiveType: '4g', rtt: 50, downlink: 10 })
        });
    }
    // 7. WebGL vendor 覆盖
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(parameter) {
        if (parameter === 37445) return 'Intel Inc.';
        if (parameter === 37446) return 'Intel Iris OpenGL Engine';
        return getParameter.apply(this, arguments);
    };
    // 8. cdc_ 变量清理
    for (const key of Object.keys(window)) {
        if (key.startsWith('cdc_') || key.startsWith('$cdc_')) delete window[key];
    }
})();
"""

    CLOUDFLARE_MARKERS = [
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

    INSTITUTIONAL_LOGIN_MARKERS = [
        "carsi",
        "shibboleth",
        "institutional login",
        "sign in through your institution",
        "选择您的机构",
    ]

    import requests
    from urllib.parse import urlparse, urljoin
    from pathlib import Path

    def normalize_doi(doi):
        return doi.strip().removeprefix("https://doi.org/").removeprefix("doi:").lower()

    def is_pdf_file(path):
        try:
            with open(path, "rb") as f:
                return f.read(5) == b"%PDF-"
        except (OSError, TypeError):
            return False

    def find_recent_pdf(directory, preferred_path, max_age_seconds=90.0):
        if isinstance(preferred_path, Path):
            if (
                preferred_path.exists()
                and preferred_path.stat().st_size > 1000
                and is_pdf_file(preferred_path)
            ):
                return preferred_path
        elif (
            os.path.exists(preferred_path)
            and os.path.getsize(preferred_path) > 1000
            and is_pdf_file(preferred_path)
        ):
            return preferred_path
        now = time.time()
        if isinstance(directory, Path):
            if not directory.exists():
                return None
            items = list(directory.iterdir())
        else:
            if not os.path.exists(directory):
                return None
            items = [os.path.join(directory, f) for f in os.listdir(directory)]
        candidates = []
        for path in items:
            if isinstance(path, str):
                if not os.path.isfile(path) or not path.lower().endswith(".pdf"):
                    continue
                size = os.path.getsize(path)
            else:
                if not path.is_file() or path.suffix.lower() != ".pdf":
                    continue
                size = path.stat().st_size
            if size <= 1000:
                continue
            mtime = (
                os.path.getmtime(str(path)) if isinstance(path, Path) else os.path.getmtime(path)
            )
            if now - mtime > max_age_seconds:
                continue
            if is_pdf_file(path):
                candidates.append(path)
        if not candidates:
            return None
        return max(candidates, key=lambda p: os.path.getmtime(str(p)))

    def move_pdf(source, target):
        import shutil

        if isinstance(source, Path) and isinstance(target, Path):
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() == target.resolve():
                return
            try:
                if target.exists():
                    target.unlink()
                source.rename(target)
            except OSError:
                shutil.copy2(source, target)
        else:
            try:
                if os.path.exists(target):
                    os.remove(target)
                os.rename(source, target)
            except (PermissionError, OSError):
                shutil.copy2(source, target)

    def same_origin_relative_url(current_url, pdf_url):
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

    def identify_publisher(doi):
        for prefix, publisher in DOI_PREFIX_MAP.items():
            if doi.startswith(prefix):
                return publisher
        return "unknown"

    def publisher_urls(doi, publisher):
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
        return {"landing": f"https://doi.org/{doi}", "pdf": None}

    def looks_like_institutional_login_needed(text):
        lowered = text.lower()
        return any(marker in lowered for marker in INSTITUTIONAL_LOGIN_MARKERS)

    PDF_URL_EXTRACTION_JS = r"""
    (function() {
      const selectors = [
        'a[href*="pdf"]', 'a[href*="pdfft"]', 'a[href*="pdfdirect"]',
        'a[href*="stampPDF"]', 'a[aria-label*="PDF" i]', 'a[title*="PDF" i]',
        'iframe[src*="pdf"]', 'iframe[src*="stampPDF"]',
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

    IEEE_ARNUMBER_JS = r"""
    (function() {
      const m = location.href.match(/document\/(\d+)/);
      if (m) return m[1];
      const html = document.documentElement.outerHTML;
      const m2 = html.match(/arnumber[=:]?(\d+)/);
      if (m2) return m2[1];
      return '';
    })()
    """

    ELSEVIER_PII_JS = r"""
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

    def extract_pdf_url_from_page(browser, doi, publisher):
        value = browser.eval(PDF_URL_EXTRACTION_JS)
        pdf_url = str(value or "")
        if pdf_url:
            return pdf_url
        current = browser.get_url()
        if any(marker in current.lower() for marker in ["pdf", "silverchair"]):
            return current
        if publisher == "elsevier":
            pii = str(browser.eval(ELSEVIER_PII_JS) or "").strip()
            if pii:
                return f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft"
        if publisher == "rsc":
            article_code = doi.split("/")[-1]
            match = re.match(r"[a-z](\d)([a-z]+)", article_code)
            if match:
                year = 2020 + int(match.group(1))
                journal = match.group(2)
                return f"https://pubs.rsc.org/en/content/articlepdf/{year}/{journal}/{article_code}"
        return None

    def prepare_rsc_pdf_url(browser, doi, pdf_url):
        browser.navigate(pdf_url, wait_seconds=10.0)
        current = browser.get_url()
        return current if current else pdf_url

    def prepare_ieee_pdf_url(browser, pdf_url):
        arnumber = str(browser.eval(IEEE_ARNUMBER_JS) or "")
        if arnumber:
            return f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arnumber}"
        return pdf_url

    def prepare_elsevier_pdf_url(browser, pdf_url, max_wait_seconds=45.0):
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
        current = pdf_url
        per_attempt_wait = max(min(max_wait_seconds / max(len(variants), 1), 12.0), 4.0)
        for variant in dict.fromkeys(variants):
            browser.navigate(variant, wait_seconds=4.0)
            if hasattr(browser, "wait_for_url_markers"):
                current = browser.wait_for_url_markers(
                    ["sciencedirectassets", "els-cdn", ".pdf"],
                    max_wait_seconds=per_attempt_wait,
                )
            else:
                current = browser.get_url()
            if "sciencedirectassets" in current or "els-cdn" in current:
                return current
        return current or pdf_url

    def sciencedirect_access_status(browser):
        expression = r"""
        (function() {
          const body = document.body ? document.body.innerText.toLowerCase() : '';
          const href = location.href.toLowerCase();
          const title = document.title || '';
          const haystack = [location.href || '', document.documentElement ? document.documentElement.outerHTML : ''].join('\n');
          const hasPii = /\/science\/article\/pii\/([A-Z0-9]+)/i.test(haystack);
          const pdfLink = document.querySelector('a[href*="pdfft"], a[href*="/pdf"], a[aria-label*="PDF" i], a[title*="PDF" i]');
          if (href.includes('pdf.sciencedirectassets.com') || href.includes('els-cdn')) return 'has_access';
          if (pdfLink || (body.includes('download') && body.includes('pdf'))) return 'has_access';
          if (
            body.includes('get access') ||
            body.includes('purchase pdf') ||
            body.includes('check for this article elsewhere') ||
            body.includes('access through your institution') ||
            body.includes('sign in via your institution') ||
            body.includes('sign in to view your account details') ||
            body.includes('not entitled') ||
            body.includes('no access') ||
            (title === 'ScienceDirect' && hasPii && !pdfLink)
          ) return 'no_access';
          return 'unknown';
        })()
        """
        value = str(browser.eval(expression) or "").strip()
        return value if value in {"has_access", "no_access", "unknown"} else "unknown"

    def click_sciencedirect_institution_login(browser):
        expression = r"""
        (function() {
          const candidates = Array.from(document.querySelectorAll('a, button, [role="button"]'));
          const needles = ['sign in via your institution', 'access through your institution', 'institution', 'get access', 'sign in'];
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
        return str(browser.eval(expression) or "")

    def wait_for_sciencedirect_authorization(
        browser,
        landing_url,
        timeout_seconds,
        poll_seconds=5.0,
    ):
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

    class CDPBrowser:
        """CDP browser wrapper — fallback version (synced from littrace.cdp_core)."""

        def __init__(self, cdp_url="http://127.0.0.1:19222"):
            self.cdp_url = cdp_url.rstrip("/")
            self.reconnect_attempts = 3
            self.ws = None
            self.msg_id = 1
            self.tab_id = None
            self._download_path = None

        def connect_new_tab(self):
            import websocket

            resp = requests.put(f"{self.cdp_url}/json/new?about:blank")
            resp.raise_for_status()
            tab = resp.json()
            self.tab_id = tab["id"]
            self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"])

        def send(self, method, params=None):
            last_error = None
            for attempt in range(self.reconnect_attempts):
                try:
                    if self.ws is None:
                        self.connect_new_tab()
                        self._reinit_after_reconnect()
                    message = {"id": self.msg_id, "method": method}
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

        def inject_stealth(self):
            self.send("Page.addScriptToEvaluateOnNewDocument", {"source": STEALTH_JS})
            print("  [stealth] 8 项覆盖已注入")

        def set_download_path(self, path):
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
                self._download_path = str(path)
            else:
                os.makedirs(path, exist_ok=True)
                self._download_path = path
            self.send(
                "Page.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": self._download_path},
            )

        def navigate(self, url, wait_seconds=8.0):
            self.send("Page.navigate", {"url": url})
            time.sleep(max(wait_seconds, 0.0))

        def eval(self, expression, await_promise=False):
            response = self.send(
                "Runtime.evaluate", {"expression": expression, "awaitPromise": await_promise}
            )
            result = response.get("result", {}).get("result", {})
            return result.get("value")

        def get_url(self):
            return str(self.eval("window.location.href") or "")

        def get_title(self):
            return str(self.eval("document.title") or "")

        def get_body_text(self, max_chars=1000):
            expression = f"document.body ? document.body.innerText.substring(0, {max_chars}) : ''"
            return str(self.eval(expression) or "")

        def is_cloudflare_challenge(self):
            haystack = f"{self.get_title()}\n{self.get_url()}\n{self.get_body_text(500)}".lower()
            return any(marker in haystack for marker in CLOUDFLARE_MARKERS)

        def wait_for_cloudflare(self, max_wait_seconds, interval=5.0):
            deadline = time.monotonic() + max(max_wait_seconds, 0.0)
            while time.monotonic() <= deadline:
                if not self.is_cloudflare_challenge():
                    return True
                time.sleep(max(interval, 0.5))
            return not self.is_cloudflare_challenge()

        def fetch_blob_to_file(self, pdf_url, target_path):
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
            if isinstance(target_path, Path):
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(content)
            else:
                os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
                with open(target_path, "wb") as f:
                    f.write(content)
            if not is_pdf_file(target_path):
                return False, "Browser fetch wrote a non-PDF response."
            return True, int(payload.get("size") or len(content))

        def trigger_anchor_download(self, pdf_url, filename):
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

        def close(self):
            if self.ws is not None:
                try:
                    self.ws.close()
                except Exception:
                    pass
                self.ws = None

        def _reconnect(self):
            self.close()
            if self.tab_id:
                try:
                    import websocket

                    tabs = requests.get(f"{self.cdp_url}/json", timeout=10.0).json()
                    for tab in tabs:
                        if tab.get("id") == self.tab_id and tab.get("webSocketDebuggerUrl"):
                            self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"])
                            self._reinit_after_reconnect()
                            return
                except Exception:
                    pass
            self.connect_new_tab()
            self._reinit_after_reconnect()

        def _reinit_after_reconnect(self):
            try:
                self.inject_stealth()
                if self._download_path:
                    self.send(
                        "Page.setDownloadBehavior",
                        {"behavior": "allow", "downloadPath": self._download_path},
                    )
            except Exception:
                pass


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

DEFAULT_CDP_URL = "http://127.0.0.1:19222"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/Desktop")
DEFAULT_EMAIL = "research@sjtu.edu.cn"
UNPAYWALL_API = "https://api.unpaywall.org/v2"
CROSSREF_API = "https://api.crossref.org/works"


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────


def _wait_for_user(prompt):
    """等待用户按 Enter，非交互模式下自动等待 60 秒"""
    if sys.stdin.isatty():
        input(prompt)
    else:
        print(prompt)
        print("  (非交互模式，自动等待 60 秒，请在浏览器窗口中完成操作...)")
        time.sleep(60)


def _surface_browser(url):
    """Bring the local Chrome tab to the foreground for user auth/verification."""
    if not url:
        return
    command = ["open", "-a", "Google Chrome", url] if sys.platform == "darwin" else [
        "python3",
        "-m",
        "webbrowser",
        url,
    ]
    subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_user_action(browser, message, wait_seconds):
    """Surface Chrome and wait while the user completes publisher auth."""
    try:
        current = browser.get_url()
    except Exception:
        current = ""
    print(message)
    if current:
        print(f"  授权页面: {current[:120]}")
    _surface_browser(current)
    if sys.stdin.isatty():
        input("  >>> 完成后按 Enter 继续...")
    else:
        print(f"  非交互模式：等待 {int(wait_seconds)} 秒后自动继续...")
        time.sleep(max(wait_seconds, 0.0))


# ──────────────────────────────────────────────
# Unpaywall OA 查询
# ──────────────────────────────────────────────


def query_unpaywall(doi, email):
    """查询 Unpaywall 获取 OA 状态和 PDF 链接"""
    url = f"{UNPAYWALL_API}/{doi}?email={email}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 422:
            return None, "Unpaywall 拒绝请求（邮箱无效）"
        if resp.status_code == 404:
            return None, "DOI 未在 Unpaywall 中找到"
        if resp.status_code != 200:
            return None, f"Unpaywall 返回 HTTP {resp.status_code}"

        data = resp.json()
        result = {
            "is_oa": data.get("is_oa", False),
            "oa_type": data.get("oa_status", "unknown"),
            "title": data.get("title", ""),
            "publisher": data.get("publisher", ""),
            "locations": [],
        }

        for loc in data.get("oa_locations", []):
            entry = {
                "host_type": loc.get("host_type"),
                "url": loc.get("url"),
                "url_for_pdf": loc.get("url_for_pdf"),
                "version": loc.get("version"),
                "license": loc.get("license"),
            }
            result["locations"].append(entry)

        return result, None
    except requests.exceptions.RequestException as e:
        return None, f"网络错误: {e}"


def try_curl_download(url, output_path, timeout=60):
    """用 curl 直接下载文件"""
    try:
        tmp_path = output_path + ".tmp"

        if "hdl.handle.net" in url or "handle" in url:
            try:
                resp = requests.get(url, timeout=15, allow_redirects=True)
                html = resp.text
                match = re.search(r'href="([^"]*bitstream[^"]*\.pdf[^"]*)"', html)
                if match:
                    pdf_link = match.group(1)
                    if pdf_link.startswith("/"):
                        pdf_link = urljoin(resp.url, pdf_link)
                    url = pdf_link
            except Exception:
                pass

        result = subprocess.run(
            [
                "curl",
                "-sL",
                "--http1.1",
                "-A",
                "Mozilla/5.0 AppleWebKit/537.36 Chrome/144.0.0.0 Safari/537.36",
                "-H",
                "Accept: application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
                "-o",
                tmp_path,
                "-w",
                "%{http_code}",
                "--max-time",
                str(timeout),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
        http_code = result.stdout.strip() if result.stdout else "000"

        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
            with open(tmp_path, "rb") as f:
                header = f.read(5)
            if header == b"%PDF-":
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rename(tmp_path, output_path)
                except (PermissionError, OSError):
                    import shutil

                    shutil.copy2(tmp_path, output_path)
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                return True, os.path.getsize(output_path)
            else:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                return False, "下载到的是 HTML 而非 PDF（可能被 Cloudflare 拦截）"

        return False, f"HTTP {http_code} 或文件过小"
    except Exception as e:
        return False, str(e)


# ──────────────────────────────────────────────
# 通用下载方案：统一决策流程
# ──────────────────────────────────────────────


def download_paper(
    doi,
    output_path,
    cdp_url=DEFAULT_CDP_URL,
    email=DEFAULT_EMAIL,
    user_wait_seconds=180,
):
    """
    通用论文下载方案

    决策流程（三步递进）:
      Step 1: Unpaywall 查 OA → 有镜像则 curl 直下
      Step 2: stealth 浏览器导航到出版商页面 → fetch+blob 下载
      Step 3: 如果 Cloudflare Captcha 无法自动通过 → 提示用户手动验证 → 继续下载
    """

    normalized = normalize_doi(doi)
    publisher = identify_publisher(normalized)
    publisher_name = PUBLISHER_NAMES.get(publisher, publisher)
    print(f"\n{'=' * 60}")
    print(f"DOI: {normalized}")
    print(f"出版商: {publisher_name}")
    print(f"输出路径: {output_path}")
    if _USING_LITTRACE_CORE:
        print("核心代码: littrace.cdp_core (共享模式)")
    else:
        print("核心代码: 内置副本 (fallback 模式)")
    print(f"{'=' * 60}\n")

    # ── Step 1: Unpaywall 查 OA ──────────────────────
    print("[Step 1] 查询 Unpaywall OA 状态...")
    oa_info, err = query_unpaywall(normalized, email)

    if oa_info:
        print(f"  OA 状态: {oa_info['oa_type']} (is_oa={oa_info['is_oa']})")
        if oa_info.get("title"):
            print(f"  论文标题: {oa_info['title'][:80]}")

        for i, loc in enumerate(oa_info["locations"]):
            pdf_url = loc.get("url_for_pdf") or loc.get("url")
            host = loc.get("host_type", "unknown")
            if not pdf_url:
                continue

            print(f"  [位置 {i + 1}] {host}: {pdf_url[:80]}...")

            if host == "repository" or "bitstream" in pdf_url or "handle" in pdf_url:
                print("  → 尝试从 repository 镜像 curl 直下...")
                success, info = try_curl_download(pdf_url, output_path)
                if success:
                    print(f"  ✅ 下载成功! {info / 1024 / 1024:.1f}MB")
                    return True, f"repository 镜像 curl 直下 ({info / 1024 / 1024:.1f}MB)"

            if host == "publisher" and oa_info["is_oa"]:
                print("  → 尝试从出版商 OA 链接 curl 直下...")
                success, info = try_curl_download(pdf_url, output_path)
                if success:
                    print(f"  ✅ 下载成功! {info / 1024 / 1024:.1f}MB")
                    return True, f"出版商 OA curl 直下 ({info / 1024 / 1024:.1f}MB)"
                else:
                    print(f"  ✗ curl 失败: {info}")

        print("  → OA 镜像下载未成功，进入浏览器方案")
    else:
        print(f"  Unpaywall 查询失败: {err}")
        print("  → 进入浏览器方案")

    # ── Step 2: stealth 浏览器 + fetch+blob ──────────────
    print("\n[Step 2] 启动 stealth 浏览器方案...")

    try:
        requests.get(f"{cdp_url}/json/version", timeout=5)
    except requests.exceptions.RequestException:
        print("  ✗ CDP 浏览器未运行，请先启动带 CDP 的 Chrome:")
        print("    google-chrome --remote-debugging-port=19222")
        return False, "CDP 浏览器未运行"

    browser = CDPBrowser(cdp_url)

    try:
        browser.connect_new_tab()
        print("  新标签页已创建")

        if hasattr(browser, "prepare_and_inject_stealth"):
            notes = browser.prepare_and_inject_stealth()
            print("  [stealth] 增强覆盖已注入")
            for note in notes:
                print(f"  [stealth] {note}")
        else:
            browser.inject_stealth()

        download_dir = os.path.dirname(output_path) or "."
        browser.set_download_path(Path(download_dir))

        urls = publisher_urls(normalized, publisher)

        landing_url = urls.get("landing")
        if landing_url:
            print(f"\n  导航到 landing page: {landing_url[:80]}...")
            browser.navigate(landing_url, wait_seconds=20.0)

            if browser.is_cloudflare_challenge():
                print("  ⚠ 检测到 Cloudflare 验证页")
                print("  等待 stealth 自动通过...")
                passed = browser.wait_for_cloudflare(max_wait_seconds=60)

                if not passed:
                    print("\n[Step 3] Cloudflare Captcha 无法自动通过")
                    _wait_for_user_action(
                        browser,
                        "  请在弹出的本地 Chrome 窗口中手动完成 Cloudflare 验证。",
                        wait_seconds=user_wait_seconds,
                    )

                    if browser.is_cloudflare_challenge():
                        print("  等待 Cloudflare 通过（最多 120 秒）...")
                        passed = browser.wait_for_cloudflare(max_wait_seconds=120)
                        if not passed:
                            print("  仍在验证页，再等 30 秒...")
                            time.sleep(30)

            body = browser.get_body_text(1200)
            if looks_like_institutional_login_needed(body):
                print("  ⚠ 检测到需要机构认证 (CARSI/Shibboleth)")
                _wait_for_user_action(
                    browser,
                    "  请在弹出的本地 Chrome 窗口中完成 CARSI/Shibboleth/机构登录。",
                    wait_seconds=user_wait_seconds,
                )

            if publisher == "elsevier":
                access_status = sciencedirect_access_status(browser)
                print(f"  ScienceDirect 访问状态: {access_status}")
                if access_status == "no_access":
                    clicked = click_sciencedirect_institution_login(browser)
                    if clicked:
                        print(f"  已尝试点击机构登录入口: {clicked[:80]}")
                        time.sleep(3)
                    _wait_for_user_action(
                        browser,
                        "  ScienceDirect 当前会话没有 PDF 机构授权。请在弹出的本地 Chrome 窗口中完成机构登录/CARSI/Shibboleth 授权。",
                        wait_seconds=0 if not sys.stdin.isatty() else user_wait_seconds,
                    )
                    if not sys.stdin.isatty():
                        print(f"  正在轮询 ScienceDirect 授权状态（最多 {int(user_wait_seconds)} 秒）...")
                    access_status = wait_for_sciencedirect_authorization(
                        browser,
                        landing_url,
                        timeout_seconds=user_wait_seconds,
                    )
                    print(f"  登录后 ScienceDirect 访问状态: {access_status}")
                    if access_status == "no_access":
                        return False, "ScienceDirect 登录后仍无 PDF 机构授权"

            print(f"  当前页面: {browser.get_title()[:60]}")

        pdf_url = urls.get("pdf") or extract_pdf_url_from_page(browser, normalized, publisher)

        if not pdf_url:
            return False, "无法找到 PDF 下载链接"

        print(f"\n  PDF URL: {pdf_url[:100]}")

        if publisher == "wiley":
            print("  → 方法: CDP setDownloadBehavior + goto")
            browser.navigate(pdf_url, wait_seconds=15.0)
            time.sleep(5)
            found = find_recent_pdf(Path(download_dir), Path(output_path))
            if found:
                move_pdf(found, Path(output_path))
                if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                    size = os.path.getsize(output_path)
                    print(f"  ✅ 下载成功! {size / 1024 / 1024:.1f}MB")
                    return True, f"CDP setDownloadBehavior + goto ({size / 1024 / 1024:.1f}MB)"

        if publisher == "rsc":
            pdf_url = prepare_rsc_pdf_url(browser, normalized, pdf_url)
            print(f"  RSC CDN URL: {pdf_url[:100]}")

        if publisher == "ieee":
            pdf_url = prepare_ieee_pdf_url(browser, pdf_url)
            print(f"  IEEE stampPDF URL: {pdf_url[:100]}")

        if publisher == "elsevier" and "pdfft" in pdf_url:
            print("  → Elsevier: 导航到 pdfft URL（等待 CDN 重定向）...")
            cdn_url = prepare_elsevier_pdf_url(browser, pdf_url, max_wait_seconds=45.0)
            if "sciencedirectassets" in cdn_url or "els-cdn" in cdn_url:
                pdf_url = cdn_url
                print(f"  CDN URL: {pdf_url[:100]}")
            elif cdn_url != pdf_url:
                pdf_url = cdn_url
                print(f"  Elsevier 候选 PDF URL: {pdf_url[:100]}")
            else:
                print(f"  ⚠ 未等到 CDN PDF URL，当前页面: {cdn_url[:100]}")

        print("  → 方法: fetch + blob + base64 回传")
        success, info = browser.fetch_blob_to_file(pdf_url, Path(output_path))

        if not success and "HTTP 403" in str(info):
            _wait_for_user_action(
                browser,
                "  PDF 请求返回 HTTP 403。请在弹出的本地 Chrome 窗口中完成账号/机构授权后，脚本会重试下载。",
                wait_seconds=user_wait_seconds,
            )
            success, info = browser.fetch_blob_to_file(pdf_url, Path(output_path))

        if success:
            if isinstance(info, int):
                print(f"  ✅ 下载成功! {info / 1024 / 1024:.1f}MB")
                return True, f"fetch+blob ({info / 1024 / 1024:.1f}MB)"
            else:
                print("  ✅ 下载成功!")
                return True, "fetch+blob"

        print(f"  fetch 失败: {info}")
        print("  → 尝试 <a download> 触发浏览器下载...")
        filename = os.path.basename(output_path)
        browser.trigger_anchor_download(pdf_url, filename)

        print("  等待浏览器下载完成 (15s)...")
        time.sleep(15)

        found = find_recent_pdf(Path(download_dir), Path(output_path))
        if found:
            move_pdf(found, Path(output_path))
            if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                size = os.path.getsize(output_path)
                print(f"  ✅ 下载成功! {size / 1024 / 1024:.1f}MB")
                return True, f"<a download> ({size / 1024 / 1024:.1f}MB)"

        return False, "所有下载方法均失败"

    finally:
        if hasattr(browser, "close_tab"):
            browser.close_tab()
        else:
            browser.close()


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="通用学术论文 PDF 下载器 - 支持七家出版商",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持出版商:
  Wiley          (10.1002)
  Springer Nature (10.1038)
  MDPI           (10.3390)
  IEEE           (10.1109)
  ACS            (10.1021)
  Elsevier       (10.1016)
  RSC            (10.1039)

示例:
  %(prog)s 10.1039/d5nr04405g
  %(prog)s 10.1002/adfm.202414678 --output ~/Desktop/paper.pdf
  %(prog)s 10.1021/acs.nano.5c21970 --cdp http://127.0.0.1:9222
        """,
    )
    parser.add_argument("doi", help="论文 DOI (如 10.1039/d5nr04405g)")
    parser.add_argument(
        "--output", "-o", default=None, help="输出文件路径 (默认: ~/Desktop/<DOI>.pdf)"
    )
    parser.add_argument(
        "--cdp",
        default=DEFAULT_CDP_URL,
        help=f"CDP 浏览器地址 (默认: {DEFAULT_CDP_URL})",
    )
    parser.add_argument(
        "--email",
        default=DEFAULT_EMAIL,
        help=f"Unpaywall 查询邮箱 (默认: {DEFAULT_EMAIL})",
    )
    parser.add_argument(
        "--user-wait-seconds",
        type=float,
        default=180,
        help="需要 Cloudflare/账号/机构授权时，非交互模式下等待用户操作的秒数。",
    )

    args = parser.parse_args()

    if not args.output:
        safe_doi = re.sub(r"[^a-zA-Z0-9._-]", "_", args.doi)
        args.output = os.path.join(DEFAULT_OUTPUT_DIR, f"{safe_doi}.pdf")

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success, info = download_paper(
        args.doi,
        args.output,
        args.cdp,
        args.email,
        user_wait_seconds=args.user_wait_seconds,
    )

    print(f"\n{'=' * 60}")
    if success:
        print("✅ 下载完成!")
        print(f"   文件: {args.output}")
        print(f"   方法: {info}")
    else:
        print(f"❌ 下载失败: {info}")
        sys.exit(1)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
