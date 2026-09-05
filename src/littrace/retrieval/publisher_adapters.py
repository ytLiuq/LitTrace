"""Hostname-aware publisher full-text link extraction.

This module only discovers links rendered by a publisher page. It never
attempts to bypass a paywall or manufacture a PDF URL when the page does not
expose one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import urljoin, urlparse


@dataclass(frozen=True)
class PublisherAdapter:
    family: str
    hosts: tuple[str, ...]
    pdf_markers: tuple[str, ...]

    def matches(self, url: str) -> bool:
        hostname = (urlparse(url).hostname or "").lower()
        return any(host in hostname for host in self.hosts)

    def extract_pdf_urls(self, base_url: str, html: str) -> list[str]:
        links: list[str] = []
        links.extend(
            unescape(value)
            for value in re.findall(
                r"<meta[^>]+name=[\"']citation_pdf_url[\"'][^>]+content=[\"']([^\"']+)",
                html,
                flags=re.IGNORECASE,
            )
        )
        links.extend(
            unescape(value)
            for value in re.findall(
                r"<(?:a|link|iframe|embed)[^>]+(?:href|src)=[\"']([^\"']+)",
                html,
                flags=re.IGNORECASE,
            )
        )
        result: list[str] = []
        seen: set[str] = set()
        for raw in links:
            absolute = urljoin(base_url, raw)
            lowered = absolute.lower()
            if not any(marker in lowered for marker in self.pdf_markers):
                continue
            if absolute not in seen:
                seen.add(absolute)
                result.append(absolute)
        return result


PUBLISHER_ADAPTERS: tuple[PublisherAdapter, ...] = (
    PublisherAdapter("acs", ("pubs.acs.org",), ("/doi/pdf", "/doi/epdf", ".pdf")),
    PublisherAdapter(
        "wiley",
        ("onlinelibrary.wiley.com", "advanced.onlinelibrary.wiley.com"),
        ("/doi/pdf", "/doi/pdfdirect", ".pdf"),
    ),
    PublisherAdapter(
        "springer_nature",
        ("link.springer.com", "nature.com"),
        ("/content/pdf", "/article/", ".pdf"),
    ),
    PublisherAdapter("rsc", ("pubs.rsc.org",), ("/articlepdf", "/pdf/", ".pdf")),
    PublisherAdapter(
        "elsevier",
        ("sciencedirect.com", "api.elsevier.com"),
        ("/science/article/pii/", "/content/article/", "/pdf", ".pdf"),
    ),
    PublisherAdapter("mdpi", ("mdpi.com",), ("/pdf", ".pdf")),
)


def adapter_for_url(url: str) -> PublisherAdapter | None:
    return next((adapter for adapter in PUBLISHER_ADAPTERS if adapter.matches(url)), None)
