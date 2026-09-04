"""Single source of truth for the publisher list LitTrace talks to.

Round 19 unified catalog — replaces three drifting lists that lived
in ``window_qt.py`` (``PUBLISHER_LINKS`` + the inline ``publisher_domains``
list inside ``_refresh_cookie_status``) and ``chrome_profiles.py``
(``PUBLISHER_COOKIE_DOMAINS``). Any new publisher we add gated-PDF
support for should be added here; everything downstream derives from
``PUBLISHERS``.

Scope note: this module deliberately does **not** replace the deeper
access-layer catalogs (``DOI_PREFIX_MAP`` / ``PUBLISHER_NAMES`` in
``access_layer/cdp_core.py``, the ``PUBLISHER_ALIASES`` family-slug
table in ``publisher_connectors.py``, etc.). Those use a different
keying (``springer_nature`` vs ``nature``, DOI prefixes, etc.) and
are owned by the access-layer code. This catalog covers the
**user-visible** concerns:

  * "which sign-in shortcut buttons does BrowserPanel render?"
  * "which domains in the cookie store mean 'logged in'?"
  * "which publishers need a clickable ✗ vs. which are always ✓?"

Anything beyond that (URL templates, search templates, DOI
prefixes) is the access layer's job.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Publisher:
    """One academic publisher LitTrace can sign the user into.

    Attributes
    ----------
    slug:
        Stable identifier used as a key in URL templates and JSON.
        Use lowercase + underscores (e.g. ``wiley``).
    display_name:
        Human label rendered in the BrowserPanel shortcut button
        (without the globe emoji prefix) and in the cookie strip.
    sign_in_url:
        The page that lands the user on a "Sign in" link after
        federated SSO redirects. Surfaced in the BrowserPanel
        shortcut row and as the ✗ anchor in the cookie strip.
    cookie_domains:
        Substrings to search the raw cookie-store bytes for. If
        any of these substrings appears, the publisher is logged
        in. Keep this conservative — over-broad matches (e.g.
        ``.com``) will produce false positives.
    requires_login:
        ``False`` for open-access archives (arXiv) — they always
        render ✓ in the cookie strip and have no sign-in shortcut.
    """

    slug: str
    display_name: str
    sign_in_url: str
    cookie_domains: tuple[str, ...]
    requires_login: bool = True

    @property
    def shortcut_label(self) -> str:
        """Globe-emoji-prefixed label rendered as the BrowserPanel
        shortcut button (e.g. ``🌐 Wiley``)."""
        return f"🌐 {self.display_name}"


# The full publisher list. Keep it ordered the way the user sees it
# in the shortcut row — that's the order the buttons render in.
PUBLISHERS: tuple[Publisher, ...] = (
    Publisher(
        slug="wiley",
        display_name="Wiley",
        sign_in_url="https://onlinelibrary.wiley.com/action/login",
        cookie_domains=("wiley.com", "onlinelibrary.wiley.com"),
    ),
    Publisher(
        slug="acs",
        display_name="ACS",
        # Round 22: ACS's real login entry point is /action/login
        # (NOT /action/showLogin which used to be the previous
        # placeholder, and NOT /action/sso which doesn't exist as a
        # login entry on pubs.acs.org). Verified against the live
        # ACS publications site.
        sign_in_url="https://pubs.acs.org/action/login",
        cookie_domains=("acs.org", "pubs.acs.org"),
    ),
    Publisher(
        slug="springer",
        display_name="Springer",
        # Round 22: ``link.springer.com/signup-login`` immediately
        # 302-redirects to the Spring Nature federated identity
        # provider at ``idp-personal-authenticator.springernature.com``;
        # the actual login session cookie lands on
        # ``springernature.com`` (not on ``springer.com`` or
        # ``link.springer.com``). Without the extra domain the
        # cookie detector returns "logged out" after a successful
        # login, and ``sentinel`` falls back to public-only fetches.
        sign_in_url="https://link.springer.com/signup-login",
        cookie_domains=(
            "springer.com",
            "link.springer.com",
            "springernature.com",
            "idp.springernature.com",
            "idp-personal-authenticator.springernature.com",
        ),
    ),
    Publisher(
        slug="elsevier",
        display_name="Elsevier",
        sign_in_url=(
            "https://www.elsevier.com/connect/"
            "elsevier-username-and-password-sign-in"
        ),
        cookie_domains=("elsevier.com", "sciencedirect.com"),
    ),
    Publisher(
        slug="nature",
        display_name="Nature",
        sign_in_url=(
            "https://idp.nature.com/authorize?response_type=cookie"
        ),
        cookie_domains=("nature.com",),
    ),
    Publisher(
        slug="rsc",
        display_name="RSC",
        sign_in_url="https://pubs.rsc.org/en/login",
        cookie_domains=("rsc.org", "pubs.rsc.org"),
    ),
    Publisher(
        slug="ieee",
        display_name="IEEE",
        sign_in_url="https://ieeexplore.ieee.org/Xplore/login.jsp",
        cookie_domains=("ieee.org", "ieeexplore.ieee.org"),
    ),
    Publisher(
        slug="mdpi",
        display_name="MDPI",
        # Round 22: ``https://www.mdpi.com/login`` returns 404 (the
        # endpoint never existed — it was carried over from the
        # legacy ``PUBLISHER_LINKS`` parallel list with no live
        # verification). The real MDPI account login lives at
        # ``/user/login`` per MDPI's current site, but that page
        # itself 302-redirects to ``login.mdpi.com/login`` (MDPI's
        # central auth gateway). The session cookie lands on
        # ``login.mdpi.com`` — without that domain in
        # ``cookie_domains`` the detector still returns "logged out"
        # after a successful login.
        sign_in_url="https://www.mdpi.com/user/login",
        cookie_domains=("mdpi.com", "login.mdpi.com"),
    ),
    Publisher(
        slug="arxiv",
        display_name="arXiv",
        sign_in_url="https://arxiv.org/login",
        cookie_domains=("arxiv.org",),
        requires_login=False,  # open access — never gated
    ),
)


def get_by_slug(slug: str) -> Publisher | None:
    """Lookup helper used by tests + access-layer code."""
    for p in PUBLISHERS:
        if p.slug == slug:
            return p
    return None


def get_by_display_name(name: str) -> Publisher | None:
    """Lookup helper used by ``_refresh_cookie_status`` which only
    knows the human-readable display name from the legacy parallel
    list."""
    for p in PUBLISHERS:
        if p.display_name == name:
            return p
    return None


def publisher_cookie_domains() -> tuple[str, ...]:
    """Flat tuple of every cookie domain across every publisher.

    This is what ``chrome_profiles._detect_publisher_cookie_domains``
    substring-matches against. Exported as a function (not a module
    constant) so callers can re-import without circularity and tests
    can monkeypatch individual publishers if they need to.
    """
    out: list[str] = []
    seen: set[str] = set()
    for pub in PUBLISHERS:
        for d in pub.cookie_domains:
            if d not in seen:
                seen.add(d)
                out.append(d)
    return tuple(out)


def sign_in_shortlinks() -> tuple[tuple[str, str], ...]:
    """``(shortcut_label, sign_in_url)`` pairs for the BrowserPanel
    shortcut row. Returns publishers that actually need a login
    shortcut (``requires_login=True``) — arXiv is excluded because
    clicking 🌐 arXiv makes no sense if the user is always logged in.
    """
    return tuple(
        (p.shortcut_label, p.sign_in_url)
        for p in PUBLISHERS
        if p.requires_login
    )


__all__ = [
    "Publisher",
    "PUBLISHERS",
    "get_by_slug",
    "get_by_display_name",
    "publisher_cookie_domains",
    "sign_in_shortlinks",
]