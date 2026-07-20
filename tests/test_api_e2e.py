"""Sandbox-runnable E2E tests for Crossref + Unpaywall API queries.

Unlike ``test_publisher_e2e.py`` which requires a CDP browser (GUI + network),
these tests only require HTTP access to Crossref and Unpaywall APIs.
They validate the first step of the three-step download pipeline:
DOI → publisher identification → Crossref metadata → Unpaywall OA lookup.

Run with: ``pytest tests/test_api_e2e.py -v``
Skip with: ``SKIP_NETWORK_E2E=1 pytest tests/test_api_e2e.py``
"""

from __future__ import annotations

import os

import httpx
import pytest

from littrace.cdp_core import identify_publisher, normalize_doi, publisher_urls

SKIP = os.environ.get("SKIP_NETWORK_E2E") == "1"
skip_reason = "Set SKIP_NETWORK_E2E=0 (or unset) to run network E2E tests."

# Seven-publisher test DOIs (from SKILL.md second-round test)
SEVEN_PUBLISHER_DOIS: list[tuple[str, str, str]] = [
    ("wiley", "10.1002/mame.202400237", "Wiley"),
    ("springer_nature", "10.1038/srep14751", "Springer Nature"),
    ("mdpi", "10.3390/s23052443", "MDPI"),
    ("ieee", "10.1109/SENSORS43011.2019.8956652", "IEEE"),
    ("acs", "10.1021/acsomega.3c04786", "ACS"),
    ("elsevier", "10.1016/j.matdes.2025.114201", "Elsevier"),
    ("rsc", "10.1039/d2ma00987k", "RSC"),
]

CROSSREF_API = "https://api.crossref.org/works"
UNPAYWALL_API = "https://api.unpaywall.org/v2"
UNPAYWALL_EMAIL = "research@sjtu.edu.cn"


# ---------------------------------------------------------------------------
# Publisher identification tests (no network)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "doi,expected_publisher",
    [(doi, pub) for pub, doi, _ in SEVEN_PUBLISHER_DOIS],
)
def test_identify_publisher_all_seven(doi: str, expected_publisher: str):
    """DOI prefix → publisher slug for all 7 publishers."""
    normalized = normalize_doi(doi)
    assert identify_publisher(normalized) == expected_publisher


@pytest.mark.parametrize(
    "doi",
    [doi for _, doi, _ in SEVEN_PUBLISHER_DOIS],
)
def test_publisher_urls_returns_landing(doi: str):
    """publisher_urls() always returns a landing URL."""
    normalized = normalize_doi(doi)
    publisher = identify_publisher(normalized)
    urls = publisher_urls(normalized, publisher)
    assert urls["landing"], f"No landing URL for {publisher}"


def test_springer_nature_has_direct_pdf_url():
    """Springer Nature should now have a direct PDF URL in publisher_urls."""
    urls = publisher_urls("10.1038/srep14751", "springer_nature")
    assert urls["pdf"] is not None
    assert "nature.com/articles/" in urls["pdf"]
    assert urls["pdf"].endswith(".pdf")


def test_wiley_has_direct_pdf_url():
    """Wiley should have a direct pdfdirect URL."""
    urls = publisher_urls("10.1002/mame.202400237", "wiley")
    assert urls["pdf"] is not None
    assert "pdfdirect" in urls["pdf"]


def test_acs_has_direct_pdf_url():
    """ACS should have a direct PDF URL."""
    urls = publisher_urls("10.1021/acsomega.3c04786", "acs")
    assert urls["pdf"] is not None
    assert urls["pdf"].startswith("https://pubs.acs.org/doi/pdf/")


# ---------------------------------------------------------------------------
# Crossref API E2E tests (require network)
# ---------------------------------------------------------------------------


@pytest.fixture
async def http_client():
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        headers={"User-Agent": "LitTrace-E2E-Test/1.0 (mailto:research@sjtu.edu.cn)"},
        follow_redirects=True,
    ) as client:
        yield client


async def _get_or_skip_network(http_client, url: str, **kwargs) -> httpx.Response:
    try:
        return await http_client.get(url, **kwargs)
    except httpx.RequestError as exc:
        pytest.skip(f"Network E2E endpoint is unavailable: {exc}")


@pytest.mark.anyio
@pytest.mark.skipif(SKIP, reason=skip_reason)
@pytest.mark.parametrize(
    "doi,expected_publisher_slug",
    [(doi, slug) for slug, doi, _ in SEVEN_PUBLISHER_DOIS],
)
async def test_crossref_resolves_all_seven_dois(http_client, doi, expected_publisher_slug):
    """Crossref API resolves each of the 7 publisher test DOIs."""
    response = await _get_or_skip_network(http_client, f"{CROSSREF_API}/{doi}")
    assert response.status_code == 200, f"Crossref returned {response.status_code} for {doi}"
    data = response.json()["message"]
    assert data.get("DOI", "").lower() == doi.lower()
    # Verify publisher name in Crossref matches expected
    crossref_publisher = data.get("publisher", "")
    expected_name = dict(
        wiley="Wiley",
        springer_nature="Springer",
        mdpi="MDPI",
        ieee="IEEE",
        acs="American Chemical Society",
        elsevier="Elsevier",
        rsc="Royal Society of Chemistry",
    ).get(expected_publisher_slug, "")
    if expected_name:
        assert expected_name.lower() in crossref_publisher.lower(), (
            f"Crossref publisher '{crossref_publisher}' does not match expected '{expected_name}'"
        )


@pytest.mark.anyio
@pytest.mark.skipif(SKIP, reason=skip_reason)
@pytest.mark.parametrize("doi", [doi for _, doi, _ in SEVEN_PUBLISHER_DOIS])
async def test_crossref_returns_title(http_client, doi):
    """Crossref returns a non-empty title for each DOI."""
    response = await _get_or_skip_network(http_client, f"{CROSSREF_API}/{doi}")
    assert response.status_code == 200
    titles = response.json()["message"].get("title", [])
    assert titles, f"No title returned for {doi}"
    assert len(titles[0]) > 10, f"Title too short for {doi}"


# ---------------------------------------------------------------------------
# Unpaywall API E2E tests (require network)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.skipif(SKIP, reason=skip_reason)
@pytest.mark.parametrize("doi", [doi for _, doi, _ in SEVEN_PUBLISHER_DOIS])
async def test_unpaywall_responds_all_seven_dois(http_client, doi):
    """Unpaywall API responds (200 or 404) for each DOI — validates API reachability."""
    response = await _get_or_skip_network(
        http_client,
        f"{UNPAYWALL_API}/{doi}",
        params={"email": UNPAYWALL_EMAIL},
    )
    assert response.status_code in (200, 404), (
        f"Unpaywall returned unexpected {response.status_code} for {doi}"
    )


@pytest.mark.anyio
@pytest.mark.skipif(SKIP, reason=skip_reason)
async def test_unpaywall_springer_nature_is_oa(http_client):
    """Springer Nature Scientific Reports article should be OA (gold)."""
    doi = "10.1038/srep14751"
    response = await _get_or_skip_network(
        http_client, f"{UNPAYWALL_API}/{doi}", params={"email": UNPAYWALL_EMAIL}
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("is_oa") is True, f"Expected is_oa=True for {doi}"
    assert data.get("oa_status") == "gold", f"Expected oa_status=gold, got {data.get('oa_status')}"


@pytest.mark.anyio
@pytest.mark.skipif(SKIP, reason=skip_reason)
async def test_unpaywall_mdpi_is_oa(http_client):
    """MDPI article should be OA (gold)."""
    doi = "10.3390/s23052443"
    response = await _get_or_skip_network(
        http_client, f"{UNPAYWALL_API}/{doi}", params={"email": UNPAYWALL_EMAIL}
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("is_oa") is True, f"Expected is_oa=True for {doi}"


@pytest.mark.anyio
@pytest.mark.skipif(SKIP, reason=skip_reason)
@pytest.mark.parametrize("doi", [doi for _, doi, _ in SEVEN_PUBLISHER_DOIS])
async def test_unpaywall_oa_locations_structure(http_client, doi):
    """Unpaywall OA locations have expected fields when present."""
    response = await _get_or_skip_network(
        http_client, f"{UNPAYWALL_API}/{doi}", params={"email": UNPAYWALL_EMAIL}
    )
    if response.status_code == 404:
        pytest.skip(f"DOI {doi} not in Unpaywall index yet")
    data = response.json()
    locations = data.get("oa_locations") or []
    for loc in locations:
        assert "host_type" in loc
        assert "url" in loc or "url_for_pdf" in loc


# ---------------------------------------------------------------------------
# Golden dataset coverage test
# ---------------------------------------------------------------------------


def test_golden_dataset_covers_all_seven_publishers():
    """Golden dataset should have at least one case for each of the 7 publishers."""
    import json
    from pathlib import Path

    from littrace.cdp_core import DOI_PREFIX_MAP

    golden_path = Path(__file__).parent.parent / "eval" / "golden" / "materials_seed.jsonl"
    if not golden_path.exists():
        pytest.skip(f"Golden dataset not found at {golden_path}")

    found_publishers: set[str] = set()
    with golden_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            for doi in case.get("expected_dois", []):
                pub = identify_publisher(normalize_doi(doi))
                if pub != "unknown":
                    found_publishers.add(pub)

    expected = set(DOI_PREFIX_MAP.values())
    missing = expected - found_publishers
    assert not missing, f"Golden dataset missing publishers: {missing}"
