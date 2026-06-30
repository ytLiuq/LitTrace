from littrace.publisher_connectors import build_publisher_search_plan
from littrace.publisher_retrieval import (
    build_browser_retrieval_plan,
    parse_publisher_article_html,
    parse_publisher_search_html,
)


def test_parse_publisher_search_html_extracts_doi_records():
    plan = build_publisher_search_plan("sensor", families=["acs"]).plans[0]
    html = """
    <html><body>
      <h2>MXene flexible pressure sensor with improved stability</h2>
      <a href="/doi/10.1021/acsnano.6b00001">doi</a>
    </body></html>
    """

    result = parse_publisher_search_html(plan, html)

    assert result.publisher_family == "acs"
    assert result.papers[0].doi == "10.1021/acsnano.6b00001"
    assert "MXene flexible pressure sensor" in result.papers[0].title


def test_browser_retrieval_plan_describes_safe_steps():
    plan = build_publisher_search_plan("sensor", families=["wiley"]).plans[0]
    browser_plan = build_browser_retrieval_plan(plan)

    assert browser_plan.requires_user_login
    assert any("Do not bypass" in step for step in browser_plan.steps)
    assert browser_plan.extract_selectors


def test_parse_publisher_article_html_extracts_enrichment_and_si():
    html = """
    <meta name="citation_article_type" content="Research Article">
    <meta name="keywords" content="MXene, flexible sensor">
    <section class="abstract">This study reports a flexible sensor with stable response.</section>
    <a href="https://example.org/supporting-info.pdf">Supporting Information</a>
    10.1021/acsnano.6b00001
    """

    enrichment = parse_publisher_article_html(html)

    assert enrichment.article_type == "Research Article"
    assert enrichment.keywords == ["MXene", "flexible sensor"]
    assert "flexible sensor" in enrichment.abstract
    assert str(enrichment.supplementary_links[0]) == "https://example.org/supporting-info.pdf"
