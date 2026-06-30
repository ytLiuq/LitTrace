from littrace.publisher_connectors import build_publisher_search_plan
from littrace.publisher_retrieval import parse_publisher_search_html


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
