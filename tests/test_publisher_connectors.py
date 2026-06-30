from littrace.models import AccessType, PaperMetadata
from littrace.publisher_connectors import (
    build_publisher_route,
    build_publisher_route_report,
    build_publisher_search_plan,
)


def test_publisher_connector_infers_materials_publishers():
    report = build_publisher_route_report(
        [
            PaperMetadata(
                paper_id="wiley",
                title="Flexible Materials",
                journal="Advanced Functional Materials",
                publisher="Wiley",
                doi="10.1002/adfm.example",
                access_type=AccessType.REQUIRES_LOGIN,
            ),
            PaperMetadata(
                paper_id="acs",
                title="Nano Sensor",
                journal="ACS Nano",
                publisher="American Chemical Society",
                doi="10.1021/acsnano.example",
            ),
            PaperMetadata(
                paper_id="mdpi",
                title="Open Paper",
                publisher="MDPI",
                pdf_url="https://example.org/paper.pdf",
                access_type=AccessType.OPEN_ACCESS,
            ),
        ]
    )

    families = [route.publisher_family for route in report.routes]
    assert families == ["wiley", "acs", "mdpi"]
    assert report.routes[0].requires_login
    assert str(report.routes[2].pdf_url) == "https://example.org/paper.pdf"


def test_publisher_connector_uses_doi_landing_page():
    route = build_publisher_route(
        PaperMetadata(
            paper_id="p1",
            title="Paper",
            doi="10.1002/adfm.example",
            publisher="Wiley",
        )
    )

    assert str(route.landing_url) == "https://doi.org/10.1002/adfm.example"
    assert "DOI landing page" in route.notes[0]


def test_publisher_search_plan_builds_official_routes():
    report = build_publisher_search_plan("MXene flexible sensor", families=["acs", "wiley"])

    assert [plan.publisher_family for plan in report.plans] == ["acs", "wiley"]
    assert "pubs.acs.org" in str(report.plans[0].query_url)
    assert "MXene+flexible+sensor" in str(report.plans[0].query_url)
    assert report.plans[0].requires_browser
