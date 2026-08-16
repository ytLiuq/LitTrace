"""Publisher connector routing test (real route inference, no I/O)."""

from __future__ import annotations

import pytest

from littrace.context import add_papers  # noqa: F401  (kept for parity with siblings)
from littrace.models import AccessType, PaperMetadata
from littrace.publisher_connectors import build_publisher_route_report


pytestmark = pytest.mark.domain


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
