from littrace.config import LitTraceConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.quality_report import build_quality_report


def test_quality_report_summarizes_workspace():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper", doi="10.1000/example")],
    )
    workspace.supplementary_links["p1"] = ["https://example.org/si.pdf"]

    report = build_quality_report(LitTraceConfig(), workspace)

    assert report.metrics["active_paper_count"] == 1.0
    assert report.metrics["supplementary_link_count"] == 1.0
    assert "citation_guard_pass" in report.metrics
