from littrace.config import LitTraceConfig
from littrace.context import add_papers
from littrace.document_composer import build_research_document_report
from littrace.models import EvidenceSpan, LiteratureWorkspace, PaperMetadata, PerformanceCell


def test_document_report_is_citation_and_evidence_backed():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Traceable MXene Sensor",
                authors=["Ada Lovelace"],
                year=2026,
                journal="ACS Nano",
                doi="10.1021/example",
            )
        ],
    )
    workspace.performance_cells.append(
        PerformanceCell(
            paper_id="p1",
            metric="sensitivity",
            value=12.5,
            unit="kPa-1",
            evidence=EvidenceSpan(
                paper_id="p1",
                section="Results",
                page=4,
                snippet="sensitivity reached 12.5 kPa-1",
                confidence=0.9,
            ),
        )
    )

    report = build_research_document_report(workspace, LitTraceConfig())

    assert "LitTrace Research Report" in report.markdown
    assert "## 摘要" in report.markdown
    assert "## 方法与证据来源" in report.markdown
    assert "## 局限性与下一步" in report.markdown
    assert "https://doi.org/10.1021/example" in report.markdown
    assert "sensitivity reached 12.5" in report.markdown
    assert report.evidence_count >= 2
    assert report.citation_records[0].paper_id == "p1"
