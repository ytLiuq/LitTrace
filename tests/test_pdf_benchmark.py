from littrace.config import LitTraceConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.pdf_benchmark import benchmark_pdf_parsing


def test_pdf_benchmark_reports_missing_local_pdfs_and_confidence():
    workspace = add_papers(
        LiteratureWorkspace(
            parsed_papers={
                "p1": {
                    "parsed": True,
                    "sections": [
                        {
                            "name": "Results",
                            "text": "sensitivity 2.3 kPa-1",
                            "evidence": {"page": 2, "confidence": 0.8},
                        }
                    ],
                }
            }
        ),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026)],
    )

    report = benchmark_pdf_parsing(workspace, LitTraceConfig())

    assert report.active_papers == 1
    assert report.local_pdf_count == 0
    assert report.parsed_count == 1
    assert report.parsed_with_page_evidence == 1
    assert report.average_evidence_confidence == 0.8
