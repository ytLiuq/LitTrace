from littrace.config import LitTraceConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.ocr.tool import ParsedPaper
from littrace.pdf_benchmark import benchmark_pdf_parsing, benchmark_single_pdf


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
    assert report.local_pdf_rate == 0.0
    assert report.parsed_rate == 1.0


def test_benchmark_single_pdf_reports_elapsed_and_chars(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    class FakeTool:
        name = "fake"
        progress_callback = None

        def parse_pdf(self, pdf_path, mode=None):
            if self.progress_callback:
                self.progress_callback({"page": 1, "total_pages": 1})
            return ParsedPaper(
                pdf_path=pdf_path,
                sections=[{"text": "hello world"}],
                parsed=True,
                parser_reports=[{"parser": "fake"}],
            )

    monkeypatch.setattr("littrace.pdf_benchmark.build_ocr_tool", lambda config, paper_lookup: FakeTool())

    report = benchmark_single_pdf(pdf, LitTraceConfig())

    assert report.parsed
    assert report.parser == "fake"
    assert report.section_count == 1
    assert report.total_chars == 11
    assert report.progress_events == [{"page": 1, "total_pages": 1}]
