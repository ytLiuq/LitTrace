from littrace.config import LitTraceConfig, StorageConfig
from littrace.context import add_papers
from littrace.models import (
    EvidenceSpan,
    LiteratureWorkspace,
    PaperMetadata,
    ParsedPaper,
    ParsedTable,
)
from littrace.evidence.parsing import parse_workspace_papers
from littrace.runtime.memory import build_session_memory
from littrace.session import create_chat_session, save_workspace


def _minimal_pdf_bytes() -> bytes:
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R>>endobj\n"
        b"4 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 20 120 Td (LitTrace PDF) Tj ET\n"
        b"endstream endobj\n"
        b"xref\n0 5\n0000000000 65535 f \n"
        b"trailer<</Root 1 0 R/Size 5>>\nstartxref\n0\n%%EOF\n"
    )


def test_parse_workspace_papers_fails_without_local_pdf():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="A materials paper",
                abstract="This method has limitations.",
                year=2026,
            )
        ],
    )

    workspace, report = parse_workspace_papers(workspace, LitTraceConfig())

    assert report["parser"] == "docling"
    assert report["missing_pdf_count"] == 1
    assert report["failed_count"] == 1
    assert report["parsed_count"] == 0
    assert workspace.parsed_papers["p1"].title == "A materials paper"
    assert "metadata/abstract fallback is disabled" in workspace.parsed_papers["p1"].error


def test_parse_workspace_papers_records_docling_quality_report(tmp_path):
    class FakeDoclingTool:
        name = "docling"

        def parse_pdf(self, pdf_path, mode=None, preferred_engines=None):
            return ParsedPaper(
                pdf_path=pdf_path,
                title="Paper",
                parsed=True,
                structured_document={"markdown": "# Title\n\n" + "body " * 200},
                sections=[{"name": "Title", "text": "body"}],
                tables=[
                    ParsedTable(
                        table_id="T1",
                        cells=[{"row": 0, "column": 0, "text": "metric"}],
                        evidence=EvidenceSpan(paper_id="p1", table_id="T1"),
                    )
                ],
            )

    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    paper = PaperMetadata(paper_id="p1", title="Paper")
    workspace = add_papers(LiteratureWorkspace(papers={"p1": paper}), [paper])
    from littrace.evidence.parsing import local_pdf_path

    pdf_path = local_pdf_path(config, paper)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4")

    workspace, report = parse_workspace_papers(workspace, config, tool=FakeDoclingTool())

    assert report["parsed_count"] == 1
    quality = workspace.context.filters.docling_quality_reports["p1"]
    assert quality["parser"] == "docling"
    assert quality["score"] > 0


def test_real_pdf_structured_document_persists_to_session_memory(tmp_path):
    class FakeDoclingTool:
        name = "docling"

        def parse_pdf(self, pdf_path, mode=None, preferred_engines=None):
            assert pdf_path.read_bytes().startswith(b"%PDF-")
            return ParsedPaper(
                pdf_path=pdf_path,
                title="Real PDF Paper",
                parsed=True,
                structured_document={
                    "schema": "littrace.docling.structured_document.v1",
                    "markdown": "# Real PDF Paper\n\n" + "structured body " * 80,
                    "outline": [{"level": 1, "title": "Real PDF Paper"}],
                },
                sections=[{"name": "Real PDF Paper", "text": "structured body"}],
                tables=[
                    ParsedTable(
                        table_id="T1",
                        cells=[{"row": 0, "column": 0, "text": "Gauge factor"}],
                        evidence=EvidenceSpan(paper_id="p1", table_id="T1"),
                    )
                ],
            )

    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            sessions_dir=tmp_path / "sessions",
        )
    )
    paper = PaperMetadata(paper_id="p1", title="Real PDF Paper")
    workspace = add_papers(LiteratureWorkspace(papers={"p1": paper}), [paper])
    from littrace.evidence.parsing import local_pdf_path

    pdf_path = local_pdf_path(config, paper)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(_minimal_pdf_bytes())

    workspace, report = parse_workspace_papers(workspace, config, tool=FakeDoclingTool())
    session = create_chat_session(config)
    save_workspace(session, workspace)
    memory = build_session_memory(workspace, session_id=session.session_id)

    assert report["parsed_count"] == 1
    structured_path = session.structured_documents_dir / "p1.json"
    assert structured_path.exists()
    assert workspace.context.filters.structured_document_paths["p1"] == str(structured_path)
    assert workspace.context.filters.structured_document_count == 1
    assert memory.document.records
    assert memory.document.records[0].content["structured_document_path"] == str(structured_path)
    assert memory.document.records[0].content["quality"]["parser"] == "docling"
