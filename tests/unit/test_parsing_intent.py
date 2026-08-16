"""Real PDF → docling → session memory integration test.

This test exercises the project's real docling integration end-to-end:
``fpdf2`` generates a real PDF with extractable text, ``DoclingOCRTool``
parses it through the real docling library, and the resulting structured
document is persisted through the real ``save_workspace`` +
``build_session_memory`` flow into the real Postgres metadata store.

The test is skipped when docling is not installed.
"""

from __future__ import annotations

import pytest

from littrace.config import LitTraceConfig, StorageConfig
from littrace.context import add_papers
from littrace.evidence.parsing import parse_workspace_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.ocr.docling_adapter import DoclingOCRTool
from littrace.runtime.memory import build_session_memory
from littrace.session import create_chat_session, save_workspace


pytestmark = pytest.mark.unit


def _build_real_pdf(path) -> None:
    """Write a real PDF with multiple lines of text using fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(50, 10, "MXene Pressure Sensor Performance Analysis")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 12)
    for line in [
        "",
        "Abstract: We report a flexible pressure sensor based on MXene nanocomposites.",
        "The sensor exhibits a sensitivity of 12.5 kPa-1 under cyclic loading.",
        "",
        "Introduction: Flexible electronics have attracted significant attention.",
        "MXene materials offer high conductivity and mechanical robustness.",
        "",
        "Results: The gauge factor was measured to be 12.5 kPa-1.",
        "Response time was 80 ms under pressure step input.",
    ]:
        pdf.cell(50, 8, line)
        pdf.ln(8)
    pdf.output(str(path))


def test_real_pdf_structured_document_persists_to_session_memory(tmp_path):
    pytest.importorskip("docling.document_converter")

    from littrace.evidence.parsing import local_pdf_path

    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            sessions_dir=tmp_path / "sessions",
        )
    )
    paper = PaperMetadata(paper_id="p1", title="MXene Pressure Sensor Performance Analysis")
    workspace = add_papers(LiteratureWorkspace(papers={"p1": paper}), [paper])

    pdf_path = local_pdf_path(config, paper)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    _build_real_pdf(pdf_path)

    # Real DoclingOCRTool against a real PDF — exercises the project's
    # actual docling integration (not a hardcoded ParsedPaper fake).
    workspace, report = parse_workspace_papers(workspace, config, tool=DoclingOCRTool())

    session = create_chat_session(config)
    save_workspace(session, workspace)
    memory = build_session_memory(workspace, session_id=session.session_id)

    assert report["parsed_count"] == 1
    assert workspace.parsed_papers["p1"].parsed
    assert "MXene" in (workspace.parsed_papers["p1"].structured_document or {}).get(
        "markdown", ""
    )

    structured_path = session.structured_documents_dir / "p1.json"
    assert structured_path.exists()
    assert workspace.context.filters.structured_document_paths["p1"] == str(structured_path)
    assert workspace.context.filters.structured_document_count == 1
    assert memory.document.records
    assert memory.document.records[0].content["structured_document_path"] == str(structured_path)
    assert memory.document.records[0].content["quality"]["parser"] == "docling"
