import builtins
from pathlib import Path
import sys
import types

import pytest

from littrace.ocr.docling_adapter import _tables_from_docling_dict, markdown_to_sections
from littrace.ocr.docling_adapter import DoclingOCRTool


def test_markdown_to_sections_preserves_heading_evidence():
    sections = markdown_to_sections(
        "# Title\n\nIntro text.\n\n## Methods\nFabrication method details.",
        "p1",
    )

    assert [section["name"] for section in sections] == ["Title", "Methods"]
    assert sections[1]["evidence"]["parser"] == "docling"
    assert "Fabrication" in sections[1]["text"]


def test_docling_pdf_parser_disables_internal_ocr(monkeypatch, tmp_path):
    captured = {}

    class FakePdfPipelineOptions:
        def __init__(self, **kwargs):
            captured["pipeline_kwargs"] = kwargs

    class FakeInputFormat:
        PDF = "pdf"

    class FakePdfFormatOption:
        def __init__(self, pipeline_options):
            captured["pipeline_options"] = pipeline_options

    class FakeDocument:
        def export_to_markdown(self):
            return "# Title\n\nBody"

        def model_dump(self):
            return {"tables": []}

    class FakeResult:
        document = FakeDocument()

    class FakeDocumentConverter:
        def __init__(self, format_options=None):
            captured["format_options"] = format_options

        def convert(self, path):
            captured["path"] = path
            return FakeResult()

    modules = {
        "docling": types.ModuleType("docling"),
        "docling.document_converter": types.SimpleNamespace(
            DocumentConverter=FakeDocumentConverter,
            PdfFormatOption=FakePdfFormatOption,
        ),
        "docling.datamodel": types.ModuleType("docling.datamodel"),
        "docling.datamodel.base_models": types.SimpleNamespace(InputFormat=FakeInputFormat),
        "docling.datamodel.pipeline_options": types.SimpleNamespace(
            PdfPipelineOptions=FakePdfPipelineOptions
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules:
            return modules[name]
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    parsed = DoclingOCRTool().parse_pdf(pdf_path)

    assert parsed.parsed
    assert captured["pipeline_kwargs"] == {"do_ocr": False}
    assert parsed.structured_document["schema"] == "littrace.docling.structured_document.v1"
    assert parsed.structured_document["markdown"] == "# Title\n\nBody"


def test_docling_tables_extract_structured_cells():
    tables = _tables_from_docling_dict(
        {
            "tables": [
                {
                    "caption": "Performance",
                    "data": {
                        "grid": [
                            [{"text": "Material"}, {"text": "Gauge factor"}],
                            [{"text": "PDMS/CNT"}, {"text": "12.5"}],
                        ]
                    },
                }
            ]
        },
        "p1",
    )

    assert tables[0].caption == "Performance"
    assert tables[0].cells[0]["text"] == "Material"
    assert tables[0].cells[3]["text"] == "12.5"


def test_docling_parses_real_pdf_into_structured_document():
    pytest.importorskip("docling.document_converter")
    pdf_path = Path("data/papers/2026/10.1039_d5nr04405g/paper.pdf")
    if not pdf_path.exists():
        pytest.skip("real PDF fixture is not available")

    parsed = DoclingOCRTool().parse_pdf(pdf_path)

    assert parsed.parsed, parsed.error
    assert parsed.structured_document["schema"] == "littrace.docling.structured_document.v1"
    assert len(parsed.structured_document["markdown"]) > 1000
    assert parsed.sections
    assert parsed.structured_document["outline"]
    assert parsed.parser_reports[0]["structured_document"]["body_items"] >= 0
