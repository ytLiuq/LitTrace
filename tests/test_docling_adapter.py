import builtins
import sys
import types

from littrace.ocr.docling_adapter import markdown_to_sections
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
