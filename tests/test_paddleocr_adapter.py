from pathlib import Path

import pytest

from littrace.config import LitTraceConfig, PaddleOCRParserConfig, ParsingConfig
from littrace.ocr.paddleocr_adapter import PaddleOCRTool, normalize_paddleocr_result
from littrace.ocr.registry import build_ocr_tool


def test_normalize_paddleocr_result_extracts_lines():
    raw = [[[[0, 0], [1, 0], [1, 1], [0, 1]], ("Hello", 0.98)]]
    lines = normalize_paddleocr_result(raw)

    assert lines == [
        {
            "text": "Hello",
            "confidence": 0.98,
            "bbox": [[0, 0], [1, 0], [1, 1], [0, 1]],
        }
    ]


def test_paddleocr_tool_reports_missing_pdf_before_rendering():
    parsed = PaddleOCRTool().parse_pdf(Path("paper.pdf"))

    assert not parsed.parsed
    assert parsed.error == "PDF file does not exist."


def test_registry_supports_paddlerocr_alias():
    tool = build_ocr_tool(
        LitTraceConfig(parsing=ParsingConfig(default_parser="paddlerocr"))
    )

    assert isinstance(tool, PaddleOCRTool)


def test_paddleocr_pdf_parser_marks_page_evidence(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    image_path = tmp_path / "page_1.png"
    image_path.write_bytes(b"png")

    def fake_render(*args, **kwargs):
        return [(1, image_path)]

    def fake_parse_image(self, image_path, mode=None, preferred_engines=None):
        from littrace.ocr.tool import ParsedPaper

        return ParsedPaper(
            pdf_path=image_path,
            sections=[
                {
                    "name": "ocr_text",
                    "text": "Method and limitation text.",
                    "evidence": {
                        "paper_id": image_path.stem,
                        "section": "ocr_text",
                        "snippet": "Method and limitation text.",
                        "parser": "paddleocr",
                        "confidence": 0.9,
                    },
                }
            ],
            parsed=True,
        )

    monkeypatch.setattr(
        "littrace.ocr.paddleocr_adapter.render_pdf_pages_to_images",
        fake_render,
    )
    monkeypatch.setattr(PaddleOCRTool, "parse_image", fake_parse_image)

    parsed = PaddleOCRTool(
        PaddleOCRParserConfig(pdf_render_scale=1.5, max_pages=1)
    ).parse_pdf(pdf_path)

    assert parsed.parsed
    assert parsed.sections[0]["evidence"]["page"] == 1
    assert parsed.parser_reports[0]["pdf_pages_rendered"] == 1


def test_render_pdf_pages_to_images_requires_pypdfium2_when_missing(monkeypatch, tmp_path):
    from littrace.ocr import paddleocr_adapter

    def fake_import(name, *args, **kwargs):
        if name == "pypdfium2":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    original_import = __builtins__["__import__"]
    monkeypatch.setitem(__builtins__, "__import__", fake_import)

    with pytest.raises(ImportError):
        paddleocr_adapter.render_pdf_pages_to_images(
            tmp_path / "paper.pdf",
            tmp_path,
        )
