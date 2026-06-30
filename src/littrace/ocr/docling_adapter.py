from __future__ import annotations

from pathlib import Path
from typing import Any

from littrace.models import EvidenceSpan
from littrace.ocr.tool import OCRMode, ParsedPaper, ParsedTable


class DoclingOCRTool:
    name = "docling"

    def parse_pdf(
        self,
        pdf_path: Path,
        mode: OCRMode = OCRMode.ACCURATE,
        preferred_engines: list[str] | None = None,
    ) -> ParsedPaper:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            return ParsedPaper(
                pdf_path=pdf_path,
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "error": "Docling is not installed. Install with: pip install -e '.[parsers]'",
                    }
                ],
                parsed=False,
                error="Docling is not installed.",
            )

        if not pdf_path.exists():
            return ParsedPaper(
                pdf_path=pdf_path,
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "error": "PDF file does not exist.",
                    }
                ],
                parsed=False,
                error="PDF file does not exist.",
            )

        try:
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            document = result.document
            markdown = document.export_to_markdown()
            raw_dict = _safe_export_dict(document)
            return ParsedPaper(
                pdf_path=pdf_path,
                title=_title_from_markdown(markdown),
                sections=markdown_to_sections(markdown, pdf_path.stem),
                tables=_tables_from_docling_dict(raw_dict, pdf_path.stem),
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "markdown_chars": len(markdown),
                        "raw_keys": sorted(raw_dict.keys()) if raw_dict else [],
                    }
                ],
                parsed=True,
            )
        except Exception as exc:
            return ParsedPaper(
                pdf_path=pdf_path,
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                ],
                parsed=False,
                error=f"{exc.__class__.__name__}: {exc}",
            )


def markdown_to_sections(markdown: str, paper_id: str) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    current_name = "document"
    current_lines: list[str] = []

    def flush() -> None:
        text = "\n".join(line for line in current_lines).strip()
        if not text:
            return
        sections.append(
            {
                "name": current_name,
                "text": text,
                "evidence": EvidenceSpan(
                    paper_id=paper_id,
                    section=current_name,
                    snippet=text[:500],
                    parser="docling",
                    confidence=0.8,
                ).model_dump(),
            }
        )

    for line in markdown.splitlines():
        if line.startswith("#"):
            flush()
            current_name = line.lstrip("#").strip() or "section"
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return sections


def _safe_export_dict(document: Any) -> dict[str, Any]:
    for method_name in ("export_to_dict", "model_dump", "dict"):
        method = getattr(document, method_name, None)
        if callable(method):
            try:
                value = method()
            except TypeError:
                continue
            if isinstance(value, dict):
                return value
    return {}


def _tables_from_docling_dict(raw: dict[str, Any], paper_id: str) -> list[ParsedTable]:
    tables = raw.get("tables")
    if not isinstance(tables, list):
        return []
    parsed_tables: list[ParsedTable] = []
    for index, table in enumerate(tables, start=1):
        caption = None
        if isinstance(table, dict):
            caption = str(table.get("caption") or table.get("label") or "") or None
        parsed_tables.append(
            ParsedTable(
                table_id=f"T{index}",
                caption=caption,
                cells=[],
                evidence=EvidenceSpan(
                    paper_id=paper_id,
                    table_id=f"T{index}",
                    snippet=caption,
                    parser="docling",
                    confidence=0.65,
                ),
            )
        )
    return parsed_tables


def _title_from_markdown(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                return title
    return None
