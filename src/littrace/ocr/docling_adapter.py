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
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
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
            pipeline_options = PdfPipelineOptions(do_ocr=False)
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
            result = converter.convert(str(pdf_path))
            document = result.document
            markdown = document.export_to_markdown()
            raw_dict = _safe_export_dict(document)
            sections = markdown_to_sections(markdown, pdf_path.stem)
            tables = _tables_from_docling_dict(raw_dict, pdf_path.stem)
            figures = _figures_from_docling_dict(raw_dict, pdf_path.stem)
            return ParsedPaper(
                pdf_path=pdf_path,
                title=_title_from_markdown(markdown),
                structured_document=_structured_document(
                    markdown=markdown,
                    raw=raw_dict,
                    sections=sections,
                    tables=tables,
                    figures=figures,
                ),
                sections=sections,
                tables=tables,
                figures=figures,
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "markdown_chars": len(markdown),
                        "raw_keys": sorted(raw_dict.keys()) if raw_dict else [],
                        "structured_document": _document_summary(raw_dict),
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
        cells: list[dict[str, object]] = []
        if isinstance(table, dict):
            caption = str(table.get("caption") or table.get("label") or "") or None
            cells = _table_cells(table)
        parsed_tables.append(
            ParsedTable(
                table_id=f"T{index}",
                caption=caption,
                cells=cells,
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


def _table_cells(table: dict[str, Any]) -> list[dict[str, object]]:
    data = table.get("data") or table.get("table_data") or table.get("cells")
    if not data:
        return []
    if isinstance(data, dict):
        grid = data.get("grid") or data.get("table_cells") or data.get("cells") or []
    else:
        grid = data
    cells: list[dict[str, object]] = []
    if not isinstance(grid, list):
        return cells
    for row_index, row in enumerate(grid):
        if isinstance(row, list):
            for col_index, value in enumerate(row):
                cells.append(
                    {
                        "row": row_index,
                        "column": col_index,
                        "text": _cell_text(value),
                    }
                )
        elif isinstance(row, dict):
            cells.append(
                {
                    "row": row.get("row_span") or row.get("start_row_offset_idx") or row_index,
                    "column": row.get("col_span") or row.get("start_col_offset_idx"),
                    "text": _cell_text(row),
                }
            )
    return [cell for cell in cells if cell.get("text")]


def _cell_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("text", "content", "caption"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        return ""
    return str(value).strip()


def _figures_from_docling_dict(raw: dict[str, Any], paper_id: str) -> list[dict[str, object]]:
    figures = raw.get("figures") or raw.get("pictures") or []
    if not isinstance(figures, list):
        return []
    parsed: list[dict[str, object]] = []
    for index, figure in enumerate(figures, start=1):
        caption = None
        if isinstance(figure, dict):
            caption = str(figure.get("caption") or figure.get("label") or "") or None
        parsed.append(
            {
                "figure_id": f"F{index}",
                "caption": caption,
                "evidence": EvidenceSpan(
                    paper_id=paper_id,
                    section="figures",
                    snippet=caption,
                    parser="docling",
                    confidence=0.6,
                ).model_dump(),
            }
        )
    return parsed


def _document_summary(raw: dict[str, Any]) -> dict[str, object]:
    if not raw:
        return {}
    body = raw.get("body") or raw.get("texts") or raw.get("groups") or []
    return {
        "tables": len(raw.get("tables") or []),
        "figures": len(raw.get("figures") or raw.get("pictures") or []),
        "body_items": len(body) if isinstance(body, list) else 0,
    }


def _structured_document(
    *,
    markdown: str,
    raw: dict[str, Any],
    sections: list[dict[str, object]],
    tables: list[ParsedTable],
    figures: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": "littrace.docling.structured_document.v1",
        "parser": "docling",
        "markdown": markdown,
        "outline": [
            {
                "name": str(section.get("name") or "section"),
                "chars": len(str(section.get("text") or "")),
            }
            for section in sections
        ],
        "tables": [
            {
                "table_id": table.table_id,
                "caption": table.caption,
                "cell_count": len(table.cells),
                "cells": table.cells,
                "evidence": table.evidence.model_dump(),
            }
            for table in tables
        ],
        "figures": figures,
        "summary": _document_summary(raw),
        "docling_raw_keys": sorted(raw.keys()) if raw else [],
    }


def _title_from_markdown(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                return title
    return None
