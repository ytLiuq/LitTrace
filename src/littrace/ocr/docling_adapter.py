from __future__ import annotations

from pathlib import Path
from typing import Any

from littrace.models import EvidenceSpan, ParsedPaper, ParsedTable
from littrace.ocr.tool import OCRMode


class DoclingOCRTool:
    name = "docling"

    def __init__(self, config=None):
        self.config = config

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
            docling_config = getattr(getattr(self.config, "parsing", None), "docling", None)
            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                generate_picture_images=mode not in {OCRMode.FAST},
                do_picture_description=bool(
                    getattr(docling_config, "describe_figures", False)
                ),
            )
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
            result = converter.convert(str(pdf_path))
            document = result.document
            raw_dict = _safe_export_dict(document)
            figures, figure_assets = _extract_figures(document, pdf_path)
            markdown = document.export_to_markdown()
            markdown = _decode_html_entities(markdown)
            markdown = _replace_image_placeholders(markdown, figures)
            # Docling only emits a figures[] entry when the PDF contains an
            # actual picture item. Many real-world PDFs (and all
            # fpdf2-generated ones) ship figure captions as plain bold
            # text. Fall back to regex-matching the markdown so RAG never
            # loses the figure + caption association.
            figures = _merge_figures_from_markdown(figures, markdown, pdf_path.stem)
            sections = markdown_to_sections(markdown, pdf_path.stem)
            tables = _tables_from_docling_dict(raw_dict, pdf_path.stem)
            return ParsedPaper(
                pdf_path=pdf_path,
                title=_title_from_markdown(markdown),
                structured_document=_structured_document(
                    markdown=markdown,
                    raw=raw_dict,
                    sections=sections,
                    tables=tables,
                    figures=figures,
                    figure_assets=figure_assets,
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
                        "figure_asset_count": len(figure_assets),
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


def _extract_figures(document: Any, pdf_path: Path) -> tuple[list[dict[str, object]], list[str]]:
    figures: list[dict[str, object]] = []
    assets: list[str] = []
    items = getattr(document, "iterate_items", lambda **_: [])(with_groups=False)
    for item, _ in items:
        label = str(getattr(getattr(item, "label", None), "value", getattr(item, "label", ""))).lower()
        if "picture" not in label and "chart" not in label:
            continue
        index = len(figures) + 1
        caption = _item_caption(item, document)
        visual_summary = _item_visual_summary(item)
        asset_path = pdf_path.parent / "docling_assets" / f"figure-{index}.png"
        try:
            image = item.get_image(document)
            if image is not None:
                asset_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(asset_path, format="PNG")
                assets.append(str(asset_path))
        except Exception:
            pass
        figures.append(
            {
                "figure_id": f"F{index}",
                "caption": caption,
                "asset_path": str(asset_path) if asset_path.exists() else None,
                "asset_ref": f"artifact://figure_image/{pdf_path.stem}/F{index}",
                "summary": visual_summary or caption or f"Extracted figure F{index}; visual summary unavailable.",
                "visual_summary_available": bool(visual_summary),
                "evidence": EvidenceSpan(
                    paper_id=pdf_path.stem,
                    section="figures",
                    snippet=caption,
                    parser="docling",
                    confidence=0.6 if caption else 0.35,
                ).model_dump(),
            }
        )
    return figures, assets


def _item_caption(item: Any, document: Any) -> str | None:
    for name in ("caption_text", "get_caption_text"):
        method = getattr(item, name, None)
        if callable(method):
            try:
                value = method(document)
                if value:
                    return str(value).strip()
            except Exception:
                pass
    return str(getattr(item, "text", "") or "").strip() or None


def _item_visual_summary(item: Any) -> str | None:
    meta = getattr(item, "meta", None)
    description = getattr(meta, "description", None)
    text = getattr(description, "text", None)
    return str(text).strip() if text else None


_FIGURE_CAPTION_RE = __import__("re").compile(
    r"^(?:\s*)(?:Figure|Fig\.|图|表)\s*(\d+)[\.\:\)]\s*(.+?)\s*$",
    __import__("re").IGNORECASE,
)


def _merge_figures_from_markdown(
    figures: list[dict[str, object]],
    markdown: str,
    paper_id: str,
) -> list[dict[str, object]]:
    """Backfill figure entries that docling missed.

    Docling's `_extract_figures` walks `document.iterate_items()` and only
    emits a figure when the PDF actually carries a picture item. Many
    real-world PDFs (and all fpdf2 / LaTeX-rendered ones) ship the
    figure caption as a plain bold text line such as
    ``Figure 1. Schematic of the ...`` with no inline image. Without
    this backfill, RAG's `parsed.figures` is empty and figure chunks
    never reach the chunker.

    Strategy: regex-match ``^(Figure|Fig.|图|表) N. <caption>`` lines
    in the markdown, and either:
      - merge with an existing docling-detected figure whose
        ``figure_id`` matches the same N, or
      - append a new synthetic figure with that caption.

    Returns the merged figure list (does not mutate the input).
    """
    detected_by_id = {str(f.get("figure_id")): f for f in figures}
    next_index = max(
        (len(detected_by_id) + 1),
        _next_figure_index(markdown),
    )
    out: list[dict[str, object]] = list(figures)
    seen_numbers: set[int] = set()
    for line in markdown.splitlines():
        match = _FIGURE_CAPTION_RE.match(line)
        if not match:
            continue
        number = int(match.group(1))
        if number in seen_numbers:
            continue
        seen_numbers.add(number)
        caption = _decode_html_entities(match.group(2).strip()).rstrip(".")
        existing = detected_by_id.get(f"F{number}")
        if existing is not None:
            if not existing.get("caption"):
                existing["caption"] = caption
                ev = existing.get("evidence") or {}
                if isinstance(ev, dict):
                    ev["snippet"] = caption
            continue
        new_figure = {
            "figure_id": f"F{number}",
            "caption": caption,
            "asset_path": None,
            "asset_ref": f"artifact://figure_image/{paper_id}/F{number}",
            "summary": caption,
            "visual_summary_available": False,
            "source": "markdown_fallback",
            "evidence": EvidenceSpan(
                paper_id=paper_id,
                section="figures",
                snippet=caption[:500],
                parser="docling+markdown",
                confidence=0.5,
            ).model_dump(),
        }
        out.append(new_figure)
        next_index = number + 1
    out.sort(key=lambda f: _figure_sort_key(str(f.get("figure_id"))))
    return out


def _next_figure_index(markdown: str) -> int:
    max_seen = 0
    for line in markdown.splitlines():
        match = _FIGURE_CAPTION_RE.match(line)
        if match:
            max_seen = max(max_seen, int(match.group(1)))
    return max_seen


def _figure_sort_key(figure_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in figure_id if ch.isdigit())
    return (int(digits) if digits else 0, figure_id)


_HTML_ENTITY_RE = __import__("re").compile(r"&(#?[a-z0-9]+);", __import__("re").IGNORECASE)


def _decode_html_entities(text: str) -> str:
    """Decode the handful of HTML entities docling emits into markdown.

    Docling escapes ``>``, ``<``, ``&`` as ``&gt;`` / ``&lt;`` / ``&amp;``
    which corrupts downstream RAG tokenization (e.g. ``&gt;5000`` no
    longer matches ``>5000``). This is intentionally minimal — only the
    entities docling actually emits, leaving numeric entities intact.
    """
    table = {"gt": ">", "lt": "<", "amp": "&", "quot": '"', "apos": "'"}
    return _HTML_ENTITY_RE.sub(
        lambda m: table.get(m.group(1).lower(), m.group(0)), text
    )


def _replace_image_placeholders(markdown: str, figures: list[dict[str, object]]) -> str:
    for figure in figures:
        ref = str(figure.get("asset_ref") or "")
        label = str(figure.get("figure_id") or "figure")
        markdown = markdown.replace("<!-- image -->", f"![{label}]({ref})", 1)
    return markdown


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
    figure_assets: list[str],
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
        "figure_assets": figure_assets,
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
