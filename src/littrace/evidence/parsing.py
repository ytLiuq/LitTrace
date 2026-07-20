from __future__ import annotations

from pathlib import Path

from littrace.access_layer.paths import paper_storage_dir
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.ocr.registry import build_ocr_tool
from littrace.ocr.tool import OCRMode, OCRTool, ParsedPaper


def parse_workspace_papers(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    tool: OCRTool | None = None,
    mode: OCRMode = OCRMode.ACCURATE,
) -> tuple[LiteratureWorkspace, dict[str, object]]:
    paper_lookup = workspace.papers
    parser = tool or build_ocr_tool(config, paper_lookup)
    parsed_count = 0
    failed_count = 0
    missing_pdf_count = 0

    for paper_id in workspace.context.active_papers:
        paper = workspace.papers[paper_id]
        pdf_path = local_pdf_path(config, paper)
        if not pdf_path.exists():
            missing_pdf_count += 1
            failed_count += 1
            workspace.parsed_papers[paper_id] = ParsedPaper(
                pdf_path=pdf_path,
                title=paper.title,
                parser_reports=[
                    {
                        "parser": getattr(parser, "name", parser.__class__.__name__),
                        "mode": mode,
                        "error": "No local PDF is available; metadata/abstract fallback is disabled.",
                    }
                ],
                parsed=False,
                error="No local PDF is available; metadata/abstract fallback is disabled.",
            )
            continue
        parsed = parser.parse_pdf(pdf_path, mode=mode)
        workspace.parsed_papers[paper_id] = parsed
        if getattr(parser, "name", parser.__class__.__name__) == "docling":
            workspace.context.filters.docling_quality_reports[paper_id] = _docling_quality_report(
                parsed
            )
        if parsed.parsed:
            parsed_count += 1
        else:
            failed_count += 1

    return workspace, {
        "parser": getattr(parser, "name", parser.__class__.__name__),
        "active_papers": len(workspace.context.active_papers),
        "parsed_count": parsed_count,
        "failed_count": failed_count,
        "missing_pdf_count": missing_pdf_count,
    }


def local_pdf_path(config: LitTraceConfig, paper: PaperMetadata) -> Path:
    return paper_storage_dir(config, paper) / "paper.pdf"


def _docling_quality_report(parsed: ParsedPaper) -> dict[str, object]:
    markdown = str(parsed.structured_document.get("markdown") or "")
    section_count = len(parsed.sections)
    empty_sections = sum(
        1 for section in parsed.sections if not str(section.get("text") or "").strip()
    )
    table_count = len(parsed.tables)
    table_cell_count = sum(len(table.cells) for table in parsed.tables)
    figure_count = len(parsed.figures)
    warnings: list[str] = []
    if parsed.parsed and len(markdown) < 500:
        warnings.append("docling_markdown_short")
    if table_count and table_cell_count == 0:
        warnings.append("docling_tables_without_cells")
    if section_count and empty_sections / max(section_count, 1) > 0.5:
        warnings.append("docling_many_empty_sections")
    if parsed.error:
        warnings.append("docling_parse_error")
    return {
        "parser": "docling",
        "parsed": parsed.parsed,
        "markdown_chars": len(markdown),
        "section_count": section_count,
        "empty_section_count": empty_sections,
        "table_count": table_count,
        "table_cell_count": table_cell_count,
        "figure_count": figure_count,
        "warnings": warnings,
        "score": _docling_quality_score(
            parsed=parsed,
            markdown_chars=len(markdown),
            section_count=section_count,
            table_count=table_count,
            table_cell_count=table_cell_count,
            warning_count=len(warnings),
        ),
    }


def _docling_quality_score(
    *,
    parsed: ParsedPaper,
    markdown_chars: int,
    section_count: int,
    table_count: int,
    table_cell_count: int,
    warning_count: int,
) -> float:
    if not parsed.parsed:
        return 0.0
    score = 0.35
    if markdown_chars >= 500:
        score += 0.25
    if section_count >= 2:
        score += 0.15
    if table_count == 0 or table_cell_count > 0:
        score += 0.15
    score -= min(0.3, 0.08 * warning_count)
    return round(max(0.0, min(1.0, score)), 3)
