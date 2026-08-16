from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from littrace.access_layer.paths import paper_storage_dir
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.ocr.registry import build_ocr_tool
from littrace.models import ParsedPaper
from littrace.ocr.tool import OCRMode, OCRTool


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

    def parse_one(paper_id: str) -> tuple[str, ParsedPaper, bool]:
        paper = workspace.papers[paper_id]
        pdf_path = local_pdf_path(config, paper)
        if not pdf_path.exists():
            return paper_id, ParsedPaper(
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
            ), True
        # Docling creates a converter per invocation, so workers must not share
        # a parser instance with mutable converter state.
        active_parser = build_ocr_tool(config, paper_lookup) if tool is None else parser
        return paper_id, active_parser.parse_pdf(pdf_path, mode=mode), False

    paper_ids = list(workspace.context.active_papers)
    use_parallel_docling = tool is None and parser.name == "docling" and config.parsing.docling_workers > 1
    if use_parallel_docling:
        with ThreadPoolExecutor(max_workers=config.parsing.docling_workers) as executor:
            parsed_results = list(executor.map(parse_one, paper_ids))
    else:
        parsed_results = [parse_one(paper_id) for paper_id in paper_ids]

    for paper_id, parsed, missing_pdf in parsed_results:
        if missing_pdf:
            missing_pdf_count += 1
            failed_count += 1
            workspace.parsed_papers[paper_id] = parsed
            continue
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
    """Canonical alias for ``target_pdf_path`` — kept for backward
    compatibility; new code should import from
    ``littrace.access_layer.paths``."""
    from littrace.access_layer.paths import target_pdf_path

    return target_pdf_path(config, paper)


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
    image_placeholders = markdown.count("<!-- image -->")
    enrichment_unavailable = any(
        "optional_docling_enrichment_unavailable" in str(report)
        for report in parsed.parser_reports
    )
    figure_assets = sum(
        1 for figure in parsed.figures
        if isinstance(figure, dict) and figure.get("asset_path")
    )
    visual_summary_count = sum(
        1 for figure in parsed.figures
        if isinstance(figure, dict) and figure.get("visual_summary_available")
    )
    if parsed.parsed and len(markdown) < 500:
        warnings.append("docling_markdown_short")
    if table_count and table_cell_count == 0:
        warnings.append("docling_tables_without_cells")
    if section_count and empty_sections / max(section_count, 1) > 0.5:
        warnings.append("docling_many_empty_sections")
    if parsed.error:
        warnings.append("docling_parse_error")
    if image_placeholders:
        warnings.append("docling_image_placeholders")
    if figure_count and figure_assets < figure_count:
        warnings.append("docling_missing_figure_assets")
    if figure_count and visual_summary_count < figure_count:
        warnings.append("docling_missing_visual_summaries")
    if enrichment_unavailable:
        warnings.append("docling_optional_enrichment_unavailable")
    return {
        "parser": "docling",
        "parsed": parsed.parsed,
        "markdown_chars": len(markdown),
        "section_count": section_count,
        "empty_section_count": empty_sections,
        "table_count": table_count,
        "table_cell_count": table_cell_count,
        "figure_count": figure_count,
        "figure_asset_count": figure_assets,
        "visual_summary_count": visual_summary_count,
        "warnings": warnings,
        "rag_eligible": (
            parsed.parsed
            and bool(section_count or table_count)
        ),
        "score": _docling_quality_score(
            parsed=parsed,
            markdown_chars=len(markdown),
            section_count=section_count,
            table_count=table_count,
            table_cell_count=table_cell_count,
            warning_count=len(warnings),
            image_placeholder_count=image_placeholders,
            missing_figure_asset_count=max(0, figure_count - figure_assets),
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
    image_placeholder_count: int = 0,
    missing_figure_asset_count: int = 0,
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
    score -= min(0.35, 0.12 * image_placeholder_count)
    score -= min(0.35, 0.12 * missing_figure_asset_count)
    return round(max(0.0, min(1.0, score)), 3)
