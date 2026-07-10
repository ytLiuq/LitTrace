from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.ocr.registry import build_ocr_tool
from littrace.ocr.tool import OCRMode
from littrace.models import LiteratureWorkspace, coerce_parsed
from littrace.parsing import local_pdf_path


class PDFBenchmarkReport(BaseModel):
    active_papers: int
    local_pdf_count: int
    parsed_count: int
    failed_count: int
    parsed_with_page_evidence: int
    average_evidence_confidence: float
    local_pdf_rate: float = 0.0
    parsed_rate: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class LivePDFBenchmarkReport(BaseModel):
    pdf_path: str
    parser: str
    parsed: bool
    elapsed_seconds: float
    section_count: int
    total_chars: int
    progress_events: list[dict[str, object]] = Field(default_factory=list)
    parser_reports: list[dict[str, object]] = Field(default_factory=list)
    error: str | None = None


def benchmark_pdf_parsing(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> PDFBenchmarkReport:
    active_ids = workspace.context.active_papers
    local_pdf_count = 0
    parsed_count = 0
    failed_count = 0
    parsed_with_page_evidence = 0
    confidences: list[float] = []
    warnings: list[str] = []

    for paper_id in active_ids:
        paper = workspace.papers[paper_id]
        if local_pdf_path(config, paper).exists():
            local_pdf_count += 1
        _raw = workspace.parsed_papers.get(paper_id)
        if not _raw:
            continue
        parsed = coerce_parsed(_raw)
        if parsed.parsed:
            parsed_count += 1
        else:
            failed_count += 1
        has_page = False
        for section in parsed.sections or []:
            if not isinstance(section, dict):
                continue
            evidence = section.get("evidence") or {}
            if isinstance(evidence, dict):
                if evidence.get("page") is not None:
                    has_page = True
                if evidence.get("confidence") is not None:
                    confidences.append(float(evidence["confidence"]))
        if has_page:
            parsed_with_page_evidence += 1

    if active_ids and local_pdf_count == 0:
        warnings.append("No local PDFs found for the active context.")
    if parsed_count == 0:
        warnings.append("No successfully parsed full-text PDFs yet.")

    average = sum(confidences) / len(confidences) if confidences else 0.0
    return PDFBenchmarkReport(
        active_papers=len(active_ids),
        local_pdf_count=local_pdf_count,
        parsed_count=parsed_count,
        failed_count=failed_count,
        parsed_with_page_evidence=parsed_with_page_evidence,
        average_evidence_confidence=round(average, 3),
        local_pdf_rate=round(local_pdf_count / len(active_ids), 3) if active_ids else 0.0,
        parsed_rate=round(parsed_count / len(active_ids), 3) if active_ids else 0.0,
        warnings=warnings,
    )


def benchmark_single_pdf(
    pdf_path: Path,
    config: LitTraceConfig,
    mode: OCRMode = OCRMode.ACCURATE,
    parse_strategy: str | None = None,
) -> LivePDFBenchmarkReport:
    if parse_strategy:
        config = config.model_copy(deep=True)
        config.parsing.parse_strategy = parse_strategy
    parser = build_ocr_tool(config, {})
    progress_events: list[dict[str, object]] = []
    if hasattr(parser, "progress_callback"):
        parser.progress_callback = progress_events.append
    start = time.monotonic()
    parsed = parser.parse_pdf(pdf_path, mode=mode)
    elapsed = time.monotonic() - start
    total_chars = sum(len(str(section.get("text") or "")) for section in parsed.sections)
    return LivePDFBenchmarkReport(
        pdf_path=str(pdf_path),
        parser=getattr(parser, "name", parser.__class__.__name__),
        parsed=parsed.parsed,
        elapsed_seconds=round(elapsed, 3),
        section_count=len(parsed.sections),
        total_chars=total_chars,
        progress_events=progress_events,
        parser_reports=parsed.parser_reports,
        error=parsed.error,
    )
