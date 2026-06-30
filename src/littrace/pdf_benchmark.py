from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.parsing import local_pdf_path


class PDFBenchmarkReport(BaseModel):
    active_papers: int
    local_pdf_count: int
    parsed_count: int
    metadata_only_count: int
    parsed_with_page_evidence: int
    average_evidence_confidence: float
    local_pdf_rate: float = 0.0
    parsed_rate: float = 0.0
    warnings: list[str] = Field(default_factory=list)


def benchmark_pdf_parsing(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> PDFBenchmarkReport:
    active_ids = workspace.context.active_papers
    local_pdf_count = 0
    parsed_count = 0
    metadata_only_count = 0
    parsed_with_page_evidence = 0
    confidences: list[float] = []
    warnings: list[str] = []

    for paper_id in active_ids:
        paper = workspace.papers[paper_id]
        if local_pdf_path(config, paper).exists():
            local_pdf_count += 1
        parsed = workspace.parsed_papers.get(paper_id)
        if not parsed:
            continue
        if parsed.get("parsed"):
            parsed_count += 1
        else:
            metadata_only_count += 1
        has_page = False
        for section in parsed.get("sections") or []:
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
        metadata_only_count=metadata_only_count,
        parsed_with_page_evidence=parsed_with_page_evidence,
        average_evidence_confidence=round(average, 3),
        local_pdf_rate=round(local_pdf_count / len(active_ids), 3) if active_ids else 0.0,
        parsed_rate=round(parsed_count / len(active_ids), 3) if active_ids else 0.0,
        warnings=warnings,
    )
