from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.attachments import check_download_presence
from littrace.citation_guard import guard_citations
from littrace.citations import citation_records_for_papers
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.pdf_benchmark import benchmark_pdf_parsing
from littrace.storyline import render_structured_storyline_report
from littrace.storyline_review import review_storyline
from littrace.tables import build_comparison_matrices


class QualityReport(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def build_quality_report(config: LitTraceConfig, workspace: LiteratureWorkspace) -> QualityReport:
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    download_presence = check_download_presence(config, workspace)
    pdf_report = benchmark_pdf_parsing(workspace, config)
    matrix = build_comparison_matrices(workspace)
    storyline_review = review_storyline(workspace)
    storyline_text = render_structured_storyline_report(workspace)
    citation_guard = guard_citations(storyline_text, workspace)
    citations = citation_records_for_papers(papers)
    full_text_reports = [
        workspace.full_text_reports[paper.paper_id]
        for paper in papers
        if paper.paper_id in workspace.full_text_reports
    ]

    active_count = len(papers)
    full_text_resolved = len(full_text_reports)
    oa_pdf_count = sum(bool(report.best_pdf_url) for report in full_text_reports)
    login_required_count = sum(
        report.login_required_candidate_count > 0 for report in full_text_reports
    )
    metrics = {
        "active_paper_count": float(active_count),
        "selected_download_count": float(len(workspace.context.selected_for_download)),
        "full_text_resolved_rate": full_text_resolved / active_count if active_count else 0.0,
        "oa_pdf_candidate_rate": oa_pdf_count / active_count if active_count else 0.0,
        "login_required_candidate_rate": login_required_count / active_count if active_count else 0.0,
        "local_pdf_rate": pdf_report.local_pdf_rate,
        "parsed_rate": pdf_report.parsed_rate,
        "performance_cell_count": float(len(workspace.performance_cells)),
        "comparison_matrix_count": float(len(matrix.matrices)),
        "storyline_claim_count": float(storyline_review.claim_count),
        "citation_count": float(len(citations)),
        "citation_guard_pass": 1.0 if citation_guard.passed else 0.0,
        "supplementary_link_count": float(sum(len(items) for items in workspace.supplementary_links.values())),
    }
    warnings = [
        *download_presence.warnings,
        *pdf_report.warnings,
        *matrix.warnings,
        *storyline_review.warnings,
        *citation_guard.warnings,
    ]
    return QualityReport(metrics=metrics, warnings=warnings)
