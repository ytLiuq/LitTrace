from __future__ import annotations

from pydantic import BaseModel, Field


class EvalMetricReport(BaseModel):
    run_id: str
    topic: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    failures: list[dict[str, object]] = Field(default_factory=list)


def retrieval_metrics() -> dict[str, float]:
    return {
        "retrieval_recall_at_20": 0.0,
        "retrieval_precision_at_20": 0.0,
        "recent_paper_ratio_2023_2026": 0.0,
        "duplicate_rate": 0.0,
    }


def parsing_metrics() -> dict[str, float]:
    return {
        "metadata_accuracy": 0.0,
        "section_extraction_accuracy": 0.0,
        "table_cell_exact_match": 0.0,
        "reference_accuracy": 0.0,
    }


def storyline_metrics() -> dict[str, float]:
    return {
        "claim_grounding_rate": 0.0,
        "citation_coverage": 0.0,
        "unsupported_claim_rate": 0.0,
    }


def full_text_metrics_from_workspace(workspace) -> dict[str, float]:
    active_ids = workspace.context.active_papers
    active_count = len(active_ids)
    reports = [
        workspace.full_text_reports[paper_id]
        for paper_id in active_ids
        if paper_id in workspace.full_text_reports
    ]
    if active_count == 0:
        return {
            "full_text_resolved_rate": 0.0,
            "verified_candidate_rate": 0.0,
            "oa_pdf_candidate_rate": 0.0,
            "login_handoff_ready_rate": 0.0,
            "parsed_full_text_rate": 0.0,
        }
    verified = sum(report.verified_candidate_count > 0 for report in reports)
    oa_pdf = sum(bool(report.best_pdf_url) for report in reports)
    login_ready = sum(report.login_required_candidate_count > 0 for report in reports)
    parsed = sum(
        bool(workspace.parsed_papers.get(paper_id, {}).get("parsed"))
        for paper_id in active_ids
    )
    return {
        "full_text_resolved_rate": len(reports) / active_count,
        "verified_candidate_rate": verified / active_count,
        "oa_pdf_candidate_rate": oa_pdf / active_count,
        "login_handoff_ready_rate": login_ready / active_count,
        "parsed_full_text_rate": parsed / active_count,
    }
