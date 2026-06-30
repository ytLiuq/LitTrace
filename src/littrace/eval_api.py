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
