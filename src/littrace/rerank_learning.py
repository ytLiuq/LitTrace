from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.evaluation.retrieval_eval import RetrievalEvalReport, run_retrieval_golden_eval


class RerankWeightCandidate(BaseModel):
    name: str
    weights: dict[str, float] = Field(default_factory=dict)


class RerankLearningReport(BaseModel):
    candidate_count: int
    best_candidate: str | None = None
    best_score: float = 0.0
    results: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


DEFAULT_RERANK_CANDIDATES = [
    RerankWeightCandidate(
        name="balanced_title_phrase",
        weights={
            "active_recall": 0.45,
            "candidate_recall": 0.2,
            "mrr": 0.35,
        },
    ),
    RerankWeightCandidate(
        name="recall_first",
        weights={
            "active_recall": 0.6,
            "candidate_recall": 0.25,
            "mrr": 0.15,
        },
    ),
    RerankWeightCandidate(
        name="mrr_first",
        weights={
            "active_recall": 0.3,
            "candidate_recall": 0.15,
            "mrr": 0.55,
        },
    ),
]


async def learn_rerank_policy_from_golden(
    config: LitTraceConfig,
    live: bool = True,
    candidates: list[RerankWeightCandidate] | None = None,
) -> RerankLearningReport:
    """Score rerank policy candidates against the current golden retrieval runner.

    The current ranker is deterministic; this report gives a reproducible objective and
    establishes the hook for future weight injection/grid search.
    """

    candidates = candidates or DEFAULT_RERANK_CANDIDATES
    eval_report = await run_retrieval_golden_eval(config, live=live)
    results = [_score_candidate(candidate, eval_report) for candidate in candidates]
    best = max(results, key=lambda item: float(item["score"]), default=None)
    warnings = list(eval_report.warnings)
    if eval_report.metrics.get("active_recall", 0.0) < 0.8:
        warnings.append(
            "Active recall below 0.8; prefer recall_first weights or expand active_context_limit."
        )
    if eval_report.metrics.get("mrr", 0.0) < 0.5:
        warnings.append("MRR below 0.5; increase title/phrase/topical specificity signals.")
    return RerankLearningReport(
        candidate_count=len(candidates),
        best_candidate=str(best["name"]) if best else None,
        best_score=float(best["score"]) if best else 0.0,
        results=results,
        warnings=warnings,
    )


def _score_candidate(
    candidate: RerankWeightCandidate, report: RetrievalEvalReport
) -> dict[str, object]:
    metrics = report.metrics
    score = sum(
        candidate.weights.get(metric, 0.0) * float(metrics.get(metric, 0.0))
        for metric in ["active_recall", "candidate_recall", "mrr"]
    )
    return {
        "name": candidate.name,
        "weights": candidate.weights,
        "score": round(score, 3),
        "metrics": metrics,
    }
