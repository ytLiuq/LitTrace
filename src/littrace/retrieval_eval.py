from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.golden_eval import _load_cases
from littrace.models import PaperSearchRequest
from littrace.workflow import run_search_preview


class RetrievalEvalCaseResult(BaseModel):
    case_id: str
    topic: str
    expected_dois: list[str] = Field(default_factory=list)
    active_doi_hits: list[str] = Field(default_factory=list)
    candidate_doi_hits: list[str] = Field(default_factory=list)
    active_count: int = 0
    candidate_count: int = 0
    active_recall: float = 0.0
    candidate_recall: float = 0.0
    mrr: float = 0.0
    diagnostics: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RetrievalEvalReport(BaseModel):
    case_count: int
    live: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: list[RetrievalEvalCaseResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class _RankedDOIs:
    active: list[str]
    candidates: list[str]


async def run_retrieval_golden_eval(
    config: LitTraceConfig,
    live: bool = True,
    limit: int = 40,
) -> RetrievalEvalReport:
    cases = _load_cases(config.eval.golden_set_dir)
    results: list[RetrievalEvalCaseResult] = []
    warnings: list[str] = []
    if not cases:
        warnings.append(f"No golden cases found under {config.eval.golden_set_dir}.")

    for case in cases:
        topic = str(case.get("topic") or "").strip()
        if not topic:
            continue
        expected = _normalize_dois(case.get("expected_dois"))
        year_min = case.get("preferred_year_min")
        request = PaperSearchRequest(
            topic=topic,
            year_min=int(year_min) if isinstance(year_min, int | float) else None,
            live=live,
            limit=limit,
            min_relevant_results=5,
        )
        workspace = await run_search_preview(request, config)
        ranked = _ranked_dois(workspace)
        active_hits = _ordered_hits(expected, ranked.active)
        candidate_hits = _ordered_hits(expected, ranked.candidates)
        diagnostics = getattr(workspace.context.filters, "search_diagnostics", None) or {}
        case_warnings = []
        if isinstance(diagnostics, dict):
            case_warnings.extend(str(item) for item in diagnostics.get("errors", [])[:5])
        results.append(
            RetrievalEvalCaseResult(
                case_id=str(case.get("case_id") or topic),
                topic=topic,
                expected_dois=sorted(expected),
                active_doi_hits=active_hits,
                candidate_doi_hits=candidate_hits,
                active_count=len(workspace.context.active_papers),
                candidate_count=int(
                    getattr(workspace.context.filters, "candidate_pool_count", None)
                    or len(workspace.papers)
                ),
                active_recall=_safe_div(len(active_hits), len(expected)),
                candidate_recall=_safe_div(len(candidate_hits), len(expected)),
                mrr=_mrr(expected, ranked.candidates),
                diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
                warnings=case_warnings,
            )
        )

    metrics = {
        "active_recall": _avg([case.active_recall for case in results]),
        "candidate_recall": _avg([case.candidate_recall for case in results]),
        "mrr": _avg([case.mrr for case in results]),
        "avg_active_count": _avg([float(case.active_count) for case in results]),
        "avg_candidate_count": _avg([float(case.candidate_count) for case in results]),
        "zero_hit_case_rate": _safe_div(
            sum(1 for case in results if not case.candidate_doi_hits),
            len(results),
        ),
    }
    return RetrievalEvalReport(
        case_count=len(results),
        live=live,
        metrics=metrics,
        cases=results,
        warnings=warnings,
    )


def _ranked_dois(workspace) -> _RankedDOIs:
    active = [
        _normalize_doi(workspace.papers[paper_id].doi)
        for paper_id in workspace.context.active_papers
        if workspace.papers[paper_id].doi
    ]
    candidate_ids = getattr(workspace.context.filters, "candidate_pool_ids", None)
    if not isinstance(candidate_ids, list):
        candidate_ids = list(workspace.papers)
    candidates = [
        _normalize_doi(workspace.papers[paper_id].doi)
        for paper_id in candidate_ids
        if isinstance(paper_id, str)
        and paper_id in workspace.papers
        and workspace.papers[paper_id].doi
    ]
    return _RankedDOIs(
        active=[doi for doi in active if doi], candidates=[doi for doi in candidates if doi]
    )


def _normalize_dois(raw: object) -> set[str]:
    if isinstance(raw, str):
        return {_normalize_doi(raw)}
    if isinstance(raw, list):
        return {_normalize_doi(str(item)) for item in raw if item}
    return set()


def _normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")


def _ordered_hits(expected: set[str], ranked: list[str]) -> list[str]:
    return [doi for doi in ranked if doi in expected]


def _mrr(expected: set[str], ranked: list[str]) -> float:
    for index, doi in enumerate(ranked, start=1):
        if doi in expected:
            return round(1 / index, 3)
    return 0.0


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 3)
