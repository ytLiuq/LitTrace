from __future__ import annotations

import json
from math import log2
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.retrieval.pgvector_store import RagSearchHit
from littrace.retrieval.rag_search import search_workspace_rag


class RagGoldEvidence(BaseModel):
    evidence_id: str | None = None
    paper_id: str | None = None
    doi: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    table_id: str | None = None
    section: str | None = None
    required_terms: list[str] = Field(default_factory=list)
    relevance: int = Field(default=1, ge=1, le=3)


class RagGoldenCase(BaseModel):
    case_id: str
    question: str
    gold_evidence: list[RagGoldEvidence] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class RagEvalCaseResult(BaseModel):
    case_id: str
    question: str
    top_k: int
    gold_evidence_count: int
    retrieved_hit_count: int
    matched_evidence_count: int
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    mrr: float
    duplicate_rate: float
    matched_evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RagEvalReport(BaseModel):
    case_count: int
    top_k: int
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: list[RagEvalCaseResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


async def run_rag_golden_eval(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    *,
    top_k: int = 10,
) -> RagEvalReport:
    top_k = max(1, int(top_k))
    cases = load_rag_golden_cases(config.eval.rag_golden_set_dir)
    results: list[RagEvalCaseResult] = []
    warnings: list[str] = []
    if not cases:
        warnings.append(f"No RAG golden cases found under {config.eval.rag_golden_set_dir}.")

    for case in cases:
        search_result = await search_workspace_rag(
            config,
            workspace,
            case.question,
            top_k=top_k,
        )
        if search_result is None:
            results.append(
                score_rag_case(
                    case,
                    [],
                    workspace,
                    top_k=top_k,
                    warnings=["The current workspace has no searchable RAG profile."],
                )
            )
            continue
        results.append(score_rag_case(case, search_result.hits, workspace, top_k=top_k))

    return RagEvalReport(
        case_count=len(results),
        top_k=top_k,
        metrics={
            "rag_recall_at_k": _avg([case.recall_at_k for case in results]),
            "rag_precision_at_k": _avg([case.precision_at_k for case in results]),
            "rag_ndcg_at_k": _avg([case.ndcg_at_k for case in results]),
            "rag_mrr": _avg([case.mrr for case in results]),
            "rag_duplicate_rate": _avg([case.duplicate_rate for case in results]),
            "rag_zero_hit_case_rate": _safe_div(
                sum(case.matched_evidence_count == 0 for case in results),
                len(results),
            ),
        },
        cases=results,
        warnings=warnings,
    )


def score_rag_case(
    case: RagGoldenCase,
    hits: list[RagSearchHit],
    workspace: LiteratureWorkspace,
    *,
    top_k: int = 10,
    warnings: list[str] | None = None,
) -> RagEvalCaseResult:
    ranked_hits = list(hits[: max(1, top_k)])
    unmatched = set(range(len(case.gold_evidence)))
    matched: list[int] = []
    gains: list[int] = []
    first_relevant_rank: int | None = None

    for rank, hit in enumerate(ranked_hits, start=1):
        candidates = [
            index
            for index in unmatched
            if _hit_matches_gold(hit, case.gold_evidence[index], workspace)
        ]
        if not candidates:
            gains.append(0)
            continue
        best = max(candidates, key=lambda index: case.gold_evidence[index].relevance)
        unmatched.remove(best)
        matched.append(best)
        gains.append(case.gold_evidence[best].relevance)
        if first_relevant_rank is None:
            first_relevant_rank = rank

    gold_count = len(case.gold_evidence)
    unique_chunks = {hit.chunk_id for hit in ranked_hits}
    matched_ids = [
        case.gold_evidence[index].evidence_id or f"gold:{case.case_id}:{index}"
        for index in matched
    ]
    return RagEvalCaseResult(
        case_id=case.case_id,
        question=case.question,
        top_k=max(1, top_k),
        gold_evidence_count=gold_count,
        retrieved_hit_count=len(ranked_hits),
        matched_evidence_count=len(matched),
        recall_at_k=_safe_div(len(matched), gold_count),
        precision_at_k=_safe_div(len(matched), len(ranked_hits)),
        ndcg_at_k=_ndcg(gains, case.gold_evidence, max(1, top_k)),
        mrr=round(1 / first_relevant_rank, 3) if first_relevant_rank else 0.0,
        duplicate_rate=_safe_div(len(ranked_hits) - len(unique_chunks), len(ranked_hits)),
        matched_evidence_ids=matched_ids,
        warnings=list(warnings or []),
    )


def load_rag_golden_cases(root: Path) -> list[RagGoldenCase]:
    cases: list[RagGoldenCase] = []
    if not root.exists():
        return cases
    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(RagGoldenCase.model_validate(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"Invalid RAG golden case at {path}:{line_number}: {exc}") from exc
    return cases


def _hit_matches_gold(
    hit: RagSearchHit,
    gold: RagGoldEvidence,
    workspace: LiteratureWorkspace,
) -> bool:
    paper = workspace.papers.get(hit.paper_id)
    if gold.chunk_id and gold.chunk_id != hit.chunk_id:
        return False
    if gold.paper_id and gold.paper_id != hit.paper_id:
        return False
    if gold.doi:
        hit_doi = _normalize_doi(paper.doi if paper is not None else None)
        if hit_doi != _normalize_doi(gold.doi):
            return False
    if gold.page is not None and gold.page != hit.page:
        return False
    if gold.table_id and _normalize(gold.table_id) != _normalize(hit.table_id):
        return False
    if gold.section and _normalize(gold.section) not in _normalize(hit.section):
        return False
    text = _normalize(hit.text)
    return all(_normalize(term) in text for term in gold.required_terms)


def _ndcg(gains: list[int], gold: list[RagGoldEvidence], top_k: int) -> float:
    dcg = sum(((2**gain) - 1) / log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_gains = sorted((item.relevance for item in gold), reverse=True)[:top_k]
    ideal = sum(
        ((2**gain) - 1) / log2(rank + 1)
        for rank, gain in enumerate(ideal_gains, start=1)
    )
    return round(dcg / ideal, 3) if ideal else 0.0


def _normalize(value: object) -> str:
    return " ".join(str(value or "").lower().replace("-", " ").replace("/", " ").split())


def _normalize_doi(value: str | None) -> str:
    return (value or "").strip().lower().removeprefix("https://doi.org/")


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _safe_div(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0
