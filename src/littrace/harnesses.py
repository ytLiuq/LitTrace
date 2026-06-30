from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.models import CitationRecord, LinkStatus, PerformanceCell, StorylineClaim


class HarnessResult(BaseModel):
    passed: bool
    score: float
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def check_citations(records: list[CitationRecord]) -> HarnessResult:
    errors: list[str] = []
    for record in records:
        if record.link_status in {LinkStatus.FAILED, LinkStatus.UNCHECKED}:
            errors.append(f"{record.paper_id}: access link is not verified")
        if not record.citation_text.strip():
            errors.append(f"{record.paper_id}: missing citation text")
    total = max(len(records), 1)
    score = (total - len(errors)) / total
    return HarnessResult(passed=not errors, score=score, errors=errors)


def check_performance_cells(cells: list[PerformanceCell]) -> HarnessResult:
    errors: list[str] = []
    warnings: list[str] = []
    for cell in cells:
        evidence = cell.evidence
        if evidence.page is None and evidence.snippet is None:
            errors.append(f"{cell.paper_id}: metric {cell.metric} lacks traceable evidence")
        if cell.higher_is_better is None:
            warnings.append(f"{cell.paper_id}: metric direction missing for {cell.metric}")
        if evidence.confidence < 0.65:
            warnings.append(f"{cell.paper_id}: low extraction confidence for {cell.metric}")
    total = max(len(cells), 1)
    score = (total - len(errors)) / total
    return HarnessResult(passed=not errors, score=score, errors=errors, warnings=warnings)


def check_storyline_claims(claims: list[StorylineClaim]) -> HarnessResult:
    errors: list[str] = []
    warnings: list[str] = []
    allowed_types = {
        "prior_solution",
        "remaining_limitation",
        "later_response",
        "solution_limit_response_chain",
        "unresolved_gap",
        "trend_by_year_and_method",
    }

    for claim in claims:
        if claim.claim_type not in allowed_types:
            errors.append(f"Unsupported storyline claim type: {claim.claim_type}")
        unique_papers = {item.paper_id for item in claim.evidence}
        if len(unique_papers) < 1:
            errors.append(f"Ungrounded storyline claim: {claim.claim}")
        if len(unique_papers) < 2 and claim.claim_type in {
            "trend_by_year_and_method",
            "later_response",
        }:
            warnings.append(f"Claim should have at least two supporting papers: {claim.claim}")
        if claim.claim_type == "solution_limit_response_chain" and len(claim.evidence) < 3:
            errors.append(f"Storyline chain lacks solution-limit-response evidence: {claim.claim}")
        if claim.confidence < 0.7:
            warnings.append(f"Low-confidence storyline claim: {claim.claim}")

    total = max(len(claims), 1)
    score = (total - len(errors)) / total
    return HarnessResult(passed=not errors, score=score, errors=errors, warnings=warnings)
