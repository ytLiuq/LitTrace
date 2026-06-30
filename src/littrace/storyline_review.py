from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.models import LiteratureWorkspace
from littrace.storyline import build_storyline_from_workspace


class StorylineReviewReport(BaseModel):
    passed: bool
    warnings: list[str] = Field(default_factory=list)
    claim_count: int = 0


def review_storyline(workspace: LiteratureWorkspace) -> StorylineReviewReport:
    claims = build_storyline_from_workspace(workspace)
    warnings: list[str] = []
    for claim in claims:
        text = claim.claim.lower()
        if any(token in text for token in ["证明", "必然", "彻底解决", "establishes that"]):
            warnings.append(f"Potential overclaim: {claim.claim}")
        if len({e.paper_id for e in claim.evidence}) < 1:
            warnings.append(f"Claim lacks paper-level evidence: {claim.claim}")
        if claim.claim_type == "solution_limit_response_chain" and len(claim.evidence) < 3:
            warnings.append(f"Chain claim lacks three evidence anchors: {claim.claim}")
    if not claims:
        warnings.append("No storyline claims available to review.")
    return StorylineReviewReport(passed=not warnings, warnings=warnings, claim_count=len(claims))
