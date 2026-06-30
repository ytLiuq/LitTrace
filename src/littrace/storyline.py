from __future__ import annotations

from collections import defaultdict

from littrace.harnesses import HarnessResult, check_storyline_claims
from littrace.models import EvidenceSpan, LiteratureWorkspace, PaperMetadata, StorylineClaim


def build_storyline_preview(papers: list[PaperMetadata]) -> list[StorylineClaim]:
    """Build a conservative evidence-first storyline preview from metadata only.

    Full narrative claims should wait for parsed full text. Metadata can support
    only cautious year/source trends, so this function intentionally emits
    low-volume claims.
    """

    by_year: dict[int, list[PaperMetadata]] = defaultdict(list)
    for paper in papers:
        if paper.year is not None:
            by_year[paper.year].append(paper)

    claims: list[StorylineClaim] = []
    recent_years = sorted(year for year in by_year if year >= 2023)
    if len(recent_years) >= 2:
        evidence = [
            EvidenceSpan(
                paper_id=paper.paper_id,
                section="metadata",
                snippet=f"{paper.year}; {paper.journal or paper.publisher or 'unknown source'}",
                confidence=0.75,
            )
            for year in recent_years[-2:]
            for paper in by_year[year][:2]
        ]
        claims.append(
            StorylineClaim(
                claim=(
                    "Recent retrieved literature is concentrated in 2023 or later; "
                    "full-text parsing is required before making solution-limit-response claims."
                ),
                claim_type="trend_by_year_and_method",
                evidence=evidence,
                confidence=0.72,
            )
        )

    return claims


def verify_storyline_preview(claims: list[StorylineClaim]) -> HarnessResult:
    return check_storyline_claims(claims)


def build_storyline_from_workspace(workspace: LiteratureWorkspace) -> list[StorylineClaim]:
    parsed_claims = _claims_from_parsed_papers(workspace)
    if parsed_claims:
        return parsed_claims
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    return build_storyline_preview(papers)


def _claims_from_parsed_papers(workspace: LiteratureWorkspace) -> list[StorylineClaim]:
    limitation_evidence: list[EvidenceSpan] = []
    method_evidence: list[EvidenceSpan] = []
    response_evidence: list[EvidenceSpan] = []
    for paper_id, parsed in workspace.parsed_papers.items():
        for section in parsed.get("sections", []):
            text = str(section.get("text") or "")
            name = str(section.get("name") or "")
            lowered = f"{name} {text}".lower()
            evidence = section.get("evidence") or {}
            if "limit" in lowered or "challenge" in lowered:
                limitation_evidence.append(
                    EvidenceSpan(
                        paper_id=paper_id,
                        section=name or "parsed_text",
                        snippet=text[:240] or name,
                        page=evidence.get("page"),
                        parser=evidence.get("parser"),
                        confidence=0.7,
                    )
                )
            if "method" in lowered or "fabrication" in lowered or "synthesis" in lowered:
                method_evidence.append(
                    EvidenceSpan(
                        paper_id=paper_id,
                        section=name or "parsed_text",
                        snippet=text[:240] or name,
                        page=evidence.get("page"),
                        parser=evidence.get("parser"),
                        confidence=0.7,
                    )
                )
            if any(token in lowered for token in ["improve", "enhance", "address", "overcome"]):
                response_evidence.append(
                    EvidenceSpan(
                        paper_id=paper_id,
                        section=name or "parsed_text",
                        snippet=text[:240] or name,
                        page=evidence.get("page"),
                        parser=evidence.get("parser"),
                        confidence=0.7,
                    )
                )

    claims: list[StorylineClaim] = []
    if method_evidence:
        claims.append(
            StorylineClaim(
                claim="前序工作给出了可追溯的方法或制备路线；这些内容可作为“前人解决了什么”的证据起点。",
                claim_type="prior_solution",
                evidence=method_evidence[:4],
                confidence=0.72,
            )
        )
    if limitation_evidence:
        claims.append(
            StorylineClaim(
                claim="部分论文明确呈现 limitation/challenge，可作为“留下什么局限”的证据锚点。",
                claim_type="remaining_limitation",
                evidence=limitation_evidence[:4],
                confidence=0.72,
            )
        )
    if response_evidence and limitation_evidence:
        claims.append(
            StorylineClaim(
                claim="后续工作中出现 improve/enhance/address/overcome 等回应性表述，可用于构建“下一代论文如何回应局限”的链条，但仍需人工复核具体因果关系。",
                claim_type="later_response",
                evidence=response_evidence[:4],
                confidence=0.72,
            )
        )
    return claims
