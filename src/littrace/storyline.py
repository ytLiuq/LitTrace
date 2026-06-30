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
    by_paper: dict[str, dict[str, list[EvidenceSpan]]] = defaultdict(
        lambda: {"method": [], "limitation": [], "response": []}
    )
    for paper_id, parsed in workspace.parsed_papers.items():
        for section in parsed.get("sections", []):
            text = str(section.get("text") or "")
            name = str(section.get("name") or "")
            lowered = f"{name} {text}".lower()
            evidence = section.get("evidence") or {}
            if "limit" in lowered or "challenge" in lowered:
                span = EvidenceSpan(
                    paper_id=paper_id,
                    section=name or "parsed_text",
                    snippet=text[:240] or name,
                    page=evidence.get("page"),
                    parser=evidence.get("parser"),
                    confidence=0.7,
                )
                limitation_evidence.append(span)
                by_paper[paper_id]["limitation"].append(span)
            if "method" in lowered or "fabrication" in lowered or "synthesis" in lowered:
                span = EvidenceSpan(
                    paper_id=paper_id,
                    section=name or "parsed_text",
                    snippet=text[:240] or name,
                    page=evidence.get("page"),
                    parser=evidence.get("parser"),
                    confidence=0.7,
                )
                method_evidence.append(span)
                by_paper[paper_id]["method"].append(span)
            if any(token in lowered for token in ["improve", "enhance", "address", "overcome"]):
                span = EvidenceSpan(
                    paper_id=paper_id,
                    section=name or "parsed_text",
                    snippet=text[:240] or name,
                    page=evidence.get("page"),
                    parser=evidence.get("parser"),
                    confidence=0.7,
                )
                response_evidence.append(span)
                by_paper[paper_id]["response"].append(span)

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
    chain_claim = _build_solution_limit_response_chain(workspace, by_paper)
    if chain_claim:
        claims.append(chain_claim)
    return claims


def _build_solution_limit_response_chain(
    workspace: LiteratureWorkspace,
    by_paper: dict[str, dict[str, list[EvidenceSpan]]],
) -> StorylineClaim | None:
    dated_papers = [
        workspace.papers[paper_id]
        for paper_id in workspace.context.active_papers
        if paper_id in by_paper and workspace.papers[paper_id].year is not None
    ]
    dated_papers.sort(key=lambda paper: (paper.year or 0, paper.title))

    prior = next((paper for paper in dated_papers if by_paper[paper.paper_id]["method"]), None)
    limitation = next(
        (paper for paper in dated_papers if by_paper[paper.paper_id]["limitation"]),
        None,
    )
    response = next(
        (
            paper
            for paper in dated_papers
            if by_paper[paper.paper_id]["response"]
            and (limitation is None or (paper.year or 0) >= (limitation.year or 0))
        ),
        None,
    )

    if not prior or not limitation or not response:
        return None

    evidence = [
        by_paper[prior.paper_id]["method"][0],
        by_paper[limitation.paper_id]["limitation"][0],
        by_paper[response.paper_id]["response"][0],
    ]
    claim = (
        f"可形成一条保守的发展链：{prior.year} 年左右的工作“{prior.title}”提供了方法或制备路线；"
        f"{limitation.year} 年的“{limitation.title}”暴露或讨论了局限/挑战；"
        f"{response.year} 年的“{response.title}”出现了回应性改进表述。"
        "这只是基于已解析片段的证据链，不应扩展为未验证的领域共识。"
    )
    return StorylineClaim(
        claim=claim,
        claim_type="solution_limit_response_chain",
        evidence=evidence,
        confidence=0.76,
    )
