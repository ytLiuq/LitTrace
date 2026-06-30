from __future__ import annotations

import re

from pydantic import BaseModel, Field

from littrace.citations import citation_records_for_papers
from littrace.models import LiteratureWorkspace


class CitationGuardReport(BaseModel):
    passed: bool
    checked_sentence_count: int
    unsupported_sentences: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


CLAIM_HINTS = [
    "表明",
    "说明",
    "提升",
    "降低",
    "解决",
    "局限",
    "回应",
    "性能",
    "sensitivity",
    "improve",
    "enhance",
    "limitation",
    "challenge",
]


def guard_citations(text: str, workspace: LiteratureWorkspace) -> CitationGuardReport:
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    records = citation_records_for_papers(papers)
    anchors = set()
    for paper in papers:
        anchors.add(paper.paper_id.lower())
        if paper.doi:
            anchors.add(paper.doi.lower())
        anchors.add(paper.title[:40].lower())
    for record in records:
        anchors.add(str(record.access_url).lower())

    checked = 0
    unsupported: list[str] = []
    for sentence in _split_sentences(text):
        lowered = sentence.lower()
        if not any(hint.lower() in lowered for hint in CLAIM_HINTS):
            continue
        checked += 1
        if not any(anchor and anchor in lowered for anchor in anchors):
            unsupported.append(sentence)

    warnings = []
    if unsupported:
        warnings.append("Some evidence-bearing sentences lack a paper id, DOI, title anchor, or access URL.")
    return CitationGuardReport(
        passed=not unsupported,
        checked_sentence_count=checked,
        unsupported_sentences=unsupported,
        warnings=warnings,
    )


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+", text.replace("\n", " "))
    return [part.strip() for part in parts if part.strip()]
