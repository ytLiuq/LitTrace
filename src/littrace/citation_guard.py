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


def remove_unsupported_sentences(text: str, report: CitationGuardReport) -> str:
    if report.passed:
        return text
    repaired = text
    for sentence in report.unsupported_sentences:
        repaired = repaired.replace(sentence, "")
    repaired = re.sub(r"\s+\n", "\n", repaired)
    repaired = re.sub(r"\n{3,}", "\n\n", repaired)
    repaired = re.sub(r" {2,}", " ", repaired)
    repaired = repaired.strip()
    if not repaired:
        return "生成内容因缺少句子级引用证据已被移除。请先解析更多全文或放宽问题范围。"
    return repaired


def _split_sentences(text: str) -> list[str]:
    normalized = text.replace("\n", " ")
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(normalized):
        if char in "。！？!?":
            sentences.append(normalized[start : index + 1].strip())
            start = index + 1
        elif char == ".":
            previous_is_digit = index > 0 and normalized[index - 1].isdigit()
            next_is_digit = index + 1 < len(normalized) and normalized[index + 1].isdigit()
            if not (previous_is_digit and next_is_digit):
                sentences.append(normalized[start : index + 1].strip())
                start = index + 1
    if start < len(normalized):
        sentences.append(normalized[start:].strip())
    return [sentence for sentence in sentences if sentence]
