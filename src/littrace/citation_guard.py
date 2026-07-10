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


# Default claim-hint keywords — used when no config overrides are provided.
# Configure via ``config.citation_guard.claim_hints`` in config.yaml.
DEFAULT_CLAIM_HINTS: list[str] = [
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

# Backwards-compatible module-level alias (tests and legacy code may import this).
CLAIM_HINTS = DEFAULT_CLAIM_HINTS


def guard_citations(
    text: str,
    workspace: LiteratureWorkspace,
    *,
    claim_hints: list[str] | None = None,
) -> CitationGuardReport:
    """Check that claim-bearing sentences have citation anchors.

    Args:
        claim_hints: Optional list of keywords to override DEFAULT_CLAIM_HINTS.
            If provided, only these keywords are used. If None, defaults are used.
    """
    hints = claim_hints if claim_hints is not None else DEFAULT_CLAIM_HINTS
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    records = citation_records_for_papers(papers)
    anchors_by_type: dict[str, set[str]] = {
        "paper_id": set(),
        "doi": set(),
        "title": set(),
        "url": set(),
    }
    for paper in papers:
        anchors_by_type["paper_id"].add(paper.paper_id.lower())
        if paper.doi:
            anchors_by_type["doi"].add(paper.doi.lower())
        anchors_by_type["title"].add(paper.title[:40].lower())
    for record in records:
        anchors_by_type["url"].add(str(record.access_url).lower())

    checked = 0
    unsupported: list[str] = []
    for sentence in _split_sentences(text):
        lowered = sentence.lower()
        if not any(hint.lower() in lowered for hint in hints):
            continue
        checked += 1
        if not _has_any_anchor(lowered, anchors_by_type):
            unsupported.append(sentence)

    warnings = []
    if unsupported:
        missing = _missing_anchor_types(unsupported, anchors_by_type)
        warnings.append(
            "Some evidence-bearing sentences lack citation anchors. Missing/weak anchor types: "
            + ", ".join(missing)
            + "."
        )
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


def _has_any_anchor(sentence: str, anchors_by_type: dict[str, set[str]]) -> bool:
    return any(
        anchor and anchor in sentence for anchors in anchors_by_type.values() for anchor in anchors
    )


def _missing_anchor_types(
    sentences: list[str],
    anchors_by_type: dict[str, set[str]],
) -> list[str]:
    missing = []
    for anchor_type, anchors in anchors_by_type.items():
        if not any(
            anchor and anchor in sentence.lower() for sentence in sentences for anchor in anchors
        ):
            missing.append(anchor_type)
    return missing or ["unknown"]


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
