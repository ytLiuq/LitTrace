"""Core type tests: paper merge, rank, and storyline harness.

These are pure-logic tests that exercise ranking/dedup/harness decisions
without any I/O or monkeypatching.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from littrace.evaluation.harnesses import check_storyline_claims
from littrace.models import (
    AccessType,
    EvidenceSpan,
    PaperMetadata,
    PaperSearchRequest,
    StorylineClaim,
)
from littrace.retrieval.search import (
    merge_papers,
    rank_papers,
)


def test_merge_papers_deduplicates_by_doi_and_prefers_oa():
    papers = merge_papers(
        [
            PaperMetadata(
                paper_id="left",
                title="Same",
                doi="10.1000/example",
                access_type=AccessType.UNAVAILABLE,
            ),
            PaperMetadata(
                paper_id="right",
                title="Same",
                doi="10.1000/example",
                pdf_url="https://example.org/paper.pdf",
                access_type=AccessType.OPEN_ACCESS,
            ),
        ]
    )
    assert len(papers) == 1
    assert papers[0].access_type == AccessType.OPEN_ACCESS
    assert str(papers[0].pdf_url) == "https://example.org/paper.pdf"


def test_rank_papers_prefers_recent_papers_when_other_signals_match():
    papers = rank_papers(
        [
            PaperMetadata(paper_id="old", title="Old", year=2016),
            PaperMetadata(paper_id="new", title="New", year=2026),
        ],
        PaperSearchRequest(topic="sensor"),
    )
    assert papers[0].paper_id == "new"


def test_storyline_harness_rejects_ungrounded_claims():
    result = check_storyline_claims(
        [
            StorylineClaim(
                claim="The field shifted.",
                claim_type="trend_by_year_and_method",
                evidence=[],
            )
        ]
    )
    assert not result.passed
