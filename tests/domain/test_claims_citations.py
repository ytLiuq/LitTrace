"""Claim verification and citation guard — pure logic, no I/O."""

from __future__ import annotations

import pytest

from littrace.citation_guard import guard_citations
from littrace.context import add_papers
from littrace.evidence.claims import register_evidence, verify_structured_claim
from littrace.models import (
    Claim,
    ClaimKind,
    EvidenceSpan,
    LiteratureWorkspace,
    PaperMetadata,
)


pytestmark = pytest.mark.domain


def test_claim_verifier_requires_an_exact_quote_for_each_evidence_id():
    workspace = LiteratureWorkspace()
    span = EvidenceSpan(paper_id="p1", page=2, snippet="Sensitivity reached 12.5 kPa-1.")
    registry = register_evidence(workspace, [span])
    claim = Claim(
        text="Sensitivity was 12.5 kPa-1.",
        claim_kind=ClaimKind.NUMERIC,
        metric="sensitivity",
        evidence_ids=[span.evidence_id],
        support_quotes={span.evidence_id: "invented quote"},
    )

    report = verify_structured_claim(claim, registry)

    assert not report.publishable
    assert "not present" in report.missing_requirements[0]


def test_citation_guard_flags_claim_without_anchor():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Traceable Sensor", doi="10.1000/example")],
    )

    report = guard_citations("该方法显著提升了性能。", workspace)

    assert not report.passed
    assert report.unsupported_sentences
    assert "Missing/weak anchor types" in report.warnings[0]
