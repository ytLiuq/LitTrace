from littrace.evidence.claims import register_evidence, verify_structured_claim
from littrace.models import Claim, ClaimKind, EvidenceSpan, LiteratureWorkspace


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


def test_claim_verifier_detects_conflicting_numeric_sources():
    workspace = LiteratureWorkspace()
    left = EvidenceSpan(
        paper_id="p1",
        page=2,
        snippet="Sensitivity reached 10 kPa-1.",
        observed_value=10,
        observed_unit="kPa-1",
    )
    right = EvidenceSpan(
        paper_id="p2",
        page=2,
        snippet="Sensitivity reached 25 kPa-1.",
        observed_value=25,
        observed_unit="kPa-1",
    )
    registry = register_evidence(workspace, [left, right])
    claim = Claim(
        text="Sensitivity was 10 kPa-1.",
        claim_kind=ClaimKind.NUMERIC,
        metric="sensitivity",
        requires_corroboration=True,
        evidence_ids=[left.evidence_id, right.evidence_id],
        support_quotes={left.evidence_id: left.snippet, right.evidence_id: right.snippet},
    )

    report = verify_structured_claim(claim, registry)

    assert report.status == "conflicted"


def test_claim_verifier_blocks_freshness_claim_without_retrieval_cutoff():
    workspace = LiteratureWorkspace()
    span = EvidenceSpan(paper_id="p1", page=2, snippet="Published in 2026.")
    registry = register_evidence(workspace, [span])
    claim = Claim(
        text="This is the latest result.",
        claim_kind=ClaimKind.FRESHNESS,
        evidence_ids=[span.evidence_id],
        support_quotes={span.evidence_id: span.snippet},
    )

    report = verify_structured_claim(claim, registry)

    assert not report.publishable
    assert "cutoff time" in report.missing_requirements[0]


def test_claim_verifier_rejects_a_paraphrase_that_the_quote_does_not_assert():
    workspace = LiteratureWorkspace()
    span = EvidenceSpan(
        paper_id="p1",
        page=2,
        snippet="Sensitivity reached 12.5 kPa-1 under cyclic loading.",
    )
    registry = register_evidence(workspace, [span])
    claim = Claim(
        text="The material is highly durable.",
        evidence_ids=[span.evidence_id],
        support_quotes={span.evidence_id: span.snippet},
    )

    report = verify_structured_claim(claim, registry)

    assert not report.publishable
    assert not report.semantic_supported
    assert "exact asserted quote" in report.missing_requirements[0]


def test_numeric_claim_requires_an_exact_observed_value_not_substring_match():
    workspace = LiteratureWorkspace()
    span = EvidenceSpan(
        paper_id="p1",
        page=2,
        snippet="Sensitivity reached 10 kPa-1.",
        observed_value=10,
        observed_unit="kPa-1",
    )
    registry = register_evidence(workspace, [span])
    claim = Claim(
        text="Sensitivity was 1 kPa-1.",
        claim_kind=ClaimKind.NUMERIC,
        metric="sensitivity",
        expected_value=1,
        expected_unit="kPa-1",
        evidence_ids=[span.evidence_id],
        support_quotes={span.evidence_id: span.snippet},
    )

    report = verify_structured_claim(claim, registry)

    assert not report.publishable
    assert "does not match" in report.missing_requirements[0]
