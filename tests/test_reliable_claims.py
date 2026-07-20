from littrace.models import ClaimStatus, EvidenceSpan, verify_claim


def test_verify_claim_requires_traceable_evidence():
    report = verify_claim("A material is stable.", [EvidenceSpan(paper_id="p1")])

    assert report.status == ClaimStatus.CANDIDATE
    assert not report.publishable


def test_verify_claim_corroborates_independent_sources():
    report = verify_claim(
        "A material is stable.",
        [
            EvidenceSpan(paper_id="p1", page=4, snippet="Stable for 100 cycles."),
            EvidenceSpan(paper_id="p2", page=6, snippet="No degradation observed."),
        ],
    )

    assert report.status == ClaimStatus.CORROBORATED
    assert report.publishable


def test_verify_claim_exposes_missing_corroboration_requirement():
    report = verify_claim(
        "A material is stable.",
        [EvidenceSpan(paper_id="p1", section="Results", snippet="Stable for 100 cycles.")],
        requires_corroboration=True,
    )

    assert report.status == ClaimStatus.SUPPORTED
    assert not report.publishable
    assert report.draftable
    assert report.missing_requirements


def test_verify_claim_marks_conflicting_independent_measurements():
    report = verify_claim(
        "The material has a sensitivity of 10 kPa-1.",
        [
            EvidenceSpan(
                paper_id="p1",
                page=4,
                snippet="Sensitivity was 10 kPa-1.",
                observed_value=10,
                observed_unit="kPa-1",
            ),
            EvidenceSpan(
                paper_id="p2",
                page=6,
                snippet="Sensitivity was 25 kPa-1.",
                observed_value=25,
                observed_unit="kPa-1",
            ),
        ],
        requires_corroboration=True,
        metric="sensitivity",
    )

    assert report.status == ClaimStatus.CONFLICTED
    assert not report.publishable
    assert report.warnings
