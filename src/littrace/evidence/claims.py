"""Claim-level evidence registration and deterministic release verification."""

from __future__ import annotations

import re
from math import isclose

from littrace.models import (
    Claim,
    ClaimKind,
    ClaimStatus,
    EvidenceSpan,
    LiteratureWorkspace,
    SourceRecord,
    VerificationReport,
    coerce_parsed,
    verify_claim,
)


def register_evidence(
    workspace: LiteratureWorkspace, evidence: list[EvidenceSpan]
) -> dict[str, EvidenceSpan]:
    """Register immutable evidence records and their source provenance."""

    for span in evidence:
        if not span.evidence_id:
            continue
        workspace.evidence_records[span.evidence_id] = span
        if span.source_record_id and span.source_record_id not in workspace.source_records:
            workspace.source_records[span.source_record_id] = SourceRecord(
                source_record_id=span.source_record_id,
                paper_id=span.paper_id,
                source_name=span.parser or span.source_kind.value,
                content_hash=span.content_hash,
                retrieved_at=span.captured_at or "",
            )
    return dict(workspace.evidence_records)


def workspace_evidence_registry(workspace: LiteratureWorkspace) -> dict[str, EvidenceSpan]:
    """Collect every evidence-bearing workspace field into the durable registry."""

    spans = list(workspace.evidence_records.values())
    for paper_id, parsed in workspace.parsed_papers.items():
        parsed_paper = coerce_parsed(parsed)
        for section in parsed_paper.sections:
            if not isinstance(section, dict):
                continue
            raw = section.get("evidence")
            if isinstance(raw, EvidenceSpan):
                spans.append(raw)
            elif isinstance(raw, dict):
                spans.append(EvidenceSpan.model_validate({"paper_id": paper_id, **raw}))
            else:
                spans.append(
                    EvidenceSpan(
                        paper_id=paper_id,
                        section=str(section.get("name") or "section"),
                        snippet=str(section.get("text") or "")[:700],
                    )
                )
    spans.extend(cell.evidence for cell in workspace.performance_cells)
    return register_evidence(workspace, spans)


def verify_structured_claim(
    claim: Claim,
    evidence_registry: dict[str, EvidenceSpan],
) -> VerificationReport:
    """Verify source identity, quote grounding, numeric agreement, and policy."""

    unknown = [
        evidence_id for evidence_id in claim.evidence_ids if evidence_id not in evidence_registry
    ]
    if unknown:
        return _blocked_claim(claim, [], f"Unknown evidence IDs: {', '.join(sorted(unknown))}")
    evidence = [evidence_registry[evidence_id] for evidence_id in claim.evidence_ids]
    if not all(span.provenance_complete for span in evidence):
        return _blocked_claim(claim, evidence, "Evidence provenance is incomplete.")
    quote_error = _quote_error(claim, evidence_registry)
    if quote_error:
        return _blocked_claim(claim, evidence, quote_error)
    numeric_error = _numeric_error(claim, evidence)
    if numeric_error:
        return _blocked_claim(claim, evidence, numeric_error)
    if (
        claim.requires_freshness or claim.claim_kind == ClaimKind.FRESHNESS
    ) and not claim.retrieval_cutoff_at:
        return _blocked_claim(
            claim,
            evidence,
            "Freshness-sensitive claim needs a retrieval cutoff time.",
            status=ClaimStatus.SUPPORTED,
        )
    semantic_error = _semantic_error(claim, evidence)
    if semantic_error:
        return _blocked_claim(claim, evidence, semantic_error, status=ClaimStatus.SUPPORTED)

    requires_corroboration = claim.requires_corroboration or claim.claim_kind in {
        ClaimKind.COMPARATIVE,
        ClaimKind.CAUSAL,
    }
    return verify_claim(
        claim.text,
        evidence,
        requires_corroboration=requires_corroboration,
        metric=claim.metric,
        claim_id=claim.claim_id,
        semantic_supported=True,
        support_quotes=claim.support_quotes,
        requires_freshness=claim.requires_freshness or claim.claim_kind == ClaimKind.FRESHNESS,
        freshness_checked_at=claim.retrieval_cutoff_at,
    ).model_copy(update={"critical": claim.critical})


def record_claim_verification(
    workspace: LiteratureWorkspace,
    claim: Claim,
    report: VerificationReport,
) -> None:
    workspace.claims = [item for item in workspace.claims if item.claim_id != claim.claim_id]
    workspace.claims.append(claim)
    workspace.claim_verification_reports = [
        item for item in workspace.claim_verification_reports if item.claim_id != claim.claim_id
    ]
    workspace.claim_verification_reports.append(report)


def _quote_error(claim: Claim, evidence_registry: dict[str, EvidenceSpan]) -> str | None:
    if set(claim.support_quotes) != set(claim.evidence_ids):
        return "Every evidence ID needs one exact support quote."
    for evidence_id, quote in claim.support_quotes.items():
        snippet = evidence_registry[evidence_id].snippet or ""
        if not quote.strip() or _normalize(quote) not in _normalize(snippet):
            return f"Support quote for {evidence_id} is not present in the registered evidence."
    return None


def _numeric_error(claim: Claim, evidence: list[EvidenceSpan]) -> str | None:
    if claim.claim_kind != ClaimKind.NUMERIC:
        return None
    observed = [span for span in evidence if isinstance(span.observed_value, int | float)]
    if not observed:
        return "Numeric claim requires structured observed values."
    expected_values = _claim_values(claim)
    if not expected_values:
        return "Numeric claim requires a machine-readable value."
    if not any(
        isclose(float(span.observed_value), value, rel_tol=1e-9, abs_tol=1e-12)
        for span in observed
        for value in expected_values
    ):
        return "Numeric value in the claim does not match registered evidence."
    if claim.metric is None:
        return "Numeric claim requires a metric name."
    if not _metric_is_grounded(claim.metric, claim, observed):
        return "Numeric claim metric is not present in the supported evidence."
    if not _units_are_compatible(claim, observed):
        return "Numeric claim unit is incompatible with registered evidence."
    if not _ranges_are_compatible(claim, observed):
        return "Numeric claim range or uncertainty does not match registered evidence."
    return None


def _semantic_error(claim: Claim, evidence: list[EvidenceSpan]) -> str | None:
    """Release only deterministic entailment, never model-declared entailment.

    Numeric claims are verified structurally above.  For non-numeric claims,
    a generated paraphrase is useful in a draft but cannot be promoted to a
    released conclusion unless the asserted text itself appears in a cited
    source quote.  This intentionally conservative boundary prevents a valid
    quote from being treated as proof of an unrelated causal or comparative
    conclusion.
    """

    if claim.claim_kind == ClaimKind.NUMERIC:
        return None
    normalized_claim = _normalize(claim.text)
    if any(normalized_claim in _normalize(span.snippet or "") for span in evidence):
        return None
    return (
        "Automatic semantic verification requires an exact asserted quote; "
        "paraphrased qualitative, comparative, causal, and freshness claims remain draft-only."
    )


def _claim_values(claim: Claim) -> list[float]:
    if claim.expected_value is not None:
        return [claim.expected_value]
    return [
        float(value) for value in re.findall(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])", claim.text)
    ]


def _metric_is_grounded(metric: str, claim: Claim, evidence: list[EvidenceSpan]) -> bool:
    normalized_metric = _normalize(metric)
    if normalized_metric in _normalize(claim.text):
        return True
    return any(normalized_metric in _normalize(span.snippet or "") for span in evidence)


def _units_are_compatible(claim: Claim, evidence: list[EvidenceSpan]) -> bool:
    expected_unit = claim.expected_unit
    if expected_unit is None:
        return True
    for span in evidence:
        if span.observed_unit is None or not isinstance(span.observed_value, int | float):
            continue
        from littrace.units import normalize_metric_unit

        expected_value, expected_normalized, _ = normalize_metric_unit(
            claim.metric or "", claim.expected_value or span.observed_value, expected_unit
        )
        observed_value, observed_normalized, _ = normalize_metric_unit(
            claim.metric or "", span.observed_value, span.observed_unit
        )
        if (
            expected_normalized == observed_normalized
            and isinstance(expected_value, int | float)
            and isinstance(observed_value, int | float)
            and isclose(float(expected_value), float(observed_value), rel_tol=1e-9, abs_tol=1e-12)
        ):
            return True
    return False


def _ranges_are_compatible(claim: Claim, evidence: list[EvidenceSpan]) -> bool:
    if (
        claim.expected_value_min is None
        and claim.expected_value_max is None
        and claim.expected_uncertainty is None
    ):
        return True
    for span in evidence:
        if (
            (
                claim.expected_value_min is None
                or claim.expected_value_min == span.observed_value_min
            )
            and (
                claim.expected_value_max is None
                or claim.expected_value_max == span.observed_value_max
            )
            and (
                claim.expected_uncertainty is None
                or claim.expected_uncertainty == span.observed_uncertainty
            )
        ):
            return True
    return False


def _blocked_claim(
    claim: Claim,
    evidence: list[EvidenceSpan],
    requirement: str,
    *,
    status: ClaimStatus = ClaimStatus.CANDIDATE,
) -> VerificationReport:
    return VerificationReport(
        claim_id=claim.claim_id,
        claim=claim.text,
        status=status,
        evidence=evidence,
        missing_requirements=[requirement],
        support_quotes=claim.support_quotes,
        critical=claim.critical,
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
