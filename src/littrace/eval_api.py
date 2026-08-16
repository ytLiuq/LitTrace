from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.evaluation.golden_eval import _load_cases
from littrace.models import LiteratureWorkspace, coerce_parsed
from littrace.evaluation.pdf_benchmark import benchmark_pdf_parsing
from littrace.evidence.storyline import build_storyline_from_workspace


class EvalMetricReport(BaseModel):
    run_id: str
    topic: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    failures: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval metrics — computed from workspace + optional golden set
# ---------------------------------------------------------------------------


def retrieval_metrics(
    workspace: LiteratureWorkspace | None = None,
    config: LitTraceConfig | None = None,
) -> dict[str, float]:
    """Compute retrieval quality metrics from the workspace.

    If *config* is provided and a golden set exists, DOI recall is computed
    against the expected DOIs.  Otherwise, structural metrics (duplicate rate,
    recent-paper ratio) are still computed from the workspace.
    """
    if workspace is None:
        return {
            "retrieval_recall_at_20": 0.0,
            "retrieval_precision_at_20": 0.0,
            "recent_paper_ratio_2023_2026": 0.0,
            "duplicate_rate": 0.0,
        }

    active_papers = [
        workspace.papers[pid] for pid in workspace.context.active_papers if pid in workspace.papers
    ]
    active_count = len(active_papers)

    # --- duplicate rate -------------------------------------------------
    seen_dois: set[str] = set()
    dup_count = 0
    for paper in active_papers:
        if paper.doi:
            key = paper.doi.lower()
            if key in seen_dois:
                dup_count += 1
            else:
                seen_dois.add(key)
    duplicate_rate = dup_count / active_count if active_count else 0.0

    # --- recent paper ratio --------------------------------------------
    recent_count = sum(1 for p in active_papers if p.year is not None and p.year >= 2023)
    recent_ratio = recent_count / active_count if active_count else 0.0

    # --- golden-set recall (if config & golden cases available) ---------
    recall = 0.0
    precision = 0.0
    if config is not None:
        cases = _load_cases(config.eval_golden_set_dir)
        if cases:
            expected_dois: set[str] = set()
            for case in cases:
                raw = case.get("expected_dois")
                if isinstance(raw, str):
                    expected_dois.add(raw.lower())
                elif isinstance(raw, list):
                    expected_dois.update(str(d).lower() for d in raw if d)
            if expected_dois:
                active_dois = {p.doi.lower() for p in active_papers if p.doi}
                hits = expected_dois & active_dois
                recall = len(hits) / len(expected_dois)
                # precision@20: of the top-20 active papers, how many are expected?
                top20 = active_papers[:20]
                top20_dois = {p.doi.lower() for p in top20 if p.doi}
                top20_hits = top20_dois & expected_dois
                precision = len(top20_hits) / len(top20) if top20 else 0.0

    return {
        "retrieval_recall_at_20": round(recall, 3),
        "retrieval_precision_at_20": round(precision, 3),
        "recent_paper_ratio_2023_2026": round(recent_ratio, 3),
        "duplicate_rate": round(duplicate_rate, 3),
    }


# ---------------------------------------------------------------------------
# Parsing metrics — computed via pdf_benchmark
# ---------------------------------------------------------------------------


def parsing_metrics(
    workspace: LiteratureWorkspace | None = None,
    config: LitTraceConfig | None = None,
) -> dict[str, float]:
    """Compute PDF parsing quality metrics.

    Delegates to ``benchmark_pdf_parsing`` for structural metrics
    (local_pdf_rate, parsed_rate, page-evidence coverage).
    """
    if workspace is None or config is None:
        return {
            "metadata_accuracy": 0.0,
            "section_extraction_accuracy": 0.0,
            "table_cell_exact_match": 0.0,
            "reference_accuracy": 0.0,
        }

    report = benchmark_pdf_parsing(workspace, config)
    active = report.active_papers or 1

    # section_extraction_accuracy: fraction of parsed papers that produced
    # at least one section with page-level evidence
    section_acc = report.parsed_with_page_evidence / active if active else 0.0

    # metadata_accuracy: fraction of active papers that have a local PDF
    # and at least some parsed content (proxy for metadata completeness)
    metadata_acc = report.parsed_count / active if active else 0.0

    # table_cell_exact_match: not measurable without a golden table set;
    # use parsed_rate as a proxy
    table_match = report.parsed_rate

    # reference_accuracy: fraction of parsed papers with 0 failures
    ref_acc = (
        (report.parsed_count - report.failed_count) / active
        if active and report.parsed_count >= report.failed_count
        else 0.0
    )

    return {
        "metadata_accuracy": round(metadata_acc, 3),
        "section_extraction_accuracy": round(section_acc, 3),
        "table_cell_exact_match": round(table_match, 3),
        "reference_accuracy": round(max(ref_acc, 0.0), 3),
    }


# ---------------------------------------------------------------------------
# Storyline metrics — computed from workspace storyline
# ---------------------------------------------------------------------------


def storyline_metrics(
    workspace: LiteratureWorkspace | None = None,
) -> dict[str, float]:
    """Compute storyline quality metrics from the workspace.

    - **claim_grounding_rate**: fraction of claims that have at least one
      evidence span with a non-empty snippet.
    - **citation_coverage**: fraction of claims whose evidence cites an
      active paper.
    - **unsupported_claim_rate**: 1 - claim_grounding_rate.
    """
    if workspace is None:
        return {
            "claim_grounding_rate": 0.0,
            "citation_coverage": 0.0,
            "unsupported_claim_rate": 0.0,
        }

    claims = build_storyline_from_workspace(workspace)
    total = len(claims)
    if total == 0:
        return {
            "claim_grounding_rate": 0.0,
            "citation_coverage": 0.0,
            "unsupported_claim_rate": 0.0,
        }

    active_paper_ids = set(workspace.context.active_papers)

    grounded = 0
    cited = 0
    for claim in claims:
        has_evidence = any((ev.snippet and ev.snippet.strip()) for ev in claim.evidence)
        if has_evidence:
            grounded += 1
        has_citation = any(ev.paper_id and ev.paper_id in active_paper_ids for ev in claim.evidence)
        if has_citation:
            cited += 1

    grounding_rate = grounded / total
    citation_cov = cited / total
    unsupported = 1.0 - grounding_rate

    return {
        "claim_grounding_rate": round(grounding_rate, 3),
        "citation_coverage": round(citation_cov, 3),
        "unsupported_claim_rate": round(unsupported, 3),
    }


# ---------------------------------------------------------------------------
# Full-text metrics — already implemented, kept as-is
# ---------------------------------------------------------------------------


def full_text_metrics_from_workspace(workspace) -> dict[str, float]:
    active_ids = workspace.context.active_papers
    active_count = len(active_ids)
    reports = [
        workspace.full_text_reports[paper_id]
        for paper_id in active_ids
        if paper_id in workspace.full_text_reports
    ]
    if active_count == 0:
        return {
            "full_text_resolved_rate": 0.0,
            "verified_candidate_rate": 0.0,
            "oa_pdf_candidate_rate": 0.0,
            "login_handoff_ready_rate": 0.0,
            "parsed_full_text_rate": 0.0,
        }
    verified = sum(report.verified_candidate_count > 0 for report in reports)
    oa_pdf = sum(bool(report.best_pdf_url) for report in reports)
    login_ready = sum(report.login_required_candidate_count > 0 for report in reports)
    parsed = sum(
        (
            workspace.parsed_papers.get(paper_id) is not None
            and coerce_parsed(workspace.parsed_papers[paper_id]).parsed
        )
        for paper_id in active_ids
    )
    return {
        "full_text_resolved_rate": len(reports) / active_count,
        "verified_candidate_rate": verified / active_count,
        "oa_pdf_candidate_rate": oa_pdf / active_count,
        "login_handoff_ready_rate": login_ready / active_count,
        "parsed_full_text_rate": parsed / active_count,
    }
