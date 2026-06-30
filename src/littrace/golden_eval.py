from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.agent_interactions import build_agent_interaction_report
from littrace.config import LitTraceConfig
from littrace.citations import citation_records_for_papers
from littrace.models import LiteratureWorkspace
from littrace.storyline import build_storyline_from_workspace


class GoldenEvalReport(BaseModel):
    golden_set_dir: str
    case_count: int
    metrics: dict[str, float] = Field(default_factory=dict)
    failures: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def run_golden_eval(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace | None = None,
) -> GoldenEvalReport:
    root = config.eval.golden_set_dir
    cases = _load_cases(root)
    warnings: list[str] = []
    if not root.exists():
        warnings.append(f"Golden set directory does not exist: {root}")
    if not cases:
        warnings.append("No golden cases found. Add JSONL files under eval/golden.")
    metrics = {
        "case_count": float(len(cases)),
        "has_expected_doi_rate": _rate(cases, "expected_dois"),
        "has_expected_metrics_rate": _rate(cases, "expected_metrics"),
        "has_expected_storyline_rate": _rate(cases, "expected_storyline_claims"),
        "has_expected_pdf_features_rate": _rate(cases, "expected_pdf_features"),
    }
    failures: list[dict[str, object]] = []
    if workspace is not None:
        workspace_metrics, failures = evaluate_workspace_against_golden(workspace, cases)
        metrics.update(workspace_metrics)
    return GoldenEvalReport(
        golden_set_dir=str(root),
        case_count=len(cases),
        metrics=metrics,
        failures=failures,
        warnings=warnings,
    )


def evaluate_workspace_against_golden(
    workspace: LiteratureWorkspace,
    cases: list[dict[str, object]],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    active_papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    active_dois = {paper.doi.lower() for paper in active_papers if paper.doi}
    active_publishers = {
        _norm(str(paper.publisher or paper.journal or "")) for paper in active_papers
    }
    active_years = [paper.year for paper in active_papers if paper.year is not None]
    metrics_available = {_norm(cell.metric) for cell in workspace.performance_cells}
    storyline_text = _storyline_search_text(workspace)
    citation_records = citation_records_for_papers(active_papers)
    interaction_report = build_agent_interaction_report(workspace)
    failures: list[dict[str, object]] = []

    expected_dois = _expected_values(cases, "expected_dois")
    found_dois = {doi for doi in expected_dois if doi.lower() in active_dois}
    _append_missing(failures, "retrieval", "expected_dois", expected_dois, found_dois)

    expected_metrics = {_norm(value) for value in _expected_values(cases, "expected_metrics")}
    found_metrics = {metric for metric in expected_metrics if metric in metrics_available}
    _append_missing(failures, "table", "expected_metrics", expected_metrics, found_metrics)

    expected_story = {_norm(value) for value in _expected_values(cases, "expected_storyline_claims")}
    found_story = {keyword for keyword in expected_story if keyword in storyline_text}
    _append_missing(failures, "storyline", "expected_storyline_claims", expected_story, found_story)

    expected_publishers = {_norm(value) for value in _expected_values(cases, "expected_publishers")}
    found_publishers = {
        publisher
        for publisher in expected_publishers
        if any(publisher in active for active in active_publishers)
    }
    _append_missing(failures, "source_router", "expected_publishers", expected_publishers, found_publishers)

    min_year = int(min(_expected_numbers(cases, "preferred_year_min") or [2023]))
    recent_count = sum(1 for year in active_years if year >= min_year)
    citation_coverage = sum(bool(record.access_url) for record in citation_records)

    metrics = {
        "golden_retrieval_doi_recall": _safe_div(len(found_dois), len(expected_dois)),
        "golden_recent_paper_ratio": _safe_div(recent_count, len(active_years)),
        "golden_expected_publisher_coverage": _safe_div(
            len(found_publishers), len(expected_publishers)
        ),
        "golden_table_metric_recall": _safe_div(len(found_metrics), len(expected_metrics)),
        "golden_storyline_keyword_coverage": _safe_div(len(found_story), len(expected_story)),
        "golden_citation_coverage": _safe_div(citation_coverage, len(citation_records)),
        "golden_agent_flow_blocked_count": float(interaction_report.blocked_count),
        "golden_agent_flow_complete_count": float(interaction_report.complete_count),
    }
    return metrics, failures


def _load_cases(root: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    if not root.exists():
        return cases
    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    return cases


def _rate(cases: list[dict[str, object]], key: str) -> float:
    if not cases:
        return 0.0
    return sum(bool(case.get(key)) for case in cases) / len(cases)


def _expected_values(cases: list[dict[str, object]], key: str) -> set[str]:
    values: set[str] = set()
    for case in cases:
        raw = case.get(key)
        if isinstance(raw, str):
            values.add(raw)
        elif isinstance(raw, list):
            values.update(str(item) for item in raw)
    return values


def _expected_numbers(cases: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for case in cases:
        raw = case.get(key)
        if isinstance(raw, int | float):
            values.append(float(raw))
    return values


def _append_missing(
    failures: list[dict[str, object]],
    agent: str,
    field: str,
    expected: set[str],
    found: set[str],
) -> None:
    missing = sorted(expected - found)
    if missing:
        failures.append({"agent": agent, "field": field, "missing": missing})


def _norm(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("/", " ").split())


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 3)


def _storyline_search_text(workspace: LiteratureWorkspace) -> str:
    parts: list[str] = []
    for claim in build_storyline_from_workspace(workspace):
        parts.append(claim.claim)
        parts.extend(evidence.snippet or "" for evidence in claim.evidence)
    return " ".join(parts).lower()
