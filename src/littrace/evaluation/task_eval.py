from __future__ import annotations

from collections.abc import Mapping
from math import isclose

from pydantic import BaseModel, Field

from littrace.models import ClaimStatus, LiteratureWorkspace, PerformanceCell
from littrace.units import normalize_metric_unit


class ExpectedPerformanceCell(BaseModel):
    paper_id: str | None = None
    doi: str | None = None
    metric: str
    value: float | str
    unit: str | None = None
    page: int | None = None
    table_id: str | None = None
    relative_tolerance: float = 1e-3
    absolute_tolerance: float = 1e-9


class TableCellEvalResult(BaseModel):
    gold_count: int
    extracted_count: int
    matched_count: int
    recall: float
    precision: float
    exact_match: float
    missing: list[dict[str, object]] = Field(default_factory=list)


class TaskEvalCaseResult(BaseModel):
    case_id: str
    doi_recall: float
    table_cell_recall: float
    required_claim_coverage: float
    critical_claim_precision: float
    citation_recall: float
    unsupported_critical_claim_rate: float
    should_abstain: bool
    passed: bool
    failures: list[str] = Field(default_factory=list)


class TaskEvalReport(BaseModel):
    case_count: int
    evidence_grounded_task_success_rate: float
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: list[TaskEvalCaseResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def evaluate_task_runs(
    cases: list[dict[str, object]],
    workspaces_by_case: Mapping[str, LiteratureWorkspace],
) -> TaskEvalReport:
    results: list[TaskEvalCaseResult] = []
    warnings: list[str] = []
    for index, case in enumerate(cases):
        case_id = str(case.get("case_id") or case.get("topic") or f"case-{index + 1}")
        workspace = workspaces_by_case.get(case_id)
        if workspace is None:
            warnings.append(f"No workspace result was supplied for task case {case_id}.")
            continue
        results.append(evaluate_workspace_task_case(case, workspace))

    return TaskEvalReport(
        case_count=len(results),
        evidence_grounded_task_success_rate=_safe_div(
            sum(result.passed for result in results),
            len(results),
        ),
        metrics={
            "doi_recall": _avg([result.doi_recall for result in results]),
            "table_cell_recall": _avg([result.table_cell_recall for result in results]),
            "required_claim_coverage": _avg(
                [result.required_claim_coverage for result in results]
            ),
            "critical_claim_precision": _avg(
                [result.critical_claim_precision for result in results]
            ),
            "citation_recall": _avg([result.citation_recall for result in results]),
            "unsupported_critical_claim_rate": _avg(
                [result.unsupported_critical_claim_rate for result in results]
            ),
        },
        cases=results,
        warnings=warnings,
    )


def evaluate_workspace_task_case(
    case: dict[str, object],
    workspace: LiteratureWorkspace,
) -> TaskEvalCaseResult:
    case_id = str(case.get("case_id") or case.get("topic") or "task")
    expected_dois = {_normalize_doi(value) for value in _string_values(case, "expected_dois")}
    active_dois = {
        _normalize_doi(workspace.papers[paper_id].doi)
        for paper_id in workspace.context.active_papers
        if paper_id in workspace.papers and workspace.papers[paper_id].doi
    }
    doi_recall = _safe_div(len(expected_dois & active_dois), len(expected_dois))
    if not expected_dois:
        doi_recall = 1.0

    raw_cells = case.get("expected_performance_cells")
    expected_cells = [
        ExpectedPerformanceCell.model_validate(item)
        for item in raw_cells
        if isinstance(item, dict)
    ] if isinstance(raw_cells, list) else []
    table_result = evaluate_performance_cells(expected_cells, workspace)
    table_recall = table_result.recall if expected_cells else 1.0

    required_claims = _string_values(case, "required_claims")
    if not required_claims:
        required_claims = _string_values(case, "expected_storyline_claims")
    publishable_reports = [
        report for report in workspace.claim_verification_reports if report.publishable
    ]
    verified_claim_text = _normalize(
        " ".join(report.claim for report in publishable_reports)
    )
    matched_claims = [
        claim for claim in required_claims if _normalize(claim) in verified_claim_text
    ]
    claim_coverage = _safe_div(len(matched_claims), len(required_claims))
    if not required_claims:
        claim_coverage = 1.0

    critical_reports = [
        report for report in workspace.claim_verification_reports if report.critical
    ]
    publishable_critical = [
        report for report in critical_reports if report.status in _PUBLISHABLE_STATUSES
    ]
    critical_precision = _safe_div(len(publishable_critical), len(critical_reports))
    citation_recall = _safe_div(
        sum(
            bool(report.evidence)
            and all(span.provenance_complete for span in report.evidence)
            for report in critical_reports
        ),
        len(critical_reports),
    )
    if not critical_reports:
        critical_precision = 1.0 if not required_claims else 0.0
        citation_recall = 1.0 if not required_claims else 0.0
    unsupported_rate = 1.0 - critical_precision if critical_reports else 0.0

    should_abstain = bool(case.get("should_abstain", False))
    failures: list[str] = []
    if should_abstain:
        if publishable_critical:
            failures.append("The task should abstain, but publishable critical claims were produced.")
    else:
        if doi_recall < 1.0:
            failures.append("Not all required papers were retrieved.")
        if table_recall < 1.0:
            failures.append("Not all expected performance cells were extracted exactly.")
        if claim_coverage < 1.0:
            failures.append("Not all required answer claims were verified.")
        if critical_precision < 1.0:
            failures.append("At least one critical claim is not publishable.")
        if citation_recall < 1.0:
            failures.append("At least one critical claim lacks traceable evidence.")

    return TaskEvalCaseResult(
        case_id=case_id,
        doi_recall=doi_recall,
        table_cell_recall=table_recall,
        required_claim_coverage=claim_coverage,
        critical_claim_precision=critical_precision,
        citation_recall=citation_recall,
        unsupported_critical_claim_rate=round(unsupported_rate, 3),
        should_abstain=should_abstain,
        passed=not failures,
        failures=failures,
    )


def evaluate_performance_cells(
    expected: list[ExpectedPerformanceCell],
    workspace: LiteratureWorkspace,
) -> TableCellEvalResult:
    unmatched_actual = set(range(len(workspace.performance_cells)))
    matched = 0
    missing: list[dict[str, object]] = []
    for gold in expected:
        match_index = next(
            (
                index
                for index in unmatched_actual
                if _cell_matches(gold, workspace.performance_cells[index], workspace)
            ),
            None,
        )
        if match_index is None:
            missing.append(gold.model_dump(mode="json"))
            continue
        unmatched_actual.remove(match_index)
        matched += 1

    return TableCellEvalResult(
        gold_count=len(expected),
        extracted_count=len(workspace.performance_cells),
        matched_count=matched,
        recall=_safe_div(matched, len(expected)),
        precision=_safe_div(matched, len(workspace.performance_cells)),
        exact_match=1.0 if matched == len(expected) else 0.0,
        missing=missing,
    )


def _cell_matches(
    gold: ExpectedPerformanceCell,
    actual: PerformanceCell,
    workspace: LiteratureWorkspace,
) -> bool:
    if gold.paper_id and gold.paper_id != actual.paper_id:
        return False
    if gold.doi:
        paper = workspace.papers.get(actual.paper_id)
        if _normalize_doi(paper.doi if paper else None) != _normalize_doi(gold.doi):
            return False
    if _normalize(gold.metric) != _normalize(actual.metric):
        return False
    if gold.page is not None and gold.page != actual.evidence.page:
        return False
    if gold.table_id and _normalize(gold.table_id) != _normalize(actual.evidence.table_id):
        return False

    expected_value, expected_unit, _ = normalize_metric_unit(
        gold.metric,
        gold.value,
        gold.unit,
    )
    actual_value, actual_unit, _ = normalize_metric_unit(
        actual.metric,
        actual.value,
        actual.unit,
    )
    if expected_unit and expected_unit != actual_unit:
        return False
    if isinstance(expected_value, int | float) and isinstance(actual_value, int | float):
        return isclose(
            float(expected_value),
            float(actual_value),
            rel_tol=gold.relative_tolerance,
            abs_tol=gold.absolute_tolerance,
        )
    return _normalize(expected_value) == _normalize(actual_value)


def _string_values(case: dict[str, object], key: str) -> list[str]:
    value = case.get(key)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _normalize(value: object) -> str:
    return " ".join(str(value or "").lower().replace("-", " ").replace("/", " ").split())


def _normalize_doi(value: object) -> str:
    return str(value or "").strip().lower().removeprefix("https://doi.org/")


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _safe_div(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


_PUBLISHABLE_STATUSES = {ClaimStatus.VERIFIED, ClaimStatus.CORROBORATED}
