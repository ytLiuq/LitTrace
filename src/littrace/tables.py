from __future__ import annotations

import re

from littrace.harnesses import HarnessResult, check_performance_cells, check_structured_artifacts
from littrace.models import (
    ComparisonMatrix,
    ComparisonMatrixReport,
    ComparisonMatrixRow,
    EvidenceSpan,
    LiteratureWorkspace,
    PerformanceCell,
    StructuredArtifact,
)
from littrace.units import normalize_metric_unit


METRIC_DIRECTIONS = {
    "sensitivity": True,
    "gauge factor": True,
    "gf": True,
    "response time": False,
    "recovery time": False,
    "limit of detection": False,
    "lod": False,
    "accuracy": True,
    "f1": True,
    "auc": True,
    "mse": False,
    "mae": False,
    "rmse": False,
    "conductivity": True,
    "specific capacitance": True,
    "capacity": True,
    "retention": True,
    "cycle retention": True,
    "selectivity": True,
    "young's modulus": None,
    "tensile strength": True,
    "strain range": True,
}

METRIC_PATTERN = re.compile(
    r"(?P<metric>sensitivity|gauge factor|response time|recovery time|limit of detection|"
    r"specific capacitance|cycle retention|young'?s modulus|tensile strength|strain range|"
    r"conductivity|capacity|retention|selectivity|accuracy|f1|auc|mse|mae|rmse|lod|gf)"
    r"[^0-9+\-.]{0,40}"
    r"(?P<value>[+-]?\d+(?:\.\d+)?)"
    r"(?:\s*(?:±|\+/-)\s*(?P<uncertainty>\d+(?:\.\d+)?))?"
    r"(?:\s*[-–]\s*(?P<value_max>\d+(?:\.\d+)?))?"
    r"\s*(?P<unit>%|ms|s|S/m|S cm-1|S/cm|mS/cm|F/g|mF/cm2|mAh/g|mAh g-1|"
    r"kPa-1|Pa-1|ppm|GPa|MPa|kPa|Pa|cycles|)?",
    re.IGNORECASE,
)


def extract_performance_cells(workspace: LiteratureWorkspace) -> tuple[LiteratureWorkspace, HarnessResult]:
    cells: list[PerformanceCell] = []
    artifacts: list[StructuredArtifact] = []
    for paper_id, parsed in workspace.parsed_papers.items():
        cells.extend(_cells_from_sections(paper_id, parsed))
        cells.extend(_cells_from_tables(paper_id, parsed))
        artifacts.extend(_structured_artifacts_from_parsed(paper_id, parsed))

    workspace.performance_cells = cells
    _store_structured_artifacts(workspace, artifacts)
    return workspace, _combine_harnesses(
        performance=check_performance_cells(cells),
        artifacts=check_structured_artifacts(artifacts),
        artifact_count=len(artifacts),
    )


def extract_structured_artifacts(workspace: LiteratureWorkspace) -> tuple[LiteratureWorkspace, HarnessResult]:
    artifacts: list[StructuredArtifact] = []
    for paper_id, parsed in workspace.parsed_papers.items():
        artifacts.extend(_structured_artifacts_from_parsed(paper_id, parsed))
    _store_structured_artifacts(workspace, artifacts)
    return workspace, check_structured_artifacts(artifacts)


def build_comparison_matrices(workspace: LiteratureWorkspace) -> ComparisonMatrixReport:
    grouped: dict[str, list[PerformanceCell]] = {}
    for cell in workspace.performance_cells:
        grouped.setdefault(cell.metric, []).append(cell)

    matrices: list[ComparisonMatrix] = []
    report_warnings: list[str] = []
    for metric, cells in sorted(grouped.items()):
        normalized_cells = [_normalized_cell(cell) for cell in cells]
        units = {cell.unit for cell in normalized_cells if cell.unit}
        datasets = {cell.dataset for cell in normalized_cells if cell.dataset}
        tasks = {cell.task for cell in normalized_cells if cell.task}
        matrix_warnings: list[str] = []
        if len(units) > 1:
            matrix_warnings.append(f"Mixed units for {metric}: {sorted(units)}")
        if not datasets:
            matrix_warnings.append("Dataset is missing for all rows; comparison may be unfair.")
        if not tasks:
            matrix_warnings.append("Task is missing for all rows; comparison may be unfair.")

        rows = [
            _matrix_row(
                workspace,
                cell,
                units=units,
                has_dataset=bool(datasets),
                has_task=bool(tasks),
            )
            for cell in normalized_cells
        ]
        rows = sorted(rows, key=_row_sort_key(metric))
        matrices.append(ComparisonMatrix(metric=metric, rows=rows, warnings=matrix_warnings))
        report_warnings.extend(matrix_warnings)

    return ComparisonMatrixReport(matrices=matrices, warnings=report_warnings)


def _normalized_cell(cell: PerformanceCell) -> PerformanceCell:
    value, unit, warning = normalize_metric_unit(cell.metric, cell.value, cell.unit)
    if warning is None:
        return cell
    update = cell.model_dump()
    update["value"] = value
    update["unit"] = unit
    evidence = dict(update["evidence"])
    snippet = evidence.get("snippet") or ""
    evidence["snippet"] = f"{snippet} [{warning}]".strip()
    update["evidence"] = evidence
    return PerformanceCell.model_validate(update)


def _cells_from_sections(paper_id: str, parsed: dict[str, object]) -> list[PerformanceCell]:
    cells: list[PerformanceCell] = []
    sections = parsed.get("sections") or []
    if not isinstance(sections, list):
        return cells
    for section in sections:
        if not isinstance(section, dict):
            continue
        text = str(section.get("text") or "")
        evidence = section.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {}
        for match in METRIC_PATTERN.finditer(text):
            metric = _normalize_metric(match.group("metric"))
            snippet = _window(text, match.start(), match.end())
            cells.append(
                PerformanceCell(
                    paper_id=paper_id,
                    dataset=_guess_dataset(snippet),
                    metric=metric,
                    value=float(match.group("value")),
                    value_min=float(match.group("value")) if match.group("value_max") else None,
                    value_max=float(match.group("value_max")) if match.group("value_max") else None,
                    uncertainty=float(match.group("uncertainty"))
                    if match.group("uncertainty")
                    else None,
                    unit=match.group("unit") or None,
                    higher_is_better=METRIC_DIRECTIONS.get(metric),
                    evidence=EvidenceSpan(
                        paper_id=paper_id,
                        section=str(section.get("name") or evidence.get("section") or "section"),
                        page=evidence.get("page"),
                        snippet=snippet,
                        parser=evidence.get("parser"),
                        confidence=float(evidence.get("confidence") or 0.7),
                    ),
                )
            )
    return cells


def _cells_from_tables(paper_id: str, parsed: dict[str, object]) -> list[PerformanceCell]:
    cells: list[PerformanceCell] = []
    tables = parsed.get("tables") or []
    if not isinstance(tables, list):
        return cells
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_id = str(table.get("table_id") or "")
        caption = str(table.get("caption") or "")
        evidence = table.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {}
        for cell in table.get("cells") or []:
            if not isinstance(cell, dict):
                continue
            text = " ".join(str(value) for value in cell.values())
            for match in METRIC_PATTERN.finditer(f"{caption} {text}"):
                metric = _normalize_metric(match.group("metric"))
                cells.append(
                    PerformanceCell(
                        paper_id=paper_id,
                        dataset=_guess_dataset(f"{caption} {text}"),
                        metric=metric,
                        value=float(match.group("value")),
                        value_min=float(match.group("value")) if match.group("value_max") else None,
                        value_max=float(match.group("value_max")) if match.group("value_max") else None,
                        uncertainty=float(match.group("uncertainty"))
                        if match.group("uncertainty")
                        else None,
                        unit=match.group("unit") or None,
                        higher_is_better=METRIC_DIRECTIONS.get(metric),
                        evidence=EvidenceSpan(
                            paper_id=paper_id,
                            table_id=table_id,
                            row_label=str(cell.get("row") or "") or None,
                            column_label=str(cell.get("column") or "") or None,
                            snippet=_window(f"{caption} {text}", match.start(), match.end()),
                            parser=evidence.get("parser"),
                            confidence=float(evidence.get("confidence") or 0.7),
                        ),
                    )
                )
    return cells


def _structured_artifacts_from_parsed(
    paper_id: str,
    parsed: dict[str, object],
) -> list[StructuredArtifact]:
    artifacts: list[StructuredArtifact] = []
    artifacts.extend(_artifacts_from_table_objects(paper_id, parsed))
    artifacts.extend(_artifacts_from_sections(paper_id, parsed))
    return artifacts


def _artifacts_from_table_objects(
    paper_id: str,
    parsed: dict[str, object],
) -> list[StructuredArtifact]:
    artifacts: list[StructuredArtifact] = []
    tables = parsed.get("tables") or []
    if not isinstance(tables, list):
        return artifacts
    for table in tables:
        if not isinstance(table, dict):
            continue
        evidence = table.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {}
        label = str(table.get("table_id") or "") or None
        caption = str(table.get("caption") or "")
        cells = table.get("cells") or []
        text = caption
        if cells:
            text = f"{caption}\n{cells}".strip()
        artifacts.append(
            StructuredArtifact(
                paper_id=paper_id,
                artifact_type="table",
                label=label,
                text=text,
                evidence=EvidenceSpan(
                    paper_id=paper_id,
                    table_id=label,
                    snippet=text[:500],
                    parser=evidence.get("parser"),
                    confidence=float(evidence.get("confidence") or 0.75),
                ),
                confidence=float(evidence.get("confidence") or 0.75),
            )
        )
    return artifacts


def _artifacts_from_sections(paper_id: str, parsed: dict[str, object]) -> list[StructuredArtifact]:
    artifacts: list[StructuredArtifact] = []
    sections = parsed.get("sections") or []
    if not isinstance(sections, list):
        return artifacts
    for section in sections:
        if not isinstance(section, dict):
            continue
        text = str(section.get("text") or "")
        evidence = section.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {}
        artifacts.extend(_caption_artifacts(paper_id, text, evidence, "figure", FIGURE_PATTERN))
        artifacts.extend(_caption_artifacts(paper_id, text, evidence, "table", TABLE_PATTERN))
        artifacts.extend(_equation_artifacts(paper_id, text, evidence))
    return artifacts


FIGURE_PATTERN = re.compile(
    r"(?P<label>(?:Fig\.?|Figure)\s*\d+[A-Za-z]?)\s*[:.\-]?\s*(?P<caption>.{20,500}?)(?=\n\s*(?:Fig\.?|Figure|Table|Eq\.?|Equation)\s*\d+|\n\s*\n|$)",
    re.IGNORECASE | re.DOTALL,
)
TABLE_PATTERN = re.compile(
    r"(?P<label>Table\s*\d+[A-Za-z]?)\s*[:.\-]?\s*(?P<caption>.{20,700}?)(?=\n\s*(?:Fig\.?|Figure|Table|Eq\.?|Equation)\s*\d+|\n\s*\n|$)",
    re.IGNORECASE | re.DOTALL,
)
EQUATION_PATTERN = re.compile(
    r"(?P<label>(?:Eq\.?|Equation)\s*\(?\d+[A-Za-z]?\)?)\s*[:.\-]?\s*(?P<body>.{3,240}?)(?=\n|$)",
    re.IGNORECASE,
)
FORMULA_HINT_PATTERN = re.compile(
    r"(?P<formula>[A-Za-z][A-Za-z0-9_{}^+\-*/=().\s]{0,80}=\s*[A-Za-z0-9_{}^+\-*/=().\s]{1,120})"
)


def _caption_artifacts(
    paper_id: str,
    text: str,
    evidence: dict[str, object],
    artifact_type: str,
    pattern: re.Pattern[str],
) -> list[StructuredArtifact]:
    artifacts = []
    for match in pattern.finditer(text):
        label = " ".join(match.group("label").split())
        artifact_text = " ".join(match.group("caption").split())
        artifacts.append(
            StructuredArtifact(
                paper_id=paper_id,
                artifact_type=artifact_type,
                label=label,
                text=artifact_text,
                evidence=EvidenceSpan(
                    paper_id=paper_id,
                    section=str(evidence.get("section") or "section"),
                    page=evidence.get("page"),
                    snippet=artifact_text[:500],
                    parser=evidence.get("parser"),
                    confidence=float(evidence.get("confidence") or 0.68),
                ),
                confidence=float(evidence.get("confidence") or 0.68),
            )
        )
    return artifacts


def _equation_artifacts(
    paper_id: str,
    text: str,
    evidence: dict[str, object],
) -> list[StructuredArtifact]:
    artifacts = []
    for pattern in (EQUATION_PATTERN, FORMULA_HINT_PATTERN):
        for match in pattern.finditer(text):
            label_value = match.groupdict().get("label") or "formula"
            label = " ".join(label_value.split()) or "formula"
            artifact_text = " ".join((match.groupdict().get("body") or match.group(0)).split())
            if len(artifact_text) < 3:
                continue
            artifacts.append(
                StructuredArtifact(
                    paper_id=paper_id,
                    artifact_type="equation" if label.lower().startswith(("eq", "equation")) else "formula",
                    label=label,
                    text=artifact_text,
                    evidence=EvidenceSpan(
                        paper_id=paper_id,
                        section=str(evidence.get("section") or "section"),
                        page=evidence.get("page"),
                        snippet=artifact_text[:500],
                        parser=evidence.get("parser"),
                        confidence=float(evidence.get("confidence") or 0.62),
                    ),
                    confidence=float(evidence.get("confidence") or 0.62),
                )
            )
    return artifacts


def _store_structured_artifacts(
    workspace: LiteratureWorkspace,
    artifacts: list[StructuredArtifact],
) -> None:
    workspace.context.filters["structured_artifacts"] = [
        artifact.model_dump(mode="json") for artifact in artifacts
    ]


def _combine_harnesses(
    performance: HarnessResult,
    artifacts: HarnessResult,
    artifact_count: int,
) -> HarnessResult:
    score = (performance.score + artifacts.score) / 2
    warnings = [
        *performance.warnings,
        *artifacts.warnings,
    ]
    if artifact_count == 0:
        warnings.append("No table, figure, formula, or equation artifacts were extracted.")
    return HarnessResult(
        passed=performance.passed and artifacts.passed,
        score=score,
        errors=[*performance.errors, *artifacts.errors],
        warnings=warnings,
    )


def _normalize_metric(metric: str) -> str:
    normalized = metric.lower().strip()
    if normalized == "gf":
        return "gauge factor"
    if normalized == "lod":
        return "limit of detection"
    if normalized in {"youngs modulus", "young's modulus"}:
        return "young's modulus"
    return normalized


def _window(text: str, start: int, end: int, radius: int = 90) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)].strip()


def _matrix_row(
    workspace: LiteratureWorkspace,
    cell: PerformanceCell,
    units: set[str],
    has_dataset: bool,
    has_task: bool,
) -> ComparisonMatrixRow:
    paper = workspace.papers.get(cell.paper_id)
    warnings: list[str] = []
    comparable = True
    if len(units) > 1:
        comparable = False
        warnings.append("Mixed units within this metric group.")
    if has_dataset and not cell.dataset:
        comparable = False
        warnings.append("Dataset missing for this row.")
    if has_task and not cell.task:
        comparable = False
        warnings.append("Task missing for this row.")
    if cell.evidence.confidence < 0.65:
        comparable = False
        warnings.append("Low-confidence evidence.")
    if cell.higher_is_better is None:
        warnings.append("Metric direction is unknown.")

    return ComparisonMatrixRow(
        paper_id=cell.paper_id,
        title=paper.title if paper else None,
        year=paper.year if paper else None,
        metric=cell.metric,
        value=cell.value,
        unit=cell.unit,
        task=cell.task,
        dataset=cell.dataset,
        method_name=cell.method_name,
        higher_is_better=cell.higher_is_better,
        comparable=comparable,
        warnings=warnings,
        evidence=cell.evidence,
    )


def _row_sort_key(metric: str):
    higher_is_better = METRIC_DIRECTIONS.get(metric)

    def sort_key(row: ComparisonMatrixRow):
        value = float(row.value) if isinstance(row.value, int | float) else 0.0
        direction = -1 if higher_is_better else 1
        return (not row.comparable, direction * value)

    return sort_key


def _guess_dataset(text: str) -> str | None:
    known_datasets = [
        "ETTm1",
        "ETTm2",
        "ETTh1",
        "ETTh2",
        "MNIST",
        "CIFAR-10",
        "human motion",
        "artificial sweat",
        "PBS",
        "electrochemical workstation",
        "cyclic bending",
        "cycling test",
    ]
    lowered = text.lower()
    for dataset in known_datasets:
        if dataset.lower() in lowered:
            return dataset
    return None
