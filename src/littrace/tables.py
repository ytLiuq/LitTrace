"""Performance-table extraction, comparison matrix construction, and OCR/artifact
need heuristics.

Canonical home for the table-extraction API. ``littrace.evidence.tables`` is a
backward-compatible re-export shim — older code paths continue to work, but
new code should import from this module directly. The legacy
``evidence.tables`` location was kept around so the rest of the codebase
didn't have to update its imports, but every external test (``tests/test_tables.py``)
and skill plugin points at this module instead.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError

from littrace.config import LitTraceConfig
from littrace.evaluation.harnesses import (
    HarnessResult,
    check_performance_cells,
    check_structured_artifacts,
    check_schema_compliance,
    SchemaCheckItem,
)
from littrace.llm import chat_completion
from littrace.log import get_logger, timed
from littrace.models import (
    ComparisonMatrix,
    ComparisonMatrixReport,
    ComparisonMatrixRow,
    EvidenceSpan,
    ExperimentalConditions,
    LiteratureWorkspace,
    ParsedPaper,
    PerformanceCell,
    StructuredArtifact,
    coerce_parsed,
)
from littrace.units import normalize_metric_unit

logger = get_logger("tables")


# ── Schema for LLM output validation (Dimension 3) ─────────────


class _LLMCellSchema(BaseModel):
    """Pydantic schema for validating each LLM-returned performance cell."""

    metric: str
    value: float
    value_min: float | None = None
    value_max: float | None = None
    uncertainty: float | None = None
    unit: str | None = None
    section: str = "section"
    snippet: str = ""
    dataset: str | None = None
    task: str | None = None
    method_name: str | None = None


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


class ArtifactNeedReport(BaseModel):
    needs_artifact_extraction: bool
    reason: str
    text_evidence_count: int = 0
    performance_cell_count: int = 0
    structured_artifact_count: int = 0
    recommended_tools: list[str] = Field(default_factory=list)
    recommended_parse_strategy: str = "text_only"
    buttons: list[dict[str, str]] = Field(default_factory=list)


def decide_artifact_extraction_need(workspace: LiteratureWorkspace) -> ArtifactNeedReport:
    text_evidence_count = sum(
        1
        for parsed in map(coerce_parsed, workspace.parsed_papers.values())
        for section in parsed.sections or []
        if isinstance(section, dict) and str(section.get("text") or "").strip()
    )
    artifact_count = len(_stored_structured_artifacts(workspace))
    performance_count = len(workspace.performance_cells)
    if performance_count and text_evidence_count:
        return ArtifactNeedReport(
            needs_artifact_extraction=False,
            reason="正文和已抽取性能指标已能支撑基础综述；图表/公式抽取可作为增强步骤。",
            text_evidence_count=text_evidence_count,
            performance_cell_count=performance_count,
            structured_artifact_count=artifact_count,
            recommended_parse_strategy="text_only",
            buttons=_ocr_choice_buttons("text_only"),
        )
    if artifact_count:
        return ArtifactNeedReport(
            needs_artifact_extraction=False,
            reason="已有结构化图表/公式/图注证据。",
            text_evidence_count=text_evidence_count,
            performance_cell_count=performance_count,
            structured_artifact_count=artifact_count,
            recommended_parse_strategy="text_only",
            buttons=_ocr_choice_buttons("text_only"),
        )
    if text_evidence_count:
        return ArtifactNeedReport(
            needs_artifact_extraction=True,
            reason="已有正文证据但缺少性能单元；建议仅在需要精确性能对比、公式解释或图中趋势时抽取图表/公式。",
            text_evidence_count=text_evidence_count,
            performance_cell_count=performance_count,
            structured_artifact_count=artifact_count,
            recommended_tools=["paddleocr", "structured_artifact_extractor"],
            recommended_parse_strategy="ocr",
            buttons=_ocr_choice_buttons("ocr"),
        )
    return ArtifactNeedReport(
        needs_artifact_extraction=True,
        reason="缺少正文和结构化证据；需要先解析全文，扫描件或复杂图表优先使用 PaddleOCR。",
        text_evidence_count=text_evidence_count,
        performance_cell_count=performance_count,
        structured_artifact_count=artifact_count,
        recommended_tools=["paddleocr"],
        recommended_parse_strategy="ocr",
        buttons=_ocr_choice_buttons("ocr"),
    )


def _ocr_choice_buttons(recommended: str) -> list[dict[str, str]]:
    return [
        {
            "id": "text_only",
            "label": "只看文字层",
            "parse_strategy": "text_only",
            "recommended": str(recommended == "text_only").lower(),
            "description": "速度快，适合可复制文字的 PDF；不会读取扫描图片里的文字。",
        },
        {
            "id": "ocr",
            "label": "使用 OCR",
            "parse_strategy": "ocr",
            "recommended": str(recommended == "ocr").lower(),
            "description": "速度慢但能处理扫描件、图中标注、复杂图表和公式附近文字。",
        },
    ]


_EXTRACTION_SYSTEM_PROMPT = """You are a materials/chemistry performance-metric extractor.
Read the parsed PDF text (sections + tables) and extract every quantitative performance metric.

Return STRICT JSON: a list of objects, each with keys:
  metric (string), value (number), value_min (number|null), value_max (number|null),
  uncertainty (number|null), unit (string|null), section (string), snippet (string),
  dataset (string|null), task (string|null), method_name (string|null)

Rules:
- Extract ONLY from the provided text; do NOT invent values.
- Extract only observations with a numeric value. Skip "not given", "N/A", qualitative
  descriptions, and multi-value prose that cannot be represented by one value plus an optional range.
- snippet must be the exact surrounding text (max 200 chars) where the value was found.
- If a value appears as a range (e.g. "0.1-0.5 S/cm"), set value_min and value_max.
- Common metrics: sensitivity, gauge factor, response time, recovery time, limit of detection,
  conductivity, specific capacitance, capacity, retention, selectivity, tensile strength,
  young's modulus, strain range, accuracy, f1, auc, mse, mae, rmse.
- If no metrics are found, return an empty list [].
- Do NOT wrap the JSON in markdown fences."""


def _build_extraction_payload(paper_id: str, parsed: ParsedPaper) -> str:
    """Build the text payload sent to the LLM for metric extraction."""
    parts: list[str] = [f"Paper ID: {paper_id}"]
    for section in parsed.sections or []:
        if not isinstance(section, dict):
            continue
        name = str(section.get("name") or "section")
        text = str(section.get("text") or "")
        if text.strip():
            parts.append(f"\n[Section: {name}]\n{text[:2000]}")
    for table in parsed.tables or []:
        caption = str(table.caption or "")
        table_id = str(table.table_id or "")
        cells = table.cells or []
        if cells:
            parts.append(f"\n[Table: {table_id}] {caption}\n{cells}")
    return "\n".join(parts)


def _parse_llm_cells(
    paper_id: str,
    raw_cells: list[dict[str, object]],
    parsed: ParsedPaper | dict[str, object],
) -> tuple[list[PerformanceCell], list[str]]:
    """Convert LLM-returned JSON objects into PerformanceCell models.

    Uses Pydantic _LLMCellSchema for schema validation (Dimension 3).
    Returns (valid_cells, schema_errors) — errors are collected for harness
    reporting instead of being silently dropped.
    """
    cells: list[PerformanceCell] = []
    schema_errors: list[str] = []
    section_evidence: dict[str, dict[str, object]] = {}

    # Support both ParsedPaper objects and plain dicts for backwards compat
    raw_sections: list = []
    if isinstance(parsed, ParsedPaper):
        raw_sections = parsed.sections or []
    elif isinstance(parsed, dict):
        raw_sections = parsed.get("sections") or []

    for section in raw_sections:
        if isinstance(section, dict):
            name = str(section.get("name") or "section")
            ev = section.get("evidence") or {}
            if isinstance(ev, dict):
                section_evidence[name] = ev

    for idx, raw in enumerate(raw_cells):
        if not isinstance(raw, dict):
            schema_errors.append(f"Item {idx}: expected object, got {type(raw).__name__}")
            continue
        try:
            # Dimension 3: Validate via Pydantic schema
            validated = _LLMCellSchema.model_validate(raw)
            metric = validated.metric.strip().lower()
            if not metric:
                schema_errors.append(f"Item {idx}: empty metric name")
                continue

            section_name = validated.section or "section"
            ev = section_evidence.get(section_name, {})
            snippet = (validated.snippet or "")[:200]
            cells.append(
                PerformanceCell(
                    paper_id=paper_id,
                    dataset=validated.dataset or _guess_dataset(snippet),
                    task=validated.task or None,
                    method_name=validated.method_name or None,
                    metric=metric,
                    value=validated.value,
                    value_min=validated.value_min,
                    value_max=validated.value_max,
                    uncertainty=validated.uncertainty,
                    unit=validated.unit or None,
                    higher_is_better=METRIC_DIRECTIONS.get(metric),
                    evidence=EvidenceSpan(
                        paper_id=paper_id,
                        section=section_name,
                        page=ev.get("page"),
                        snippet=snippet,
                        parser=ev.get("parser"),
                        confidence=0.85,
                    ),
                )
            )
        except ValidationError as exc:
            # Collect the first error message for harness reporting
            errors = exc.errors()
            if errors:
                loc = ".".join(str(x) for x in errors[0]["loc"])
                msg = errors[0]["msg"]
                schema_errors.append(f"Item {idx} ({loc}): {msg}")
            else:
                schema_errors.append(f"Item {idx}: validation error")
        except (ValueError, TypeError) as exc:
            schema_errors.append(f"Item {idx}: {exc}")
    return cells, schema_errors


async def extract_performance_cells(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig | None = None,
) -> tuple[LiteratureWorkspace, HarnessResult]:
    """Extract performance metrics from parsed papers using LLM.

    A secondary regex pass catches values the LLM missed (e.g. in dense tables).
    When the LLM is intentionally unavailable, extraction degrades explicitly to
    the regex/structured-table pass and records a harness warning. Runtime LLM
    failures still raise so transient provider errors are not hidden.
    """
    if config is None:
        from littrace.config import load_config

        config = load_config()

    cells: list[PerformanceCell] = []
    artifacts: list[StructuredArtifact] = []
    all_schema_errors: list[str] = []
    llm_fallback_warnings: list[str] = []
    total_raw_items = 0

    for paper_id, parsed in workspace.parsed_papers.items():
        parsed = coerce_parsed(parsed)
        artifacts.extend(_structured_artifacts_from_parsed(paper_id, parsed))

        if not parsed.parsed:
            continue

        payload = _build_extraction_payload(paper_id, parsed)
        if not payload.strip():
            continue

        extraction_config = config.model_copy(deep=True)
        extraction_config.llm.request_timeout_seconds = max(
            config.llm.request_timeout_seconds,
            config.llm.metric_extraction_timeout_seconds,
        )
        with timed("llm_metric_extract", paper_id=paper_id):
            reply = await chat_completion(
                extraction_config,
                _EXTRACTION_SYSTEM_PROMPT,
                payload,
                workspace=None,
            )

        if not reply.used_llm and reply.error in {"missing_api_key", "llm_disabled"}:
            raw_cells = []
            llm_fallback_warnings.append(
                f"{paper_id}: LLM metric extraction skipped ({reply.error}); "
                "regex-only fallback used."
            )
        elif not reply.used_llm:
            raise RuntimeError(f"LLM metric extraction failed for paper {paper_id}: {reply.error}")
        else:
            try:
                raw_cells = json.loads(reply.text)
                if not isinstance(raw_cells, list):
                    raw_cells = []
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "llm_json_parse_error", extra={"paper_id": paper_id, "error": str(exc)}
                )
                raw_cells = []

        total_raw_items += len(raw_cells)
        # Dimension 3: schema-validated parsing with error collection
        llm_cells, schema_errors = _parse_llm_cells(paper_id, raw_cells, parsed)
        cells.extend(llm_cells)
        if schema_errors:
            all_schema_errors.extend(schema_errors)
            logger.warning(
                "schema_validation_errors",
                extra={
                    "paper_id": paper_id,
                    "error_count": len(schema_errors),
                    "errors": schema_errors[:5],
                },
            )

        # Secondary regex pass: catch values the LLM missed
        regex_cells = _cells_from_sections(paper_id, parsed)
        regex_cells.extend(_cells_from_tables(paper_id, parsed))
        existing_snippets = {c.evidence.snippet for c in llm_cells}
        missed = [c for c in regex_cells if c.evidence.snippet not in existing_snippets]
        if missed:
            logger.info(
                "regex_supplement",
                extra={
                    "paper_id": paper_id,
                    "llm_count": len(llm_cells),
                    "regex_extra": len(missed),
                },
            )
            cells.extend(missed)

    workspace.performance_cells = cells
    _store_structured_artifacts(workspace, artifacts)

    # Dimension 3: Run schema compliance harness if there were any LLM items
    schema_result = HarnessResult(passed=True, score=1.0)
    if total_raw_items > 0:
        schema_report = check_schema_compliance(
            [
                SchemaCheckItem(
                    source="tables.py:extract_performance_cells",
                    total_items=total_raw_items,
                    valid_items=total_raw_items - len(all_schema_errors),
                    invalid_items=all_schema_errors,
                    schema_name="PerformanceCell",
                )
            ]
        )
        if not schema_report.passed:
            logger.warning(
                "schema_harness_failed",
                extra={
                    "score": schema_report.score,
                    "errors": schema_report.errors[:5],
                },
            )
        schema_result = schema_report.to_result()

    harness = _combine_harnesses(
        performance=check_performance_cells(cells),
        artifacts=check_structured_artifacts(artifacts),
        schema=schema_result,
        artifact_count=len(artifacts),
    )
    harness.warnings.extend(llm_fallback_warnings)
    logger.info(
        "extraction_done",
        extra={
            "total_cells": len(cells),
            "total_artifacts": len(artifacts),
            "harness_score": harness.score,
            "schema_errors": len(all_schema_errors),
        },
    )
    return workspace, harness


def extract_structured_artifacts(
    workspace: LiteratureWorkspace,
) -> tuple[LiteratureWorkspace, HarnessResult]:
    artifacts: list[StructuredArtifact] = []
    for paper_id, parsed in workspace.parsed_papers.items():
        parsed = coerce_parsed(parsed)
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
        condition_complete = [cell for cell in normalized_cells if cell.conditions.complete]
        condition_profiles = {_condition_profile(cell.conditions) for cell in condition_complete}
        matrix_warnings: list[str] = []
        if len(units) > 1:
            matrix_warnings.append(f"Mixed units for {metric}: {sorted(units)}")
        if not datasets:
            matrix_warnings.append("Dataset is missing for all rows; comparison may be unfair.")
        if not tasks:
            matrix_warnings.append("Task is missing for all rows; comparison may be unfair.")
        elif len(tasks) > 1:
            matrix_warnings.append("Mixed tasks within this metric group; rows are not comparable.")
        if len(datasets) > 1:
            matrix_warnings.append(
                "Mixed datasets within this metric group; rows are not comparable."
            )
        if not condition_complete:
            matrix_warnings.append("Experimental conditions are incomplete for all rows.")
        elif len(condition_profiles) > 1:
            matrix_warnings.append(
                "Experimental conditions differ within this metric group; rows are not comparable."
            )

        rows = [
            _matrix_row(
                workspace,
                cell,
                units=units,
                has_dataset=bool(datasets),
                has_task=bool(tasks),
                has_complete_conditions=bool(condition_complete),
                mixed_datasets=len(datasets) > 1,
                mixed_tasks=len(tasks) > 1,
                mixed_conditions=len(condition_profiles) > 1,
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
    if cell.value_min is not None:
        update["value_min"] = normalize_metric_unit(cell.metric, cell.value_min, cell.unit)[0]
    if cell.value_max is not None:
        update["value_max"] = normalize_metric_unit(cell.metric, cell.value_max, cell.unit)[0]
    if cell.uncertainty is not None:
        update["uncertainty"] = normalize_metric_unit(cell.metric, cell.uncertainty, cell.unit)[0]
    evidence = dict(update["evidence"])
    snippet = evidence.get("snippet") or ""
    evidence["snippet"] = f"{snippet} [{warning}]".strip()
    update["evidence"] = evidence
    return PerformanceCell.model_validate(update)


def _cells_from_sections(paper_id: str, parsed: ParsedPaper) -> list[PerformanceCell]:
    cells: list[PerformanceCell] = []
    sections = parsed.sections or []
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
                    conditions=_conditions_from_text(snippet),
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


def _cells_from_tables(paper_id: str, parsed: ParsedPaper) -> list[PerformanceCell]:
    cells: list[PerformanceCell] = []
    tables = parsed.tables or []
    for table in tables:
        table_id = str(table.table_id or "")
        caption = str(table.caption or "")
        evidence = table.evidence
        for cell in table.cells or []:
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
                        value_max=float(match.group("value_max"))
                        if match.group("value_max")
                        else None,
                        uncertainty=float(match.group("uncertainty"))
                        if match.group("uncertainty")
                        else None,
                        unit=match.group("unit") or None,
                        conditions=_conditions_from_text(f"{caption} {text}"),
                        higher_is_better=METRIC_DIRECTIONS.get(metric),
                        evidence=EvidenceSpan(
                            paper_id=paper_id,
                            table_id=table_id,
                            row_label=str(cell.get("row") or "") or None,
                            column_label=str(cell.get("column") or "") or None,
                            snippet=_window(f"{caption} {text}", match.start(), match.end()),
                            parser=evidence.parser,
                            confidence=evidence.confidence or 0.7,
                        ),
                    )
                )
    return cells


def _structured_artifacts_from_parsed(
    paper_id: str,
    parsed: ParsedPaper,
) -> list[StructuredArtifact]:
    artifacts: list[StructuredArtifact] = []
    artifacts.extend(_artifacts_from_table_objects(paper_id, parsed))
    artifacts.extend(_artifacts_from_sections(paper_id, parsed))
    return artifacts


def _stored_structured_artifacts(workspace: LiteratureWorkspace) -> list[StructuredArtifact]:
    raw = getattr(workspace.context.filters, "structured_artifacts", [])
    if not isinstance(raw, list):
        return []
    artifacts: list[StructuredArtifact] = []
    for item in raw:
        if isinstance(item, StructuredArtifact):
            artifacts.append(item)
        elif isinstance(item, dict):
            artifacts.append(StructuredArtifact.model_validate(item))
    return artifacts


def _artifacts_from_table_objects(
    paper_id: str,
    parsed: ParsedPaper,
) -> list[StructuredArtifact]:
    artifacts: list[StructuredArtifact] = []
    for table in parsed.tables or []:
        evidence = table.evidence
        label = str(table.table_id or "") or None
        caption = str(table.caption or "")
        cells = table.cells or []
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
                    parser=evidence.parser,
                    confidence=evidence.confidence or 0.75,
                ),
                confidence=evidence.confidence or 0.75,
            )
        )
    return artifacts


def _artifacts_from_sections(paper_id: str, parsed: ParsedPaper) -> list[StructuredArtifact]:
    artifacts: list[StructuredArtifact] = []
    sections = parsed.sections or []
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
                    artifact_type="equation"
                    if label.lower().startswith(("eq", "equation"))
                    else "formula",
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
    workspace.context.filters.structured_artifacts = [
        artifact.model_dump(mode="json") for artifact in artifacts
    ]


def _combine_harnesses(
    performance: HarnessResult,
    artifacts: HarnessResult,
    schema: HarnessResult,
    artifact_count: int,
) -> HarnessResult:
    score = (performance.score + artifacts.score + schema.score) / 3
    warnings = [
        *performance.warnings,
        *artifacts.warnings,
        *schema.warnings,
    ]
    if artifact_count == 0:
        warnings.append("No table, figure, formula, or equation artifacts were extracted.")
    return HarnessResult(
        passed=performance.passed and artifacts.passed and schema.passed,
        score=score,
        errors=[*performance.errors, *artifacts.errors, *schema.errors],
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
    has_complete_conditions: bool,
    mixed_datasets: bool,
    mixed_tasks: bool,
    mixed_conditions: bool,
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
    if mixed_datasets:
        comparable = False
        warnings.append("Dataset differs within this metric group.")
    if has_task and not cell.task:
        comparable = False
        warnings.append("Task missing for this row.")
    if mixed_tasks:
        comparable = False
        warnings.append("Task differs within this metric group.")
    if has_complete_conditions and not cell.conditions.complete:
        comparable = False
        warnings.append("Experimental conditions are incomplete for this row.")
    if mixed_conditions:
        comparable = False
        warnings.append("Experimental conditions differ within this metric group.")
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
        conditions=cell.conditions,
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


def _conditions_from_text(text: str) -> ExperimentalConditions:
    lowered = text.lower()
    protocol = None
    if "cyclic" in lowered or "cycle" in lowered:
        protocol = "cyclic loading"
    elif "tensile" in lowered:
        protocol = "tensile test"
    environment = None
    if "room temperature" in lowered:
        environment = "room temperature"
    elif "humidity" in lowered:
        environment = "humidity-controlled"
    loading_match = re.search(r"\b\d+(?:\.\d+)?\s*(?:pa|kpa|mpa|% strain)\b", lowered)
    return ExperimentalConditions(
        test_protocol=protocol,
        environment=environment,
        loading_range=loading_match.group(0) if loading_match else None,
    )


def _condition_profile(
    conditions: ExperimentalConditions,
) -> tuple[str, str, str, str, str, int | None]:
    return (
        conditions.material_system or "",
        conditions.device_structure or "",
        conditions.test_protocol or "",
        conditions.environment or "",
        conditions.loading_range or "",
        conditions.sample_count,
    )


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


__all__ = [
    "ArtifactNeedReport",
    "build_comparison_matrices",
    "decide_artifact_extraction_need",
    "extract_performance_cells",
    "extract_structured_artifacts",
]