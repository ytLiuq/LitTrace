from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.evaluation.harnesses import check_performance_cells, check_storyline_claims
from littrace.evaluation.pdf_benchmark import benchmark_pdf_parsing
from littrace.models import LiteratureWorkspace
from littrace.skill_runner import build_comparison_matrix_skill, build_storyline_skill


class QualityAuditReport(BaseModel):
    component: str
    passed: bool
    score: float
    findings: list[str] = Field(default_factory=list)


def audit_parser(config: LitTraceConfig, workspace: LiteratureWorkspace) -> QualityAuditReport:
    report = benchmark_pdf_parsing(workspace, config)
    score = 0.5 * report.local_pdf_rate + 0.5 * report.parsed_rate
    return QualityAuditReport(
        component="PDF/OCR parsing",
        passed=score >= 0.6,
        score=round(score, 3),
        findings=list(report.warnings),
    )


def audit_tables(workspace: LiteratureWorkspace) -> QualityAuditReport:
    harness = check_performance_cells(workspace.performance_cells)
    matrix = build_comparison_matrix_skill(workspace)
    return QualityAuditReport(
        component="table extraction",
        passed=harness.passed and bool(workspace.performance_cells),
        score=round(harness.score if workspace.performance_cells else 0.0, 3),
        findings=[*harness.errors, *harness.warnings, *matrix.warnings],
    )


def audit_storyline(workspace: LiteratureWorkspace) -> QualityAuditReport:
    claims = build_storyline_skill(workspace)
    harness = check_storyline_claims(claims)
    return QualityAuditReport(
        component="storyline evidence",
        passed=harness.passed and bool(claims),
        score=round(harness.score if claims else 0.0, 3),
        findings=[*harness.errors, *harness.warnings],
    )
