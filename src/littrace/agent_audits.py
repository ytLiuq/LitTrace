from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.harnesses import check_performance_cells, check_storyline_claims
from littrace.models import LiteratureWorkspace
from littrace.pdf_benchmark import benchmark_pdf_parsing
from littrace.storyline import build_storyline_from_workspace
from littrace.tables import build_comparison_matrices


class AgentAuditReport(BaseModel):
    agent: str
    passed: bool
    score: float
    findings: list[str] = Field(default_factory=list)


def audit_parser_agent(config: LitTraceConfig, workspace: LiteratureWorkspace) -> AgentAuditReport:
    report = benchmark_pdf_parsing(workspace, config)
    findings = list(report.warnings)
    score = 0.5 * report.local_pdf_rate + 0.5 * report.parsed_rate
    return AgentAuditReport(
        agent="PDF/OCR Parser",
        passed=score >= 0.6,
        score=round(score, 3),
        findings=findings,
    )


def audit_table_agent(workspace: LiteratureWorkspace) -> AgentAuditReport:
    harness = check_performance_cells(workspace.performance_cells)
    matrix = build_comparison_matrices(workspace)
    findings = [*harness.errors, *harness.warnings, *matrix.warnings]
    score = harness.score if workspace.performance_cells else 0.0
    return AgentAuditReport(
        agent="Table Extractor",
        passed=harness.passed and bool(workspace.performance_cells),
        score=round(score, 3),
        findings=findings,
    )


def audit_storyline_agent(workspace: LiteratureWorkspace) -> AgentAuditReport:
    claims = build_storyline_from_workspace(workspace)
    harness = check_storyline_claims(claims)
    return AgentAuditReport(
        agent="Storyline Verifier",
        passed=harness.passed and bool(claims),
        score=round(harness.score if claims else 0.0, 3),
        findings=[*harness.errors, *harness.warnings],
    )
