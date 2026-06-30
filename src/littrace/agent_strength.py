from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.agents import agent_runtime_statuses
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.quality_report import build_quality_report


class AgentStrengthReport(BaseModel):
    name: str
    score: float
    level: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class AgentPortfolioReport(BaseModel):
    average_score: float
    agents: list[AgentStrengthReport]
    recommendations: list[str] = Field(default_factory=list)


BASE_TOOL_WEIGHTS = {
    "Source Router": 0.82,
    "Citation Verifier": 0.86,
    "Access Manager": 0.84,
    "Publisher Connector": 0.78,
    "PDF/OCR Parser": 0.72,
    "Table Extractor": 0.76,
    "Research Planner": 0.84,
    "Research Writer": 0.82,
    "Eval Auditor": 0.8,
    "Storyline Verifier": 0.78,
}


def build_agent_portfolio_report(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
) -> AgentPortfolioReport:
    quality = build_quality_report(config, workspace)
    agents = [
        _agent_strength(status.name, quality.metrics, bool(status.remaining_work))
        for status in agent_runtime_statuses()
    ]
    average = sum(agent.score for agent in agents) / max(len(agents), 1)
    recommendations = []
    if quality.metrics.get("local_pdf_rate", 0.0) < 0.5 and quality.metrics.get("active_paper_count", 0):
        recommendations.append("Increase local PDF coverage to strengthen Parser/Table/Storyline agents.")
    if quality.metrics.get("citation_guard_pass", 0.0) < 1.0:
        recommendations.append("Resolve citation guard warnings before treating narrative output as final.")
    if quality.metrics.get("comparison_matrix_count", 0.0) == 0 and quality.metrics.get("active_paper_count", 0):
        recommendations.append("Run parsing and table extraction to strengthen the Table Extractor.")
    return AgentPortfolioReport(
        average_score=round(average, 3),
        agents=agents,
        recommendations=recommendations,
    )


def _agent_strength(
    name: str,
    metrics: dict[str, float],
    has_remaining_work: bool,
) -> AgentStrengthReport:
    score = BASE_TOOL_WEIGHTS.get(name, 0.7)
    strengths: list[str] = ["Has executable tools and tests."]
    gaps: list[str] = []

    if name in {"PDF/OCR Parser", "Table Extractor", "Storyline Verifier"}:
        parsed_rate = metrics.get("parsed_rate", 0.0)
        score += 0.12 * parsed_rate
        if parsed_rate == 0:
            gaps.append("Needs local PDFs and parsing output for stronger real-paper performance.")
    if name == "Table Extractor":
        matrix_count = metrics.get("comparison_matrix_count", 0.0)
        score += 0.08 if matrix_count else 0.0
        if not matrix_count:
            gaps.append("No comparison matrix has been generated in the current workspace.")
    if name == "Storyline Verifier":
        claim_count = metrics.get("storyline_claim_count", 0.0)
        score += min(claim_count, 3.0) * 0.025
        if not claim_count:
            gaps.append("No storyline claims available in the current workspace.")
    if name == "Citation Verifier":
        score += 0.08 * metrics.get("citation_guard_pass", 0.0)
    if name == "Access Manager":
        score += 0.06 * metrics.get("local_pdf_rate", 0.0)
    if name == "Eval Auditor":
        score += 0.05 if metrics else 0.0
    if name == "Research Planner":
        score += 0.04 if metrics.get("active_paper_count", 0.0) else 0.0
    if has_remaining_work:
        score -= 0.03
    score = max(0.0, min(score, 0.96))
    level = "strong" if score >= 0.82 else "solid" if score >= 0.7 else "developing"
    if level == "strong":
        strengths.append("Meets the current strong-agent threshold.")
    return AgentStrengthReport(
        name=name,
        score=round(score, 3),
        level=level,
        strengths=strengths,
        gaps=gaps,
    )
