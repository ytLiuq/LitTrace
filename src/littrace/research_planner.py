from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.models import LiteratureWorkspace


class ResearchPlanStep(BaseModel):
    agent: str
    action: str
    rationale: str
    expected_output: str
    handoff_to: str | None = None
    requires: list[str] = Field(default_factory=list)
    quality_gate: str | None = None


class ResearchPlan(BaseModel):
    topic: str
    steps: list[ResearchPlanStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def build_research_plan(topic: str, workspace: LiteratureWorkspace) -> ResearchPlan:
    active_count = len(workspace.context.active_papers)
    has_pdfs = bool(workspace.parsed_papers)
    has_tables = bool(workspace.performance_cells)
    steps = [
        ResearchPlanStep(
            agent="Source Router",
            action="Route sources and build publisher search plan",
            rationale="Materials/chemistry topics need scholarly graph and publisher-native coverage.",
            expected_output="Source route list and publisher search URLs.",
            handoff_to="Search/Retrieval Agent",
            requires=["topic", "discipline", "recency preference"],
            quality_gate="Plan includes publisher routes and recency constraints.",
        ),
        ResearchPlanStep(
            agent="Search/Retrieval Agent",
            action="Retrieve recent papers and merge duplicates",
            rationale="The user prefers recent literature and DOI-level traceability.",
            expected_output="Deduplicated active literature context.",
            handoff_to="Citation Verifier",
            requires=["source route list", "publisher search URLs"],
            quality_gate="Every retained paper keeps DOI or source URL metadata.",
        ),
    ]
    if active_count:
        steps.append(
            ResearchPlanStep(
                agent="Citation Verifier",
                action="Audit citations and access links",
                rationale="Every paper-specific answer needs citation text and a usable access URL.",
                expected_output="Citation audit with cached link status.",
                handoff_to="Access Manager",
                requires=["active papers", "DOI or source URL"],
                quality_gate="Citation records include access URLs and link status.",
            )
        )
        steps.append(
            ResearchPlanStep(
                agent="Access Manager",
                action="Plan downloads and resume local PDFs",
                rationale="Gated papers require authorized user login while OA PDFs can be downloaded.",
                expected_output="Download/resume report and local PDF readiness.",
                handoff_to="PDF/OCR Parser",
                requires=["citation records", "selected download IDs"],
                quality_gate="No gated content is bypassed; login handoff is explicit.",
            )
        )
    if active_count and not has_pdfs:
        steps.append(
            ResearchPlanStep(
                agent="PDF/OCR Parser",
                action="Parse local PDFs or request attachments",
                rationale="Storylines and performance tables need page-aware evidence.",
                expected_output="Parsed sections, tables, and evidence spans.",
                handoff_to="Table Extractor",
                requires=["local PDFs", "attached PDFs", "paper metadata"],
                quality_gate="Parsed evidence includes parser, page/section, and confidence.",
            )
        )
    if has_pdfs and not has_tables:
        steps.append(
            ResearchPlanStep(
                agent="Table Extractor",
                action="Extract and normalize performance metrics",
                rationale="Materials comparison requires units, ranges, uncertainty, and provenance.",
                expected_output="Comparison matrices with warnings.",
                handoff_to="Storyline Verifier",
                requires=["parsed papers", "table candidates", "evidence spans"],
                quality_gate="Cells include metric, value, unit or comparability warning, and evidence.",
            )
        )
    steps.append(
        ResearchPlanStep(
            agent="Storyline Verifier",
            action="Build and review solution-limit-response chain",
            rationale="Narratives must be grounded in paper-level evidence, not broad claims.",
            expected_output="Structured storyline report and reviewer warnings.",
            handoff_to="Research Writer",
            requires=["active papers", "parsed evidence", "comparison matrices"],
            quality_gate="Claims are constrained to evidence-backed solution-limit-response links.",
        )
    )
    warnings = []
    if not active_count:
        warnings.append("No active papers yet; start with retrieval.")
    return ResearchPlan(topic=topic, steps=steps, warnings=warnings)
