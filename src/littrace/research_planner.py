from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.models import LiteratureWorkspace


class ResearchPlanStep(BaseModel):
    component: str
    action: str
    rationale: str
    expected_output: str
    next_component: str | None = None
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
            component="route_sources skill",
            action="Route sources and build publisher search plan",
            rationale="Materials/chemistry topics need scholarly graph and publisher-native coverage.",
            expected_output="Source route list and publisher search URLs.",
            next_component="search_papers skill",
            requires=["topic", "discipline", "recency preference"],
            quality_gate="Plan includes publisher routes and recency constraints.",
        ),
        ResearchPlanStep(
            component="search_papers skill",
            action="Retrieve recent papers and merge duplicates",
            rationale="The user prefers recent literature and DOI-level traceability.",
            expected_output="Deduplicated active literature context.",
            next_component="citation gate",
            requires=["source route list", "publisher search URLs"],
            quality_gate="Every retained paper keeps DOI or source URL metadata.",
        ),
    ]
    if active_count:
        steps.append(
            ResearchPlanStep(
                component="citation gate",
                action="Audit citations and access links",
                rationale="Every paper-specific answer needs citation text and a usable access URL.",
                expected_output="Citation audit with cached link status.",
                next_component="resolve_workspace_full_text skill",
                requires=["active papers", "DOI or source URL"],
                quality_gate="Citation records include access URLs and link status.",
            )
        )
        steps.append(
            ResearchPlanStep(
                component="resolve_workspace_full_text skill",
                action="Resolve DOI-level PDF, XML, landing-page, and login-required candidates",
                rationale="Full text should be resolved from DOI metadata, OA locations, and publisher links before download.",
                expected_output="Full-text candidate reports with best OA PDF or authorized landing page.",
                next_component="build_download_plan skill",
                requires=["active papers", "DOI", "citation records"],
                quality_gate="Each paper has OA candidates, a publisher landing page, or login-required handoff.",
            )
        )
        steps.append(
            ResearchPlanStep(
                component="build_download_plan skill",
                action="Plan downloads and resume local PDFs",
                rationale="Gated papers require authorized user login while OA PDFs can be downloaded.",
                expected_output="Download/resume report and local PDF readiness.",
                next_component="parse_workspace_papers skill",
                requires=["citation records", "selected download IDs"],
                quality_gate="No gated content is bypassed; login handoff is explicit.",
            )
        )
    if active_count and not has_pdfs:
        steps.append(
            ResearchPlanStep(
                component="parse_workspace_papers skill",
                action="Parse local PDFs or request attachments",
                rationale="Storylines and performance tables need page-aware evidence.",
                expected_output="Parsed sections, tables, and evidence spans.",
                next_component="extract_performance_cells skill",
                requires=["local PDFs", "attached PDFs", "paper metadata"],
                quality_gate="Parsed evidence includes parser, page/section, and confidence.",
            )
        )
    if has_pdfs and not has_tables:
        steps.append(
            ResearchPlanStep(
                component="extract_performance_cells skill",
                action="Extract and normalize performance metrics",
                rationale="Materials comparison requires units, ranges, uncertainty, and provenance.",
                expected_output="Comparison matrices with warnings.",
                next_component="storyline gate",
                requires=["parsed papers", "table candidates", "evidence spans"],
                quality_gate="Cells include metric, value, unit or comparability warning, and evidence.",
            )
        )
    steps.append(
        ResearchPlanStep(
            component="storyline gate",
            action="Build and review solution-limit-response chain",
            rationale="Narratives must be grounded in paper-level evidence, not broad claims.",
            expected_output="Structured storyline report and reviewer warnings.",
            next_component="evidence-grounded writing skill",
            requires=["active papers", "parsed evidence", "comparison matrices"],
            quality_gate="Claims are constrained to evidence-backed solution-limit-response links.",
        )
    )
    warnings = []
    if not active_count:
        warnings.append("No active papers yet; start with retrieval.")
    return ResearchPlan(topic=topic, steps=steps, warnings=warnings)
