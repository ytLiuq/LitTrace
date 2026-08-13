"""Runtime component status for the single Coordinator architecture."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RuntimeComponentStatus(BaseModel):
    name: str
    role_layer: str
    runtime: str
    implemented: bool
    workflow_node: str | None = None
    callable_tools: list[str] = Field(default_factory=list)
    remaining_work: list[str] = Field(default_factory=list)


def component_statuses() -> list[RuntimeComponentStatus]:
    return [
        RuntimeComponentStatus(
            name="LitTrace Coordinator",
            role_layer="primary agent",
            runtime="Single Coordinator + Skills",
            implemented=True,
            callable_tools=[
                "build_research_plan",
                "search_papers",
                "resolve_workspace_full_text",
                "build_download_plan",
                "parse_workspace_papers",
                "extract_performance_cells",
                "build_storyline_from_workspace",
                "build_research_document_report",
            ],
        ),
        RuntimeComponentStatus(
            name="Citation and Evidence Gates",
            role_layer="deterministic quality gates",
            runtime="Local validators",
            implemented=True,
            callable_tools=[
                "audit_citation_links",
                "guard_citations",
                "check_hallucination_grounding",
                "check_storyline_claims",
                "check_performance_cells",
                "evaluate_publication",
            ],
        ),
        RuntimeComponentStatus(
            name="Evaluation Gates",
            role_layer="deterministic quality gates",
            runtime="Local metrics and regression checks",
            implemented=True,
            callable_tools=["build_quality_report", "run_golden_eval"],
            remaining_work=["Expand the curated materials/chemistry golden set."],
        ),
        RuntimeComponentStatus(
            name="Optional Reviewer",
            role_layer="temporary restricted agent",
            runtime="Bounded read-only LLM review",
            implemented=True,
            workflow_node="optional_review",
            callable_tools=["review_evidence_bundle"],
        ),
    ]
