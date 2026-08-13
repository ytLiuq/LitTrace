from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.evidence.storyline import build_storyline_from_workspace
from littrace.models import LiteratureWorkspace


class WorkflowTransition(BaseModel):
    source: str
    target: str
    artifact: str
    required_inputs: list[str] = Field(default_factory=list)
    quality_gate: str
    status: str
    blocking_if_missing: bool = True
    notes: list[str] = Field(default_factory=list)


class WorkflowStatusReport(BaseModel):
    transitions: list[WorkflowTransition]
    ready_count: int
    blocked_count: int
    complete_count: int
    recommended_next_steps: list[str] = Field(default_factory=list)


def build_workflow_status(workspace: LiteratureWorkspace) -> WorkflowStatusReport:
    has_papers = bool(workspace.context.active_papers)
    has_full_text = bool(workspace.full_text_reports)
    has_parsed = bool(workspace.parsed_papers)
    has_cells = bool(workspace.performance_cells)
    has_storyline = bool(build_storyline_from_workspace(workspace))
    has_guard_reports = bool(workspace.guard_reports)
    has_document = bool(getattr(workspace.context.filters, "document_report", None))
    has_review = bool(getattr(workspace.context.filters, "autonomous_loop_report", None))

    transitions = [
        WorkflowTransition(
            source="LitTrace Coordinator",
            target="search_papers skill",
            artifact="ranked literature context",
            required_inputs=["topic", "source constraints"],
            quality_gate="Results preserve DOI/access provenance and are identity-resolved.",
            status="complete" if has_papers else "ready",
        ),
        WorkflowTransition(
            source="LitTrace Coordinator",
            target="full-text access skills",
            artifact="verified full-text candidates and download plan",
            required_inputs=["active papers", "explicit download selection"],
            quality_gate="Access policy separates open, login-required, and unavailable content.",
            status="complete" if has_full_text else ("ready" if has_papers else "blocked"),
        ),
        WorkflowTransition(
            source="LitTrace Coordinator",
            target="parse_workspace_papers skill",
            artifact="page-aware parsed documents",
            required_inputs=["authorized local PDFs"],
            quality_gate="Parsed evidence records parser, location, and confidence.",
            status="complete" if has_parsed else ("ready" if has_papers else "blocked"),
        ),
        WorkflowTransition(
            source="LitTrace Coordinator",
            target="extraction and synthesis skills",
            artifact="performance cells, storyline, and report",
            required_inputs=["parsed documents", "active literature context"],
            quality_gate="Synthesis uses traceable evidence and surfaces comparability warnings.",
            status=(
                "complete"
                if has_document or (has_cells and has_storyline)
                else ("ready" if has_parsed else "blocked")
            ),
        ),
        WorkflowTransition(
            source="LitTrace Coordinator",
            target="citation and evidence gates",
            artifact="publication decision and deterministic findings",
            required_inputs=["draft", "citations", "evidence spans"],
            quality_gate="Unsupported claims cannot be released.",
            status="complete" if has_guard_reports else ("ready" if has_papers else "blocked"),
        ),
        WorkflowTransition(
            source="LitTrace Coordinator",
            target="Optional Reviewer",
            artifact="read-only structured critique",
            required_inputs=["draft", "bounded evidence bundle"],
            quality_gate="Reviewer cannot search, mutate workspace, or publish; rounds are bounded.",
            status="complete" if has_review else ("ready" if has_document else "blocked"),
            blocking_if_missing=False,
            notes=["Optional for high-value reports; deterministic gates remain mandatory."],
        ),
    ]
    return WorkflowStatusReport(
        transitions=transitions,
        ready_count=sum(item.status == "ready" for item in transitions),
        blocked_count=sum(item.status == "blocked" for item in transitions),
        complete_count=sum(item.status == "complete" for item in transitions),
        recommended_next_steps=_recommended_next_steps(transitions),
    )


def _recommended_next_steps(transitions: list[WorkflowTransition]) -> list[str]:
    next_steps: list[str] = []
    for transition in transitions:
        if transition.status == "ready" and transition.target not in next_steps:
            next_steps.append(transition.target)
    return next_steps[:4]
