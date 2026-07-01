from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.models import LiteratureWorkspace
from littrace.storyline import build_storyline_from_workspace


class AgentHandoff(BaseModel):
    from_agent: str
    to_agent: str
    artifact: str
    required_inputs: list[str] = Field(default_factory=list)
    quality_gate: str
    status: str
    blocking_if_missing: bool = True
    notes: list[str] = Field(default_factory=list)


class AgentInteractionReport(BaseModel):
    handoffs: list[AgentHandoff]
    ready_count: int
    blocked_count: int
    complete_count: int
    recommended_next_agents: list[str] = Field(default_factory=list)


def build_agent_interaction_report(workspace: LiteratureWorkspace) -> AgentInteractionReport:
    has_routes = bool(workspace.context.filters.get("source_routes"))
    has_papers = bool(workspace.context.active_papers)
    has_download_selection = bool(workspace.context.selected_for_download)
    has_full_text_reports = bool(workspace.full_text_reports)
    has_parsed = bool(workspace.parsed_papers)
    has_cells = bool(workspace.performance_cells)
    has_storyline = bool(build_storyline_from_workspace(workspace))
    has_guard_reports = bool(workspace.guard_reports)
    has_document = bool(workspace.context.filters.get("document_report"))
    has_autonomous_loop = bool(workspace.context.filters.get("autonomous_loop_report"))

    handoffs = [
        AgentHandoff(
            from_agent="Research Planner",
            to_agent="Source Router",
            artifact="research plan and source priorities",
            required_inputs=["topic", "discipline", "recency preference"],
            quality_gate="Plan names source families and recency constraints before retrieval.",
            status="complete" if has_routes else "ready",
        ),
        AgentHandoff(
            from_agent="Source Router",
            to_agent="Search/Retrieval Agent",
            artifact="source routes and publisher search plan",
            required_inputs=["source route list", "publisher search URLs"],
            quality_gate="Retrieval preserves DOI/access URLs and deduplicates by DOI/title.",
            status="complete" if has_papers else ("ready" if has_routes else "blocked"),
            notes=[] if has_routes else ["Run source routing or a search preview first."],
        ),
        AgentHandoff(
            from_agent="Search/Retrieval Agent",
            to_agent="Citation Verifier",
            artifact="active literature context",
            required_inputs=["paper metadata", "DOI or source URL"],
            quality_gate="Every active paper has a citation string and best access URL.",
            status="ready" if has_papers else "blocked",
        ),
        AgentHandoff(
            from_agent="Citation Verifier",
            to_agent="FullText Resolver",
            artifact="citation records with access URLs",
            required_inputs=["active papers", "DOI", "access URLs"],
            quality_gate="Resolver should produce OA PDF/XML, publisher landing, or login-required candidates.",
            status="ready" if has_papers else "blocked",
        ),
        AgentHandoff(
            from_agent="FullText Resolver",
            to_agent="Access Manager",
            artifact="full-text candidate report",
            required_inputs=["full-text candidates", "selected download IDs"],
            quality_gate="Open PDFs may be downloaded; login-required papers use user handoff only.",
            status="complete" if has_full_text_reports else ("ready" if has_papers else "blocked"),
            notes=[] if has_download_selection else ["No papers selected for download yet."],
        ),
        AgentHandoff(
            from_agent="Access Manager",
            to_agent="PDF/OCR Parser",
            artifact="local PDFs or user-uploaded files",
            required_inputs=["session paper folder", "paper.pdf or attached PDF"],
            quality_gate="Parsed output includes parser name, confidence, and evidence spans.",
            status="complete" if has_parsed else ("ready" if has_papers else "blocked"),
            blocking_if_missing=False,
            notes=[] if has_parsed else ["Parser can fall back to metadata-only evidence."],
        ),
        AgentHandoff(
            from_agent="PDF/OCR Parser",
            to_agent="Table Extractor",
            artifact="parsed full text, tables, and page evidence",
            required_inputs=["parsed papers", "table candidates", "evidence spans"],
            quality_gate="Performance cells require metric, value, unit/comparability warning, and evidence.",
            status="complete" if has_cells else ("ready" if has_parsed else "blocked"),
        ),
        AgentHandoff(
            from_agent="Table Extractor",
            to_agent="Storyline Verifier",
            artifact="comparison matrices and material performance evidence",
            required_inputs=["active papers", "parsed snippets", "performance cells"],
            quality_gate="Narrative claims follow solution-limit-response and cite paper-level evidence.",
            status="complete" if has_storyline else ("ready" if has_papers else "blocked"),
            notes=[] if has_cells else ["Storyline can start from metadata, but table-backed claims are stronger."],
        ),
        AgentHandoff(
            from_agent="Storyline Verifier",
            to_agent="Research Writer",
            artifact="verified storyline claims",
            required_inputs=["storyline claims", "citation records", "comparison warnings"],
            quality_gate="Writer removes unsupported paper-specific sentences before replying.",
            status="ready" if has_storyline else "blocked",
        ),
        AgentHandoff(
            from_agent="Research Writer",
            to_agent="Citation Verifier",
            artifact="draft answer",
            required_inputs=["draft text", "active citations"],
            quality_gate="Final answer includes citation records and accessible links for literature claims.",
            status="complete" if has_guard_reports else ("ready" if has_papers else "blocked"),
        ),
        AgentHandoff(
            from_agent="Storyline Verifier",
            to_agent="Document Composer",
            artifact="storyline claims, matrices, structured artifacts, and citations",
            required_inputs=["active papers", "citation records", "quality metrics"],
            quality_gate="Report sections must cite active papers and surface harness warnings.",
            status="complete" if has_document else ("ready" if has_papers else "blocked"),
            blocking_if_missing=False,
            notes=[] if has_storyline else ["Document Composer can draft context reports, but full-text storyline evidence improves it."],
        ),
        AgentHandoff(
            from_agent="Document Composer",
            to_agent="Eval Auditor",
            artifact="auditable research report",
            required_inputs=["markdown report", "evidence spans", "citation records"],
            quality_gate="Report exposes citation/storyline warnings instead of hiding evidence gaps.",
            status="ready" if has_document else "blocked",
            blocking_if_missing=False,
        ),
        AgentHandoff(
            from_agent="Research Writer",
            to_agent="Autonomous Review Council",
            artifact="draft answer or research report",
            required_inputs=["draft text", "active workspace", "quality metrics"],
            quality_gate="Reviewer council must surface citation, storyline, and table objections before finalizing.",
            status="complete" if has_autonomous_loop else ("ready" if has_papers else "blocked"),
            blocking_if_missing=False,
        ),
        AgentHandoff(
            from_agent="Autonomous Review Council",
            to_agent="Research Planner",
            artifact="replan actions from failed review gates",
            required_inputs=["critique list", "failed harnesses", "workspace state"],
            quality_gate="Failed review rounds produce explicit recovery actions such as parse_full_text or extract_tables.",
            status="ready" if has_autonomous_loop else "blocked",
            blocking_if_missing=False,
        ),
        AgentHandoff(
            from_agent="Eval Auditor",
            to_agent="All Agents",
            artifact="quality report and regression metrics",
            required_inputs=["workspace", "parser benchmark", "table harness", "citation guard"],
            quality_gate="Known gaps are surfaced as optimization targets, not hidden as success.",
            status="ready",
            blocking_if_missing=False,
        ),
    ]
    ready_count = sum(1 for handoff in handoffs if handoff.status == "ready")
    blocked_count = sum(1 for handoff in handoffs if handoff.status == "blocked")
    complete_count = sum(1 for handoff in handoffs if handoff.status == "complete")
    return AgentInteractionReport(
        handoffs=handoffs,
        ready_count=ready_count,
        blocked_count=blocked_count,
        complete_count=complete_count,
        recommended_next_agents=_recommended_next_agents(handoffs),
    )


def _recommended_next_agents(handoffs: list[AgentHandoff]) -> list[str]:
    agents: list[str] = []
    for handoff in handoffs:
        if handoff.status == "ready" and handoff.to_agent not in agents:
            agents.append(handoff.to_agent)
    return agents[:4]
