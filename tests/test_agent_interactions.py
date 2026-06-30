from littrace.agent_interactions import build_agent_interaction_report
from littrace.context import add_papers
from littrace.models import EvidenceSpan, LiteratureWorkspace, PaperMetadata, PerformanceCell


def test_agent_interactions_recommend_retrieval_when_empty():
    report = build_agent_interaction_report(LiteratureWorkspace())

    assert report.handoffs
    assert "Source Router" in report.recommended_next_agents
    assert report.blocked_count > 0


def test_agent_interactions_progress_after_evidence_exists():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="MXene sensor", year=2026)],
    )
    workspace.context.filters["source_routes"] = ["OpenAlex", "Crossref"]
    workspace.parsed_papers["p1"] = {"sections": [{"text": "sensitivity 12 kPa-1"}]}
    workspace.performance_cells.append(
        PerformanceCell(
            paper_id="p1",
            metric="sensitivity",
            value=12.0,
            unit="kPa^-1",
            evidence=EvidenceSpan(paper_id="p1", section="Results", snippet="sensitivity 12 kPa-1"),
        )
    )

    report = build_agent_interaction_report(workspace)

    assert report.complete_count >= 3
    assert any(handoff.to_agent == "Storyline Verifier" for handoff in report.handoffs)
