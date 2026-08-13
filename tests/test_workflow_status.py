from littrace.context import add_papers
from littrace.models import EvidenceSpan, LiteratureWorkspace, PaperMetadata, PerformanceCell
from littrace.workflow_status import build_workflow_status


def test_workflow_status_recommends_retrieval_when_empty():
    report = build_workflow_status(LiteratureWorkspace())

    assert report.transitions
    assert "search_papers skill" in report.recommended_next_steps
    assert report.blocked_count > 0


def test_workflow_status_progresses_after_evidence_exists():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="MXene sensor", year=2026)],
    )
    workspace.context.filters.source_routes = ["OpenAlex", "Crossref"]
    workspace.parsed_papers["p1"] = {"sections": [{"text": "sensitivity 12 kPa-1"}]}
    workspace.performance_cells.append(
        PerformanceCell(
            paper_id="p1",
            metric="sensitivity",
            value=12.0,
            unit="kPa^-1",
            evidence=EvidenceSpan(
                paper_id="p1",
                section="Results",
                snippet="sensitivity 12 kPa-1",
            ),
        )
    )

    report = build_workflow_status(workspace)

    assert report.complete_count >= 2
    assert any(
        transition.target == "extraction and synthesis skills"
        for transition in report.transitions
    )
