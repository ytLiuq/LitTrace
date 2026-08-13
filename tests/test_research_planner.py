from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.research_planner import build_research_plan


def test_research_plan_starts_with_retrieval_when_empty():
    plan = build_research_plan("MXene sensor", LiteratureWorkspace())

    assert plan.warnings
    assert plan.steps[0].component == "route_sources skill"
    assert plan.steps[0].next_component == "search_papers skill"
    assert plan.steps[0].quality_gate


def test_research_plan_includes_access_and_citation_when_context_exists():
    workspace = add_papers(LiteratureWorkspace(), [PaperMetadata(paper_id="p1", title="Paper")])

    plan = build_research_plan("MXene sensor", workspace)

    components = [step.component for step in plan.steps]
    assert "citation gate" in components
    assert "build_download_plan skill" in components
