from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.research_planner import build_research_plan


def test_research_plan_starts_with_retrieval_when_empty():
    plan = build_research_plan("MXene sensor", LiteratureWorkspace())

    assert plan.warnings
    assert plan.steps[0].agent == "Source Router"


def test_research_plan_includes_access_and_citation_when_context_exists():
    workspace = add_papers(LiteratureWorkspace(), [PaperMetadata(paper_id="p1", title="Paper")])

    plan = build_research_plan("MXene sensor", workspace)

    agents = [step.agent for step in plan.steps]
    assert "Citation Verifier" in agents
    assert "Access Manager" in agents
