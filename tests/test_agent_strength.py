from littrace.agent_strength import build_agent_portfolio_report
from littrace.config import LitTraceConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata


def test_agent_portfolio_reports_all_core_agents():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper", doi="10.1000/example")],
    )

    report = build_agent_portfolio_report(LitTraceConfig(), workspace)

    names = {agent.name for agent in report.agents}
    assert "Research Planner" in names
    assert "Research Writer" in names
    assert "Eval Auditor" in names
    assert report.average_score > 0
