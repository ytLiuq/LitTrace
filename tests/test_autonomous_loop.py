import pytest

from littrace.autonomous_loop import run_autonomous_research_loop
from littrace.config import LLMConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata


@pytest.mark.anyio
async def test_autonomous_loop_reports_empty_workspace():
    report = await run_autonomous_research_loop(
        LitTraceConfig(llm=LLMConfig(enabled=False)),
        "总结当前文献",
        LiteratureWorkspace(),
    )

    assert not report.passed
    assert "search_papers" in report.replan_actions


@pytest.mark.anyio
async def test_autonomous_loop_replans_when_full_text_missing():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Traceable Sensor Paper",
                year=2026,
                doi="10.1000/example",
            )
        ],
    )

    report = await run_autonomous_research_loop(
        LitTraceConfig(llm=LLMConfig(enabled=False)),
        "请比较性能并讲发展脉络",
        workspace,
    )

    assert report.rounds
    assert "parse_full_text_with_paddleocr" in report.replan_actions
    assert "多 agent 复核后的限制说明" in report.final_answer
