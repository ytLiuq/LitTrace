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


@pytest.mark.anyio
async def test_autonomous_loop_can_execute_safe_replan_actions(monkeypatch):
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Traceable Sensor Paper", year=2026)],
    )

    def fake_parse(workspace, config):
        workspace.parsed_papers["p1"] = {
            "sections": [
                {
                    "name": "Results",
                    "text": "Method improves sensitivity and discusses limitation.",
                    "evidence": {"page": 1, "parser": "fake"},
                }
            ],
            "parsed": True,
        }
        return workspace, {"parsed_count": 1, "metadata_only_count": 0}

    monkeypatch.setattr("littrace.autonomous_loop.parse_workspace_papers", fake_parse)

    report = await run_autonomous_research_loop(
        LitTraceConfig(llm=LLMConfig(enabled=False)),
        "请自动重规划并比较性能",
        workspace,
        auto_replan=True,
    )

    assert "parse_full_text_with_paddleocr" in report.executed_replan_actions
    assert workspace.parsed_papers
