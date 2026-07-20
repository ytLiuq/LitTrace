import pytest

from littrace.autonomous_loop import run_autonomous_research_loop
from littrace.config import LLMConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.llm import LLMReply
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
async def test_autonomous_loop_raises_when_llm_disabled_with_papers():
    """LLM disabled → RuntimeError (no degradation)."""
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

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await run_autonomous_research_loop(
            LitTraceConfig(llm=LLMConfig(enabled=False)),
            "请比较性能并讲发展脉络",
            workspace,
        )


@pytest.mark.anyio
async def test_autonomous_loop_raises_when_llm_disabled_even_with_parsed(monkeypatch):
    """LLM disabled → RuntimeError even when papers are parsed (no degradation)."""
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Traceable Sensor Paper", year=2026)],
    )

    async def fake_parse(workspace, config):
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
        return workspace, {"parsed_count": 1, "failed_count": 0}

    monkeypatch.setattr("littrace.autonomous_loop.parse_workspace_skill", fake_parse)

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await run_autonomous_research_loop(
            LitTraceConfig(llm=LLMConfig(enabled=False)),
            "请自动重规划并比较性能",
            workspace,
            auto_replan=True,
        )


@pytest.mark.anyio
async def test_autonomous_loop_rechecks_publication_gate_before_final_answer(monkeypatch):
    workspace = add_papers(
        LiteratureWorkspace(
            parsed_papers={
                "p1": {
                    "parsed": True,
                    "sections": [{"name": "Results", "text": "Full text evidence."}],
                }
            }
        ),
        [PaperMetadata(paper_id="p1", title="Traceable Paper", year=2026)],
    )

    async def fake_writer(*args, **kwargs):
        return LLMReply(text="修订前的研究结论。", used_llm=True)

    monkeypatch.setattr("littrace.autonomous_loop.write_evidence_grounded_answer", fake_writer)
    report = await run_autonomous_research_loop(
        LitTraceConfig(llm=LLMConfig(enabled=True, api_key="test-key")),
        "总结当前文献",
        workspace,
        enable_smart_debate=False,
    )

    assert not report.release_ready
    assert not report.passed
    assert report.release_blockers
    assert "修订前的研究结论。" not in report.final_answer
    assert "未通过最终发布检查" in report.final_answer
