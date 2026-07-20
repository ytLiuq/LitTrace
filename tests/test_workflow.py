import pytest

from littrace.config import LLMConfig, LitTraceConfig
from littrace.models import PaperSearchRequest
from littrace.workflow import run_research_graph
from littrace.tool_contracts import ToolExecutionLedger


@pytest.mark.anyio
async def test_run_research_graph_returns_workspace_audit_and_download_plan():
    result = await run_research_graph(
        PaperSearchRequest(topic="MXene flexible sensor", live=False),
        LitTraceConfig(),
    )

    assert result.workspace.context.active_papers
    assert result.citation_audit is not None
    assert result.download_plan is not None
    assert result.publisher_routes is not None
    assert result.workflow_trace is not None
    assert [step.node for step in result.workflow_trace.steps[:2]] == [
        "plan_sources",
        "search_papers",
    ]
    assert "candidate_pool_count" in result.workflow_trace.steps[1].outputs
    assert "downloaded_full_text_count" in result.workflow_trace.steps[1].outputs
    assert "parsed_full_text_count" in result.workflow_trace.steps[1].outputs
    assert result.workflow_trace.steps[1].next_reason


@pytest.mark.anyio
async def test_run_research_graph_can_skip_optional_nodes():
    result = await run_research_graph(
        PaperSearchRequest(topic="MXene flexible sensor", live=False),
        LitTraceConfig(),
        audit_citations_enabled=False,
        plan_downloads_enabled=False,
    )

    assert result.workspace.context.active_papers
    assert result.citation_audit is None
    assert result.download_plan is None
    assert result.publisher_routes is not None
    assert result.workflow_trace is not None
    assert any(step.next_node == "route_publishers" for step in result.workflow_trace.steps)


@pytest.mark.anyio
async def test_run_research_graph_uses_supplied_task_ledger():
    ledger = ToolExecutionLedger()

    await run_research_graph(
        PaperSearchRequest(topic="MXene flexible sensor", live=False),
        LitTraceConfig(),
        audit_citations_enabled=False,
        plan_downloads_enabled=False,
        route_publishers_enabled=False,
        tool_ledger=ledger,
    )

    assert any(key.startswith("search_papers:v1:search_papers:") for key in ledger.cached_results)


@pytest.mark.anyio
async def test_run_research_graph_can_build_storyline_preview():
    result = await run_research_graph(
        PaperSearchRequest(topic="MXene flexible sensor", live=False),
        LitTraceConfig(),
        audit_citations_enabled=False,
        plan_downloads_enabled=False,
        parse_full_text_enabled=True,
        extract_tables_enabled=True,
        build_storyline_enabled=True,
    )

    assert result.storyline is not None
    assert result.parse_report is not None
    assert result.table_harness is not None
    assert result.comparison_matrix is not None
    assert result.workspace.parsed_papers


@pytest.mark.anyio
async def test_run_research_graph_can_run_autonomous_review(monkeypatch):
    from littrace.llm import LLMReply

    async def _fake_chat(config, system_prompt, user_message, workspace=None, **kwargs):
        return LLMReply(text="多 agent 审稿后的研究结论。", used_llm=True)

    async def _fake_write(config, question, workspace):
        return LLMReply(text="多 agent 审稿后的研究结论。", used_llm=True)

    monkeypatch.setattr("littrace.research_writer.chat_completion", _fake_chat)
    monkeypatch.setattr("littrace.autonomous_loop.chat_completion", _fake_chat)
    monkeypatch.setattr("littrace.research_writer.write_evidence_grounded_answer", _fake_write)
    monkeypatch.setattr("littrace.autonomous_loop.write_evidence_grounded_answer", _fake_write)

    result = await run_research_graph(
        PaperSearchRequest(topic="MXene flexible sensor", live=False),
        LitTraceConfig(llm=LLMConfig(enabled=True, api_key="fake-key")),
        audit_citations_enabled=False,
        plan_downloads_enabled=False,
        route_publishers_enabled=False,
        autonomous_review_enabled=True,
    )

    assert result.autonomous_loop_report is not None
    assert result.workspace.context.filters.autonomous_loop_report is not None
