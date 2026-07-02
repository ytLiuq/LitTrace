import pytest

from littrace.chat import _format_search_result_reply, handle_chat
from littrace.config import LLMConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.models import ChatRequest, LiteratureWorkspace, PaperMetadata


def _offline_config() -> LitTraceConfig:
    return LitTraceConfig(llm=LLMConfig(intent_parser_enabled=False))


@pytest.mark.anyio
async def test_chat_reports_intent_parser_missing_key_without_fallback():
    response, workspace = await handle_chat(
        ChatRequest(message="帮我找几篇柔性压力传感器论文"),
        LiteratureWorkspace(),
        LitTraceConfig(llm=LLMConfig(api_key=None, enabled=True, intent_parser_enabled=True)),
    )

    assert response.action == "intent_parse_error"
    assert "没有配置 LLM API key" in response.reply
    assert not workspace.context.active_papers


@pytest.mark.anyio
async def test_chat_search_updates_workspace():
    response, workspace = await handle_chat(
        ChatRequest(message="检索 MXene flexible sensor 的最新论文", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    assert response.action == "search"
    assert workspace.context.active_papers
    assert response.citations
    assert response.publisher_routes is not None
    assert response.research_result is not None
    assert response.research_result.workflow_trace is not None
    assert any(
        step.node == "minimum_evidence_gate"
        for step in response.research_result.workflow_trace.steps
    )


@pytest.mark.anyio
async def test_chat_research_request_runs_search_instead_of_help():
    response, workspace = await handle_chat(
        ChatRequest(message="我想了解一下薄膜压敏传感阵列的相关文献，请帮我调研一下", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    assert response.action == "search"
    assert workspace.context.active_papers
    assert "mock/开发样例检索" in response.reply


def test_search_reply_mentions_expanded_year_range():
    workspace = LiteratureWorkspace()
    workspace.context.filters["search_mode"] = "live"
    reply = _format_search_result_reply(
        "rare topic",
        workspace,
        expanded_year_range=True,
        original_year_min=2024,
    )

    assert "已自动扩大到不限年份" in reply


def test_search_reply_refuses_analysis_below_five_papers():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id=f"p{i}", title=f"Paper {i}") for i in range(4)],
    )
    workspace.context.filters["search_mode"] = "live"

    reply = _format_search_result_reply("rare topic", workspace)

    assert "少于最低分析门槛 5 篇" in reply
    assert "我先不做分析型结论" in reply


def test_search_reply_lists_all_results_above_five():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id=f"p{i}", title=f"Paper {i}") for i in range(7)],
    )
    workspace.context.filters["search_mode"] = "live"

    reply = _format_search_result_reply("topic", workspace)

    assert "已找到 7 篇真实候选文献" in reply
    assert "7. Paper 6" in reply


@pytest.mark.anyio
async def test_chat_show_and_hide_context():
    workspace = LiteratureWorkspace()
    response, workspace = await handle_chat(
        ChatRequest(message="隐藏上下文"),
        workspace,
        _offline_config(),
    )
    assert response.action == "hide_context"
    assert not workspace.context.visible_to_user

    response, workspace = await handle_chat(
        ChatRequest(message="显示上下文"),
        workspace,
        _offline_config(),
    )
    assert response.action == "show_context"
    assert workspace.context.visible_to_user


@pytest.mark.anyio
async def test_chat_help_for_unknown_intent():
    response, _ = await handle_chat(
        ChatRequest(message="你好"),
        LiteratureWorkspace(),
        LitTraceConfig(llm=LLMConfig(enabled=False, intent_parser_enabled=False)),
    )

    assert response.action == "help"


@pytest.mark.anyio
async def test_chat_composite_search_and_table():
    response, workspace = await handle_chat(
        ChatRequest(message="检索 2024 年后的 AFM 和 ACS Nano，先别下载，生成性能对比表", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    assert response.action == "search"
    assert "我先按" in response.reply
    assert response.comparison_matrix is None
    assert any("不足 5 篇" in warning for warning in response.warnings)
    assert "download" not in response.action
    assert workspace.context.filters["year_min"] is None
    assert workspace.context.filters["expanded_year_range_from"] == 2024


@pytest.mark.anyio
async def test_chat_mock_search_does_not_pass_minimum_evidence_gate():
    response, _ = await handle_chat(
        ChatRequest(message="检索 碳基PDMS柔性薄膜传感器长时间受压漂移", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    assert "mock/开发样例检索" in response.reply
    gate = [
        step
        for step in response.research_result.workflow_trace.steps
        if step.node == "minimum_evidence_gate"
    ][-1]
    assert gate.status == "blocked"
    assert gate.outputs["real_search"] is False


@pytest.mark.anyio
async def test_chat_refuses_to_summarize_mock_context_routes():
    _, workspace = await handle_chat(
        ChatRequest(message="检索 碳基PDMS柔性薄膜传感器长时间受压漂移", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    response, _ = await handle_chat(
        ChatRequest(message="请总结这些论文的主要路线"),
        workspace,
        LitTraceConfig(llm=LLMConfig(enabled=False, intent_parser_enabled=False)),
    )

    assert "mock/开发样例" in response.reply
    assert "不能基于这些内容总结研究路线" in response.reply
    assert "mock202" not in response.reply


@pytest.mark.anyio
async def test_chat_trace_starts_with_intent_parsing():
    response, _ = await handle_chat(
        ChatRequest(message="检索 carbon PDMS pressure sensor drift", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    first = response.research_result.workflow_trace.steps[0]
    assert first.node == "parse_user_intent"
    assert first.outputs["minimum_required"] == 5
    assert first.outputs["query_variant_count"] >= 1


@pytest.mark.anyio
async def test_chat_can_select_downloads_by_index():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(paper_id="p1", title="First"),
            PaperMetadata(paper_id="p2", title="Second"),
        ],
    )

    response, workspace = await handle_chat(
        ChatRequest(message="选择第 1、2 篇下载"),
        workspace,
        LitTraceConfig(llm=LLMConfig(enabled=False, intent_parser_enabled=False)),
    )

    assert response.action == "select_downloads"
    assert workspace.context.selected_for_download == ["p1", "p2"]

    response, workspace = await handle_chat(
        ChatRequest(message="取消选择第 2 篇"),
        workspace,
        LitTraceConfig(llm=LLMConfig(enabled=False, intent_parser_enabled=False)),
    )

    assert workspace.context.selected_for_download == ["p1"]


@pytest.mark.anyio
async def test_chat_reports_agent_status():
    response, _ = await handle_chat(
        ChatRequest(message="agent状态"),
        LiteratureWorkspace(),
        LitTraceConfig(llm=LLMConfig(enabled=False, intent_parser_enabled=False)),
    )

    assert response.action == "agent_status"
    assert "Publisher Connector" in response.reply


@pytest.mark.anyio
async def test_chat_runs_autonomous_review_loop():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Traceable Paper", year=2026, doi="10.1000/example")],
    )

    response, workspace = await handle_chat(
        ChatRequest(message="请多轮反驳并修订当前结论"),
        workspace,
        LitTraceConfig(llm=LLMConfig(enabled=False, intent_parser_enabled=False)),
    )

    assert response.action == "autonomous_review"
    assert "多 agent 审稿" in response.reply
    assert "autonomous_loop_report" in workspace.context.filters
