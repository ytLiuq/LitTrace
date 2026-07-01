import pytest

from littrace.chat import handle_chat
from littrace.config import LLMConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.models import ChatRequest, LiteratureWorkspace, PaperMetadata


@pytest.mark.anyio
async def test_chat_search_updates_workspace():
    response, workspace = await handle_chat(
        ChatRequest(message="检索 MXene flexible sensor 的最新论文", live=False),
        LiteratureWorkspace(),
        LitTraceConfig(),
    )

    assert response.action == "search"
    assert workspace.context.active_papers
    assert response.citations
    assert response.publisher_routes is not None


@pytest.mark.anyio
async def test_chat_show_and_hide_context():
    workspace = LiteratureWorkspace()
    response, workspace = await handle_chat(
        ChatRequest(message="隐藏上下文"),
        workspace,
        LitTraceConfig(),
    )
    assert response.action == "hide_context"
    assert not workspace.context.visible_to_user

    response, workspace = await handle_chat(
        ChatRequest(message="显示上下文"),
        workspace,
        LitTraceConfig(),
    )
    assert response.action == "show_context"
    assert workspace.context.visible_to_user


@pytest.mark.anyio
async def test_chat_help_for_unknown_intent():
    response, _ = await handle_chat(
        ChatRequest(message="你好"),
        LiteratureWorkspace(),
        LitTraceConfig(llm=LLMConfig(enabled=False)),
    )

    assert response.action == "help"


@pytest.mark.anyio
async def test_chat_composite_search_and_table():
    response, workspace = await handle_chat(
        ChatRequest(message="检索 2024 年后的 AFM 和 ACS Nano，先别下载，生成性能对比表", live=False),
        LiteratureWorkspace(),
        LitTraceConfig(),
    )

    assert response.action == "search"
    assert "已围绕" in response.reply
    assert response.comparison_matrix is not None
    assert "download" not in response.action
    assert workspace.context.filters["year_min"] == 2024


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
        LitTraceConfig(llm=LLMConfig(enabled=False)),
    )

    assert response.action == "select_downloads"
    assert workspace.context.selected_for_download == ["p1", "p2"]

    response, workspace = await handle_chat(
        ChatRequest(message="取消选择第 2 篇"),
        workspace,
        LitTraceConfig(llm=LLMConfig(enabled=False)),
    )

    assert workspace.context.selected_for_download == ["p1"]


@pytest.mark.anyio
async def test_chat_reports_agent_status():
    response, _ = await handle_chat(
        ChatRequest(message="agent状态"),
        LiteratureWorkspace(),
        LitTraceConfig(llm=LLMConfig(enabled=False)),
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
        LitTraceConfig(llm=LLMConfig(enabled=False)),
    )

    assert response.action == "autonomous_review"
    assert "多 agent 审稿" in response.reply
    assert "autonomous_loop_report" in workspace.context.filters
