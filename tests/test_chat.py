import pytest

from littrace.chat import handle_chat
from littrace.config import LitTraceConfig
from littrace.models import ChatRequest, LiteratureWorkspace


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
        LitTraceConfig(),
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
