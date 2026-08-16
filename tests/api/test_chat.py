"""Chat API smoke test — exercises a real search flow with offline config."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.api

from littrace.chat import handle_chat
from littrace.config import LLMConfig, LitTraceConfig
from littrace.models import ChatRequest, LiteratureWorkspace


def _offline_config() -> LitTraceConfig:
    return LitTraceConfig(llm=LLMConfig(intent_parser_enabled=False))


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
