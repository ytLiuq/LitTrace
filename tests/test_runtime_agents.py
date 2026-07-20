import pytest

from littrace.config import APIConfig, LitTraceConfig
from littrace.models import LiteratureWorkspace, PaperSearchRequest
from littrace.runtime.agents import RetrievalAccessParsingAgent
from littrace.runtime.messages import AgentMessage


@pytest.mark.anyio
async def test_retrieval_agent_returns_react_trace_and_tool_results():
    agent = RetrievalAccessParsingAgent()
    message = AgentMessage(
        sender="test",
        receiver="retrieval_access_parsing",
        intent="search",
        payload=PaperSearchRequest(topic="MXene sensor", live=False).model_dump(),
    )

    result, workspace = await agent.run(
        message,
        LiteratureWorkspace(),
        LitTraceConfig(api=APIConfig(enable_live_search=False)),
    )

    assert result.status == "completed"
    assert result.react_trace is not None
    assert result.react_trace.steps
    assert "search_papers" in result.react_trace.allowed_tools
    assert result.react_trace.planned_actions == ["search_papers"]
    assert result.react_trace.stop_reason == "completed"
    assert result.react_trace.steps[-1].next_action == "finish"
    assert any(artifact.kind == "tool_result" for artifact in result.artifacts)
    assert workspace.context.active_papers
