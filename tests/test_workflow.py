import pytest

from littrace.config import LLMConfig, LitTraceConfig
from littrace.models import PaperSearchRequest
from littrace.workflow import run_research_graph


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
async def test_run_research_graph_can_run_autonomous_review():
    result = await run_research_graph(
        PaperSearchRequest(topic="MXene flexible sensor", live=False),
        LitTraceConfig(llm=LLMConfig(enabled=False)),
        audit_citations_enabled=False,
        plan_downloads_enabled=False,
        route_publishers_enabled=False,
        autonomous_review_enabled=True,
    )

    assert result.autonomous_loop_report is not None
    assert "autonomous_loop_report" in result.workspace.context.filters
