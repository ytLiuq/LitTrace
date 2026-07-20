import pytest

from littrace.config import LitTraceConfig, StorageConfig
from littrace.context import add_papers
from littrace.models import AccessType, LiteratureWorkspace, PaperMetadata
from littrace.session import create_chat_session
from littrace.skill_runner import (
    build_comparison_matrix_skill,
    build_download_plan_skill,
    build_quality_report_skill,
    build_storyline_skill,
    export_session_bundle_skill,
)
from littrace.tool_contracts import ToolResult
from littrace.models import PaperSearchRequest
from littrace.skill_runner import search_papers_skill
from littrace.tool_contracts import ToolExecutionLedger


@pytest.mark.anyio
async def test_build_download_plan_skill_wraps_contract_runner():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Open paper",
                access_type=AccessType.OPEN_ACCESS,
            )
        ],
    )

    plan = await build_download_plan_skill(LitTraceConfig(), workspace)

    assert plan.items[0].paper_id == "p1"
    assert plan.downloadable_count >= 1


@pytest.mark.anyio
async def test_export_session_bundle_skill_wraps_contract_runner(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026)],
    )

    paths = await export_session_bundle_skill(session, workspace, config)

    assert "research_report_draft" in paths
    assert paths["release_ready"] == "false"
    assert "quality" in paths
    assert all(paths.values())


@pytest.mark.anyio
async def test_search_skill_reuses_task_ledger_result():
    ledger = ToolExecutionLedger()
    request = PaperSearchRequest(topic="MXene", live=False)

    first = await search_papers_skill(
        request,
        LitTraceConfig(),
        ledger=ledger,
        idempotency_key="same-task-search",
    )
    second = await search_papers_skill(
        request,
        LitTraceConfig(),
        ledger=ledger,
        idempotency_key="same-task-search",
    )

    assert first.tool_result.ok
    assert second.tool_result.metadata["idempotency_reused"] is True


def test_sync_synthesis_skills_wrap_contract_runner(monkeypatch):
    from littrace import skill_runner

    calls: list[str] = []

    def fake_run_sync_tool(contract, func, payload, **kwargs):
        calls.append(contract.name)
        return ToolResult(
            tool=contract.name,
            contract_id=contract.contract_id,
            ok=True,
            output=f"wrapped:{contract.name}",
            started_at="2026-01-01T00:00:00",
            elapsed_ms=0,
        )

    monkeypatch.setattr(skill_runner, "run_sync_tool", fake_run_sync_tool)
    workspace = LiteratureWorkspace()

    assert build_storyline_skill(workspace) == "wrapped:build_storyline_from_workspace"
    assert build_quality_report_skill(LitTraceConfig(), workspace) == "wrapped:quality_report"
    assert build_comparison_matrix_skill(workspace) == "wrapped:build_comparison_matrices"
    assert calls == [
        "build_storyline_from_workspace",
        "quality_report",
        "build_comparison_matrices",
    ]
