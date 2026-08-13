from __future__ import annotations

from littrace.config import LitTraceConfig, StorageConfig
from littrace.models import (
    AccessType,
    EvidenceSpan,
    DownloadExecutionResult,
    FullTextResolutionReport,
    LiteratureWorkspace,
    PaperMetadata,
    PaperSearchResult,
    ParsedPaper,
    PerformanceCell,
)
from littrace.evaluation.quality_report import QualityReport
from littrace.sentinel.agent import LiteratureSentinel
from littrace.sentinel.state import AccessTask, Watchlist
from littrace.sentinel.storage import (
    load_access_queue,
    save_sentinel_state,
    save_sentinel_workspace,
)
from littrace.skill_runner import SearchSkillResult
from littrace.tool_contracts import ToolResult
from littrace.evaluation.harnesses import HarnessResult


def test_sentinel_run_builds_digest_and_access_queue(monkeypatch, tmp_path):
    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            sessions_dir=tmp_path / "sessions",
        ),
        api=LitTraceConfig().api.model_copy(update={"enable_live_search": False}),
    )
    watchlist = Watchlist(
        watchlist_id=f"mxene_sensor_{tmp_path.name}",
        topic="MXene flexible piezoresistive sensors",
        objective="monitor mxene sensors",
        query_variants=["MXene sensor"],
        year_min=2024,
    )
    agent = LiteratureSentinel(config, watchlist)

    papers = [
        PaperMetadata(
            paper_id="p1",
            title="Open access MXene sensor",
            year=2026,
            doi="10.1000/open",
            access_type=AccessType.OPEN_ACCESS,
            pdf_url="https://example.org/open.pdf",
        ),
        PaperMetadata(
            paper_id="p2",
            title="Login required MXene sensor",
            year=2026,
            doi="10.1000/login",
            access_type=AccessType.REQUIRES_LOGIN,
        ),
    ]

    async def fake_search(request, config, context=None):
        return SearchSkillResult(
            result=PaperSearchResult(request=request, papers=papers),
            diagnostics=None,
            use_live=False,
            tool_result=ToolResult(
                tool="search_papers",
                contract_id="search_papers:v1",
                ok=True,
                output=PaperSearchResult(request=request, papers=papers),
                started_at="2026-01-01T00:00:00",
                elapsed_ms=1.0,
            ),
        )

    async def fake_resolve(workspace, config, context=None):
        workspace.full_text_reports["p1"] = FullTextResolutionReport(
            paper_id="p1",
            best_pdf_url="https://example.org/open.pdf",
            best_landing_url=None,
            login_required_candidate_count=0,
        )
        workspace.full_text_reports["p2"] = FullTextResolutionReport(
            paper_id="p2",
            best_pdf_url=None,
            best_landing_url="https://example.org/login",
            login_required_candidate_count=1,
        )
        return workspace

    async def fake_download(config, workspace, request, context=None):
        return DownloadExecutionResult(
            items=[], downloaded_count=1, requires_login_count=0, skipped_count=0
        )

    async def fake_parse(workspace, config, context=None):
        workspace.parsed_papers["p1"] = ParsedPaper(
            parsed=True,
            title="Open access MXene sensor",
            structured_document={
                "schema": "littrace.docling.structured_document.v1",
                "markdown": "# Open",
            },
            sections=[{"name": "Intro", "text": "Body"}],
            parser_reports=[{"parser": "docling"}],
        )
        return workspace, {"parsed_count": 1, "warnings": []}

    async def fake_extract(workspace, config, context=None):
        workspace.performance_cells = [
            PerformanceCell(
                paper_id="p1",
                metric="gauge factor",
                value=12.0,
                evidence=EvidenceSpan(paper_id="p1", snippet="GF=12"),
            )
        ]
        return workspace, HarnessResult(passed=True, score=1.0, warnings=[])

    def fake_quality(config, workspace, context=None):
        return QualityReport(metrics={"overall_score": 0.9}, warnings=["minor"])

    monkeypatch.setattr("littrace.sentinel.agent.search_papers_skill", fake_search)
    monkeypatch.setattr("littrace.sentinel.agent.resolve_workspace_full_text_skill", fake_resolve)
    monkeypatch.setattr("littrace.sentinel.agent.execute_downloads_skill", fake_download)
    monkeypatch.setattr("littrace.sentinel.agent.parse_workspace_skill", fake_parse)
    monkeypatch.setattr("littrace.sentinel.agent.extract_tables_skill", fake_extract)
    monkeypatch.setattr("littrace.sentinel.agent.build_quality_report_skill", fake_quality)

    result = __import__("asyncio").run(agent.run())

    assert result.summary.new_candidates_count == 2
    assert result.summary.downloaded_count == 1
    assert result.summary.parsed_count == 1
    assert result.summary.access_task_count == 1
    assert result.digest_path is not None
    assert result.run_dir is not None
    assert result.resource_pack.missing_evidence == []
    assert result.state.access_queue[0].paper_id == "p2"
    assert (tmp_path / "sessions" / "sentinel" / watchlist.watchlist_id / "digests").exists()
    assert [task.paper_id for task in load_access_queue(result.store)] == ["p2"]
    assert (
        tmp_path
        / "sessions"
        / "sentinel"
        / watchlist.watchlist_id
        / "evidence_base"
        / "runs"
        / result.summary.run_id
        / "papers.jsonl"
    ).exists()
    workspace_dir = tmp_path / "sessions" / "sentinel" / watchlist.watchlist_id / "workspace"
    assert (workspace_dir / "evidence" / "spans.json").exists()
    assert (workspace_dir / "releases").exists()


def test_sentinel_resume_after_login_uses_access_queue(monkeypatch, tmp_path):
    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            sessions_dir=tmp_path / "sessions",
        ),
    )
    watchlist = Watchlist(
        watchlist_id="mxene_sensor",
        topic="MXene flexible piezoresistive sensors",
        objective="monitor mxene sensors",
    )
    agent = LiteratureSentinel(config, watchlist)
    workspace = LiteratureWorkspace()
    workspace.papers["p2"] = PaperMetadata(
        paper_id="p2",
        title="Login required MXene sensor",
        year=2026,
        doi="10.1000/login",
        access_type=AccessType.REQUIRES_LOGIN,
    )
    workspace.context.active_papers = ["p2"]
    state = agent.load_state()
    state.seen_paper_ids = ["p2"]
    state.access_queue = [
        AccessTask(
            paper_id="p2",
            title="Login required MXene sensor",
            doi="10.1000/login",
            reason="requires_institution_login",
        )
    ]
    save_sentinel_workspace(agent.store, workspace)
    save_sentinel_state(agent.store, state)

    async def fake_download(config, workspace, request, context=None):
        assert request.paper_ids == ["p2"]
        return DownloadExecutionResult(
            items=[], downloaded_count=1, requires_login_count=0, skipped_count=0
        )

    async def fake_parse(workspace, config, context=None):
        workspace.parsed_papers["p2"] = ParsedPaper(
            parsed=True,
            title="Login required MXene sensor",
            structured_document={
                "schema": "littrace.docling.structured_document.v1",
                "markdown": "# Login",
            },
            sections=[{"name": "Intro", "text": "Body"}],
            parser_reports=[{"parser": "docling"}],
        )
        return workspace, {"parsed_count": 1, "warnings": []}

    monkeypatch.setattr("littrace.sentinel.agent.execute_downloads_skill", fake_download)
    monkeypatch.setattr("littrace.sentinel.agent.parse_workspace_skill", fake_parse)

    result = __import__("asyncio").run(agent.resume_after_login())

    assert result.summary.downloaded_count == 1
    assert result.summary.parsed_count == 1
    assert result.summary.access_task_count == 0
    assert result.state.access_queue == []
    assert load_access_queue(agent.store) == []
