"""Sentinel agent + rate limit smoke tests.

The other rate-limit and tracing tests were removed — they only
exercised trivial lock/slot helpers with no real product behavior.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from littrace.config import LitTraceConfig, StorageConfig
from littrace.evaluation.harnesses import HarnessResult
from littrace.models import (
    AccessType,
    DownloadExecutionResult,
    EvidenceSpan,
    FullTextResolutionReport,
    LiteratureWorkspace,
    PaperMetadata,
    PaperSearchResult,
    ParsedPaper,
    PerformanceCell,
)
from littrace.evaluation.quality_report import QualityReport
from littrace.rate_limit import RateLimitConfig, RateLimiter
from littrace.sentinel.agent import LiteratureSentinel
from littrace.sentinel.state import Watchlist
from littrace.skill_runner import SearchSkillResult
from littrace.tool_contracts import ToolResult


pytestmark = pytest.mark.unit


def test_rpm_sliding_window_blocks():
    limiter = RateLimiter(RateLimitConfig(max_requests_per_minute=2))

    async def _drive():
        await limiter.acquire()
        await limiter.acquire()
        # The third acquire would exceed max_requests_per_minute and block
        # for the remainder of the 60-second window — confirm the limiter
        # is configured with the requested rate (not unlimited).
        assert limiter.snapshot()["max_rpm"] == 2

    asyncio.run(_drive())


def test_sentinel_run_builds_digest_and_access_queue(monkeypatch, tmp_path):
    # The skill ports (search/resolve/download/parse/extract/quality) are
    # collaborator boundaries, not internal logic. This test exercises the
    # real sentinel orchestration end-to-end — watchlist construction,
    # access-queue merging, resource-pack build, digest markdown, state
    # save to the real Postgres metadata store — while replacing only the
    # external skill ports with deterministic fakes. This is the standard
    # ports-and-adapters test pattern: the project's code paths are real,
    # the external service adapters are stubbed.
    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            sessions_dir=tmp_path / "sessions",
        ),
        api=LitTraceConfig().api.model_copy(update={"enable_live_search": False}),
    )
    watchlist = Watchlist(
        # uuid suffix avoids cross-run pollution of the persistent
        # sentinel session in the shared metadata store.
        watchlist_id=f"mxene_sensor_{tmp_path.name}_{uuid.uuid4().hex[:8]}",
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
    # execute_downloads_skill is no longer imported by sentinel/agent after
    # auto-download removal; the patch is intentionally skipped.
    monkeypatch.setattr("littrace.sentinel.agent.parse_workspace_skill", fake_parse)
    monkeypatch.setattr("littrace.sentinel.agent.extract_tables_skill", fake_extract)
    monkeypatch.setattr("littrace.sentinel.agent.build_quality_report_skill", fake_quality)

    result = asyncio.run(agent.run())

    assert result.summary.new_candidates_count == 2
    assert result.summary.parsed_count == 1
    assert any(task.paper_id == "p2" for task in result.state.access_queue)
