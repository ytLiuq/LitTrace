from __future__ import annotations

import asyncio
from types import SimpleNamespace

from littrace.models import (
    AccessType,
    DownloadExecutionItem,
    DownloadExecutionResult,
    LiteratureWorkspace,
    PaperMetadata,
    PaperSearchRequest,
    PaperSearchResult,
)
from littrace.topic_search import run_topic_search


def test_topic_search_counts_rag_ready_and_keeps_all_candidates(monkeypatch, tmp_path):
    from littrace.config import LitTraceConfig, StorageConfig
    from littrace.skills._helpers import SearchSkillResult

    config = LitTraceConfig(
        storage=StorageConfig(
            sessions_dir=tmp_path / "sessions",
            paper_library_dir=tmp_path / "papers",
        )
    )
    session = SimpleNamespace(session_id="topic-test")
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="MXene pressure sensor one",
            doi="10.1000/p1",
            access_type=AccessType.OPEN_ACCESS,
        ),
        PaperMetadata(
            paper_id="p2",
            title="MXene pressure sensor two",
            doi="10.1000/p2",
            access_type=AccessType.REQUIRES_LOGIN,
        ),
    ]
    request = PaperSearchRequest(topic="MXene pressure sensor", limit=10, live=False)
    holder = {"workspace": LiteratureWorkspace()}

    async def fake_search(request, config):
        return SearchSkillResult(
            result=PaperSearchResult(request=request, papers=papers),
            diagnostics=None,
            use_live=False,
            tool_result=SimpleNamespace(ok=True),
        )

    async def fake_download(config, papers, request):
        return DownloadExecutionResult(
            items=[
                DownloadExecutionItem(
                    paper_id=paper.paper_id,
                    action="download",
                    status="downloaded",
                )
                for paper in papers
            ],
            downloaded_count=2,
            requires_login_count=0,
            skipped_count=0,
        )

    async def fake_parse(config, limit, **kwargs):
        return SimpleNamespace(parsed=2, warnings=[])

    async def fake_embed(config, limit, **kwargs):
        return SimpleNamespace(
            processed=2,
            ready_paper_ids=["p1", "p2"],
            warnings=[],
        )

    def fake_save(session, workspace, config=None):
        holder["workspace"] = workspace
        workspace.context.filters.workspace_revision += 1

    monkeypatch.setattr("littrace.topic_search.search_papers_skill", fake_search)
    monkeypatch.setattr("littrace.topic_search.execute_downloads", fake_download)
    monkeypatch.setattr("littrace.topic_search.enqueue_parse_job", lambda *a, **kw: None)
    monkeypatch.setattr("littrace.topic_search.run_pending_parse_jobs", fake_parse)
    monkeypatch.setattr("littrace.topic_search.run_pending_embedding_jobs", fake_embed)
    monkeypatch.setattr(
        "littrace.topic_search.artifact_registry_from_config",
        lambda _config: SimpleNamespace(
            find_in_session=lambda artifact_id, session_id: SimpleNamespace(
                backend="local",
                bucket=None,
                object_key=f"{artifact_id}.pdf",
                sha256="sha",
                size_bytes=1,
                content_type="application/pdf",
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "littrace.topic_search.artifact_store_from_config",
        lambda _config: SimpleNamespace(exists=lambda _ref: True),
        raising=False,
    )
    monkeypatch.setattr("littrace.topic_search.save_workspace", fake_save)
    monkeypatch.setattr(
        "littrace.topic_search.load_workspace",
        lambda session: holder["workspace"],
    )

    result = asyncio.run(
        run_topic_search(config, session, request, requested_rag_ready=2)
    )

    assert result.candidate_count == 2
    assert result.downloaded_count == 2
    assert result.parsed_count == 2
    assert result.embedded_count == 2
    assert result.rag_ready_count == 2
    assert set(result.workspace.context.active_papers) == {"p1", "p2"}
    assert result.workspace.context.filters.paper_pipeline_status == {
        "p1": "rag_ready",
        "p2": "rag_ready",
    }
