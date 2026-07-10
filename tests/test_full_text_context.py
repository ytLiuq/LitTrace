import pytest

from littrace.config import LitTraceConfig, StorageConfig
from littrace.context import add_ranked_candidate_papers
from littrace.full_text_context import build_full_text_context
from littrace.models import (
    AccessType,
    FullTextCandidate,
    FullTextResolutionReport,
    LiteratureWorkspace,
    PaperMetadata,
    PaperSearchRequest,
)
from littrace.parsing import local_pdf_path


@pytest.mark.anyio
async def test_full_text_context_keeps_only_downloaded_and_parsed_papers(tmp_path, monkeypatch):
    config = LitTraceConfig(
        storage=StorageConfig(
            paper_library_dir=tmp_path / "papers",
            metadata_dir=tmp_path / "metadata",
            cache_dir=tmp_path / "cache",
            sessions_dir=tmp_path / "sessions",
        )
    )
    request = PaperSearchRequest(topic="carbon PDMS pressure sensor drift", live=True)
    papers = [
        PaperMetadata(
            paper_id="good",
            title="Carbon PDMS pressure sensor drift stability",
            doi="10.1000/good",
            pdf_url="https://example.org/good.pdf",
            access_type=AccessType.OPEN_ACCESS,
            year=2026,
        ),
        PaperMetadata(
            paper_id="login",
            title="ACS gated pressure sensor",
            doi="10.1000/login",
            source_urls=["https://doi.org/10.1000/login"],
            access_type=AccessType.REQUIRES_LOGIN,
            year=2026,
        ),
        PaperMetadata(paper_id="bad", title="", year=2026),
    ]
    workspace = add_ranked_candidate_papers(LiteratureWorkspace(), papers, request, active_limit=10)

    async def fake_resolve(_papers, _config):
        reports = []
        for paper in _papers:
            candidates = []
            if paper.paper_id == "good":
                candidates.append(
                    FullTextCandidate(
                        paper_id=paper.paper_id,
                        url="https://example.org/good.pdf",
                        source="test",
                        access_type=AccessType.OPEN_ACCESS,
                        is_pdf=True,
                    )
                )
            reports.append(
                FullTextResolutionReport(
                    paper_id=paper.paper_id,
                    doi=paper.doi,
                    candidates=candidates,
                    best_pdf_url="https://example.org/good.pdf"
                    if paper.paper_id == "good"
                    else None,
                    login_required_candidate_count=1 if paper.paper_id == "login" else 0,
                )
            )
        return reports

    async def fake_execute(_config, selected, _request):
        for paper in selected:
            path = local_pdf_path(_config, paper)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"%PDF-1.4")

        class Result:
            items = []

        return Result()

    def fake_parse(workspace, _config):
        workspace.parsed_papers["good"] = {
            "parsed": True,
            "sections": [
                {
                    "text": "carbon PDMS pressure sensor drift stability",
                    "evidence": {"page": 1, "parser": "fake"},
                }
            ],
        }
        return workspace, {"parsed_count": 1, "failed_count": 0, "missing_pdf_count": 0}

    monkeypatch.setattr("littrace.full_text_context.resolve_full_text_for_papers", fake_resolve)
    monkeypatch.setattr("littrace.full_text_context.execute_downloads", fake_execute)
    monkeypatch.setattr("littrace.full_text_context.parse_workspace_papers", fake_parse)

    result = await build_full_text_context(workspace, request, config)

    assert result.candidate_count == 3
    assert result.valid_candidate_count == 2
    assert result.downloaded_count == 1
    assert result.parsed_count == 1
    assert result.workspace.context.active_papers == ["good"]
    assert result.workspace.context.filters.requires_login_candidate_ids == ["login"]
    assert result.workspace.context.filters.active_context_source == "downloaded_full_text_pdfs"
