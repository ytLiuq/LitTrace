from littrace.access import plan_download
from littrace.config import DownloadMode, LitTraceConfig, PaperDownloadConfig
from littrace.context import add_papers, apply_context_update
from littrace.citations import best_access_url, citation_for_paper, format_apa_like_citation
from littrace.harnesses import check_storyline_claims
from littrace.models import (
    AccessType,
    ContextUpdate,
    EvidenceSpan,
    LiteratureWorkspace,
    PaperMetadata,
    StorylineClaim,
)
from littrace.source_router import route_sources
from littrace.search import SearchDiagnostics, merge_papers, rank_papers
from littrace.models import PaperSearchRequest


def test_source_router_prioritizes_recent_materials_sources():
    routes = route_sources("materials chemistry", wants_recent=True)
    names = [route.name for route in routes]
    assert "openalex" in names
    assert "publisher:wiley" in names
    assert names.index("openalex") < names.index("publisher:wiley")


def test_download_selected_only_downloads_selected_paper():
    config = LitTraceConfig(
        paper_download=PaperDownloadConfig(mode=DownloadMode.DOWNLOAD_SELECTED)
    )
    paper = PaperMetadata(
        paper_id="p1",
        title="A paper",
        access_type=AccessType.OPEN_ACCESS,
    )
    assert plan_download(config, paper, {"p1"}) == "download"
    assert plan_download(config, paper, set()) == "skip_unselected"


def test_storyline_harness_rejects_ungrounded_claims():
    result = check_storyline_claims(
        [
            StorylineClaim(
                claim="The field shifted.",
                claim_type="trend_by_year_and_method",
                evidence=[],
            )
        ]
    )
    assert not result.passed


def test_storyline_harness_accepts_grounded_limitation_claim():
    result = check_storyline_claims(
        [
            StorylineClaim(
                claim="Earlier work left stability under cyclic loading unresolved.",
                claim_type="remaining_limitation",
                evidence=[EvidenceSpan(paper_id="p1", page=8, snippet="limited cycling test")],
                confidence=0.8,
            )
        ]
    )
    assert result.passed


def test_context_update_can_exclude_and_select_papers():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(paper_id="p1", title="First"),
            PaperMetadata(paper_id="p2", title="Second"),
        ],
    )
    workspace = apply_context_update(
        workspace,
        ContextUpdate(exclude_paper_ids=["p2"], select_for_download=["p1", "p2"]),
    )
    assert workspace.context.active_papers == ["p1"]
    assert workspace.context.excluded_papers == ["p2"]
    assert workspace.context.selected_for_download == ["p1"]


def test_merge_papers_deduplicates_by_doi_and_prefers_oa():
    papers = merge_papers(
        [
            PaperMetadata(
                paper_id="left",
                title="Same",
                doi="10.1000/example",
                access_type=AccessType.METADATA_ONLY,
            ),
            PaperMetadata(
                paper_id="right",
                title="Same",
                doi="10.1000/example",
                pdf_url="https://example.org/paper.pdf",
                access_type=AccessType.OPEN_ACCESS,
            ),
        ]
    )
    assert len(papers) == 1
    assert papers[0].access_type == AccessType.OPEN_ACCESS
    assert str(papers[0].pdf_url) == "https://example.org/paper.pdf"


def test_rank_papers_prefers_recent_papers_when_other_signals_match():
    papers = rank_papers(
        [
            PaperMetadata(paper_id="old", title="Old", year=2016),
            PaperMetadata(paper_id="new", title="New", year=2026),
        ],
        PaperSearchRequest(topic="sensor"),
    )
    assert papers[0].paper_id == "new"


def test_search_diagnostics_defaults_to_empty_counts():
    diagnostics = SearchDiagnostics(live_attempted=True)
    assert diagnostics.live_attempted
    assert diagnostics.source_counts == {}


def test_citation_uses_pdf_url_before_doi():
    paper = PaperMetadata(
        paper_id="p1",
        title="Traceable Sensors",
        authors=["Ada Lovelace", "Grace Hopper"],
        year=2026,
        journal="Advanced Functional Materials",
        doi="10.1002/adfm.202600001",
        pdf_url="https://example.org/paper.pdf",
    )
    citation = citation_for_paper(paper)
    assert str(citation.access_url) == "https://example.org/paper.pdf"
    assert "Lovelace" in citation.citation_text
    assert "https://doi.org/10.1002/adfm.202600001" in citation.citation_text


def test_best_access_url_falls_back_to_doi():
    paper = PaperMetadata(
        paper_id="p1",
        title="Traceable Sensors",
        doi="10.1002/adfm.202600001",
    )
    assert best_access_url(paper) == "https://doi.org/10.1002/adfm.202600001"


def test_format_citation_handles_missing_authors_and_year():
    paper = PaperMetadata(paper_id="p1", title="Untimed Work")
    assert format_apa_like_citation(paper).startswith("Unknown author. (n.d.).")
