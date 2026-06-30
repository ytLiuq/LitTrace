from littrace.citation_guard import guard_citations, remove_unsupported_sentences
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata


def test_citation_guard_flags_claim_without_anchor():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Traceable Sensor", doi="10.1000/example")],
    )

    report = guard_citations("该方法显著提升了性能。", workspace)

    assert not report.passed
    assert report.unsupported_sentences


def test_citation_guard_accepts_claim_with_doi_anchor():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Traceable Sensor", doi="10.1000/example")],
    )

    report = guard_citations("该方法显著提升了性能，证据来自 10.1000/example。", workspace)

    assert report.passed


def test_remove_unsupported_sentences_deletes_unguarded_claims():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Traceable Sensor", doi="10.1000/example")],
    )
    text = "该方法显著提升了性能。背景介绍。证据来自 10.1000/example 的结果说明性能提升。"
    report = guard_citations(text, workspace)

    repaired = remove_unsupported_sentences(text, report)

    assert "该方法显著提升了性能" not in repaired
    assert "10.1000/example" in repaired
