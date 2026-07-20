from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata


def test_adding_doi_equivalent_records_keeps_identity_decisions():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(paper_id="index", title="Paper", doi="10.1000/example"),
            PaperMetadata(paper_id="publisher", title="Paper version", doi="10.1000/example"),
        ],
    )

    work = workspace.canonical_works["doi:10.1000/example"]
    assert sorted(work.version_paper_ids) == ["index", "publisher"]
    assert len(workspace.resolution_decisions) == 2
