from littrace.config import LitTraceConfig, StorageConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.session import create_chat_session
from littrace.supplementary import attach_supplementary_file, register_supplementary_links


def test_register_supplementary_links_deduplicates():
    workspace = LiteratureWorkspace()
    register_supplementary_links(workspace, "p1", ["https://example.org/si.pdf", "https://example.org/si.pdf"])

    assert workspace.supplementary_links["p1"] == ["https://example.org/si.pdf"]


def test_attach_supplementary_file_copies_to_artifacts(tmp_path):
    source = tmp_path / "si.pdf"
    source.write_bytes(b"%PDF-1.4")
    session = create_chat_session(LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path / "sessions")))
    workspace = add_papers(LiteratureWorkspace(), [PaperMetadata(paper_id="p1", title="Paper")])

    result = attach_supplementary_file(workspace, session, "p1", source)

    assert result.attached
    assert result.target_path
    assert workspace.supplementary_links["p1"] == [result.target_path]
