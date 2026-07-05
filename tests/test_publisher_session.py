from littrace.config import LitTraceConfig, StorageConfig
from littrace.context import add_papers
from littrace.login_flow import AuthorizedPdfFetchResult
from littrace.models import AccessType, LiteratureWorkspace, PaperMetadata
from littrace.publisher_session import build_publisher_session_e2e_report


def test_publisher_session_e2e_reports_gated_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "littrace.publisher_session.fetch_authorized_pdf_after_user_auth",
        lambda *_args, **_kwargs: AuthorizedPdfFetchResult(
            paper_id="acs",
            attempted=True,
            opened_pdf=True,
            session_name="littrace-acs-pdf",
            pdf_url="https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c06548",
        ),
    )
    config = LitTraceConfig(storage=StorageConfig(paper_library_dir=tmp_path / "papers"))
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="acs",
                title="ACS paper",
                publisher="American Chemical Society",
                source_urls=["https://doi.org/10.1021/acsomega.2c06548"],
                access_type=AccessType.REQUIRES_LOGIN,
            )
        ],
    )

    workspace, report = build_publisher_session_e2e_report(
        config,
        workspace,
        publisher_family="American Chemical Society",
        timeout_seconds=0.1,
    )

    assert report.planned_count == 1
    assert report.authorized_pdf_fetches
    assert report.authorized_pdf_fetches[0].opened_pdf
    assert not report.completed
    assert report.target_paths
    assert report.warnings
