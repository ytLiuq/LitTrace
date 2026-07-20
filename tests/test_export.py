from littrace.config import LitTraceConfig, StorageConfig
from littrace.context import add_papers
from littrace.export import (
    export_session_bundle,
    render_bibtex,
    render_markdown_brief,
    render_numbered_references,
    render_ris,
)
from littrace.models import EvidenceSpan, LiteratureWorkspace, PaperMetadata, PerformanceCell
from littrace.session import create_chat_session


def test_render_markdown_brief_includes_references():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Traceable Flexible Sensor",
                authors=["Ada Lovelace"],
                year=2026,
                journal="ACS Nano",
                doi="10.1021/example",
            )
        ],
    )

    markdown = render_markdown_brief(workspace)

    assert "LitTrace Research Brief" in markdown
    assert "https://doi.org/10.1021/example" in markdown
    assert "DRAFT - NOT FOR PUBLICATION" in markdown


def test_render_bibtex_includes_doi():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Traceable Flexible Sensor",
                authors=["Ada Lovelace"],
                year=2026,
                doi="10.1021/example",
            )
        ],
    )

    bibtex = render_bibtex(workspace)

    assert "@article" in bibtex
    assert "doi = {10.1021/example}" in bibtex


def test_export_session_bundle_writes_artifacts(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026)],
    )

    paths = export_session_bundle(session, workspace, config)

    assert set(paths) == {
        "markdown_draft",
        "research_report_draft",
        "research_report_draft_json",
        "release_ready",
        "release_blockers",
        "autonomous_review",
        "bibtex",
        "ris",
        "acs",
        "nature",
        "quality",
        "supplementary",
        "json",
    }
    artifact_paths = [
        __import__("pathlib").Path(path)
        for key, path in paths.items()
        if key not in {"release_ready", "release_blockers"}
    ]
    assert all((tmp_path / session.session_id) in path.parents for path in artifact_paths)
    assert paths["release_ready"] == "false"
    assert "research_report" not in paths


def test_export_session_bundle_publishes_only_after_claim_gate_passes(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026)],
    )
    workspace.performance_cells.append(
        PerformanceCell(
            paper_id="p1",
            metric="sensitivity",
            value=12.5,
            unit="kPa-1",
            evidence=EvidenceSpan(
                paper_id="p1",
                page=4,
                snippet="Sensitivity reached 12.5 kPa-1.",
            ),
        )
    )

    paths = export_session_bundle(session, workspace, config)

    assert paths["release_ready"] == "true"
    assert "research_report" in paths
    assert "research_report_json" in paths


def test_render_ris_and_numbered_references():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Traceable Flexible Sensor",
                authors=["Ada Lovelace"],
                year=2026,
                journal="ACS Nano",
                doi="10.1021/example",
            )
        ],
    )

    assert "TY  - JOUR" in render_ris(workspace)
    assert "DO  - 10.1021/example" in render_ris(workspace)
    assert "(1) Ada Lovelace" in render_numbered_references(workspace, style="acs")
    assert "1. Ada Lovelace" in render_numbered_references(workspace, style="nature")
