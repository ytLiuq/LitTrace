from littrace.config import LitTraceConfig, PublicationPolicyConfig
from littrace.context import add_papers
from littrace.evidence.document_composer import build_research_document_report
from littrace.models import (
    ClaimStatus,
    EvidenceSpan,
    LiteratureWorkspace,
    PaperMetadata,
    ParsedPaper,
    PerformanceCell,
)


def test_document_report_is_citation_and_evidence_backed():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Traceable MXene Sensor",
                authors=["Ada Lovelace"],
                year=2026,
                journal="ACS Nano",
                doi="10.1021/example",
            )
        ],
    )
    workspace.performance_cells.append(
        PerformanceCell(
            paper_id="p1",
            metric="sensitivity",
            value=12.5,
            unit="kPa-1",
            evidence=EvidenceSpan(
                paper_id="p1",
                section="Results",
                page=4,
                snippet="sensitivity reached 12.5 kPa-1",
                confidence=0.9,
            ),
        )
    )
    workspace.context.filters.autonomous_loop_report = {
        "passed": False,
        "score": 0.71,
        "replan_actions": ["parse_full_text_with_paddleocr"],
        "executed_replan_actions": ["extract_tables_and_structured_artifacts"],
        "rounds": [
            {
                "round_index": 1,
                "score": 0.71,
                "passed": False,
                "critiques": [
                    {
                        "reviewer": "Evidence Reviewer",
                        "severity": "warning",
                        "finding": "需要更多全文证据。",
                    }
                ],
            }
        ],
        "final_answer": "修订后的保守结论。",
    }

    report = build_research_document_report(workspace, LitTraceConfig())

    assert "LitTrace Research Report" in report.markdown
    assert "## 摘要" in report.markdown
    assert "## 方法与证据来源" in report.markdown
    assert "## 质量门与可选审稿" in report.markdown
    assert "Evidence Reviewer" in report.markdown
    assert "修订后的保守结论。" not in report.markdown
    assert "## 局限性与下一步" in report.markdown
    assert "https://doi.org/10.1021/example" in report.markdown
    assert "sensitivity reached 12.5" in report.markdown
    assert report.evidence_count >= 2
    assert report.citation_records[0].paper_id == "p1"
    assert report.release_ready
    assert report.quality_metrics["verified_claim_count"] == 1.0
    assert "## Claim Verification" in report.markdown
    assert "**verified**" in report.markdown
    assert report.release_snapshot is not None
    assert report.release_snapshot.release_ready
    assert report.release_snapshot.report_hash
    assert report.release_snapshot.config_hash


def test_document_report_marks_single_source_storyline_as_supported():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Method Paper", year=2025)],
    )
    workspace.parsed_papers["p1"] = ParsedPaper(
        parsed=True,
        sections=[
            {
                "name": "Methods",
                "text": "The fabrication method was optimized for a flexible sensor.",
                "evidence": {"page": 3, "parser": "pymupdf"},
            }
        ],
    )

    report = build_research_document_report(workspace, LitTraceConfig())

    assert not report.release_ready
    assert any(item.status == ClaimStatus.SUPPORTED for item in report.verification_reports)
    assert "Automatic semantic verification requires an exact asserted quote" in report.markdown
    assert "DRAFT - NOT FOR PUBLICATION" in report.markdown


def test_document_report_blocks_untraceable_metric_claims():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Untraceable Metric", year=2025)],
    )
    workspace.performance_cells.append(
        PerformanceCell(
            paper_id="p1",
            metric="sensitivity",
            value=4.2,
            evidence=EvidenceSpan(paper_id="p1"),
        )
    )

    report = build_research_document_report(workspace, LitTraceConfig())

    assert not report.release_ready
    assert report.verification_reports[0].status == ClaimStatus.CANDIDATE
    assert report.release_blockers
    assert report.quality_metrics["non_publishable_claim_count"] == 1.0
    assert "Release gate: **blocked**" in report.markdown


def test_non_strict_policy_can_publish_verified_sections_with_withheld_claims():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Method Paper", year=2025)],
    )
    workspace.performance_cells.append(
        PerformanceCell(
            paper_id="p1",
            metric="sensitivity",
            value=4.2,
            unit="kPa-1",
            evidence=EvidenceSpan(paper_id="p1", page=4, snippet="Sensitivity reached 4.2 kPa-1."),
        )
    )
    workspace.parsed_papers["p1"] = ParsedPaper(
        parsed=True,
        sections=[
            {
                "name": "Methods",
                "text": "The fabrication method was optimized for a flexible sensor.",
                "evidence": {"page": 3, "parser": "pymupdf"},
            }
        ],
    )

    report = build_research_document_report(
        workspace,
        LitTraceConfig(publication_policy=PublicationPolicyConfig(strict_all_claims=False)),
    )

    assert report.release_ready
    assert any(not item.publishable for item in report.verification_reports)
    assert "fabrication method was optimized" not in report.markdown
    assert "Publication: withheld pending verification" in report.markdown
