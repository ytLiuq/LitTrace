from littrace.config import EvalConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.golden_eval import run_golden_eval
from littrace.models import EvidenceSpan, LiteratureWorkspace, PaperMetadata, PerformanceCell


def test_golden_eval_reads_jsonl_cases(tmp_path):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "cases.jsonl").write_text(
        '{"topic":"sensor","expected_dois":["10.1000/a"],"expected_metrics":["sensitivity"]}\n',
        encoding="utf-8",
    )

    report = run_golden_eval(LitTraceConfig(eval=EvalConfig(golden_set_dir=golden)))

    assert report.case_count == 1
    assert report.metrics["has_expected_doi_rate"] == 1.0
    assert report.metrics["has_expected_metrics_rate"] == 1.0
    assert "has_expected_pdf_features_rate" in report.metrics


def test_golden_eval_scores_workspace_against_real_task(tmp_path):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "cases.jsonl").write_text(
        '{"topic":"MXene pressure sensor","preferred_year_min":2024,'
        '"expected_dois":["10.1021/acs.nanolett.5c01464"],'
        '"expected_publishers":["American Chemical Society"],'
        '"expected_metrics":["sensitivity"],'
        '"expected_storyline_claims":["interlayer engineering"]}\n',
        encoding="utf-8",
    )
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Nanoscale Interlayer Engineering Enhances MXene-Based Flexible Pressure Sensor",
                year=2025,
                publisher="American Chemical Society (ACS)",
                doi="10.1021/acs.nanolett.5c01464",
            )
        ],
    )
    workspace.context.filters["source_routes"] = ["Crossref"]
    workspace.performance_cells.append(
        PerformanceCell(
            paper_id="p1",
            metric="sensitivity",
            value=1.0,
            unit="kPa^-1",
            evidence=EvidenceSpan(paper_id="p1", section="abstract"),
        )
    )
    workspace.parsed_papers["p1"] = {
        "sections": [
            {
                "name": "Method",
                "text": "Interlayer engineering fabrication improves MXene pressure sensor performance.",
            }
        ]
    }

    report = run_golden_eval(
        LitTraceConfig(eval=EvalConfig(golden_set_dir=golden)),
        workspace,
    )

    assert report.metrics["golden_retrieval_doi_recall"] == 1.0
    assert report.metrics["golden_recent_paper_ratio"] == 1.0
    assert report.metrics["golden_table_metric_recall"] == 1.0
    assert report.failures == []
