from littrace.config import EvalConfig, LitTraceConfig
from littrace.golden_eval import run_golden_eval


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
