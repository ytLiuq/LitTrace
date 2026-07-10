import pytest

from littrace.config import EvalConfig, LitTraceConfig
from littrace.retrieval_eval import run_retrieval_golden_eval


@pytest.mark.anyio
async def test_retrieval_golden_eval_scores_live_pipeline(monkeypatch, tmp_path):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "cases.jsonl").write_text(
        '{"case_id":"mxene","topic":"MXene flexible pressure sensor",'
        '"preferred_year_min":2024,'
        '"expected_dois":["10.1021/acs.nanolett.5c01464"]}\n',
        encoding="utf-8",
    )

    from littrace.context import add_ranked_candidate_papers
    from littrace.models import LiteratureWorkspace, PaperMetadata

    async def fake_preview(request, config):
        workspace = LiteratureWorkspace()
        workspace.context.filters.search_mode = "live"
        return add_ranked_candidate_papers(
            workspace,
            [
                PaperMetadata(
                    paper_id="hit",
                    title="Nanoscale Interlayer Engineering Enhances MXene-Based Flexible Pressure Sensor",
                    doi="10.1021/acs.nanolett.5c01464",
                    year=2025,
                    publisher="American Chemical Society",
                ),
                PaperMetadata(
                    paper_id="miss",
                    title="Generic flexible sensor",
                    doi="10.1000/miss",
                    year=2025,
                ),
            ],
            request,
            active_limit=15,
        )

    monkeypatch.setattr("littrace.retrieval_eval.run_search_preview", fake_preview)

    report = await run_retrieval_golden_eval(
        LitTraceConfig(eval=EvalConfig(golden_set_dir=golden)),
        live=True,
    )

    assert report.case_count == 1
    assert report.metrics["candidate_recall"] == 1.0
    assert report.metrics["active_recall"] == 1.0
    assert report.metrics["mrr"] == 1.0
