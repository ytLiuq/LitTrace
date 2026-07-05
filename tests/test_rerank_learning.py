import pytest

from littrace.config import LitTraceConfig
from littrace.rerank_learning import learn_rerank_policy_from_golden
from littrace.retrieval_eval import RetrievalEvalReport


@pytest.mark.anyio
async def test_learn_rerank_policy_scores_candidates(monkeypatch):
    async def fake_eval(config, live=True):
        return RetrievalEvalReport(
            case_count=1,
            live=live,
            metrics={"active_recall": 0.8, "candidate_recall": 1.0, "mrr": 0.5},
        )

    monkeypatch.setattr("littrace.rerank_learning.run_retrieval_golden_eval", fake_eval)

    report = await learn_rerank_policy_from_golden(LitTraceConfig())

    assert report.candidate_count >= 3
    assert report.best_candidate
    assert report.best_score > 0
