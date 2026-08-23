import pytest

from littrace.config import EvalConfig, LitTraceConfig
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.rag_eval import (
    RagGoldenCase,
    RagGoldEvidence,
    run_rag_golden_eval,
    score_rag_case,
)
from littrace.retrieval.pgvector_store import RagSearchHit


def _hit(
    chunk_id: str,
    paper_id: str,
    text: str,
    *,
    page: int | None = None,
    table_id: str | None = None,
) -> RagSearchHit:
    return RagSearchHit(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        embedding=[],
        score=0.9,
        chunk_hash=f"hash:{chunk_id}",
        page=page,
        table_id=table_id,
    )


def test_score_rag_case_counts_unique_gold_evidence_once():
    workspace = LiteratureWorkspace(
        papers={
            "p1": PaperMetadata(paper_id="p1", title="Paper 1", doi="10.1000/a"),
            "p2": PaperMetadata(paper_id="p2", title="Paper 2", doi="10.1000/b"),
        }
    )
    case = RagGoldenCase(
        case_id="rag-1",
        question="Which evidence reports sensitivity and response time?",
        gold_evidence=[
            RagGoldEvidence(
                evidence_id="gold-a",
                doi="10.1000/a",
                table_id="T1",
                required_terms=["sensitivity"],
                relevance=3,
            ),
            RagGoldEvidence(
                evidence_id="gold-b",
                doi="10.1000/b",
                page=4,
                required_terms=["response time"],
                relevance=2,
            ),
        ],
    )
    hits = [
        _hit("c1", "p1", "Sensitivity reached 12.5 kPa-1.", table_id="T1"),
        _hit("c2", "p2", "A generic introduction.", page=1),
        _hit("c3", "p1", "Sensitivity reached 12.5 kPa-1.", table_id="T1"),
    ]

    result = score_rag_case(case, hits, workspace, top_k=3)

    assert result.recall_at_k == 0.5
    assert result.precision_at_k == 0.333
    assert result.mrr == 1.0
    assert result.ndcg_at_k > 0.7
    assert result.matched_evidence_ids == ["gold-a"]


@pytest.mark.anyio
async def test_rag_golden_eval_warns_without_workspace_profile(tmp_path):
    golden = tmp_path / "rag-golden"
    golden.mkdir()
    (golden / "cases.jsonl").write_text(
        '{"case_id":"rag-1","question":"Find sensitivity evidence",'
        '"gold_evidence":[{"paper_id":"p1","required_terms":["sensitivity"]}]}\n',
        encoding="utf-8",
    )
    config = LitTraceConfig(eval=EvalConfig(rag_golden_set_dir=golden))

    report = await run_rag_golden_eval(config, LiteratureWorkspace(), top_k=5)

    assert report.case_count == 1
    assert report.metrics["rag_recall_at_k"] == 0.0
    assert report.metrics["rag_zero_hit_case_rate"] == 1.0
    assert report.cases[0].warnings
