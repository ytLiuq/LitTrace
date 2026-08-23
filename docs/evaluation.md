# LitTrace Evaluation

LitTrace separates three different evaluation layers:

1. **Literature retrieval** checks whether external sources return the expected
   papers and rank them well.
2. **RAG evidence retrieval** checks whether the session index returns the
   expected paper, page, table, section, or evidence text.
3. **Evidence-grounded task success** checks whether a completed workspace
   retrieved the required papers, extracted exact performance cells, verified
   required claims, and attached traceable evidence to every critical claim.

## RAG Golden Cases

RAG cases live under `eval/rag_golden/*.jsonl`. Each line has this shape:

```json
{
  "case_id": "sensor-performance-rag",
  "question": "What evidence reports sensitivity and response time?",
  "tags": ["sensor", "performance"],
  "gold_evidence": [
    {
      "evidence_id": "paper-a-table-2",
      "doi": "10.1000/example",
      "page": 6,
      "table_id": "Table 2",
      "section": "Results",
      "required_terms": ["sensitivity", "response time"],
      "relevance": 3
    }
  ]
}
```

Only fields that have been human-reviewed should be populated. DOI-only labels
measure paper-level RAG recall. Adding page, table, section, and required terms
turns the same case into a stricter evidence-level benchmark.

The report includes Recall@K, Precision@K, nDCG@K, MRR, duplicate rate, and
zero-hit case rate. A golden evidence item is counted once even when duplicate
chunks retrieve the same evidence.

## Exact Performance Cells

Task cases under `eval/golden/*.jsonl` may include:

```json
{
  "expected_performance_cells": [
    {
      "doi": "10.1000/example",
      "metric": "response time",
      "value": 0.045,
      "unit": "s",
      "page": 6,
      "table_id": "Table 2",
      "relative_tolerance": 0.001
    }
  ]
}
```

Unit normalization is applied before numeric comparison. Precision, recall, and
exact-match are reported only against these reviewed structured labels. A
successful PDF parse is not treated as table-cell correctness.

## Evidence-Grounded Task Success

`/eval/task-golden` evaluates one completed workspace against one named task.
A non-abstention task passes only when:

- every expected DOI is active;
- every reviewed performance cell is matched;
- every required claim appears in a publishable verification report;
- every critical claim is verified or corroborated;
- every critical claim has complete evidence provenance.

Cases with `"should_abstain": true` pass only when the workspace does not
produce publishable critical claims.

## API

With the FastAPI service running:

```bash
curl "http://127.0.0.1:8000/eval/retrieval-golden?live=true"
curl "http://127.0.0.1:8000/eval/rag-golden?top_k=10"
curl "http://127.0.0.1:8000/eval/task-golden?case_id=CASE_ID"
curl -X POST "http://127.0.0.1:8000/eval/end-to-end"
curl "http://127.0.0.1:8000/quality"
```

Run RAG evaluation only after the current session has parsed documents and a
fresh RAG index. The API returns a warning instead of inventing scores when the
workspace has no searchable RAG profile.
