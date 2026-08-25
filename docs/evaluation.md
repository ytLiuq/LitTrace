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

## Rollout → eval pipeline (Round 10)

The side-channel rollout JSONL (Round 2C `RolloutRecorder`) and the
offline eval harness (`littrace.evaluation.harnesses`) are bridged
by `littrace.evaluation.rollout_eval`. One LitTrace session
produces one JSONL file with five event types
(`session_meta`, `turn_context`, `turn_start`, `event`,
`turn_complete`, `compaction`, `system_error`); the converter
groups events by `turn_id` and emits typed items the existing
harness checks already accept:

| Harness check          | Items produced from rollout                              |
|------------------------|---------------------------------------------------------|
| `check_citations`      | `CitationRecord` extracted from `item/completed`         |
| `check_retry_health`   | `RetryHealthItem` aggregated from MCP tool calls         |
| (custom)               | `TurnRecord` / `ToolCallRecord` for operator-defined checks |

Run the converter end-to-end against a rollouts directory:

```bash
littrace eval-from-rollout data/sessions/*/rollouts \
    --checks check_citations,check_retry_health \
    --report eval.json
```

Or invoke it from a CI script:

```bash
uv run python scripts/run_rollout_to_eval.py \
    data/sessions/<id>/rollouts --report eval.json
```

The output JSON looks like:

```json
{
  "rollout_path": "data/sessions/abc/rollouts",
  "checks": ["check_citations", "check_retry_health"],
  "reports": [
    {"check": "check_citations", "passed": false, "score": 0.0,
     "item_count": 2, "errors": ["..."], "warnings": []},
    {"check": "check_retry_health", "passed": true, "score": 1.0,
     "item_count": 2, "errors": [], "warnings": []}
  ],
  "summary": {"sessions": 2, "turns": 2, "tool_calls": 3, "errors": 1}
}
```

### Limitations

  * `RetryHealthItem.failure_rate` requires an `error_code` on
    the per-turn bucket; the rollout's `system_error` event
    plumbs that field but per-tool `item/completed` failures
    are not currently annotated. A future round can extend
    `RolloutRecorder` to capture per-tool status without
    breaking the wire format.
  * `CitationRecord.access_url` is a required `HttpUrl`; when
    the rollout does not capture an access URL the converter
    substitutes `http://about.local/citation-placeholder` so
    `check_citations` still flags the row as `UNCHECKED` and
    surfaces the remediation hint.
  * Cross-turn repeats are NOT treated as retries; the
    aggregator only counts a repeat when the same method
    fires more than once in the same `turn_id`.
