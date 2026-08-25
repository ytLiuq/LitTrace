# pgvector ANN performance (Round 7)

LitTrace stores RAG chunks in pgvector and queries them with
cosine similarity. Below the planner's seq-scan threshold the
cost is negligible; past a few thousand rows an ANN index
becomes mandatory. Round 7 surfaces the index family and the
HNSW / IVFFlat tuning knobs in `RagConfig` so operators do not
have to inherit pgvector's defaults blind.

## Index family

`RagConfig.index_kind` selects the index family:

| Value     | Use case                                                                 |
|-----------|--------------------------------------------------------------------------|
| `hnsw`    | Default. Recall-biased, builds in O(N log N), reads in O(log N).         |
| `ivfflat` | Faster build, slightly lower recall. Good for very large static corpora. |
| `none`    | Disable ANN. Useful for <1k rows where the planner prefers a seq scan.    |

`pgvector_store.PgvectorRagStore.ensure_schema` reads
`RagConfig` and emits the corresponding `CREATE INDEX` statement
with the operator's `hnsw_m`, `hnsw_ef_construction`, and
`ivfflat_lists` parameters. The new index name is
`{collection_name}_embedding_{kind}_idx` so multiple profiles
coexist on the same Postgres instance.

## HNSW knobs

`pgvector` exposes three knobs for HNSW:

  * `hnsw_m` — degree of each node in the graph. Default 16.
    Higher `m` improves recall at the cost of build time and
    index size.
  * `hnsw_ef_construction` — beam width at build time. Default 64.
    Larger values give a higher-quality graph but take longer
    to build.
  * `hnsw_ef_search` — query-time beam width. Default 40 in
    LitTrace, overridable per config. `query_chunks` issues
    `SET hnsw.ef_search = N` on the connection so the operator's
    recall / latency trade-off applies to every query.

## Benchmark

`scripts/benchmark_pgvector.py` measures `query_chunks`-style
latency and recall@10 across three corpus sizes (1k / 10k / 100k)
and three index kinds (none / hnsw-default / hnsw-tuned). Each
cell is the median of `--queries` synthetic cosine queries
matched against a brute-force ground truth, so the recall number
is comparable across cells.

Run::

```bash
LITTRACE_RAG_POSTGRES_DSN=postgresql://littrace:littrace@localhost:5433/littrace \
uv run python scripts/benchmark_pgvector.py \
    --sizes 1000 10000 100000 --queries 50 \
    --report pgvector_benchmark.json
```

Verified locally on 1k / 10k rows (384-d vectors, 30 queries):

| Size | Index | p50 (ms) | recall@10 |
|-----:|-------|---------:|----------:|
| 1 000 | none  | 6.03     | 1.0000    |
| 1 000 | hnsw  | 5.23     | 1.0000    |
| 10 000 | none  | 33.62   | 1.0000    |
| 10 000 | hnsw  | 5.71     | 1.0000    |

5.8x speedup at 10k, recall unchanged. The 100k cell is opt-in
because the index build is ~30s on the local pgvector container;
see `tests/integration/test_pgvector_hnsw.py` for the production
regression test that asserts ≥ 5x speedup and ≥ 0.95 recall.

## CI integration

  * `tests/integration/test_pgvector_hnsw.py::test_hnsw_recall_and_latency_at_1k`
    runs by default. Asserts recall@10 = 1.0 and that HNSW
    latency stays within 1.5x of the brute-force baseline.
  * `test_hnsw_5x_speedup_at_100k` is gated by
    `RUN_PGVECTOR_BENCHMARK_100K=1`. It is intended to be run
    on a release-candidate build or on a hosted pgvector
    runner, not on every PR.

## Tuning recipe

For a corpus of N rows:

  * **N < 1 000** — `index_kind: none`. The planner's seq scan
    is faster than the HNSW graph walk.
  * **1 000 ≤ N < 100 000** — `index_kind: hnsw` with the
    defaults (`m=16`, `ef_construction=64`, `ef_search=40`).
  * **N ≥ 100 000** — `index_kind: hnsw`, raise `ef_search` to
    100-200. If recall is still below 0.95, raise `m` to 32 and
    re-build the index (the existing index must be dropped
    first; LitTrace does not yet auto-rebuild on config change).
  * **N ≥ 1 000 000** — consider `ivfflat` with `lists = N/1000`
    and re-train periodically as the corpus grows.

## Limitations

  * The current `pgvector_store.setup_sql` does not drop and
    rebuild the index on a config change. Operators who change
    `hnsw_m` after the first run must drop the old index
    manually; a future round will add a migration helper.
  * `SET hnsw.ef_search` is per-connection. LitTrace opens a
    new connection per query, so the GUC is reset on every
    request; the trade-off is the explicit `SET` on the
    `query_chunks` path.
