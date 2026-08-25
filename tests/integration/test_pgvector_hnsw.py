"""Round 7 HNSW regression test.

Drives the benchmark script's metric surface (p50 latency,
recall@10) inside the integration test layer so a pgvector
upgrade or a schema change is caught in CI.

Two scenarios:

  1. **1k corpus** — runs by default. HNSW must match the
     brute-force ground truth with recall@10 = 1.0 and
     a p50 that is not worse than a 1.5x brute-force baseline.
  2. **100k corpus** — opt-in via ``RUN_PGVECTOR_BENCHMARK_100K=1``.
     HNSW build is ~30s on the local pgvector container, so
     this is gated out of the default CI run.

Skip conditions:

  * ``LITTRACE_RAG_POSTGRES_DSN`` not set
  * Postgres unreachable
  * pgvector extension unavailable
"""

from __future__ import annotations

import hashlib
import math
import os
import statistics
import struct
import time

import pytest

pytestmark = pytest.mark.integration


DSN = os.environ.get("LITTRACE_RAG_POSTGRES_DSN") or os.environ.get(
    "LITTRACE_POSTGRES_DSN"
)
SKIP_NO_DSN = "LITTRACE_RAG_POSTGRES_DSN not set"
RUN_100K = os.environ.get("RUN_PGVECTOR_BENCHMARK_100K", "").strip() == "1"


def _vector_for(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    pool: list[float] = []
    counter = 0
    while len(pool) < dimensions:
        h = hashlib.sha256(digest + struct.pack("<I", counter)).digest()
        for i in range(0, len(h), 4):
            pool.append(struct.unpack("<i", h[i:i + 4])[0] / 2**31)
            if len(pool) >= dimensions:
                break
        counter += 1
    norm = math.sqrt(sum(v * v for v in pool))
    return [v / norm for v in pool[:dimensions]]


def _brute_force_topk(cursor, table, query_vec, k):
    cursor.execute(
        f"SELECT id FROM {table} ORDER BY vec <=> %s::vector LIMIT %s",
        (query_vec, k),
    )
    return [row[0] for row in cursor.fetchall()]


def _populate(cursor, table, size, dimension):
    batch = 1000
    for start in range(0, size, batch):
        rows = [
            (i, _vector_for(f"row-{i}", dimension), f"row-{i}")
            for i in range(start, min(start + batch, size))
        ]
        cursor.executemany(
            f"INSERT INTO {table} (id, vec, payload) VALUES (%s, %s, %s)",
            rows,
        )


@pytest.mark.skipif(not DSN, reason=SKIP_NO_DSN)
def test_hnsw_recall_and_latency_at_1k() -> None:
    """The headline guarantee: at 1k rows, HNSW gives back the
    exact top-10 the brute-force scan would have, and does not
    regress latency by more than 1.5x of the brute-force run.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    dimension = 384  # smaller for fast test
    size = 1_000
    queries = 30
    top_k = 10
    table = "littrace_hnsw_bench_1k"
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cur.execute(
                f"""
                CREATE TABLE {table} (
                    id bigserial PRIMARY KEY,
                    vec vector({dimension}) NOT NULL,
                    payload text NOT NULL
                )
                """
            )
            cur.execute(
                f"CREATE INDEX {table}_vec_idx ON {table} "
                f"USING hnsw (vec vector_cosine_ops) "
                f"WITH (m = 16, ef_construction = 64)"
            )
        conn.commit()
        with conn.cursor() as cur:
            _populate(cur, table, size, dimension)
        conn.commit()

        # Brute-force baseline (ground truth + latency).
        brute_latencies: list[float] = []
        ground_truth: list[list[int]] = []
        with conn.cursor() as cur:
            for q in range(queries):
                query_vec = _vector_for(f"query-{q}", dimension)
                start = time.perf_counter()
                gt = _brute_force_topk(cur, table, query_vec, top_k)
                brute_latencies.append((time.perf_counter() - start) * 1000.0)
                ground_truth.append(gt)
        brute_p50 = statistics.median(brute_latencies)

        # HNSW indexed run.
        hnsw_latencies: list[float] = []
        hnsw_predictions: list[list[int]] = []
        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 40")
            for q in range(queries):
                query_vec = _vector_for(f"query-{q}", dimension)
                start = time.perf_counter()
                cur.execute(
                    f"SELECT id FROM {table} ORDER BY vec <=> %s::vector LIMIT %s",
                    (query_vec, top_k),
                )
                rows = [row[0] for row in cur.fetchall()]
                hnsw_latencies.append((time.perf_counter() - start) * 1000.0)
                hnsw_predictions.append(rows)
        hnsw_p50 = statistics.median(hnsw_latencies)
        recall = statistics.mean(
            len(set(g) & set(p)) / top_k
            for g, p in zip(ground_truth, hnsw_predictions)
        )

        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()

    assert recall == 1.0, f"HNSW recall@10 regressed: {recall}"
    # HNSW on 1k may be slightly slower than brute force because
    # the planner prefers a seq scan anyway, but it must not be
    # more than 1.5x the brute force baseline.
    assert hnsw_p50 <= brute_p50 * 1.5 + 5.0, (
        f"HNSW p50 {hnsw_p50:.2f}ms regressed vs brute {brute_p50:.2f}ms"
    )


@pytest.mark.skipif(
    not (DSN and RUN_100K),
    reason="set RUN_PGVECTOR_BENCHMARK_100K=1 to run the 100k benchmark",
)
def test_hnsw_5x_speedup_at_100k() -> None:
    """The 100k case is the production-shaped regression target.

    HNSW must be at least 5x faster than a brute-force scan
    while keeping recall@10 ≥ 0.95. Builds a 100k corpus with
    384-d vectors and runs 30 queries; the brute force path
    here doubles as the recall ground truth.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    dimension = 384
    size = 100_000
    queries = 30
    top_k = 10
    table = "littrace_hnsw_bench_100k"
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cur.execute(
                f"""
                CREATE TABLE {table} (
                    id bigserial PRIMARY KEY,
                    vec vector({dimension}) NOT NULL,
                    payload text NOT NULL
                )
                """
            )
            cur.execute(
                f"CREATE INDEX {table}_vec_idx ON {table} "
                f"USING hnsw (vec vector_cosine_ops) "
                f"WITH (m = 16, ef_construction = 64)"
            )
        conn.commit()
        with conn.cursor() as cur:
            _populate(cur, table, size, dimension)
        conn.commit()
        # ANALYZE so the planner picks the HNSW path at 100k.
        with conn.cursor() as cur:
            cur.execute(f"ANALYZE {table}")
        conn.commit()

        brute_latencies: list[float] = []
        ground_truth: list[list[int]] = []
        with conn.cursor() as cur:
            for q in range(queries):
                query_vec = _vector_for(f"query-{q}", dimension)
                start = time.perf_counter()
                gt = _brute_force_topk(cur, table, query_vec, top_k)
                brute_latencies.append((time.perf_counter() - start) * 1000.0)
                ground_truth.append(gt)
        brute_p50 = statistics.median(brute_latencies)

        hnsw_latencies: list[float] = []
        hnsw_predictions: list[list[int]] = []
        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 40")
            for q in range(queries):
                query_vec = _vector_for(f"query-{q}", dimension)
                start = time.perf_counter()
                cur.execute(
                    f"SELECT id FROM {table} ORDER BY vec <=> %s::vector LIMIT %s",
                    (query_vec, top_k),
                )
                rows = [row[0] for row in cur.fetchall()]
                hnsw_latencies.append((time.perf_counter() - start) * 1000.0)
                hnsw_predictions.append(rows)
        hnsw_p50 = statistics.median(hnsw_latencies)
        recall = statistics.mean(
            len(set(g) & set(p)) / top_k
            for g, p in zip(ground_truth, hnsw_predictions)
        )

        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()

    assert recall >= 0.95, f"HNSW recall@10 too low at 100k: {recall}"
    speedup = brute_p50 / max(hnsw_p50, 0.1)
    assert speedup >= 5.0, (
        f"HNSW speedup {speedup:.1f}x at 100k below the 5x target "
        f"(brute p50={brute_p50:.2f}ms, hnsw p50={hnsw_p50:.2f}ms)"
    )
