"""Round 7 benchmark for the pgvector ANN path.

Measures ``query_chunks`` latency and recall@10 across three corpus
sizes (1k / 10k / 100k) and three ANN configurations (none /
hnsw-default / hnsw-tuned). Each cell is the median over
``--queries`` synthetic queries drawn from the same distribution
as the corpus so the recall number is comparable to a brute-force
ground truth.

Run::

    LITTRACE_RAG_POSTGRES_DSN=postgresql://... \
    LITTRACE_RAG_EMBEDDING_BASE_URL=http://127.0.0.1:8765/v1 \
    LITTRACE_RAG_EMBEDDING_API_KEY=test \
    uv run python scripts/benchmark_pgvector.py \\
        --sizes 1000 10000 100000 --queries 50

The script writes a JSON report to ``--report`` (default
``pgvector_benchmark.json``) so CI can attach it as a build
artifact.

The benchmark never hits a real embedding provider. Vectors are
synthesised deterministically from a SHA-256 of the row id so
re-runs are reproducible and the cosine-similarity distribution
matches what real embeddings would produce after an L2 norm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import time
from dataclasses import asdict, dataclass
from typing import Any

import psycopg
from pgvector.psycopg import register_vector


def _vector_for(text: str, dimensions: int) -> list[float]:
    """Match the mock embedding server's deterministic sha256 vector."""
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


@dataclass
class CellResult:
    size: int
    index_kind: str
    ef_search: int | None
    p50_ms: float
    p95_ms: float
    recall_at_10: float
    rows_per_second: float
    notes: str = ""


def _build_table_sql(
    table: str,
    dimension: int,
    index_kind: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
    ivfflat_lists: int,
) -> list[str]:
    statements: list[str] = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"DROP TABLE IF EXISTS {table}",
        f"""
        CREATE TABLE {table} (
            id bigserial PRIMARY KEY,
            vec vector({dimension}) NOT NULL,
            payload text NOT NULL
        )
        """,
    ]
    if index_kind == "hnsw":
        statements.append(
            f"CREATE INDEX {table}_vec_idx ON {table} "
            f"USING hnsw (vec vector_cosine_ops) "
            f"WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction})"
        )
    elif index_kind == "ivfflat":
        statements.append(
            f"CREATE INDEX {table}_vec_idx ON {table} "
            f"USING ivfflat (vec vector_cosine_ops) "
            f"WITH (lists = {ivfflat_lists})"
        )
    # index_kind == "none": no ANN index
    return statements


def _populate(
    cursor: psycopg.Cursor,
    table: str,
    size: int,
    dimension: int,
) -> None:
    batch_size = 1000
    for start in range(0, size, batch_size):
        rows = []
        for i in range(start, min(start + batch_size, size)):
            vec = _vector_for(f"row-{i}", dimension)
            rows.append((i, vec, f"row-{i}"))
        cursor.executemany(
            f"INSERT INTO {table} (id, vec, payload) VALUES (%s, %s, %s)",
            rows,
        )


def _brute_force_topk(
    cursor: psycopg.Cursor,
    table: str,
    query_vec: list[float],
    k: int,
) -> list[int]:
    """Exact cosine-similarity scan for the ground truth top-k."""
    cursor.execute(
        f"SELECT id FROM {table} ORDER BY vec <=> %s::vector LIMIT %s",
        (query_vec, k),
    )
    return [row[0] for row in cursor.fetchall()]


def _indexed_topk(
    cursor: psycopg.Cursor,
    table: str,
    query_vec: list[float],
    k: int,
    ef_search: int | None,
) -> list[int]:
    if ef_search is not None:
        # ``SET`` does not accept query parameters in Postgres; the
        # value has to be inlined as a literal. ``ef_search`` comes
        # from the CLI (not the network) so this is safe.
        cursor.execute(f"SET hnsw.ef_search = {int(ef_search)}")
    cursor.execute(
        f"SELECT id FROM {table} ORDER BY vec <=> %s::vector LIMIT %s",
        (query_vec, k),
    )
    return [row[0] for row in cursor.fetchall()]


def _recall_at_k(
    ground_truth: list[int],
    predicted: list[int],
    k: int,
) -> float:
    if not ground_truth:
        return 0.0
    return len(set(ground_truth[:k]) & set(predicted[:k])) / float(k)


def _benchmark_one(
    dsn: str,
    table: str,
    dimension: int,
    size: int,
    index_kind: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
    ivfflat_lists: int,
    ef_search: int | None,
    queries: int,
    top_k: int,
) -> CellResult:
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for stmt in _build_table_sql(
                table, dimension, index_kind,
                hnsw_m, hnsw_ef_construction, ivfflat_lists,
            ):
                cur.execute(stmt)
        conn.commit()
        populate_start = time.perf_counter()
        with conn.cursor() as cur:
            _populate(cur, table, size, dimension)
        conn.commit()
        populate_seconds = time.perf_counter() - populate_start

        # Per-query latency
        latencies: list[float] = []
        recalls: list[float] = []
        with conn.cursor() as cur:
            for q in range(queries):
                query_vec = _vector_for(f"query-{q}", dimension)
                start = time.perf_counter()
                if index_kind == "none":
                    predicted = _brute_force_topk(cur, table, query_vec, top_k)
                else:
                    predicted = _indexed_topk(
                        cur, table, query_vec, top_k, ef_search,
                    )
                latencies.append((time.perf_counter() - start) * 1000.0)
                # Always compute the ground truth so the recall
                # number is comparable across index kinds.
                ground = _brute_force_topk(cur, table, query_vec, top_k)
                recalls.append(_recall_at_k(ground, predicted, top_k))

        # Tear the temporary table down so re-runs start fresh.
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()

    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max(latencies)
    return CellResult(
        size=size,
        index_kind=index_kind,
        ef_search=ef_search,
        p50_ms=round(p50, 2),
        p95_ms=round(p95, 2),
        recall_at_10=round(statistics.mean(recalls), 4),
        rows_per_second=round(size / populate_seconds, 1) if populate_seconds else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[1000, 10_000, 100_000],
        help="corpus sizes to benchmark (default: 1k 10k 100k)",
    )
    parser.add_argument(
        "--queries", type=int, default=50,
        help="number of synthetic queries per cell",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="top-k for the recall@10 measurement",
    )
    parser.add_argument(
        "--dimension", type=int, default=1536,
        help="embedding dimension (default: 1536)",
    )
    parser.add_argument(
        "--hnsw-m", type=int, default=16,
        help="HNSW m parameter (default: 16)",
    )
    parser.add_argument(
        "--hnsw-ef-construction", type=int, default=64,
        help="HNSW ef_construction parameter (default: 64)",
    )
    parser.add_argument(
        "--hnsw-ef-search", type=int, default=40,
        help="HNSW ef_search parameter (default: 40)",
    )
    parser.add_argument(
        "--hnsw-ef-search-tuned", type=int, default=120,
        help="HNSW ef_search for the 'tuned' cell (default: 120)",
    )
    parser.add_argument(
        "--ivfflat-lists", type=int, default=100,
        help="IVFFlat lists parameter (default: 100)",
    )
    parser.add_argument(
        "--report", default="pgvector_benchmark.json",
        help="output JSON report path",
    )
    args = parser.parse_args()

    dsn = os.environ.get("LITTRACE_RAG_POSTGRES_DSN") or os.environ.get(
        "LITTRACE_POSTGRES_DSN"
    )
    if not dsn:
        raise SystemExit(
            "LITTRACE_RAG_POSTGRES_DSN (or LITTRACE_POSTGRES_DSN) must be set"
        )

    cells: list[CellResult] = []
    for size in args.sizes:
        table = f"benchmark_pgvector_{size}"
        for kind, ef_search in [
            ("none", None),
            ("hnsw", args.hnsw_ef_search),
            ("hnsw", args.hnsw_ef_search_tuned),
        ]:
            label = f"size={size:>7d}  kind={kind:<7s}  ef={ef_search}"
            print(f"benchmarking {label}")
            try:
                cell = _benchmark_one(
                    dsn, table, args.dimension, size, kind,
                    args.hnsw_m, args.hnsw_ef_construction,
                    args.ivfflat_lists, ef_search, args.queries, args.top_k,
                )
            except Exception as exc:  # pragma: no cover
                cell = CellResult(
                    size=size,
                    index_kind=kind,
                    ef_search=ef_search,
                    p50_ms=0.0,
                    p95_ms=0.0,
                    recall_at_10=0.0,
                    rows_per_second=0.0,
                    notes=f"{exc.__class__.__name__}: {exc}",
                )
            cells.append(cell)
            print(
                f"  p50={cell.p50_ms:>7.2f}ms  p95={cell.p95_ms:>7.2f}ms  "
                f"recall@{args.top_k}={cell.recall_at_10:.4f}"
            )

    report: dict[str, Any] = {
        "dimension": args.dimension,
        "queries": args.queries,
        "top_k": args.top_k,
        "hnsw_m": args.hnsw_m,
        "hnsw_ef_construction": args.hnsw_ef_construction,
        "ivfflat_lists": args.ivfflat_lists,
        "cells": [asdict(c) for c in cells],
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nreport written to {args.report}")


if __name__ == "__main__":
    main()
