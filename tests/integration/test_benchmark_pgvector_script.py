"""Round 7 CR pass 3 coverage: smoke-test the
``scripts/benchmark_pgvector.py`` CLI.

The benchmark script is the operator-facing entry point
for the round 7 HNSW tuning recommendations. Without a
test, a regression in the script's argument parsing or
the underlying ``_benchmark_one`` math would only
surface when an operator manually runs the script.

The test runs the script against a tiny 100-row corpus
so it stays under a few seconds and CI-friendly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration

DSN = os.environ.get("LITTRACE_RAG_POSTGRES_DSN") or os.environ.get(
    "LITTRACE_POSTGRES_DSN"
)
SKIP_NO_DSN = "LITTRACE_RAG_POSTGRES_DSN not set"
SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "benchmark_pgvector.py"


def _run_benchmark(tmp_path: Path, *args: str) -> dict[str, object]:
    report_path = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--report", str(report_path), *args],
        env={**os.environ, "LITTRACE_RAG_POSTGRES_DSN": DSN or ""},
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"benchmark_pgvector.py failed (rc={proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    assert report_path.exists()
    return json.loads(report_path.read_text(encoding="utf-8"))


@pytest.mark.skipif(not DSN, reason=SKIP_NO_DSN)
def test_benchmark_runs_against_tiny_corpus(tmp_path: Path) -> None:
    """The script's --sizes / --queries / --top-k flags
    propagate through ``subprocess.run`` and the resulting
    JSON report has the expected schema.
    """
    report = _run_benchmark(
        tmp_path, "--sizes", "100", "--queries", "5", "--top-k", "5",
    )
    # Top-level keys produced by the script.
    assert {"dimension", "queries", "top_k", "cells"} <= set(report)
    # One cell per (size, kind, ef) combination. The script
    # benchmarks ``none`` + ``hnsw(ef=default)`` +
    # ``hnsw(ef=tuned)`` so the cell count is 3 for a
    # single size.
    assert len(report["cells"]) == 3
    cell_kinds = {cell["index_kind"] for cell in report["cells"]}
    assert cell_kinds == {"none", "hnsw"}
    # Recall is reported as ``recall_at_<top_k>`` — the
    # round 7 CR pass rewrote the script to project the
    # dataclass field ``recall_at_k`` into a per-call key
    # that matches the operator's --top-k argument.
    for cell in report["cells"]:
        assert 0.0 <= cell["recall_at_5"] <= 1.0
        # Latency figures are millisecond floats.
        assert cell["p50_ms"] >= 0.0
        assert cell["p95_ms"] >= cell["p50_ms"]
    # Sanity: the script echoes the requested top-k in
    # the summary block.
    assert report["top_k"] == 5


@pytest.mark.skipif(not DSN, reason=SKIP_NO_DSN)
def test_benchmark_dsn_missing_fails_cleanly(tmp_path: Path) -> None:
    """When neither ``LITTRACE_RAG_POSTGRES_DSN`` nor
    ``LITTRACE_POSTGRES_DSN`` is set, the script exits with
    a clear error instead of silently writing an empty
    report.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("LITTRACE_RAG_POSTGRES_DSN", "LITTRACE_POSTGRES_DSN")}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--sizes", "50", "--queries", "2"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "LITTRACE_RAG_POSTGRES_DSN" in proc.stderr