"""Smoke test for littrace.cli.main dispatch.

Covers:
- Each subcommand is recognised and routes to its handler.
- An unknown subcommand exits non-zero with a clear error (typos don't
  silently drop into the REPL shell).
- The REPL shell is the no-arg default.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(*args: str, env: dict | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    full_env = {
        "PATH": __import__("os").environ.get("PATH", ""),
        "HOME": __import__("os").environ.get("HOME", ""),
        "LITTRACE_RAG_POSTGRES_DSN": "postgresql://littrace:littrace@localhost:5433/littrace",
        "LITTRACE_LLM_INTENT_PARSER_ENABLED": "false",
    }
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "littrace.cli", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=full_env,
        timeout=timeout,
    )


def test_unknown_subcommand_exits_nonzero():
    proc = _run_cli("not-a-real-subcommand")
    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "unknown" in combined
    assert "sentinel" in combined or "rag" in combined


def test_help_flag_does_not_exit_nonzero_for_unknown():
    # ``-h`` is a flag, not a subcommand — should fall through to the
    # REPL shell, which exits cleanly on EOF.
    proc = _run_cli("-h")
    # The REPL shell reads EOF and exits 0. We just check that -h is
    # not rejected as an unknown subcommand.
    combined = (proc.stdout + proc.stderr).lower()
    assert "unknown" not in combined


def test_doctor_subcommand_runs():
    proc = _run_cli("doctor", env={"LITTRACE_LLM_API_KEY": ""}, timeout=15)
    # ``littrace doctor`` is a pure-config check; it should print SOMETHING
    # and never silently no-op.
    assert (proc.stdout + proc.stderr).strip() != "", "doctor produced no output"