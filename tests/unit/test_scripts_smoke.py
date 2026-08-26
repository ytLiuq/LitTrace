"""Round 7 CR pass 3 coverage: every script under
``scripts/`` that exposes ``--help`` must continue to do so
without raising. The test pins down the operator's
day-to-day entry points so a regression in
``argparse`` setup is caught in CI instead of in the
shell.

Scripts that always run a network/DB workload on import
(no ``--help`` branch, run the main flow immediately) are
out of scope for this smoke test; they have their own
live integration tests under ``tests/live/``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Scripts that DO support ``--help`` and are expected to
# exit 0 without doing any real work. ``run_research.py``
# and ``run_rollout_to_eval.py`` are in this set even
# though they delegate to ``littrace`` subcommands —
# ``--help`` should never touch the network or Postgres.
HELP_CLEAN_SCRIPTS: tuple[str, ...] = (
    "extract_tables.py",
    "parse.py",
    "search.py",
    "quality.py",
    "run_research.py",
    "run_rollout_to_eval.py",
    "generate_config_schema.py",
    "benchmark_docling_workers.py",
    "migrate_workspace_to_postgres.py",
    "smoke_codex_app_server.py",
    # The four real-workload e2e drivers also support --help
    # because they are CLI tools; we just don't run their
    # main flow here.
    "run_dual_path_30_e2e.py",
    "run_seven_publisher_download_e2e.py",
    "run_topic_download_e2e.py",
)


@pytest.mark.parametrize("script_name", HELP_CLEAN_SCRIPTS)
def test_script_help_exits_cleanly(script_name: str) -> None:
    script = SCRIPTS_DIR / script_name
    assert script.exists(), f"script not found: {script}"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"{script_name} --help failed (rc={proc.returncode})\n"
        f"STDOUT:\n{proc.stdout[:500]}\nSTDERR:\n{proc.stderr[:500]}"
    )
