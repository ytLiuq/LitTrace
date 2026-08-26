"""Stand-alone entry point for the rollout → harness eval pipeline.

Round 10 step 2: thin wrapper that forwards ``sys.argv`` to
``littrace eval-from-rollout`` so the converter can be invoked
from CI without depending on the ``littrace`` console-script
being installed (e.g. inside the ``scripts/`` directory's own
unit test runner).

Usage::

    uv run python scripts/run_rollout_to_eval.py \\
        data/sessions/<id>/rollouts --report eval.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from littrace.cli import _run_eval_from_rollout_command

    parser = argparse.ArgumentParser(
        description=(
            "Stand-alone entry point for the rollout → harness eval "
            "pipeline. Forwards to ``littrace eval-from-rollout`` so "
            "the converter can be invoked from CI without depending "
            "on the ``littrace`` console-script being installed."
        ),
    )
    parser.add_argument(
        "rollout_path",
        help="Path to a rollout JSONL file or a directory of files.",
    )
    parser.add_argument(
        "--checks",
        default="check_citations,check_retry_health",
        help=(
            "Comma-separated list of check names. "
            "Default: check_citations,check_retry_health"
        ),
    )
    parser.add_argument(
        "--report",
        help="Path to write the JSON report. Optional.",
    )
    args = parser.parse_args()

    argv = [args.rollout_path]
    if args.checks:
        argv += ["--checks", args.checks]
    if args.report:
        argv += ["--report", args.report]
    _run_eval_from_rollout_command(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
