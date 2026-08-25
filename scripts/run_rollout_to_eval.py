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

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from littrace.cli import _run_eval_from_rollout_command

    if len(sys.argv) < 2:
        print(
            "usage: run_rollout_to_eval.py <rollout-dir-or-file> "
            "[--checks check_citations,check_retry_health] [--report out.json]",
            file=sys.stderr,
        )
        return 2
    _run_eval_from_rollout_command(sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
