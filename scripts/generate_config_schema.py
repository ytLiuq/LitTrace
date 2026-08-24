"""Generate a machine-readable JSON schema for LitTraceConfig.

Round 4 P3 step 13 of 15.

The config has 30+ nested Pydantic models with hundreds of
fields. ``pyproject.toml``-style hand-written docs drift; the
schema stays in lock-step with the source because Pydantic's
``model_json_schema`` walks the same model the runtime parses.

The output is checked into ``docs/config.schema.json`` so a
downstream documentation site, an editor plugin, or a third-party
SDK generator can pick it up without depending on LitTrace at
runtime. ``LitTraceConfig`` already powers the production config
parser, so the schema is always in sync.

Usage::

    uv run python scripts/generate_config_schema.py

Re-run after any field rename, type change, or new sub-config.
CI does not run this automatically yet — keep the diff under
review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from littrace.config import LitTraceConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/config.schema.json"),
        help="where to write the JSON schema (default: docs/config.schema.json)",
    )
    args = parser.parse_args(argv)

    schema = LitTraceConfig.model_json_schema()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({len(schema.get('properties', {}))} top-level fields)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
