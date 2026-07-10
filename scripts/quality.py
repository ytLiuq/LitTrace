#!/usr/bin/env python3
"""Get a 14-dimension quality report for the current workspace.

Usage:
    python scripts/quality.py [--workspace workspace.json]
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from littrace.config import load_config
from littrace.models import LiteratureWorkspace
from littrace.quality_report import build_quality_report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate quality report")
    parser.add_argument(
        "--workspace",
        default="workspace.json",
        help="Workspace JSON file path",
    )
    args = parser.parse_args()

    config = load_config()

    if os.path.exists(args.workspace):
        workspace = LiteratureWorkspace.model_validate_json(open(args.workspace).read())
    else:
        print(f"Workspace file not found: {args.workspace}")
        sys.exit(1)

    report = build_quality_report(config, workspace)
    result = {
        "metrics": report.metrics,
        "warnings": report.warnings,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
