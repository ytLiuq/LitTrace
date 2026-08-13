#!/usr/bin/env python3
"""Extract performance metrics from parsed papers into comparison matrices.

Usage:
    python scripts/extract_tables.py
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from littrace.config import load_config
from littrace.models import LiteratureWorkspace
from littrace.evidence.tables import build_comparison_matrices, extract_performance_cells


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract performance metrics")
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

    workspace, harness = await extract_performance_cells(workspace, config)
    matrix = build_comparison_matrices(workspace)

    result = {
        "performance_cell_count": len(workspace.performance_cells),
        "harness_score": harness.score,
        "harness_passed": harness.passed,
        "matrix_count": len(matrix.matrices),
        "matrices": [
            {
                "metric": m.metric,
                "row_count": len(m.rows),
                "warnings": m.warnings,
            }
            for m in matrix.matrices
        ],
        "sample_cells": [
            {
                "paper_id": c.paper_id,
                "metric": c.metric,
                "value": c.value,
                "unit": c.unit,
                "section": c.evidence.section,
                "snippet": c.evidence.snippet[:150],
            }
            for c in workspace.performance_cells[:10]
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
