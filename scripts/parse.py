#!/usr/bin/env python3
"""Parse downloaded PDFs into traceable text, tables, and page evidence.

Usage:
    python scripts/parse.py --strategy auto
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from littrace.config import load_config
from littrace.models import LiteratureWorkspace
from littrace.evidence.parsing import parse_workspace_papers


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse downloaded PDFs")
    parser.add_argument(
        "--strategy",
        choices=["auto", "text_only", "ocr"],
        default="auto",
        help="Parsing strategy",
    )
    parser.add_argument(
        "--workspace",
        default="workspace.json",
        help="Workspace JSON file path (default: workspace.json)",
    )
    args = parser.parse_args()

    config = load_config()

    # Load or create workspace
    if os.path.exists(args.workspace):
        workspace = LiteratureWorkspace.model_validate_json(open(args.workspace).read())
    else:
        print(f"Workspace file not found: {args.workspace}")
        print("Run scripts/search.py first to create a workspace.")
        sys.exit(1)

    workspace, report = parse_workspace_papers(workspace, config)
    result = {
        "parsed_count": report.get("parsed_count", 0),
        "failed_count": report.get("failed_count", 0),
        "total_parsed": len(workspace.parsed_papers),
        "details": [
            {
                "paper_id": pid,
                "parsed": p.get("parsed", False),
                "section_count": len(p.get("sections", [])),
                "table_count": len(p.get("tables", [])),
            }
            for pid, p in workspace.parsed_papers.items()
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
