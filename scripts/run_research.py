#!/usr/bin/env python3
"""Run the full end-to-end research workflow.

Usage:
    python scripts/run_research.py "MXene flexible sensor" --limit 15
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from littrace.config import load_config
from littrace.models import PaperSearchRequest
from littrace.retrieval.search import build_query_variants
from littrace.workflow import run_research_graph


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run full research workflow")
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("--limit", type=int, default=15, help="Max papers")
    parser.add_argument("--year-min", type=int, default=2023, help="Min year")
    parser.add_argument("--autonomous-review", action="store_true", help="Enable autonomous review")
    args = parser.parse_args()

    config = load_config()
    query_variants = build_query_variants(args.topic)
    request = PaperSearchRequest(
        topic=args.topic,
        year_min=args.year_min,
        limit=args.limit,
        live=True,
        query_variants=query_variants,
    )

    result = await run_research_graph(
        request,
        config,
        audit_citations_enabled=True,
        plan_downloads_enabled=False,
        route_publishers_enabled=True,
        parse_full_text_enabled=True,
        extract_tables_enabled=True,
        build_storyline_enabled=True,
        compose_document_enabled=True,
        autonomous_review_enabled=args.autonomous_review,
    )

    workspace = result.workspace
    summary = {
        "paper_count": len(workspace.context.active_papers),
        "parsed_count": len(workspace.parsed_papers),
        "performance_cell_count": len(workspace.performance_cells),
        "workflow_steps": len(result.workflow_trace.steps) if result.workflow_trace else 0,
        "comparison_matrices": len(result.comparison_matrix.matrices)
        if result.comparison_matrix
        else 0,
        "storyline_claims": len(result.storyline) if result.storyline else 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
