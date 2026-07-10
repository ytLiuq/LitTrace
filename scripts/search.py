#!/usr/bin/env python3
"""Search for academic papers on a materials/chemistry topic.

Usage:
    python scripts/search.py "MXene flexible sensor" --limit 10 --year-min 2023
"""

import asyncio
import json
import sys
import os

# Ensure src is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from littrace.config import load_config
from littrace.models import PaperSearchRequest
from littrace.workflow import run_search_preview


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Search for academic papers")
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("--limit", type=int, default=15, help="Max papers")
    parser.add_argument("--year-min", type=int, default=2023, help="Min year")
    parser.add_argument("--live", action="store_true", default=True, help="Live search")
    parser.add_argument("--no-live", dest="live", action="store_false", help="Mock search")
    args = parser.parse_args()

    config = load_config()
    request = PaperSearchRequest(
        topic=args.topic,
        year_min=args.year_min,
        limit=args.limit,
        live=args.live,
    )
    workspace = await run_search_preview(request, config)

    papers = [workspace.papers[pid] for pid in workspace.context.active_papers]
    result = {
        "paper_count": len(papers),
        "search_mode": workspace.context.filters.get("search_mode", "unknown"),
        "papers": [
            {
                "id": p.paper_id,
                "title": p.title,
                "year": p.year,
                "journal": p.journal,
                "doi": p.doi,
                "access_type": p.access_type.value if p.access_type else None,
            }
            for p in papers
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
