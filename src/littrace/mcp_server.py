"""MCP Server for LitTrace — exposes core research tools via Model Context Protocol.

Usage:
    python -m littrace.mcp_server

Or in an MCP client config (e.g. Claude Desktop):
    {
      "mcpServers": {
        "littrace": {
          "command": "python",
          "args": ["-m", "littrace.mcp_server"]
        }
      }
    }

Tools exposed:
    - search_papers: Search for papers on a topic
    - get_context: View the active literature context
    - parse_full_text: Parse downloaded PDFs
    - extract_tables: Extract performance metrics into comparison matrices
    - build_storyline: Build evidence-grounded research storyline
    - run_research: Full end-to-end research workflow
    - quality_report: Get 14-dimension quality metrics
    - research_report: Compose an auditable research report
"""

from __future__ import annotations

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from littrace.config import LitTraceConfig, load_config
from littrace.log import get_logger
from littrace.skill_runner import (
    build_comparison_matrix_skill,
    build_quality_report_skill,
    build_research_report_skill,
    build_storyline_skill,
    extract_tables_skill,
    parse_workspace_skill,
)

logger = get_logger("mcp_server")

app = Server("littrace")

# Module-level state (MCP servers are single-session by design)
_config: LitTraceConfig = load_config()
_workspace = None  # LiteratureWorkspace, lazily imported


def _get_workspace():
    """Get or create the module-level workspace."""
    global _workspace
    if _workspace is None:
        from littrace.models import LiteratureWorkspace

        _workspace = LiteratureWorkspace()
    return _workspace


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available LitTrace tools."""
    return [
        Tool(
            name="search_papers",
            description=(
                "Search for academic papers on a materials/chemistry topic. "
                "Returns paper metadata (title, authors, year, journal, DOI, access type)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Research topic (e.g. 'MXene flexible sensors')",
                    },
                    "year_min": {
                        "type": "integer",
                        "description": "Minimum publication year (default: 2023)",
                        "default": 2023,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of papers to return (default: 15)",
                        "default": 15,
                    },
                    "live": {
                        "type": "boolean",
                        "description": "Use live search (OpenAlex/Crossref) vs mock data",
                        "default": True,
                    },
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="get_context",
            description=(
                "View the current active literature context — papers, parsed status, "
                "performance cells, and storyline claims."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="parse_full_text",
            description=(
                "Parse downloaded PDFs into traceable text, tables, and page evidence "
                "using Docling or PaddleOCR. Requires papers to have local PDFs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "parse_strategy": {
                        "type": "string",
                        "enum": ["auto", "text_only", "ocr"],
                        "description": "Parsing strategy: text_only=Docling, ocr=PaddleOCR, auto=use default",
                        "default": "auto",
                    },
                },
            },
        ),
        Tool(
            name="extract_tables",
            description=(
                "Extract performance metrics from parsed papers into comparison matrices. "
                "Uses LLM to read parsed sections/tables and extract quantitative metrics "
                "(sensitivity, conductivity, gauge factor, etc.). Requires parsed full text."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="build_storyline",
            description=(
                "Build an evidence-grounded research storyline from the active papers. "
                "Generates solution-limit-response chains with evidence spans."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="run_research",
            description=(
                "Run the full end-to-end research workflow: search → parse → extract tables → "
                "build storyline → compose document → autonomous review."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Research topic",
                    },
                    "year_min": {
                        "type": "integer",
                        "default": 2023,
                    },
                    "limit": {
                        "type": "integer",
                        "default": 15,
                    },
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="quality_report",
            description=(
                "Get a 14-dimension quality report covering paper count, full-text resolution rate, "
                "parsed rate, performance cells, citation guard pass rate, and more."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="research_report",
            description="Compose an auditable research report from the current workspace evidence.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a LitTrace tool and return the result as text."""
    global _workspace
    workspace = _get_workspace()

    try:
        if name == "search_papers":
            from littrace.workflow import run_search_preview
            from littrace.models import PaperSearchRequest

            topic = arguments.get("topic", "")
            year_min = arguments.get("year_min", 2023)
            limit = arguments.get("limit", 15)
            live = arguments.get("live", True)

            request = PaperSearchRequest(
                topic=topic,
                year_min=year_min,
                limit=limit,
                live=live,
            )
            workspace = await run_search_preview(request, _config)
            _workspace = workspace

            papers = [workspace.papers[pid] for pid in workspace.context.active_papers]
            summary = {
                "paper_count": len(papers),
                "search_mode": getattr(workspace.context.filters, "search_mode", "unknown"),
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
            return [
                TextContent(type="text", text=json.dumps(summary, ensure_ascii=False, indent=2))
            ]

        elif name == "get_context":
            papers = [workspace.papers[pid] for pid in workspace.context.active_papers]
            context = {
                "active_paper_count": len(papers),
                "parsed_paper_count": len(workspace.parsed_papers),
                "performance_cell_count": len(workspace.performance_cells),
                "search_mode": getattr(workspace.context.filters, "search_mode", None),
                "papers": [
                    {
                        "id": p.paper_id,
                        "title": p.title,
                        "year": p.year,
                        "doi": p.doi,
                        "has_full_text": p.paper_id in workspace.parsed_papers,
                    }
                    for p in papers
                ],
            }
            return [
                TextContent(type="text", text=json.dumps(context, ensure_ascii=False, indent=2))
            ]

        elif name == "parse_full_text":
            from littrace.models import coerce_parsed

            strategy = arguments.get("parse_strategy", "auto")
            parse_config = _config
            if strategy in {"text_only", "ocr"}:
                parse_config = _config.model_copy(deep=True)
                parse_config.parsing.parse_strategy = strategy
            workspace, report = await parse_workspace_skill(workspace, parse_config)
            _workspace = workspace

            result = {
                "parsed_count": report.get("parsed_count", 0),
                "failed_count": report.get("failed_count", 0),
                "total_papers": len(workspace.parsed_papers),
                "details": [
                    {
                        "paper_id": pid,
                        "parsed": (_cp := coerce_parsed(p)).parsed,
                        "section_count": len(_cp.sections or []),
                        "table_count": len(_cp.tables or []),
                    }
                    for pid, p in workspace.parsed_papers.items()
                ],
            }
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "extract_tables":
            workspace, harness = await extract_tables_skill(workspace, _config)
            matrix = build_comparison_matrix_skill(workspace)
            _workspace = workspace

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
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "build_storyline":
            claims = build_storyline_skill(workspace)
            result = {
                "claim_count": len(claims),
                "claims": [
                    {
                        "type": c.claim_type,
                        "claim": c.claim,
                        "confidence": c.confidence,
                        "evidence_count": len(c.evidence),
                        "evidence": [
                            {
                                "paper_id": e.paper_id,
                                "section": e.section,
                                "page": e.page,
                                "snippet": e.snippet[:150],
                            }
                            for e in c.evidence
                        ],
                    }
                    for c in claims
                ],
            }
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "run_research":
            from littrace.workflow import run_research_graph
            from littrace.models import PaperSearchRequest
            from littrace.retrieval.search import build_query_variants

            topic = arguments.get("topic", "")
            year_min = arguments.get("year_min", 2023)
            limit = arguments.get("limit", 15)

            query_variants = build_query_variants(topic)
            request = PaperSearchRequest(
                topic=topic,
                year_min=year_min,
                limit=limit,
                live=True,
                query_variants=query_variants,
            )
            result = await run_research_graph(
                request,
                _config,
                audit_citations_enabled=True,
                plan_downloads_enabled=False,
                route_publishers_enabled=True,
                parse_full_text_enabled=True,
                extract_tables_enabled=True,
                build_storyline_enabled=True,
                compose_document_enabled=True,
                autonomous_review_enabled=False,
            )
            workspace = result.workspace
            _workspace = workspace

            summary = {
                "paper_count": len(workspace.context.active_papers),
                "parsed_count": len(workspace.parsed_papers),
                "performance_cell_count": len(workspace.performance_cells),
                "workflow_steps": len(result.workflow_trace.steps) if result.workflow_trace else 0,
                "comparison_matrices": len(result.comparison_matrix.matrices)
                if result.comparison_matrix
                else 0,
            }
            return [
                TextContent(type="text", text=json.dumps(summary, ensure_ascii=False, indent=2))
            ]

        elif name == "quality_report":
            report = build_quality_report_skill(_config, workspace)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"metrics": report.metrics, "warnings": report.warnings},
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            ]

        elif name == "research_report":
            report = await build_research_report_skill(workspace, _config)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"sections": len(report.sections), "warnings": report.warnings},
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            ]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as exc:
        logger.error("tool_error", extra={"tool": name, "error": str(exc)})
        return [
            TextContent(
                type="text", text=f"Error executing {name}: {exc.__class__.__name__}: {exc}"
            )
        ]


async def main():
    """Run the MCP server over stdio."""
    logger.info("mcp_server_starting")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
