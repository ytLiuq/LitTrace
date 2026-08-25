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
import os
import secrets
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from littrace.codex_runtime.gateway import LitTraceToolGateway, app_server_tool_specs
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
APP_SERVER_GATEWAY = os.environ.get("LITTRACE_MCP_GATEWAY", "").strip() == "1"

# Round 13 step 3: lazy, process-wide gateway singleton so the
# third-party ``littrace.mcp_servers`` plugins are registered
# exactly once per process. The gateway's ``external_handlers``
# and ``external_tools`` maps survive across tool calls; the
# per-call state (workspace, workspace_sha256, etc.) is still
# reconstructed from the request's threadId inside
# ``gateway.call``.
_GATEWAY: "LitTraceToolGateway | None" = None
_GATEWAY_PLUGINS_APPLIED = False


def _get_gateway() -> "LitTraceToolGateway":
    """Lazily construct + plugin-load the gateway singleton.

    First call wires up the in-tree gateway, then runs the
    marketplace discovery and registers every
    ``littrace.mcp_servers`` plugin against it. Subsequent
    calls reuse the same gateway instance so third-party
    tools installed via ``pip install`` between requests
    are picked up at the next cold start.
    """
    global _GATEWAY, _GATEWAY_PLUGINS_APPLIED
    if _GATEWAY is None:
        from littrace.state_db import state_store_from_config
        _GATEWAY = LitTraceToolGateway(_config, state_store_from_config(_config))
    if not _GATEWAY_PLUGINS_APPLIED:
        try:
            from littrace.marketplace import list_plugins
            from littrace.marketplace.discovery import ENTRY_POINT_MCP_SERVERS
            result = list_plugins()
            warnings = result.apply(
                mcp_gateway=_GATEWAY,
            )
            for warning in warnings:
                log.warning(
                    "external MCP plugin load failed: %s",
                    warning,
                )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "external MCP plugin scan failed: %s",
                exc,
            )
        _GATEWAY_PLUGINS_APPLIED = True
    return _GATEWAY

# Module-level state (MCP servers are single-session by design)
_config: LitTraceConfig = load_config(os.environ.get("LITTRACE_CONFIG_PATH", "config.yaml"))
_workspace = None  # LiteratureWorkspace, lazily imported


def _get_or_create_mcp_token() -> str:
    """Return the auth token for this MCP server.

    If ``LITTRACE_MCP_TOKEN`` is set, use it (so the operator can pin a
    stable token in the host MCP config). Otherwise generate a fresh
    token at startup and print it once to stderr — the operator has to
    paste it into the MCP client config. This avoids a no-auth default
    while still allowing zero-config first-run.
    """
    explicit = os.environ.get("LITTRACE_MCP_TOKEN", "").strip()
    if explicit:
        return explicit
    generated = "mcp-" + secrets.token_urlsafe(24)
    print(
        f"[mcp] LITTRACE_MCP_TOKEN not set. Generated one-shot token:\n"
        f"      {generated}\n"
        f"      Paste it into the MCP client config or set LITTRACE_MCP_TOKEN={generated}",
        file=sys.stderr,
    )
    return generated


MCP_TOKEN = "" if APP_SERVER_GATEWAY else _get_or_create_mcp_token()
"""The token every call_tool() invocation must include.

Clients send it as the ``token`` argument; a missing or mismatched
value is rejected with an explicit error so a misconfigured client
fails loud rather than silently polluting the workspace.
"""


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
    if APP_SERVER_GATEWAY:
        return [Tool(**spec) for spec in app_server_tool_specs()]
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
async def call_tool(name: str, arguments: dict) -> list[TextContent] | CallToolResult:
    """Execute a LitTrace tool and return the result as text."""
    global _workspace
    if APP_SERVER_GATEWAY:
        meta = app.request_context.meta
        thread_id = None
        if meta is not None:
            thread_id = (meta.model_extra or {}).get("threadId")
        try:
            gateway = _get_gateway()
            payload = await gateway.call(
                name,
                arguments,
                codex_thread_id=str(thread_id or ""),
            )
            # Round 4 P0 step 3: wrap the tool's raw dict in the
            # McpResponse envelope so the MCP client gets
            # {success, data, warnings, error?} on every call. The
            # legacy 8-tool branch (below) keeps its hand-built
            # dict — the envelope is opt-in for the app-server
            # path only because codex-harness clients need the
            # explicit shape.
            from littrace.codex_runtime.gateway import (
                McpResponse,
                ok_envelope,
            )
            envelope = ok_envelope(payload if isinstance(payload, dict) else {})
            return [
                TextContent(
                    type="text",
                    text=envelope.model_dump_json(),
                )
            ]
        except Exception as exc:  # noqa: BLE001 - MCP must return a structured tool error
            logger.error("gateway_tool_error", extra={"tool": name, "error": str(exc)})
            from littrace.codex_runtime.errors import CodexErrorCode
            from littrace.codex_runtime.gateway import err_envelope
            error_code = getattr(exc, "error_code", CodexErrorCode.OTHER)
            envelope = err_envelope(
                error_code,
                f"{exc.__class__.__name__}: {exc}",
            )
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=envelope.model_dump_json(),
                    )
                ],
            )
    workspace = _get_workspace()

    token = (arguments or {}).get("token", "")
    if not token or token != MCP_TOKEN:
        logger.warning("mcp_auth_failed", extra={"tool": name, "got_token": bool(token)})
        return [
            TextContent(
                type="text",
                text=(
                    "Authentication failed. Pass the configured LITTRACE_MCP_TOKEN "
                    "as the 'token' argument to every call_tool invocation. "
                    "If you have not configured a token, set LITTRACE_MCP_TOKEN "
                    "in the server process environment or paste the one printed "
                    "to stderr at startup."
                ),
            )
        ]

    try:
        if name == "search_papers":
            from littrace.models import PaperSearchRequest
            from littrace.workflow import run_search_preview

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
            from littrace.models import PaperSearchRequest
            from littrace.retrieval.search import build_query_variants
            from littrace.workflow import run_research_graph

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

    except Exception as exc:  # noqa: BLE001 - legacy MCP boundary returns tool errors
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
