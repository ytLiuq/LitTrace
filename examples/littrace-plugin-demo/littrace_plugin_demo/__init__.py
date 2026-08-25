"""Demo third-party LitTrace MCP plugin (round 13 sample).

Install with::

    pip install -e examples/littrace-plugin-demo

Then run::

    littrace plugin list
    # == littrace-plugin-demo ==
    #   [littrace.mcp_servers] vendor_search -> littrace_plugin_demo.register

The plugin adds a single MCP tool, ``vendor_search``, that the
Codex App Server can invoke alongside the 15 built-in LitTrace
tools. The handler echoes the query back as a single fake
result so the test fixture does not depend on a real vendor
catalog.
"""

from __future__ import annotations

from typing import Any

from littrace.codex_runtime.gateway import LitTraceToolGateway


async def vendor_search(
    name: str, args: dict[str, Any], *, codex_thread_id: str,
) -> dict[str, Any]:
    """Echo the query back as a single fake vendor hit.

    Real plugins will hit an external API or database. The
    fixture handler keeps the example self-contained so a CI
    job can install + import the package without network
    access.
    """
    return {
        "results": [
            {"id": "1", "title": f"result for {args.get('q')!r}"},
        ],
    }


def register(gateway: LitTraceToolGateway) -> None:
    """Entry point invoked by ``littrace.marketplace.discovery``.

    The callable signature is the canonical plugin contract:
    ``register(gateway)``. The gateway exposes two helpers a
    plugin can use: ``register_external_tool`` to install one
    MCP tool, and the underlying ``external_handlers`` /
    ``external_tools`` maps if a plugin needs to replace a
    previously-installed tool.
    """
    gateway.register_external_tool(
        name="vendor_search",
        spec={
            "name": "vendor_search",
            "description": "Search the vendor catalog.",
            "inputSchema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
        handler=vendor_search,
    )


__all__ = ["register", "vendor_search"]