# littrace-plugin-demo

A 30-line third-party LitTrace plugin that demonstrates the round
13 entry-point surface. Install it into any LitTrace deployment
and it shows up under `littrace plugin list` plus the App
Server tool catalog.

This is **documentation / sample** only — the package is not
published to PyPI. Copy the directory into a new repository
when you start writing a real plugin.

## What it does

Adds one MCP tool, `vendor_search`, that the Codex App Server
can call alongside the 15 built-in LitTrace tools. The handler
echoes the query back as a single fake result so the test
fixture does not depend on a real vendor catalog.

## Project layout

```
littrace-plugin-demo/
├── pyproject.toml
├── README.md
└── littrace_plugin_demo/
    └── __init__.py
```

## `pyproject.toml`

```toml
[project]
name = "littrace-plugin-demo"
version = "0.1.0"
description = "Demo third-party LitTrace MCP plugin"
requires-python = ">=3.11"

[project.entry-points."littrace.mcp_servers"]
vendor_search = "littrace_plugin_demo:register"
```

## `littrace_plugin_demo/__init__.py`

```python
from littrace.codex_runtime.gateway import LitTraceToolGateway


async def vendor_search(name, args, *, codex_thread_id):
    return {
        "results": [{"id": "1", "title": f"result for {args.get('q')!r}"}],
    }


def register(gateway: LitTraceToolGateway) -> None:
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
```

## Install + verify

```bash
pip install -e .

littrace plugin list
# == littrace-plugin-demo ==
#   [littrace.mcp_servers] vendor_search -> littrace_plugin_demo.register

littrace plugin info vendor_search
# group: littrace.mcp_servers
# name: vendor_search
# dist: littrace-plugin-demo
# callable: littrace_plugin_demo.register
# doc: ...
```

Then start a chat:

```
> search the vendor catalog for "MXene pressure sensor"
```

The Codex App Server routes the `vendor_search` call through
the gateway to the third-party handler.

## Authoring tips

  * The plugin contract is intentionally minimal:
    `async def handler(name, args, *, codex_thread_id) -> dict`.
    Return any JSON-serialisable object.
  * The gateway's `register_external_tool` accepts a JSON-Schema
    `spec` matching what the App Server expects; missing
    `inputSchema` is allowed but the model cannot call the
    tool without it.
  * The plugin must not crash during `register()`. A single
    broken plugin is recorded as a warning and the rest of the
    suite still loads — see the round 13 design notes for
    the failure-mode contract.