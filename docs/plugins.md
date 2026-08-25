# LitTrace Plugin Authoring Guide (Round 13)

LitTrace ships a small, opinionated core and lets the rest of the
ecosystem grow through Python entry points. Any package on the
Python path can extend LitTrace without forking the repository —
the same model `pluggy`, `stevedore`, and the `setuptools`
entry-point spec have been using for a decade.

## Three entry-point groups

| Group                  | Callable signature                                | What it sees                                |
|------------------------|---------------------------------------------------|--------------------------------------------|
| `littrace.skills`      | `register(registry)`                                | a `SkillRegistry` the plugin mutates        |
| `littrace.mcp_servers` | `register(gateway)`                                | a `LitTraceToolGateway` the plugin mutates  |
| `littrace.harnesses`   | `register(registry)`                                | a `HarnessRegistry` the plugin mutates      |

The plugin's `register`` callable runs once per LitTrace process
start. The CLI / API surface the user invokes decides which
registries are populated: `mcp_server.py` applies MCP plugins
against the gateway; the eval harness applies harness plugins
against its registry; etc.

## Authoring a third-party MCP plugin

The minimum viable plugin is a 30-line Python package. Place
the following in `littrace_plugin_demo/__init__.py`:

```python
from littrace.codex_runtime.gateway import LitTraceToolGateway


async def my_search(name, args, *, codex_thread_id):
    return {"results": [{"id": "1", "title": args.get("q")}]}


def register(gateway: LitTraceToolGateway) -> None:
    gateway.register_external_tool(
        name="vendor_search",
        spec={
            "name": "vendor_search",
            "description": "Search the vendor catalog.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "q": {"type": "string"}
                },
                "required": ["q"],
            },
        },
        handler=my_search,
    )
```

Then declare the entry point in `pyproject.toml`:

```toml
[project.entry-points."littrace.mcp_servers"]
vendor_search = "littrace_plugin_demo:register"
```

After `pip install littrace-plugin-demo` the App Server's tool
catalog automatically grows by one tool. The CLI surfaces it
under `littrace plugin list`.

## Authoring a third-party skill plugin

Skills follow the in-tree pattern: a `register(registry)` callable
that calls `registry.add("<name>")`. The plugin's name shows up
under `list_skills()` with a `source: <dist>:<entry>` tag so an
operator can tell bundled from installed.

## Authoring a third-party harness check

Harness plugins register a check via the existing
`HarnessRegistry.register` decorator:

```python
from littrace.evaluation.harnesses import HarnessRegistry, HarnessReport, Severity


def register(registry: HarnessRegistry) -> None:
    @registry.register(name="check_workspace_size", group="custom")
    def check_workspace_size(items, config=None):
        return HarnessReport(
            check_name="check_workspace_size",
            passed=len(items) <= 200,
            score=1.0,
            findings=[],
            item_count=len(items),
        )
```

## Discovery & introspection

The CLI surfaces every discovered plugin:

```bash
$ littrace plugin list
== littrace-plugin-demo ==
  [littrace.mcp_servers     ] vendor_search                     -> littrace_plugin_demo:register

$ litprobe plugin info vendor_search
group: littrace.mcp_servers
name: vendor_search
dist: littrace-plugin-demo
callable: littrace_plugin_demo.register
doc: ...
```

## Failure modes

  * **Plugin import error** — the offending entry point is
    recorded in the discovery result's ``failures`` list with
    the exception class and message; the rest of the catalog
    still loads. The CLI surfaces them under
    ``load failures:``.
  * **Plugin raises during ``register``** — the discovery's
    ``apply`` method catches the exception, records a warning,
    and continues. The host process never crashes because of a
    single broken plugin.
  * **Tool name collision** — the gateway routes plugin tools
    before the built-in allowlist, so a plugin whose name
    shadows a built-in wins. The round 13 merge in
    `CodexAppServerChatService._enabled_mcp_tools` puts
    built-in names first and rejects duplicates, so a typical
    `pip install` cannot shadow an existing tool by accident.

## Versioning

Plugins must declare a version via the standard `project.version`
field. LitTrace records the distribution name in the discovery
result so an operator can map a runtime behaviour back to the
installed package.