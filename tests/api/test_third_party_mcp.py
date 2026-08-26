"""Round 13 tests for third-party MCP server integration.

The ``LitTraceToolGateway`` accepts a third-party plugin's
``register_external_tool(name, spec, handler)`` call and the
``call`` method delegates to the registered handler before
falling back to the built-in tool allowlist. These tests
exercise the contract directly so a future refactor cannot
break the plugin surface.
"""

from __future__ import annotations

import asyncio

import pytest

from littrace.codex_runtime.gateway import (
    APP_SERVER_TOOL_NAMES,
    LitTraceToolGateway,
)


pytestmark = pytest.mark.api


def _gateway() -> LitTraceToolGateway:
    """Build a gateway against a stubbed state store so the
    built-in tools don't need a real Postgres connection.
    """
    from littrace.config import LitTraceConfig

    return LitTraceToolGateway(LitTraceConfig(), state_store=None)


def test_register_external_tool_rejects_invalid_arguments() -> None:
    gw = _gateway()
    with pytest.raises(ValueError):
        gw.register_external_tool(name="", spec={"name": ""}, handler=lambda *a, **k: {})
    with pytest.raises(ValueError):
        gw.register_external_tool(name="ok", spec="not-a-dict", handler=lambda *a, **k: {})


def test_external_tool_routes_before_builtin_allowlist() -> None:
    """A plugin tool whose name collides with a built-in
    name MUST win, because ``external_handlers`` is checked
    first. The inverse case (built-in wins) is impossible
    because the round 13 merge in
    ``CodexAppServerChatService._enabled_mcp_tools`` puts the
    built-in name first and rejects duplicates.
    """
    gw = _gateway()
    seen: list[str] = []

    async def handler(name, args, *, codex_thread_id):
        seen.append(name)
        return {"plugin": True, "args": args}

    # Pick a built-in name and override it via a plugin.
    builtin = next(iter(APP_SERVER_TOOL_NAMES))
    gw.register_external_tool(
        name=builtin,
        spec={"name": builtin, "description": "plugin shadow"},
        handler=handler,
    )

    async def scenario():
        result = await gw.call(
            builtin, {"x": 1}, codex_thread_id="thr-test",
        )
        assert result == {"plugin": True, "args": {"x": 1}}
        assert seen == [builtin]

    asyncio.run(scenario())


def test_external_tool_specs_merges_with_builtins() -> None:
    """``list_external_tool_specs()`` returns the merged
    catalog so a future round can advertise it on
    ``thread/start``.
    """
    gw = _gateway()
    gw.register_external_tool(
        name="vendor_search",
        spec={"name": "vendor_search", "description": "demo"},
        handler=lambda name, args, *, codex_thread_id: {},
    )
    gw.register_external_tool(
        name="vendor_lookup",
        spec={"name": "vendor_lookup", "description": "demo"},
        handler=lambda name, args, *, codex_thread_id: {},
    )
    specs = gw.list_external_tool_specs()
    assert {spec["name"] for spec in specs} == {"vendor_search", "vendor_lookup"}
    # Built-in names do NOT leak into the external specs list
    # — that is a separate surface the App Server aggregates
    # on its own.
    assert all(spec["name"] not in APP_SERVER_TOOL_NAMES for spec in specs)


def test_call_unknown_tool_still_raises() -> None:
    """An unknown tool name raises ``PermissionError`` whether
    or not the external registry is empty.
    """
    gw = _gateway()

    async def scenario():
        with pytest.raises(PermissionError):
            await gw.call("not_a_tool", {}, codex_thread_id="thr-test")

    asyncio.run(scenario())


def test_handler_returning_non_dict_is_wrapped() -> None:
    """The gateway passes the handler's return value through
    verbatim. Plugins can return any JSON-serialisable object;
    the App Server wraps the value in an McpResponse envelope
    on the receiving side, so the gateway does not need to
    transform the payload.
    """
    gw = _gateway()

    async def handler(name, args, *, codex_thread_id):
        return [{"id": "1", "title": "first"}]

    gw.register_external_tool(
        name="search",
        spec={"name": "search"},
        handler=handler,
    )

    async def scenario():
        result = await gw.call(
            "search", {"q": "x"}, codex_thread_id="thr-test",
        )
        assert result == [{"id": "1", "title": "first"}]

    asyncio.run(scenario())

def test_register_external_tool_rejects_name_spec_mismatch() -> None:
    """Round 13 fix: a plugin whose ``spec["name"]`` does not
    match the registration ``name`` is rejected. The gateway
    keys ``external_tools`` / ``external_handlers`` on the
    registration name; the App Server advertises on
    ``spec["name"]``. Allowing a mismatch would advertise
    one tool to the model but route another, which is
    impossible to debug from the caller side.
    """
    gw = _gateway()
    with pytest.raises(ValueError, match="must match"):
        gw.register_external_tool(
            name="vendor_search",
            spec={
                "name": "shadow_search",
                "description": "spec name does not match key",
            },
            handler=lambda name, args, *, codex_thread_id: {},
        )
    # The tool was NOT installed.
    assert "vendor_search" not in gw.external_tools
    assert "vendor_search" not in gw.external_handlers


def test_register_external_tool_allows_missing_spec_name() -> None:
    """A plugin that does not set ``spec["name"]`` is still
    accepted; the App Server only sees the registration
    name. The fix only fires when ``spec["name"]`` is
    explicitly set to a conflicting value.
    """
    gw = _gateway()
    gw.register_external_tool(
        name="vendor_search",
        spec={"description": "no spec name; spec key is canonical"},
        handler=lambda name, args, *, codex_thread_id: {},
    )
    assert "vendor_search" in gw.external_tools


def test_list_external_specs_merge_with_builtins_in_gateway() -> None:
    """Round 13 fix: ``list_external_tool_specs()`` returns
    third-party specs that ``list_tools()`` under
    ``LITTRACE_MCP_GATEWAY=1`` will merge with the built-ins.
    Previously the gateway only returned the built-ins, so
    R13 plugin tools were installed but invisible to the
    App Server.
    """
    gw = _gateway()
    gw.register_external_tool(
        name="vendor_search",
        spec={
            "name": "vendor_search",
            "description": "vendor catalog",
            "inputSchema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
        handler=lambda name, args, *, codex_thread_id: {},
    )
    specs = gw.list_external_tool_specs()
    assert {s["name"] for s in specs} == {"vendor_search"}
    # The spec round-trips through the gateway without losing
    # the JSON-Schema body — that is what the App Server
    # uses to teach the model how to call the tool.
    assert specs[0]["inputSchema"]["required"] == ["q"]
