"""Tests for the round 13 entry-point discovery and the
plugin list / info CLI surface.

The discovery code reads from ``importlib.metadata``, so the
tests cannot reliably depend on whatever plugins are installed
in the test environment. Instead we install a fake distribution
into ``entry_points`` via ``monkeypatch`` so we can assert on a
known plugin catalog regardless of what ``pip install`` has
loaded.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from littrace.marketplace.discovery import (
    ALL_ENTRY_POINT_GROUPS,
    ENTRY_POINT_HARNESSES,
    ENTRY_POINT_MCP_SERVERS,
    ENTRY_POINT_SKILLS,
    DiscoveryResult,
    PluginEntry,
    discover_plugins,
)


pytestmark = pytest.mark.unit


class _FakeDist:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeEntryPoint:
    def __init__(self, name: str, load_result: Any, dist_name: str) -> None:
        self.name = name
        self._load_result = load_result
        self.dist = _FakeDist(dist_name)

    def load(self) -> Any:
        result = self._load_result
        if isinstance(result, Exception):
            raise result
        return result


class _FakeEntryPoints:
    def __init__(self, eps: list[_FakeEntryPoint]) -> None:
        self._eps = eps

    def __iter__(self):  # pragma: no cover - not used
        yield from self._eps


def _patch_entry_points(monkeypatch, eps_by_group: dict[str, list[_FakeEntryPoint]]) -> None:
    """Replace ``importlib.metadata.entry_points(group=...)``."""

    def fake_entry_points(*, group: str) -> _FakeEntryPoints:
        return _FakeEntryPoints(eps_by_group.get(group, []))

    monkeypatch.setattr(
        "littrace.marketplace.discovery.importlib_metadata.entry_points",
        fake_entry_points,
    )


def test_discover_plugins_returns_empty_when_no_plugins(monkeypatch) -> None:
    _patch_entry_points(monkeypatch, {})
    result = discover_plugins()
    assert isinstance(result, DiscoveryResult)
    assert result.entries == []
    assert result.failures == []
    assert result.by_group(ENTRY_POINT_SKILLS) == []


def test_discover_plugins_collects_three_groups(monkeypatch) -> None:
    """The round 13 expansion covers three groups; the test
    installs one plugin per group and asserts every entry
    point shows up in the correct bucket.
    """

    def skill_register(registry):
        registry.add("hello-skill")

    def mcp_register(gateway):
        gateway.register_external_tool(
            name="vendor_search",
            spec={"name": "vendor_search", "description": "demo"},
            handler=lambda name, args, *, codex_thread_id: {"ok": True},
        )

    def harness_register(registry):
        registry.register(name="custom_check", group="custom")(
            lambda items, config=None: None
        )

    eps = {
        ENTRY_POINT_SKILLS: [
            _FakeEntryPoint("hello-skill", skill_register, "littrace-plugin-demo"),
        ],
        ENTRY_POINT_MCP_SERVERS: [
            _FakeEntryPoint("vendor_search", mcp_register, "littrace-plugin-demo"),
        ],
        ENTRY_POINT_HARNESSES: [
            _FakeEntryPoint("custom_check", harness_register, "littrace-plugin-demo"),
        ],
    }
    _patch_entry_points(monkeypatch, eps)
    result = discover_plugins()
    assert len(result.entries) == 3
    assert len(result.by_group(ENTRY_POINT_SKILLS)) == 1
    assert len(result.by_group(ENTRY_POINT_MCP_SERVERS)) == 1
    assert len(result.by_group(ENTRY_POINT_HARNESSES)) == 1
    assert all(e.dist == "littrace-plugin-demo" for e in result.entries)


def test_apply_routes_each_group_to_matching_registry(monkeypatch) -> None:
    """``DiscoveryResult.apply`` invokes the entry-point's
    ``register`` callable against the right registry and only
    the right registry; mismatched groups are ignored.
    """

    captured: dict[str, list[str]] = {"skills": [], "mcp": [], "harnesses": []}

    class _SkillsRegistry:
        def add(self, name: str) -> None:
            captured["skills"].append(name)

    class _HarnessesRegistry:
        def register(self, name: str, group: str):
            def decorator(func):
                captured["harnesses"].append(name)
                return func
            return decorator

    class _Gateway:
        def register_external_tool(self, *, name, spec, handler) -> None:
            captured["mcp"].append(name)

    def skill_register(registry):
        registry.add("from-skill")

    def mcp_register(gateway):
        gateway.register_external_tool(
            name="from-mcp",
            spec={"name": "from-mcp"},
            handler=lambda name, args, *, codex_thread_id: {},
        )

    def harness_register(registry):
        registry.register(name="from-harness", group="custom")(
            lambda items, config=None: None
        )

    eps = {
        ENTRY_POINT_SKILLS: [
            _FakeEntryPoint("from-skill", skill_register, "demo"),
        ],
        ENTRY_POINT_MCP_SERVERS: [
            _FakeEntryPoint("from-mcp", mcp_register, "demo"),
        ],
        ENTRY_POINT_HARNESSES: [
            _FakeEntryPoint("from-harness", harness_register, "demo"),
        ],
    }
    _patch_entry_points(monkeypatch, eps)
    result = discover_plugins()
    warnings = result.apply(
        skills_registry=_SkillsRegistry(),
        mcp_gateway=_Gateway(),
        harnesses_registry=_HarnessesRegistry(),
    )
    assert warnings == []
    assert captured["skills"] == ["from-skill"]
    assert captured["mcp"] == ["from-mcp"]
    assert captured["harnesses"] == ["from-harness"]


def test_apply_skips_groups_without_a_matching_registry(monkeypatch) -> None:
    """A partial deploy (only skills_registry) silently ignores
    MCP / harness plugins — discovery is read-only at apply time.
    """

    captured: list[str] = []

    class _SkillsRegistry:
        def add(self, name: str) -> None:
            captured.append(name)

    def skill_register(registry):
        registry.add("only-skill")

    def mcp_register(gateway):
        raise AssertionError(
            "MCP plugins must not be invoked when no "
            "mcp_gateway was passed to apply()"
        )

    eps = {
        ENTRY_POINT_SKILLS: [
            _FakeEntryPoint("only-skill", skill_register, "demo"),
        ],
        ENTRY_POINT_MCP_SERVERS: [
            _FakeEntryPoint("never-called", mcp_register, "demo"),
        ],
    }
    _patch_entry_points(monkeypatch, eps)
    result = discover_plugins()
    warnings = result.apply(skills_registry=_SkillsRegistry())
    assert warnings == []
    assert captured == ["only-skill"]


def test_apply_collects_warnings_without_raising(monkeypatch) -> None:
    """A plugin that raises during ``register`` is recorded as
    a warning; the rest of the suite still loads.
    """

    def good_register(registry):
        registry.add("good")

    def bad_register(registry):
        raise RuntimeError("boom")

    eps = {
        ENTRY_POINT_SKILLS: [
            _FakeEntryPoint("good", good_register, "demo"),
            _FakeEntryPoint("bad", bad_register, "demo"),
        ],
    }
    _patch_entry_points(monkeypatch, eps)
    result = discover_plugins()
    warnings = result.apply(skills_registry=type("R", (), {"add": lambda self, n: None})())
    assert len(warnings) == 1
    assert "bad" in warnings[0]
    assert "RuntimeError" in warnings[0]


def test_load_failure_recorded_in_failures(monkeypatch) -> None:
    """A plugin whose ``EntryPoint.load()`` raises (missing
    dependency) is captured in ``failures`` and skipped so the
    remaining plugins still install.
    """

    def good_register(registry):
        registry.add("ok")

    eps = {
        ENTRY_POINT_SKILLS: [
            _FakeEntryPoint("ok", good_register, "demo"),
            _FakeEntryPoint(
                "broken",
                load_result=ImportError("dep missing"),
                dist_name="demo",
            ),
        ],
    }
    _patch_entry_points(monkeypatch, eps)
    result = discover_plugins()
    assert len(result.entries) == 1
    assert result.entries[0].name == "ok"
    assert any("ImportError" in f for f in result.failures)


def test_discover_plugins_only_scans_requested_groups(monkeypatch) -> None:
    """Callers can ask for one group at a time (e.g.
    ``list_skills()`` should not pull MCP entries).
    """

    _patch_entry_points(
        monkeypatch,
        {
            ENTRY_POINT_SKILLS: [
                _FakeEntryPoint("s", lambda r: None, "demo"),
            ],
            ENTRY_POINT_MCP_SERVERS: [
                _FakeEntryPoint("m", lambda r: None, "demo"),
            ],
        },
    )
    skills_only = discover_plugins(groups=(ENTRY_POINT_SKILLS,))
    assert [e.name for e in skills_only.entries] == ["s"]
    all_groups = discover_plugins()
    assert sorted(e.name for e in all_groups.entries) == ["m", "s"]
    assert ALL_ENTRY_POINT_GROUPS == (
        ENTRY_POINT_SKILLS,
        ENTRY_POINT_MCP_SERVERS,
        ENTRY_POINT_HARNESSES,
    )


def test_plugin_info_lookup(monkeypatch) -> None:
    """``plugin_info(name)`` matches by entry-point name OR
    distribution name; ``None`` when nothing matches.
    """

    def skill_register(registry):
        registry.add("littrace-plugin-x:foo")

    eps = {
        ENTRY_POINT_SKILLS: [
            _FakeEntryPoint("foo", skill_register, "littrace-plugin-x"),
        ],
    }
    _patch_entry_points(monkeypatch, eps)
    from littrace.marketplace import plugin_info
    info = plugin_info("foo")
    assert info is not None
    assert info["group"] == ENTRY_POINT_SKILLS
    assert info["name"] == "foo"
    assert info["dist"] == "littrace-plugin-x"
    assert plugin_info("missing") is None
    assert plugin_info("littrace-plugin-x")["name"] == "foo"


def test_list_skills_merges_intree_and_third_party(monkeypatch) -> None:
    """``list_skills()`` returns in-tree rows plus every
    third-party ``littrace.skills`` entry point. The ``source``
    tag tells them apart.
    """

    def skill_register(registry):
        registry.add("vendor-x:my-skill")

    eps = {
        ENTRY_POINT_SKILLS: [
            _FakeEntryPoint("my-skill", skill_register, "vendor-x"),
        ],
    }
    _patch_entry_points(monkeypatch, eps)
    from littrace.marketplace import list_skills
    rows = list_skills()
    sources = {row["name"]: row["source"] for row in rows}
    # In-tree entries from the round 4 catalog.
    assert sources.get("$skill-creator") == "in-tree"
    assert sources.get("$review-agent") == "in-tree"
    # Third-party entry carries the dist name.
    assert sources.get("my-skill") == "vendor-x:my-skill"