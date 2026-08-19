"""Plugin registry for LitTrace skills.

The registry is a thin singleton that maps ``ToolContract.name`` strings
to their :class:`SkillManifest`. Built-in skills self-attach at import time
(see ``littrace.skills.__init__``); third-party skills are picked up by
:func:`discover` from the ``littrace.skills`` entry-point group.
"""
from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any, Callable

from littrace.tool_contracts import ToolContract


@dataclass(frozen=True)
class SkillManifest:
    """Static metadata about a registered skill.

    Attributes:
        name: Stable skill identifier — must equal ``contract.name``.
        run: Callable that performs the work. May be sync or async.
        contract: The :class:`ToolContract` describing policies (network,
            idempotency, budget, etc.) — sourced from
            ``LITTRACE_TOOL_CONTRACTS`` in :mod:`littrace.tool_contracts`.
        plugin_origin: Where this skill came from. ``"builtin"`` for skills
            shipped with LitTrace; ``"external:<entry_point_name>"`` for
            third-party plugins discovered via entry points.
    """

    name: str
    run: Callable[..., Any]
    contract: ToolContract
    plugin_origin: str = "builtin"


class SkillRegistry:
    """In-memory map from skill name to its manifest."""

    def __init__(self) -> None:
        self._items: dict[str, SkillManifest] = {}

    def add(self, manifest: SkillManifest) -> None:
        """Register ``manifest``. Overwrites if ``name`` already exists."""
        self._items[manifest.name] = manifest

    def get(self, name: str) -> SkillManifest:
        """Return the manifest for ``name`` or raise ``KeyError``."""
        return self._items[name]

    def has(self, name: str) -> bool:
        return name in self._items

    def all(self) -> list[SkillManifest]:
        return list(self._items.values())

    def clear(self) -> None:
        """Reset the registry. Intended for tests."""
        self._items.clear()


_REGISTRY = SkillRegistry()


def registry() -> SkillRegistry:
    """Return the module-global :class:`SkillRegistry` singleton."""
    return _REGISTRY


def discover(entry_point_group: str = "littrace.skills") -> SkillRegistry:
    """Walk ``importlib.metadata`` entry points and register each plugin.

    Each entry point must resolve to a function that accepts a single
    :class:`SkillRegistry` argument and calls ``registry.add(...)`` for
    every manifest it ships. Built-in skills do not need to be advertised
    via entry points — they self-register on package import.
    """
    eps = importlib.metadata.entry_points(group=entry_point_group)
    for ep in eps:
        register_fn = ep.load()
        register_fn(_REGISTRY, plugin_origin=f"external:{ep.name}")
    return _REGISTRY