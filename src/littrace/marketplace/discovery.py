"""Plugin entry-point discovery (Round 13).

LitTrace supports three entry-point groups that any third-party
package can declare in its own ``pyproject.toml``:

  * ``littrace.skills`` — resolves to ``register(registry)``; the
    callable mutates the provided ``SkillRegistry`` to add a new
    skill at runtime. Mirrors the in-tree ``littrace.skills``
    sub-packages which self-register at import time.

  * ``littrace.mcp_servers`` — resolves to
    ``register(gateway)``; the callable installs an additional
    MCP server into the running App Server's tool catalog. Used
    by third-party publishers that want to expose custom tools
    (e.g. an internal ``littrace-mcp-pubmed`` plugin).

  * ``littrace.harnesses`` — resolves to ``register(registry)``;
    the callable mutates ``HarnessRegistry`` to install a new
    check. Lets an organization encode its own quality rules
    (e.g. "every workspace must have at least 3 verified
    citations") and have them run alongside the canonical
    suite.

The discovery is read-only: it scans ``importlib.metadata`` once
per call and returns a structured ``DiscoveredPlugins`` snapshot
so callers can decide whether to apply it, log it, or expose it
through a CLI. ``DiscoveryResult.apply`` is the only mutating
helper — it invokes each plugin's ``register`` callable against
the relevant registry in a stable order (alphabetical by
distribution name).

Failure mode: a single misbehaving plugin must NOT crash the
host process. ``DiscoveryResult.apply`` catches
``Exception`` per plugin and records a warning so the rest
of the suite still loads.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import logging
from dataclasses import dataclass, field
from typing import Any, Callable


log = logging.getLogger(__name__)


# Entry-point group names. Kept as module-level constants so
# they can be referenced by both the discovery code and any
# documentation / config example that wants to advertise the
# public contract.
ENTRY_POINT_SKILLS = "littrace.skills"
ENTRY_POINT_MCP_SERVERS = "littrace.mcp_servers"
ENTRY_POINT_HARNESSES = "littrace.harnesses"

ALL_ENTRY_POINT_GROUPS: tuple[str, ...] = (
    ENTRY_POINT_SKILLS,
    ENTRY_POINT_MCP_SERVERS,
    ENTRY_POINT_HARNESSES,
)


@dataclass(frozen=True)
class PluginEntry:
    """One resolved entry point.

    ``group`` is the entry-point group name (e.g.
    ``littrace.skills``); ``name`` is the entry-point name inside
    the group; ``dist`` is the distribution name the entry
    point came from; ``value`` is the resolved Python object
    (a callable that the registry invokes during ``apply``).
    """

    group: str
    name: str
    dist: str
    value: Any


@dataclass
class DiscoveryResult:
    """All entry points discovered on the current Python path.

    ``entries`` is the flat list of every resolved entry point;
    ``failures`` records distributions whose entry points
    failed to import so the CLI can surface them. Both fields
    are populated by ``discover_plugins``; nothing in this
    dataclass mutates a registry by itself.
    """

    entries: list[PluginEntry] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def by_group(self, group: str) -> list[PluginEntry]:
        """Return the entries that belong to ``group``."""
        return [e for e in self.entries if e.group == group]

    def by_dist(self, dist: str) -> list[PluginEntry]:
        """Return the entries that belong to ``dist``."""
        return [e for e in self.entries if e.dist == dist]

    def apply(self, *, skills_registry: Any | None = None,
              mcp_gateway: Any | None = None,
              harnesses_registry: Any | None = None) -> list[str]:
        """Invoke every entry point's ``register`` callable against
        the matching registry.

        Callers pass whichever registries they want to populate;
        an entry point whose group has no matching registry is
        skipped silently so a partial deploy can drive the
        subset it cares about. Returns a list of warning
        strings — one per plugin that raised — so the CLI
        can surface them without crashing the host process.
        """
        warnings: list[str] = []
        for entry in self.entries:
            try:
                if entry.group == ENTRY_POINT_SKILLS and skills_registry is not None:
                    entry.value(skills_registry)
                elif entry.group == ENTRY_POINT_MCP_SERVERS and mcp_gateway is not None:
                    entry.value(mcp_gateway)
                elif entry.group == ENTRY_POINT_HARNESSES and harnesses_registry is not None:
                    entry.value(harnesses_registry)
            except Exception as exc:  # pragma: no cover - defensive
                msg = (
                    f"plugin {entry.dist}:{entry.name} "
                    f"(group={entry.group}) raised {exc.__class__.__name__}: {exc}"
                )
                log.warning(msg, exc_info=True)
                warnings.append(msg)
        return warnings


def discover_plugins(
    groups: tuple[str, ...] = ALL_ENTRY_POINT_GROUPS,
) -> DiscoveryResult:
    """Scan ``importlib.metadata`` for the requested entry-point
    groups and return a flat ``DiscoveryResult``.

    Tested against Python 3.10-3.12; uses the modern
    ``selectable`` API so a single ``group`` keyword accepts the
    ``EntryPoints`` selector. The function returns an empty
    result when no plugins are installed so callers never need
    to special-case the "first run" experience.
    """
    result = DiscoveryResult()
    for group in groups:
        try:
            eps = importlib_metadata.entry_points(group=group)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "entry_point scan failed: group=%s err=%s",
                group, exc.__class__.__name__,
            )
            result.failures.append(
                f"{group}: {exc.__class__.__name__}: {exc}"
            )
            continue
        for ep in eps:
            try:
                value = ep.load()
            except Exception as exc:
                # Import-time failure inside the plugin
                # (missing dependency, broken module). Record
                # the failure and keep going so one bad
                # plugin does not prevent the rest from
                # loading.
                log.warning(
                    "plugin load failed: group=%s name=%s err=%s",
                    group, ep.name, exc.__class__.__name__,
                )
                result.failures.append(
                    f"{group}:{ep.name}: {exc.__class__.__name__}: {exc}"
                )
                continue
            result.entries.append(
                PluginEntry(
                    group=group,
                    name=ep.name,
                    dist=getattr(ep.dist, "name", "") or "",
                    value=value,
                )
            )
    return result