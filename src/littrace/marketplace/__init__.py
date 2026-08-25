"""Marketplace + third-party plugin surface (Round 13).

Round 4 P3 step 15 introduced a stub catalog here. Round 13
turns the stub into a thin façade over the entry-point
discovery in ``littrace.marketplace.discovery``.

Three surface helpers cover the common CLI queries:

  * ``list_skills()`` — in-tree stub catalog (unchanged) plus
    any third-party ``littrace.skills`` entry points the
    installer has dropped on the Python path. The in-tree and
    third-party lists are merged in a deterministic order so
    a test can assert on the catalog snapshot.

  * ``list_plugins()`` — full ``DiscoveryResult`` view of every
    entry point across all three groups; suitable for the
    ``littrace plugin list`` CLI.

  * ``plugin_info(name)`` — single-plugin lookup by entry-point
    name (matches across all groups) or by distribution name.
    Returns the entry points plus any load failures so the CLI
    can surface them without crashing.

The in-tree skill catalog stays hand-written for now; future
rounds can replace it with a real registry fetch. The
``install()`` helper stays a no-op because third-party plugins
arrive via ``pip install``, not via this surface.
"""

from __future__ import annotations

import logging
from typing import Any

from littrace.marketplace.discovery import (
    ALL_ENTRY_POINT_GROUPS,
    DiscoveryResult,
    ENTRY_POINT_HARNESSES,
    ENTRY_POINT_MCP_SERVERS,
    ENTRY_POINT_SKILLS,
    PluginEntry,
    discover_plugins,
)


log = logging.getLogger(__name__)


# In-tree skill catalog (kept hand-written; mirrors the round
# 4 P3 stub so existing tests that import ``list_skills()``
# keep working unchanged).
_INTREE_SKILLS: tuple[dict[str, Any], ...] = (
    {
        "name": "$skill-creator",
        "description": (
            "Generate a SKILL.md + run.py skeleton for a new skill "
            "from a free-form prompt."
        ),
        "status": "available",
        "source": "in-tree",
    },
    {
        "name": "$review-agent",
        "description": (
            "Self-check the current workspace before submission. "
            "0-100 score plus a list of items that need attention."
        ),
        "status": "available",
        "source": "in-tree",
    },
)


def list_skills() -> list[dict[str, Any]]:
    """Return the in-tree stub catalog plus any third-party
    ``littrace.skills`` entry points.

    The third-party rows are recorded under ``source ==
    "<dist>:<entry>"`` so a downstream caller can tell which
    entries are bundled with LitTrace and which arrived via
    ``pip install``.
    """
    rows = list(_INTREE_SKILLS)
    for entry in discover_plugins(groups=(ENTRY_POINT_SKILLS,)).by_group(
        ENTRY_POINT_SKILLS
    ):
        rows.append({
            "name": entry.name,
            "description": getattr(entry.value, "__doc__", "") or "",
            "status": "available",
            "source": f"{entry.dist}:{entry.name}" if entry.dist else entry.name,
        })
    return rows


def list_plugins() -> DiscoveryResult:
    """Return every entry point discovered on the current Python
    path, across all three supported groups. Convenience wrapper
    around ``discover_plugins()`` with the default group tuple.
    """
    return discover_plugins()


def plugin_info(name: str) -> dict[str, Any] | None:
    """Look up a single entry point by entry-point name or by
    distribution name.

    Returns ``None`` when the name does not match anything; the
    CLI surfaces that as a 404-style error.
    """
    result = list_plugins()
    for entry in result.entries:
        if entry.name == name or entry.dist == name:
            return {
                "group": entry.group,
                "name": entry.name,
                "dist": entry.dist,
                "callable": f"{entry.value.__module__}.{entry.value.__qualname__}",
                "doc": (entry.value.__doc__ or "").strip().splitlines()[0]
                if entry.value.__doc__ else "",
            }
    return None


def install(name: str) -> dict[str, Any]:
    """Install ``name`` from the marketplace.

    In-tree skills are always available; third-party plugins
    arrive via ``pip install <dist>``. This helper stays a
    no-op that records an INFO-level log so callers can
    confirm their install intent reached the marketplace layer.
    """
    log.info(
        "marketplace_install_request: name=%s "
        "(third-party plugins arrive via pip install)",
        name,
    )
    return {"name": name, "status": "no-op"}


__all__ = (
    "ALL_ENTRY_POINT_GROUPS",
    "DiscoveryResult",
    "ENTRY_POINT_HARNESSES",
    "ENTRY_POINT_MCP_SERVERS",
    "ENTRY_POINT_SKILLS",
    "PluginEntry",
    "discover_plugins",
    "install",
    "list_plugins",
    "list_skills",
    "plugin_info",
)