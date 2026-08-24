"""Stub marketplace catalog.

Round 4 P3 step 15 of 15.

codex-harness exposes a Marketplace of installable skills. LitTrace
ships a stub catalog here so the in-tree skills can be enumerated
through the same surface — the production path can be swapped in
later by replacing this module with one that fetches the catalog
from a real registry.

The interface is intentionally minimal: ``list_skills()`` returns a
list of dicts (name / description / status), and ``install(name)``
is currently a no-op (in-tree skills are always available — they
self-register via ``littrace.skills`` entry-point).
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


_AVAILABLE = (
    {
        "name": "$skill-creator",
        "description": (
            "Generate a SKILL.md + run.py skeleton for a new skill "
            "from a free-form prompt."
        ),
        "status": "available",
    },
    {
        "name": "$review-agent",
        "description": (
            "Self-check the current workspace before submission. "
            "0-100 score plus a list of items that need attention."
        ),
        "status": "available",
    },
)


def list_skills() -> list[dict[str, Any]]:
    """Return the (currently stub) marketplace catalog."""
    return list(_AVAILABLE)


def install(name: str) -> dict[str, Any]:
    """Install ``name`` from the marketplace.

    In-tree skills are always available — this is a no-op that
    records an INFO-level log so callers can confirm their install
    intent reached the marketplace layer.
    """
    log.info("marketplace_install_request: name=%s (in-tree skills are always available)", name)
    return {"name": name, "status": "no-op (in-tree)"}
