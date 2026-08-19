"""``build_quality_metrics`` skill — fills the long-missing wrapper.

The contract was registered in :mod:`littrace.tool_contracts` from day one
but no ``*_skill`` wrapper shipped. This package is the new home.
"""
from __future__ import annotations

from littrace.tool_contracts import tool_contract

from littrace.skills.registry import SkillManifest, registry
from .run import run


CONTRACT = tool_contract("build_quality_metrics")


def register() -> None:
    registry().add(
        SkillManifest(name="build_quality_metrics", run=run, contract=CONTRACT)
    )


register()


__all__ = ["run", "register", "CONTRACT"]