"""``build_comparison_matrices`` skill."""
from __future__ import annotations

from littrace.tool_contracts import tool_contract

from littrace.skills.registry import SkillManifest, registry
from .run import run


CONTRACT = tool_contract("build_comparison_matrices")


def register() -> None:
    registry().add(
        SkillManifest(
            name="build_comparison_matrices", run=run, contract=CONTRACT
        )
    )


register()


__all__ = ["run", "register", "CONTRACT"]