"""``extract_performance_cells`` skill."""
from __future__ import annotations

from littrace.tool_contracts import tool_contract

from littrace.skills.registry import SkillManifest, registry
from .run import run


CONTRACT = tool_contract("extract_performance_cells")


def register() -> None:
    registry().add(
        SkillManifest(
            name="extract_performance_cells", run=run, contract=CONTRACT
        )
    )


register()


__all__ = ["run", "register", "CONTRACT"]