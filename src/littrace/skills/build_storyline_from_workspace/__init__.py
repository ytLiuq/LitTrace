"""``build_storyline_from_workspace`` skill."""
from __future__ import annotations

from littrace.tool_contracts import tool_contract

from littrace.skills.registry import SkillManifest, registry
from .run import run


CONTRACT = tool_contract("build_storyline_from_workspace")


def register() -> None:
    registry().add(
        SkillManifest(
            name="build_storyline_from_workspace", run=run, contract=CONTRACT
        )
    )


register()


__all__ = ["run", "register", "CONTRACT"]