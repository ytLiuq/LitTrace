"""``execute_downloads`` skill."""
from __future__ import annotations

from littrace.tool_contracts import tool_contract

from littrace.skills.registry import SkillManifest, registry
from .run import run


CONTRACT = tool_contract("execute_downloads")


def register() -> None:
    registry().add(
        SkillManifest(name="execute_downloads", run=run, contract=CONTRACT)
    )


register()


__all__ = ["run", "register", "CONTRACT"]