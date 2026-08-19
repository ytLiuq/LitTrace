"""``quality_report`` skill."""
from __future__ import annotations

from littrace.tool_contracts import tool_contract

from littrace.skills.registry import SkillManifest, registry
from .run import run


CONTRACT = tool_contract("quality_report")


def register() -> None:
    registry().add(
        SkillManifest(name="quality_report", run=run, contract=CONTRACT)
    )


register()


__all__ = ["run", "register", "CONTRACT"]