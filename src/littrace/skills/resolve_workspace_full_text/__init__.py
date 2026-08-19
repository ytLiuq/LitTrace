"""``resolve_workspace_full_text`` skill."""
from __future__ import annotations

from littrace.tool_contracts import tool_contract

from littrace.skills.registry import SkillManifest, registry
from .run import run


CONTRACT = tool_contract("resolve_workspace_full_text")


def register() -> None:
    registry().add(
        SkillManifest(
            name="resolve_workspace_full_text", run=run, contract=CONTRACT
        )
    )


register()


__all__ = ["run", "register", "CONTRACT"]