"""``search_papers`` skill — wraps live/Mock search clients with a ToolContract."""
from __future__ import annotations

from littrace.tool_contracts import tool_contract

from littrace.skills._helpers import SearchSkillResult
from littrace.skills.registry import SkillManifest, registry
from .run import run


CONTRACT = tool_contract("search_papers")


def register() -> None:
    registry().add(
        SkillManifest(name="search_papers", run=run, contract=CONTRACT)
    )


register()


__all__ = ["run", "register", "CONTRACT", "SearchSkillResult"]