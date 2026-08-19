"""``parse_workspace_papers`` skill — OCR / Docling parse entry point."""
from __future__ import annotations

from littrace.tool_contracts import tool_contract

from littrace.skills.registry import SkillManifest, registry
from .run import run


CONTRACT = tool_contract("parse_workspace_papers")


def register() -> None:
    registry().add(
        SkillManifest(name="parse_workspace_papers", run=run, contract=CONTRACT)
    )


register()


__all__ = ["run", "register", "CONTRACT"]