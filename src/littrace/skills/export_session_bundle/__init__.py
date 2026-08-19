"""``export_session_bundle`` skill."""
from __future__ import annotations

from littrace.tool_contracts import tool_contract

from littrace.skills.registry import SkillManifest, registry
from .run import run


CONTRACT = tool_contract("export_session_bundle")


def register() -> None:
    registry().add(
        SkillManifest(name="export_session_bundle", run=run, contract=CONTRACT)
    )


register()


__all__ = ["run", "register", "CONTRACT"]