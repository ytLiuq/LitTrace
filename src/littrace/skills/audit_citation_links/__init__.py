"""``audit_citation_links`` skill."""
from __future__ import annotations

from littrace.tool_contracts import tool_contract

from littrace.skills.registry import SkillManifest, registry
from .run import run


CONTRACT = tool_contract("audit_citation_links")


def register() -> None:
    registry().add(
        SkillManifest(name="audit_citation_links", run=run, contract=CONTRACT)
    )


register()


__all__ = ["run", "register", "CONTRACT"]