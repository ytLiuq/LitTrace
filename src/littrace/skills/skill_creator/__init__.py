"""$skill-creator — emit a SKILL.md + run.py skeleton from a free-form prompt.

Round 4 P3 step 15 of 15.

The $skill-creator skill mirrors the codex-harness skill that lets
the chat path author new skills on the fly. The current LitTrace
implementation is a stub that returns a templated SKILL.md body and
the run.py scaffold so the call shape is wired end-to-end. A future
iteration will let the model actually generate the run.py body via a
follow-up ``turn/compact``-style loop.
"""

from __future__ import annotations

import textwrap
from typing import Any

from littrace.models import LiteratureWorkspace
from littrace.session import ChatSession
from littrace.skills.registry import SkillManifest, registry
from littrace.tool_contracts import tool_contract


_NAME = "skill_creator"
_DESCRIPTION = (
    "Generate a SKILL.md + run.py skeleton for a new skill from a "
    "free-form prompt. The scaffold follows the layout used by the "
    "in-tree skills (sub-package with __init__.py, SKILL.md, run.py) "
    "so the resulting package can be dropped under "
    "``littrace/skills/<name>`` and discovered by the registry."
)


SKILL_TEMPLATE = textwrap.dedent(
    """\
    # {name}

    {description}

    ## When to use

    {when_to_use}

    ## Inputs

    - ``session``: LitTrace session id (``str``)
    - ``workspace``: the canonical ``LiteratureWorkspace`` for the session

    ## Output

    The skill's ``run.py`` returns a ``ToolResult`` (or whatever
    Pydantic model the skill chooses) that the Mcp Gateway exposes
    back to Codex.
    """
).lstrip()


RUN_TEMPLATE = textwrap.dedent(
    '''\
    """{description}"""

    from __future__ import annotations

    from typing import Any

    from littrace.models import LiteratureWorkspace
    from littrace.session import ChatSession


    async def run(
        session: ChatSession,
        workspace: LiteratureWorkspace,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Main entry point for the $skill-creator scaffold.

        Replace the body with the real implementation. The Mcp
        Gateway calls this function with the canonical workspace
        snapshot, so the skill must not mutate the workspace
        in place — copy first if it needs to.
        """
        return {{"status": "stub", "skill": "{name}"}}
    '''
).lstrip()


def run(
    session: ChatSession,
    workspace: LiteratureWorkspace,
    *,
    skill_name: str,
    description: str,
    when_to_use: str = "Use when the user asks for this skill.",
) -> dict[str, Any]:
    """Return a SKILL.md + run.py scaffold for ``skill_name``."""
    skill_md = SKILL_TEMPLATE.format(
        name=skill_name,
        description=description,
        when_to_use=when_to_use,
    )
    run_py = RUN_TEMPLATE.format(name=skill_name, description=description)
    return {
        "skill_md": skill_md,
        "run_py": run_py,
        "name": skill_name,
        "instructions": [
            f"mkdir -p littrace/skills/{skill_name}",
            f"write SKILL.md and run.py under littrace/skills/{skill_name}/",
            "import the new sub-package from littrace.skills.__init__",
        ],
    }


def register() -> None:
    registry().add(
        SkillManifest(
            name=_NAME, run=run, contract=tool_contract(_NAME),
        )
    )


register()


__all__ = ["run", "register", "CONTRACT"]
