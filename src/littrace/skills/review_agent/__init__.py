"""$review-agent — self-check the current workspace before submission.

Round 4 P3 step 15 of 15.

A minimal ``$review-agent`` skill that scores the workspace on a
few cheap invariants (paper count, citation count, evidence
record count, parse coverage) and returns a 0-100 score plus
the list of items that need attention. It does not call the LLM
or use any codex-harness integration — LitTrace already has
``self_audit`` for the heavy LLM-driven review; this is a fast
"is the workspace even populated" check.
"""

from __future__ import annotations

from typing import Any

from littrace.models import LiteratureWorkspace
from littrace.session import ChatSession
from littrace.skills.registry import SkillManifest, registry
from littrace.tool_contracts import tool_contract


_NAME = "review_agent"
_DESCRIPTION = (
    "Self-check the current workspace before submission. Returns a "
    "0-100 score plus a list of items that need attention. Cheap, "
    "deterministic, no LLM call."
)


def _score_workspace(workspace: LiteratureWorkspace) -> tuple[int, list[str]]:
    findings: list[str] = []
    paper_count = len(workspace.papers)
    if paper_count == 0:
        findings.append("workspace has no papers")
    elif paper_count < 3:
        findings.append(f"only {paper_count} papers (recommend >= 3)")

    parsed = sum(1 for p in workspace.parsed_papers.values() if p.parsed)
    if paper_count and parsed < paper_count:
        findings.append(
            f"{paper_count - parsed}/{paper_count} papers unparsed"
        )

    evidence = len(workspace.evidence_records)
    if evidence == 0:
        findings.append("no evidence records on workspace")

    cells = len(workspace.performance_cells)
    if paper_count and cells == 0:
        findings.append("no performance cells extracted")

    claims = len(workspace.claims)
    if paper_count and claims == 0:
        findings.append("no claims extracted")

    score = max(0, 100 - 15 * len(findings))
    return score, findings


def run(session: ChatSession, workspace: LiteratureWorkspace) -> dict[str, Any]:
    score, findings = _score_workspace(workspace)
    return {
        "score": score,
        "findings": findings,
        "session_id": session.session_id,
    }


def register() -> None:
    registry().add(
        SkillManifest(
            name=_NAME, run=run, contract=tool_contract(_NAME),
        )
    )


register()


__all__ = ["run", "register", "CONTRACT"]
