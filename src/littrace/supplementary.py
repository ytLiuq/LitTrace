from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.models import LiteratureWorkspace
from littrace.session import ChatSession


class SupplementaryAttachResult(BaseModel):
    paper_id: str
    attached: bool
    target_path: str | None = None
    links: list[str] = Field(default_factory=list)
    error: str | None = None


def register_supplementary_links(
    workspace: LiteratureWorkspace,
    paper_id: str,
    links: list[str],
) -> LiteratureWorkspace:
    existing = workspace.supplementary_links.setdefault(paper_id, [])
    for link in links:
        if link not in existing:
            existing.append(link)
    return workspace


def attach_supplementary_file(
    workspace: LiteratureWorkspace,
    session: ChatSession,
    paper_id: str,
    source_path: str | Path,
) -> SupplementaryAttachResult:
    if paper_id not in workspace.papers:
        return SupplementaryAttachResult(paper_id=paper_id, attached=False, error="Unknown paper id")
    source = Path(source_path).expanduser()
    if not source.exists():
        return SupplementaryAttachResult(
            paper_id=paper_id,
            attached=False,
            error=f"Supplementary file does not exist: {source}",
        )
    target_dir = session.artifacts_dir / "supplementary" / paper_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    workspace = register_supplementary_links(workspace, paper_id, [str(target)])
    return SupplementaryAttachResult(
        paper_id=paper_id,
        attached=True,
        target_path=str(target),
        links=workspace.supplementary_links.get(paper_id, []),
    )
