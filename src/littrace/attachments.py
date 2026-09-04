from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.artifact_registry import artifact_registry_from_config
from littrace.artifact_store import BlobRef, artifact_store_from_config
from littrace.access_layer.paths import target_pdf_path
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace


class AttachmentResult(BaseModel):
    paper_id: str
    attached: bool
    target_path: str
    error: str | None = None


class DownloadPresenceItem(BaseModel):
    paper_id: str
    title: str
    expected_path: str
    exists: bool
    selected_for_download: bool = False


class DownloadPresenceReport(BaseModel):
    items: list[DownloadPresenceItem]
    ready_to_parse_count: int
    missing_count: int
    warnings: list[str] = Field(default_factory=list)


def attach_pdf_to_paper(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    paper_id: str,
    source_path: str | Path,
) -> AttachmentResult:
    if paper_id not in workspace.papers:
        return AttachmentResult(
            paper_id=paper_id,
            attached=False,
            target_path="",
            error="Paper is not in the current workspace.",
        )
    source = Path(source_path).expanduser()
    paper = workspace.papers[paper_id]
    target = target_pdf_path(config, paper)
    if not source.exists():
        return AttachmentResult(
            paper_id=paper_id,
            attached=False,
            target_path=str(target),
            error=f"Source PDF does not exist: {source}",
        )
    if source.suffix.lower() != ".pdf":
        return AttachmentResult(
            paper_id=paper_id,
            attached=False,
            target_path=str(target),
            error="Only PDF files can be attached.",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return AttachmentResult(paper_id=paper_id, attached=True, target_path=str(target))


def check_download_presence(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    *,
    session_id: str | None = None,
) -> DownloadPresenceReport:
    selected = set(workspace.context.selected_for_download)
    items: list[DownloadPresenceItem] = []
    stored_ids: set[str] = set()
    if session_id:
        try:
            registry = artifact_registry_from_config(config)
            store = artifact_store_from_config(config)
            for record in registry.list_for_session(session_id=session_id):
                if record.kind != "paper_pdf" or not record.paper_id or not record.sha256:
                    continue
                if store.exists(BlobRef(
                    backend=record.backend,
                    bucket=record.bucket,
                    object_key=record.object_key,
                    sha256=record.sha256,
                    size_bytes=record.size_bytes,
                    content_type=record.content_type,
                )):
                    stored_ids.add(record.paper_id)
        except Exception as exc:  # noqa: BLE001 - presence is best effort
            warnings = [f"Object-storage audit unavailable: {exc.__class__.__name__}: {exc}"]
        else:
            warnings = []
    else:
        warnings = []
    for paper_id in workspace.context.active_papers:
        paper = workspace.papers[paper_id]
        expected = target_pdf_path(config, paper)
        local_exists = expected.exists()
        items.append(
            DownloadPresenceItem(
                paper_id=paper_id,
                title=paper.title,
                expected_path=str(expected),
                exists=local_exists or paper_id in stored_ids,
                selected_for_download=paper_id in selected,
            )
        )
    ready = sum(item.exists for item in items)
    missing = len(items) - ready
    if missing:
        warnings.append(f"{missing} active papers do not have local or object-storage PDFs yet.")
    return DownloadPresenceReport(
        items=items,
        ready_to_parse_count=ready,
        missing_count=missing,
        warnings=warnings,
    )
