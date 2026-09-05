"""Shared source-adapter contracts with diagnosable failures and health state."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
from typing import Protocol, TypeVar

from pydantic import BaseModel, Field

from littrace.models import (
    LiteratureWorkspace,
    PaperMetadata,
    PaperSearchRequest,
    SourceEvent,
    SourceRecord,
)


class SourceFailureClass(StrEnum):
    TRANSIENT = "transient"
    SOURCE_UNAVAILABLE = "source_unavailable"
    INVALID_INPUT = "invalid_input"
    POLICY_BLOCKED = "policy_blocked"


class SourceHealth(BaseModel):
    source_name: str
    checked_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    healthy: bool
    request_count: int = 0
    failure_count: int = 0
    last_failure_class: SourceFailureClass | None = None


OutputT = TypeVar("OutputT")


class SourceAdapter(Protocol[OutputT]):
    name: str

    async def fetch(self, *args, **kwargs) -> OutputT:
        """Fetch and normalize a source response without asserting its truth."""


class SourceResult(BaseModel):
    source_name: str
    ok: bool
    retrieved_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    failure_class: SourceFailureClass | None = None
    error: str | None = None
    cache_hit: bool = False
    stale: bool = False


def classify_source_exception(exc: Exception) -> SourceFailureClass:
    name = exc.__class__.__name__.lower()
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in {401, 403}:
        return SourceFailureClass.POLICY_BLOCKED
    if status_code == 429 or status_code in {408, 425, 500, 502, 503, 504}:
        return SourceFailureClass.TRANSIENT
    if "timeout" in name or "transport" in name or "connect" in name:
        return SourceFailureClass.TRANSIENT
    if "permission" in name or "policy" in name:
        return SourceFailureClass.POLICY_BLOCKED
    if "value" in name or "validation" in name:
        return SourceFailureClass.INVALID_INPUT
    return SourceFailureClass.SOURCE_UNAVAILABLE


def record_search_provenance(
    workspace: LiteratureWorkspace,
    request: PaperSearchRequest,
    papers: list[PaperMetadata],
    source_health: dict[str, SourceHealth],
    *,
    cache_state: str = "miss",
) -> list[SourceEvent]:
    """Append replayable source-attempt summaries to a workspace.

    The ledger intentionally stores a response fingerprint rather than a raw
    network payload. Raw source documents can be attached separately when a
    provider's licensing terms permit archival.
    """

    request_payload = request.model_dump(mode="json")
    request_fingerprint = sha256(
        json.dumps(request_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    response_hash = sha256(
        json.dumps(
            [paper.model_dump(mode="json") for paper in papers], ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    events: list[SourceEvent] = []
    for name, health in source_health.items():
        event_fingerprint = "\0".join(
            [name, request_fingerprint, health.checked_at, str(health.healthy), response_hash]
        )
        event = SourceEvent(
            event_id=f"source:{sha256(event_fingerprint.encode()).hexdigest()[:16]}",
            source_name=name,
            request_fingerprint=request_fingerprint,
            request=request_payload,
            response_hash=response_hash if health.healthy else None,
            retrieved_at=health.checked_at,
            status="completed" if health.healthy else "failed",
            cache_state=cache_state,
            failure_class=health.last_failure_class.value if health.last_failure_class else None,
            paper_ids=[paper.paper_id for paper in papers],
        )
        workspace.source_events.append(event)
        events.append(event)
        for paper in papers:
            record_id = f"{event.event_id}:{paper.paper_id}"
            workspace.source_records.setdefault(
                record_id,
                SourceRecord(
                    source_record_id=record_id,
                    paper_id=paper.paper_id,
                    source_name=name,
                    source_url=str(paper.source_urls[0]) if paper.source_urls else None,
                    retrieved_at=health.checked_at,
                    content_hash=response_hash if health.healthy else None,
                    failure_class=event.failure_class,
                ),
            )
    workspace.context.filters.source_health = {
        name: health.model_dump(mode="json") for name, health in source_health.items()
    }
    return events
