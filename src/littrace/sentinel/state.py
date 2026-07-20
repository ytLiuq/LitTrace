from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Watchlist(BaseModel):
    watchlist_id: str
    topic: str
    objective: str | None = None
    query_variants: list[str] = Field(default_factory=list)
    year_min: int = 2024
    frequency: Literal["daily", "weekly"] = "daily"
    preferred_sources: list[str] = Field(default_factory=list)
    auto_download_open_access: bool = True
    auto_download_requires_login: bool = False
    keep_pdfs: bool = True


class RetryTask(BaseModel):
    task_id: str = Field(default_factory=lambda: f"retry_{uuid4().hex[:12]}")
    paper_id: str
    reason: str
    attempts: int = 0
    next_retry_at: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class AccessTask(BaseModel):
    task_id: str = Field(default_factory=lambda: f"access_{uuid4().hex[:12]}")
    paper_id: str
    title: str
    doi: str | None = None
    publisher: str | None = None
    landing_url: str | None = None
    reason: Literal[
        "requires_institution_login",
        "cloudflare_or_mfa",
        "publisher_login_required",
        "cdp_session_unavailable",
    ]
    suggested_action: str = "open_in_browser"
    retry_after_login: bool = True
    attempts: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class DigestRecord(BaseModel):
    digest_id: str = Field(default_factory=lambda: f"digest_{uuid4().hex[:12]}")
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    digest_path: str
    paper_count: int = 0
    alert_count: int = 0
    claim_count: int = 0


class SentinelState(BaseModel):
    schema_version: str = "littrace.sentinel_state.v1"
    watchlist: Watchlist
    last_run_at: str | None = None
    seen_paper_ids: list[str] = Field(default_factory=list)
    rejected_paper_ids: list[str] = Field(default_factory=list)
    retry_queue: list[RetryTask] = Field(default_factory=list)
    access_queue: list[AccessTask] = Field(default_factory=list)
    digest_history: list[DigestRecord] = Field(default_factory=list)
    evidence_base_version: str = "v1"
    warnings: list[str] = Field(default_factory=list)


class SentinelRunSummary(BaseModel):
    run_id: str
    watchlist_id: str
    topic: str
    started_at: str
    finished_at: str | None = None
    new_candidates_count: int = 0
    downloaded_count: int = 0
    parsed_count: int = 0
    access_task_count: int = 0
    digest_path: str | None = None
    resource_pack_path: str | None = None
    quality_score: float | None = None
    warnings: list[str] = Field(default_factory=list)
