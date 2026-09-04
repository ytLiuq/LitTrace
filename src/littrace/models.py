from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from math import isclose
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


def _coerce_str_to_http_url(value: object) -> object:
    """Coerce ``str`` → ``HttpUrl`` so upstream API responses (Crossref /
    OpenAlex / Unpaywall) that hand back plain strings no longer trigger
    ``PydanticSerializationUnexpectedValue`` warnings during model_dump.

    Non-string values pass through unchanged so existing HttpUrl instances
    are not double-wrapped.
    """
    if isinstance(value, str):
        try:
            return HttpUrl(value)
        except Exception:
            return value
    return value


def _coerce_http_url_fields(data: object, field_names: list[str]) -> object:
    """Apply :func:`_coerce_str_to_http_url` to ``field_names`` inside
    ``data`` (dict or pydantic-incompatible raw input). Tolerant of bad
    inputs (returns ``data`` unchanged if it is not a dict)."""
    if not isinstance(data, dict):
        return data
    for field_name in field_names:
        if field_name in data:
            data[field_name] = _coerce_str_to_http_url(data[field_name])
        nested = data.get(field_name)
        if isinstance(nested, list):
            data[field_name] = [_coerce_str_to_http_url(item) for item in nested]
    return data


class AccessType(StrEnum):
    OPEN_ACCESS = "open_access"
    REQUIRES_LOGIN = "requires_login"
    UNAVAILABLE = "unavailable"
    USER_UPLOAD = "user_upload"


class LinkStatus(StrEnum):
    VERIFIED_200 = "verified_200"
    VERIFIED_REDIRECT = "verified_redirect"
    REQUIRES_LOGIN = "requires_login"
    FAILED = "failed"
    UNCHECKED = "unchecked"


class ClaimStatus(StrEnum):
    VERIFIED = "verified"
    CORROBORATED = "corroborated"
    SUPPORTED = "supported"
    CANDIDATE = "candidate"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"


class ClaimKind(StrEnum):
    QUALITATIVE = "qualitative"
    NUMERIC = "numeric"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    FRESHNESS = "freshness"


class EvidenceSourceKind(StrEnum):
    PRIMARY_DOCUMENT = "primary_document"
    STRUCTURED_TABLE = "structured_table"
    METADATA = "metadata"
    USER_SUPPLIED = "user_supplied"


class ArtifactKind(StrEnum):
    """Canonical artifact kinds used in chat_trail, async_tasks, and
    workspace.context.filters.artifact_index.

    Replaces ~40 string-literal sites across downloads.py, session.py,
    session_metrics.py, artifact_ops.py with one enum reference.
    """

    PAPER_PDF = "paper_pdf"
    SUPPLEMENTARY = "supplementary"
    STRUCTURED_DOCUMENT = "structured_document"
    WORKSPACE = "workspace"
    WORKSPACE_SNAPSHOT = "workspace_snapshot"
    MESSAGES = "messages"
    MEMORY = "memory"
    ARTIFACTS = "artifacts"

    @classmethod
    def embeddable(cls) -> set[str]:
        """Kinds that are submitted to the embedding queue."""
        return {cls.PAPER_PDF, cls.SUPPLEMENTARY, cls.STRUCTURED_DOCUMENT}


class IntentAction(StrEnum):
    """All recognised intent actions. Replaces ~25 string-literal sites
    in intent.py, chat.py, workflow.py with one enum reference."""

    SEARCH = "search"
    PARSE = "parse"
    TABLE = "table"
    STORYLINE = "storyline"
    DOCUMENT = "document"
    DOWNLOAD = "download"
    AUTONOMOUS_REVIEW = "autonomous_review"
    SELECT_DOWNLOADS = "select_downloads"
    DESELECT_DOWNLOADS = "deselect_downloads"
    LIST_CONTEXT = "list_context"
    HIDE_CONTEXT = "hide_context"
    SHOW_CONTEXT = "show_context"
    CANCEL_PENDING_INTENT = "cancel_pending_intent"
    INTENT_PARSE_ERROR = "intent_parse_error"
    CLARIFY = "clarify"
    COMPONENT_STATUS = "component_status"
    RESEARCH_BACKGROUND_SET = "research_background_set"
    RESEARCH_BACKGROUND_REQUIRED = "research_background_required"


class AsyncTaskStatus(StrEnum):
    """Lifecycle status for an async_tasks row.

    Replaces 5 string-literal sites in state_db.py / lifecycle.py /
    rag_jobs.py with one enum reference.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD = "dead"


class VerificationReport(BaseModel):
    claim_id: str | None = None
    claim: str
    status: ClaimStatus
    evidence: list["EvidenceSpan"] = Field(default_factory=list)
    semantic_supported: bool = False
    support_quotes: dict[str, str] = Field(default_factory=dict)
    freshness_checked_at: str | None = None
    critical: bool = True
    warnings: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)

    @property
    def publishable(self) -> bool:
        """Whether the claim can appear in a released report or answer."""

        return self.status in {ClaimStatus.VERIFIED, ClaimStatus.CORROBORATED}

    @property
    def draftable(self) -> bool:
        return self.status in {
            ClaimStatus.VERIFIED,
            ClaimStatus.CORROBORATED,
            ClaimStatus.SUPPORTED,
        }


def verify_claim(
    claim: str,
    evidence: list["EvidenceSpan"],
    *,
    requires_corroboration: bool = False,
    metric: str | None = None,
    claim_id: str | None = None,
    semantic_supported: bool = False,
    support_quotes: dict[str, str] | None = None,
    requires_freshness: bool = False,
    freshness_checked_at: str | None = None,
) -> VerificationReport:
    """Classify a claim from traceable evidence without inventing certainty.

    ``observed_value`` and ``observed_unit`` on evidence spans are optional.
    When independent sources disagree on a measurement they are being used to
    corroborate, the claim is explicitly conflicted rather than averaged away.
    """
    traceable = [span for span in evidence if span.has_location]
    source_ids = {span.source_record_id or span.paper_id for span in traceable}
    if not evidence:
        return VerificationReport(
            claim_id=claim_id,
            claim=claim,
            status=ClaimStatus.UNKNOWN,
            missing_requirements=["No evidence spans were provided."],
        )
    if not traceable:
        return VerificationReport(
            claim_id=claim_id,
            claim=claim,
            status=ClaimStatus.CANDIDATE,
            evidence=evidence,
            missing_requirements=["Evidence needs a page, section, table, or quoted snippet."],
        )
    conflicts = _measurement_conflicts(traceable, metric) if requires_corroboration else []
    if conflicts:
        return VerificationReport(
            claim_id=claim_id,
            claim=claim,
            status=ClaimStatus.CONFLICTED,
            evidence=traceable,
            semantic_supported=semantic_supported,
            support_quotes=support_quotes or {},
            warnings=conflicts,
            missing_requirements=[
                "Independent evidence must agree before this claim can be released."
            ],
        )
    if len(source_ids) >= 2:
        return VerificationReport(
            claim_id=claim_id,
            claim=claim,
            status=ClaimStatus.CORROBORATED,
            evidence=traceable,
            semantic_supported=semantic_supported,
            support_quotes=support_quotes or {},
            freshness_checked_at=freshness_checked_at,
        )
    if requires_corroboration:
        return VerificationReport(
            claim_id=claim_id,
            claim=claim,
            status=ClaimStatus.SUPPORTED,
            evidence=traceable,
            missing_requirements=["A second independent source is required for corroboration."],
        )
    missing_requirements: list[str] = []
    if requires_freshness and not freshness_checked_at:
        missing_requirements.append("Freshness-sensitive claim needs a retrieval cutoff time.")
    return VerificationReport(
        claim_id=claim_id,
        claim=claim,
        status=ClaimStatus.SUPPORTED if missing_requirements else ClaimStatus.VERIFIED,
        evidence=traceable,
        semantic_supported=semantic_supported,
        support_quotes=support_quotes or {},
        freshness_checked_at=freshness_checked_at,
        missing_requirements=missing_requirements,
    )


def _measurement_conflicts(evidence: list["EvidenceSpan"], metric: str | None) -> list[str]:
    """Return deterministic conflicts for numeric evidence from distinct sources."""

    observations: list[tuple[str, float, str | None]] = []
    for span in evidence:
        if not isinstance(span.observed_value, int | float):
            continue
        value, unit = _normalize_observed_measurement(
            metric, span.observed_value, span.observed_unit
        )
        observations.append((span.source_record_id or span.paper_id, float(value), unit))
    source_ids = {source_id for source_id, _, _ in observations}
    if len(source_ids) < 2:
        return []

    units = {unit for _, _, unit in observations}
    if len(units) > 1:
        return ["Independent evidence uses incompatible measurement units."]
    baseline = observations[0][1]
    if any(
        not isclose(value, baseline, rel_tol=0.05, abs_tol=1e-12)
        for _, value, _ in observations[1:]
    ):
        unit = next(iter(units)) or "unitless"
        values = ", ".join(f"{value:g}" for _, value, _ in observations)
        return [f"Independent evidence disagrees on the measured value ({values} {unit})."]
    return []


def _normalize_observed_measurement(
    metric: str | None,
    value: float | int,
    unit: str | None,
) -> tuple[float, str | None]:
    if metric is None:
        return float(value), unit
    from littrace.units import normalize_metric_unit

    normalized_value, normalized_unit, _ = normalize_metric_unit(metric, value, unit)
    return float(normalized_value), normalized_unit


class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    publisher: str | None = None
    doi: str | None = None
    abstract: str | None = None
    citation_count: int | None = None
    source_urls: list[HttpUrl] = Field(default_factory=list)
    pdf_url: HttpUrl | None = None
    access_type: AccessType = AccessType.UNAVAILABLE
    relevance_score: float | None = None
    recency_score: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_url_fields(cls, data: object) -> object:
        # Upstream APIs (Crossref / OpenAlex / Unpaywall) hand back plain
        # string URLs. Without this coercion, Pydantic emits
        # PydanticSerializationUnexpectedValue warnings on every model_dump
        # round-trip during the sentinel run. Coercion preserves the
        # HttpUrl type for downstream consumers.
        return _coerce_http_url_fields(
            data, ["source_urls", "pdf_url"]
        )


class SourceRecord(BaseModel):
    source_record_id: str
    paper_id: str
    source_name: str
    source_url: str | None = None
    retrieved_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    content_hash: str | None = None
    raw_artifact_path: str | None = None
    failure_class: str | None = None


class SourceEvent(BaseModel):
    """Immutable summary of one external-source retrieval attempt."""

    event_id: str
    source_name: str
    request_fingerprint: str
    request: dict[str, object] = Field(default_factory=dict)
    response_hash: str | None = None
    retrieved_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "completed"
    cache_state: str = "miss"
    failure_class: str | None = None
    paper_ids: list[str] = Field(default_factory=list)


class CanonicalWork(BaseModel):
    work_id: str
    canonical_paper_id: str
    doi: str | None = None
    version_paper_ids: list[str] = Field(default_factory=list)


class ResolutionDecision(BaseModel):
    decision_id: str
    canonical_work_id: str
    candidate_paper_ids: list[str]
    strategy: str
    confidence: float
    reason: str
    decided_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class FullTextCandidate(BaseModel):
    paper_id: str
    url: HttpUrl
    source: str
    content_type: str = "landing_page"
    access_type: AccessType = AccessType.UNAVAILABLE
    requires_login: bool = False
    is_pdf: bool = False
    is_xml: bool = False
    confidence: float = 0.0
    verified: bool = False
    status_code: int | None = None
    checked_content_type: str | None = None
    note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_url_field(cls, data: object) -> object:
        return _coerce_http_url_fields(data, ["url"])


class FullTextResolutionReport(BaseModel):
    paper_id: str
    doi: str | None = None
    candidates: list[FullTextCandidate] = Field(default_factory=list)
    best_pdf_url: HttpUrl | None = None
    best_landing_url: HttpUrl | None = None
    open_access_candidate_count: int = 0

    @model_validator(mode="before")
    @classmethod
    def _coerce_url_fields(cls, data: object) -> object:
        return _coerce_http_url_fields(
            data, ["best_pdf_url", "best_landing_url"]
        )
    login_required_candidate_count: int = 0
    verified_candidate_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class PaperSearchRequest(BaseModel):
    topic: str
    discipline: str = "materials chemistry"
    year_min: int | None = 2023
    # Round 17: honour an upper bound on publication year so the
    # ``DailyConfigDialog`` "年份区间" upper bound reaches the
    # retrieval layer. ``None`` means "no upper bound" and is the
    # backward-compat default (older ``PaperSearchRequest``
    # payloads from pre-Round-17 callers didn't have the field).
    year_max: int | None = None
    limit: int = 40
    min_relevant_results: int = 5
    wants_recent: bool = True
    live: bool | None = None
    query_variants: list[str] = Field(default_factory=list)


class ResearchTask(BaseModel):
    task_id: str | None = None
    topic: str
    requested_actions: list[str] = Field(default_factory=list)
    year_min: int | None = None
    journals: list[str] = Field(default_factory=list)
    evidence_policy: str = "verified_or_corroborated"
    requires_freshness: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @model_validator(mode="after")
    def _populate_task_id(self) -> "ResearchTask":
        if self.task_id is None:
            payload = "\0".join([self.topic, *self.requested_actions, str(self.year_min)])
            self.task_id = f"task:{sha256(payload.encode()).hexdigest()[:16]}"
        return self


class PaperSearchResult(BaseModel):
    request: PaperSearchRequest
    papers: list[PaperMetadata]


class TopicRetrievalPolicy(BaseModel):
    """LLM-derived, session-scoped constraints for literature discovery."""

    canonical_topic: str
    query_variants: list[str] = Field(default_factory=list)
    required_concept_groups: list[list[str]] = Field(default_factory=list)
    excluded_concepts: list[str] = Field(default_factory=list)
    boost_concepts: list[str] = Field(default_factory=list)


class DOIBackfillRequest(BaseModel):
    dois: list[str] = Field(default_factory=list)


class WorkspaceFilters(BaseModel):
    """Explicit, serializable metadata describing the current workspace state.

    ``extra="ignore"`` (was ``"forbid"``) lets old persisted sessions load
    cleanly when a new field is added or an old field is renamed — the
    extra keys are silently dropped rather than crashing the load path.
    """

    model_config = ConfigDict(extra="ignore")

    # Search-related
    search_mode: str | None = None
    topic: str | None = None
    search_query: str | None = None
    research_background: str | None = None
    research_retrieval_policy: TopicRetrievalPolicy | None = None
    research_background_status: str | None = None
    research_background_rejection_reason: str | None = None
    research_background_set_at: str | None = None
    research_background_last_sync_at: str | None = None
    research_background_last_downloaded_count: int = 0
    research_background_last_parsed_count: int = 0
    discipline: str | None = None
    expanded_year_range_from: int | None = None
    year_min: int | None = None
    journals: list[str] = Field(default_factory=list)
    active_context_limit: int | None = None
    candidate_pool_count: int = 0
    candidate_pool_ids: list[str] = Field(default_factory=list)
    valid_candidate_count: int = 0
    ranking_policy: str | None = None
    search_diagnostics: dict[str, object] | None = None
    publisher_search_plan: dict[str, object] | None = None
    search_completed_at: str | None = None
    source_health: dict[str, dict[str, object]] = Field(default_factory=dict)
    workspace_revision: int = 0

    # Parse-related
    parsed_full_text_count: int = 0
    downloaded_full_text_count: int = 0
    requested_rag_ready_count: int = 0
    rag_ready_count: int = 0
    paper_pipeline_status: dict[str, str] = Field(default_factory=dict)

    # Full-text context
    full_text_context_warnings: list[str] = Field(default_factory=list)
    pending_intent: dict[str, object] | None = None

    # Storyline / structured artifacts
    storyline_claim_count: int = 0
    structured_artifacts: list[dict[str, object]] = Field(default_factory=list)
    structured_document_count: int = 0
    structured_document_paths: dict[str, str] = Field(default_factory=dict)
    docling_quality_reports: dict[str, dict[str, object]] = Field(default_factory=dict)
    artifact_index: dict[str, object] = Field(default_factory=dict)
    workspace_snapshot_count: int = 0
    rag_profile: dict[str, object] | None = None
    rag_enabled: bool = False
    rag_backend: str | None = None
    rag_last_refreshed_at: str | None = None
    rag_chunk_count: int = 0
    rag_stale_chunk_count: int = 0
    rag_paper_count: int = 0
    rag_last_query: str | None = None
    rag_last_hit_count: int = 0
    rag_source_routes: list[str] = Field(default_factory=list)
    rag_refresh_report: dict[str, object] | None = None

    # Reports (stored as model_dump dicts for serialization)
    document_report: dict[str, object] | None = None
    autonomous_loop_report: dict[str, object] | None = None

    # Publisher retrieval
    publisher_retrievals: list[dict[str, object]] = Field(default_factory=list)

    # Source routes (publisher routing results — list of route names)
    source_routes: list[str] = Field(default_factory=list)


class LiteratureContext(BaseModel):
    visible_to_user: bool = True
    active_papers: list[str] = Field(default_factory=list)
    excluded_papers: list[str] = Field(default_factory=list)
    pinned_papers: list[str] = Field(default_factory=list)
    # Round 19: per-paper importance (1 = normal, 2 = important,
    # 3 = critical). The GUI renders a 🔥 / ⭐ marker so the user
    # can spot core papers at a glance when the active list grows
    # past ~10 entries.
    importance_levels: dict[str, int] = Field(default_factory=dict)
    selected_for_download: list[str] = Field(default_factory=list)
    filters: WorkspaceFilters = Field(default_factory=WorkspaceFilters)


class LiteratureWorkspace(BaseModel):
    context: LiteratureContext = Field(default_factory=LiteratureContext)
    papers: dict[str, PaperMetadata] = Field(default_factory=dict)
    parsed_papers: dict[str, "ParsedPaper"] = Field(default_factory=dict)
    performance_cells: list["PerformanceCell"] = Field(default_factory=list)
    supplementary_links: dict[str, list[str]] = Field(default_factory=dict)
    guard_reports: list[dict[str, object]] = Field(default_factory=list)
    full_text_reports: dict[str, FullTextResolutionReport] = Field(default_factory=dict)
    source_records: dict[str, SourceRecord] = Field(default_factory=dict)
    source_events: list[SourceEvent] = Field(default_factory=list)
    canonical_works: dict[str, CanonicalWork] = Field(default_factory=dict)
    resolution_decisions: list[ResolutionDecision] = Field(default_factory=list)
    evidence_records: dict[str, "EvidenceSpan"] = Field(default_factory=dict)
    claims: list["Claim"] = Field(default_factory=list)
    claim_verification_reports: list[VerificationReport] = Field(default_factory=list)
    release_snapshots: list["ReleaseSnapshot"] = Field(default_factory=list)


class ContextUpdate(BaseModel):
    visible_to_user: bool | None = None
    include_paper_ids: list[str] = Field(default_factory=list)
    exclude_paper_ids: list[str] = Field(default_factory=list)
    pin_paper_ids: list[str] = Field(default_factory=list)
    unpin_paper_ids: list[str] = Field(default_factory=list)
    select_for_download: list[str] = Field(default_factory=list)
    deselect_for_download: list[str] = Field(default_factory=list)
    filters: WorkspaceFilters | None = None


class DownloadPlanItem(BaseModel):
    paper_id: str
    title: str
    access_type: AccessType
    decision: str
    can_download: bool = False
    requires_login: bool = False
    target_dir: str


class DownloadPlan(BaseModel):
    items: list[DownloadPlanItem]
    target_root: str
    requires_login_count: int = 0
    downloadable_count: int = 0


class DownloadExecutionRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    dry_run: bool = False
    session_id: str | None = None
    # Where to land the downloaded PDF:
    # - "local_and_storage" (default): write the per-session paper_library_dir
    #   AND mirror to the artifact object store. Used by user-initiated paths
    #   (chat intent, /downloads/execute, attach-pdf).
    # - "storage_only": write only to the object store. Used by background
    #   paths (daily_update, sentinel) so user machines don't get surprise
    #   PDF files under their working directory.
    target: Literal["local_and_storage", "storage_only"] = "local_and_storage"


class DownloadExecutionItem(BaseModel):
    paper_id: str
    action: str
    status: str
    target_path: str | None = None
    task_id: str | None = None
    storage_ref: dict[str, object] | None = None
    login_url: HttpUrl | None = None
    login_instructions: list[str] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_url_field(cls, data: object) -> object:
        return _coerce_http_url_fields(data, ["login_url"])


class DownloadExecutionResult(BaseModel):
    items: list[DownloadExecutionItem]
    downloaded_count: int = 0
    requires_login_count: int = 0
    skipped_count: int = 0


class PublisherDownloadProgress(BaseModel):
    publisher: str
    total: int = 0
    queued: int = 0
    active: int = 0
    completed: int = 0
    requires_login: int = 0
    failed: int = 0
    percent: float = 0.0


class CitationRecord(BaseModel):
    paper_id: str
    citation_text: str
    access_url: HttpUrl
    link_status: LinkStatus = LinkStatus.UNCHECKED
    doi: str | None = None
    checked_url: HttpUrl | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_url_fields(cls, data: object) -> object:
        return _coerce_http_url_fields(data, ["access_url", "checked_url"])
    status_code: int | None = None
    requires_login: bool = False
    error: str | None = None


class CitationAudit(BaseModel):
    records: list[CitationRecord]
    passed: bool
    score: float
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ResearchRunRequest(BaseModel):
    search: PaperSearchRequest
    audit_citations: bool = True
    plan_downloads: bool = True
    route_publishers: bool = True
    parse_full_text: bool = False
    extract_tables: bool = False
    build_storyline: bool = False
    compose_document: bool = False
    autonomous_review: bool = False
    auto_replan: bool = False


class WorkflowTraceStep(BaseModel):
    node: str
    status: str
    reason: str
    inputs: dict[str, object] = Field(default_factory=dict)
    outputs: dict[str, object] = Field(default_factory=dict)
    next_node: str | None = None
    next_reason: str | None = None


class WorkflowTrace(BaseModel):
    steps: list[WorkflowTraceStep] = Field(default_factory=list)


class ResearchRunResult(BaseModel):
    workspace: LiteratureWorkspace
    citation_audit: CitationAudit | None = None
    download_plan: DownloadPlan | None = None
    publisher_routes: object | None = None
    workflow_status: object | None = None
    parse_report: dict[str, object] | None = None
    table_harness: dict[str, object] | None = None
    comparison_matrix: "ComparisonMatrixReport | None" = None
    storyline: list["StorylineClaim"] | None = None
    document_report: "ResearchDocumentReport | None" = None
    autonomous_loop_report: "ReviewLoopReport | None" = None
    workflow_trace: WorkflowTrace | None = None


class ChatRequest(BaseModel):
    message: str
    live: bool | None = None
    session_id: str | None = None
    research_background: str | None = None


class WorkspaceSummary(BaseModel):
    """API-friendly projection of a full ``LiteratureWorkspace``.

    Strips every large payload (parsed text, structured documents, evidence
    spans, comparison matrices) so the API response stays under 100 KB
    even for a 200-paper session. Callers needing the full payload can
    hit ``/sessions/{id}/export``.
    """

    session_id: str
    paper_count: int = 0
    active_paper_count: int = 0
    parsed_paper_count: int = 0
    excluded_paper_count: int = 0
    papers: list["PaperMetadata"] = Field(default_factory=list)
    active_papers: list[str] = Field(default_factory=list)
    excluded_papers: list[str] = Field(default_factory=list)
    # ``context`` is a slim mirror of ``workspace.context`` so callers that
    # previously read ``workspace.context.active_papers`` keep working
    # without a full ``LiteratureWorkspace`` payload.
    context: dict[str, object] = Field(default_factory=dict)
    filters: dict[str, object] = Field(default_factory=dict)
    topic: str | None = None
    research_background: str | None = None
    research_background_status: str | None = None

    @classmethod
    def from_workspace(
        cls, workspace: "LiteratureWorkspace"
    ) -> "WorkspaceSummary":
        return cls(
            session_id=str(getattr(workspace, "session_id", "") or ""),
            paper_count=len(workspace.papers),
            active_paper_count=len(workspace.context.active_papers),
            parsed_paper_count=len(workspace.parsed_papers),
            excluded_paper_count=len(workspace.context.excluded_papers),
            papers=list(workspace.papers.values()),
            active_papers=list(workspace.context.active_papers),
            excluded_papers=list(workspace.context.excluded_papers),
            context=workspace.context.model_dump(mode="json"),
            filters=workspace.context.filters.model_dump(mode="json"),
            topic=workspace.context.filters.topic,
            research_background=workspace.context.filters.research_background,
            research_background_status=workspace.context.filters.research_background_status,
        )


class ChatResponse(BaseModel):
    reply: str
    action: str
    session_id: str | None = None
    session_root: str | None = None
    intent_confidence: float | None = None
    ambiguous_intent: bool = False
    ambiguity_reasons: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    research_result: ResearchRunResult | None = None
    citations: list[CitationRecord] = Field(default_factory=list)
    download_plan: DownloadPlan | None = None
    publisher_routes: object | None = None
    comparison_matrix: "ComparisonMatrixReport | None" = None
    workflow_trace: WorkflowTrace | None = None
    warnings: list[str] = Field(default_factory=list)
    # API responses carry a bounded summary. The complete workspace remains
    # canonical in session_state and is passed internally between services.
    workspace: "WorkspaceSummary | None" = None


class EvidenceSpan(BaseModel):
    paper_id: str
    evidence_id: str | None = None
    source_record_id: str | None = None
    section: str | None = None
    page: int | None = None
    table_id: str | None = None
    row_label: str | None = None
    column_label: str | None = None
    snippet: str | None = None
    parser: str | None = None
    parser_version: str | None = None
    content_hash: str | None = None
    captured_at: str | None = None
    source_kind: EvidenceSourceKind = EvidenceSourceKind.PRIMARY_DOCUMENT
    observed_value: float | str | None = None
    observed_unit: str | None = None
    observed_value_min: float | None = None
    observed_value_max: float | None = None
    observed_uncertainty: float | None = None
    confidence: float = 0.0

    @model_validator(mode="after")
    def _populate_provenance(self) -> "EvidenceSpan":
        location = "|".join(
            str(value or "")
            for value in (
                self.section,
                self.page,
                self.table_id,
                self.row_label,
                self.column_label,
                self.snippet,
            )
        )
        self.source_record_id = self.source_record_id or f"paper:{self.paper_id}"
        self.content_hash = self.content_hash or sha256(location.encode()).hexdigest()
        self.captured_at = self.captured_at or datetime.now(UTC).isoformat()
        self.parser_version = self.parser_version or f"{self.parser or 'manual'}:v1"
        self.evidence_id = self.evidence_id or (f"ev:{self.paper_id}:{self.content_hash[:16]}")
        return self

    @property
    def has_location(self) -> bool:
        return bool(self.snippet or self.page is not None or self.section or self.table_id)

    @property
    def provenance_complete(self) -> bool:
        return bool(
            self.evidence_id
            and self.source_record_id
            and self.content_hash
            and self.captured_at
            and self.parser_version
            and self.has_location
        )


class Claim(BaseModel):
    claim_id: str | None = None
    text: str
    claim_kind: ClaimKind = ClaimKind.QUALITATIVE
    evidence_ids: list[str] = Field(min_length=1)
    support_quotes: dict[str, str] = Field(default_factory=dict)
    metric: str | None = None
    expected_value: float | None = None
    expected_unit: str | None = None
    expected_value_min: float | None = None
    expected_value_max: float | None = None
    expected_uncertainty: float | None = None
    requires_corroboration: bool = False
    requires_freshness: bool = False
    retrieval_cutoff_at: str | None = None
    critical: bool = True
    # Answer claims may be faithful translations of a verbatim source quote.
    # Derived claims retain stricter semantic verification.
    claim_origin: str = "derived"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @model_validator(mode="after")
    def _populate_claim_id(self) -> "Claim":
        if self.claim_id is None:
            payload = "\0".join([self.text, *sorted(self.evidence_ids)])
            self.claim_id = f"claim:{sha256(payload.encode()).hexdigest()[:16]}"
        return self


class ReleaseSnapshot(BaseModel):
    snapshot_id: str
    created_at: str
    workspace_hash: str
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    report_hash: str | None = None
    config_hash: str | None = None
    release_ready: bool
    release_blockers: list[str] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(default_factory=dict)


class ParsedTable(BaseModel):
    table_id: str
    caption: str | None = None
    cells: list[dict[str, object]] = Field(default_factory=list)
    evidence: EvidenceSpan


class ParsedPaper(BaseModel):
    pdf_path: Path | None = None
    title: str | None = None
    abstract: str | None = None
    structured_document: dict[str, object] = Field(default_factory=dict)
    sections: list[dict[str, object]] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    figures: list[dict[str, object]] = Field(default_factory=list)
    equations: list[dict[str, object]] = Field(default_factory=list)
    parser_reports: list[dict[str, object]] = Field(default_factory=list)
    parsed: bool = False
    error: str | None = None


def coerce_parsed(value: object) -> ParsedPaper:
    """Coerce a dict or ParsedPaper into a ParsedPaper.

    Needed because ``dict.__setitem__`` bypasses Pydantic validation,
    so ``workspace.parsed_papers["p1"] = {...}`` stores a raw dict.
    """
    if isinstance(value, ParsedPaper):
        return value
    if isinstance(value, dict):
        return ParsedPaper.model_validate(value)
    return ParsedPaper()


class PerformanceCell(BaseModel):
    paper_id: str
    task: str | None = None
    dataset: str | None = None
    metric: str
    value: float | str
    value_min: float | None = None
    value_max: float | None = None
    uncertainty: float | None = None
    unit: str | None = None
    higher_is_better: bool | None = None
    method_name: str | None = None
    conditions: "ExperimentalConditions" = Field(default_factory=lambda: ExperimentalConditions())
    evidence: EvidenceSpan


class ExperimentalConditions(BaseModel):
    material_system: str | None = None
    device_structure: str | None = None
    test_protocol: str | None = None
    environment: str | None = None
    loading_range: str | None = None
    sample_count: int | None = None

    @property
    def complete(self) -> bool:
        return bool(self.test_protocol and self.environment)


class StructuredArtifact(BaseModel):
    paper_id: str
    artifact_type: str
    label: str | None = None
    text: str
    evidence: EvidenceSpan
    confidence: float = 0.0


class ComparisonMatrixRow(BaseModel):
    paper_id: str
    title: str | None = None
    year: int | None = None
    metric: str
    value: float | str
    unit: str | None = None
    task: str | None = None
    dataset: str | None = None
    method_name: str | None = None
    conditions: ExperimentalConditions = Field(default_factory=ExperimentalConditions)
    higher_is_better: bool | None = None
    comparable: bool = True
    warnings: list[str] = Field(default_factory=list)
    evidence: EvidenceSpan


class ComparisonMatrix(BaseModel):
    metric: str
    rows: list[ComparisonMatrixRow]
    warnings: list[str] = Field(default_factory=list)


class ComparisonMatrixReport(BaseModel):
    matrices: list[ComparisonMatrix]
    warnings: list[str] = Field(default_factory=list)


class ResearchDocumentSection(BaseModel):
    title: str
    body: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class ResearchDocumentReport(BaseModel):
    title: str
    markdown: str
    sections: list[ResearchDocumentSection] = Field(default_factory=list)
    citation_records: list[CitationRecord] = Field(default_factory=list)
    verification_reports: list[VerificationReport] = Field(default_factory=list)
    release_ready: bool = True
    release_blockers: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    quality_metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    release_snapshot: ReleaseSnapshot | None = None


class ReviewFinding(BaseModel):
    reviewer: str
    severity: str = "info"
    finding: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    suggested_fix: str | None = None


class ReviewRound(BaseModel):
    round_index: int
    writer_draft: str
    critiques: list[ReviewFinding] = Field(default_factory=list)
    revised_draft: str
    passed: bool
    score: float
    replan_actions: list[str] = Field(default_factory=list)
    executed_replan_actions: list[str] = Field(default_factory=list)


class ReviewLoopReport(BaseModel):
    objective: str
    final_answer: str
    rounds: list[ReviewRound] = Field(default_factory=list)
    passed: bool
    score: float
    replan_actions: list[str] = Field(default_factory=list)
    executed_replan_actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    release_ready: bool = False
    release_blockers: list[str] = Field(default_factory=list)


class StorylineClaim(BaseModel):
    claim: str
    claim_type: str
    evidence: list[EvidenceSpan]
    confidence: float = 0.0


# Resolve forward references
LiteratureWorkspace.model_rebuild()
ResearchRunResult.model_rebuild()
ChatResponse.model_rebuild()


# ---------------------------------------------------------------------------
# Round 8: thread steering + review + cancel-with-reason request / response
# models. Kept here (instead of in routes/research.py) so the
# generated OpenAPI surface shows the same schema regardless of
# where the route is mounted, and so a generated client can
# share types with the SSE streaming route.
# ---------------------------------------------------------------------------


class ChatSteerRequest(BaseModel):
    """Body of ``POST /chat/steer``.

    Mirrors the codex-harness ``turn/steer`` request parameters:
    ``turn_id`` identifies the active turn the input lands on,
    ``text`` is the additional user input, and the optional
    ``client_user_message_id`` echoes back in the corresponding
    ``userMessage`` item so the caller can correlate stream
    notifications.
    """

    turn_id: str
    text: str
    client_user_message_id: str | None = None
    session_id: str | None = None


class ChatSteerResponse(BaseModel):
    turn_id: str
    thread_id: str
    client_user_message_id: str | None = None
    session_id: str | None = None


class ChatReviewRequest(BaseModel):
    """Body of ``POST /chat/review``."""

    target: dict[str, object] | None = None
    session_id: str | None = None


class ChatReviewResponse(BaseModel):
    turn_id: str
    thread_id: str
    status: str
    review_text: str = ""
    exit_item: dict[str, object] | None = None
    session_id: str | None = None


class ChatCancelRequest(BaseModel):
    """Body of ``POST /chat/{turn_id}/cancel``.

    ``reason`` is a free-form short string. The service layer
    stamps it into ``agent_thread_bindings.last_error`` so an
    operator can later distinguish user-triggered cancellations
    (Esc) from worker-triggered ones (compaction) from
    model-loop bailouts.
    """

    reason: str
    session_id: str | None = None


class ChatCancelResponse(BaseModel):
    turn_id: str
    session_id: str | None = None
    reason: str
    acknowledged: bool
