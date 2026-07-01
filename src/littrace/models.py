from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class AccessType(StrEnum):
    OPEN_ACCESS = "open_access"
    REQUIRES_LOGIN = "requires_login"
    UNAVAILABLE = "unavailable"
    METADATA_ONLY = "metadata_only"
    USER_UPLOAD = "user_upload"


class LinkStatus(StrEnum):
    VERIFIED_200 = "verified_200"
    VERIFIED_REDIRECT = "verified_redirect"
    REQUIRES_LOGIN = "requires_login"
    FAILED = "failed"
    UNCHECKED = "unchecked"


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
    access_type: AccessType = AccessType.METADATA_ONLY
    relevance_score: float | None = None
    recency_score: float | None = None


class FullTextCandidate(BaseModel):
    paper_id: str
    url: HttpUrl
    source: str
    content_type: str = "landing_page"
    access_type: AccessType = AccessType.METADATA_ONLY
    requires_login: bool = False
    is_pdf: bool = False
    is_xml: bool = False
    confidence: float = 0.0
    verified: bool = False
    status_code: int | None = None
    checked_content_type: str | None = None
    note: str | None = None


class FullTextResolutionReport(BaseModel):
    paper_id: str
    doi: str | None = None
    candidates: list[FullTextCandidate] = Field(default_factory=list)
    best_pdf_url: HttpUrl | None = None
    best_landing_url: HttpUrl | None = None
    open_access_candidate_count: int = 0
    login_required_candidate_count: int = 0
    verified_candidate_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class PaperSearchRequest(BaseModel):
    topic: str
    discipline: str = "materials chemistry"
    year_min: int | None = 2023
    limit: int = 20
    wants_recent: bool = True
    live: bool | None = None


class PaperSearchResult(BaseModel):
    request: PaperSearchRequest
    papers: list[PaperMetadata]


class DOIBackfillRequest(BaseModel):
    dois: list[str] = Field(default_factory=list)


class LiteratureContext(BaseModel):
    visible_to_user: bool = True
    active_papers: list[str] = Field(default_factory=list)
    excluded_papers: list[str] = Field(default_factory=list)
    pinned_papers: list[str] = Field(default_factory=list)
    selected_for_download: list[str] = Field(default_factory=list)
    filters: dict[str, object] = Field(default_factory=dict)


class LiteratureWorkspace(BaseModel):
    context: LiteratureContext = Field(default_factory=LiteratureContext)
    papers: dict[str, PaperMetadata] = Field(default_factory=dict)
    parsed_papers: dict[str, dict[str, object]] = Field(default_factory=dict)
    performance_cells: list["PerformanceCell"] = Field(default_factory=list)
    supplementary_links: dict[str, list[str]] = Field(default_factory=dict)
    guard_reports: list[dict[str, object]] = Field(default_factory=list)
    full_text_reports: dict[str, FullTextResolutionReport] = Field(default_factory=dict)


class ContextUpdate(BaseModel):
    visible_to_user: bool | None = None
    include_paper_ids: list[str] = Field(default_factory=list)
    exclude_paper_ids: list[str] = Field(default_factory=list)
    pin_paper_ids: list[str] = Field(default_factory=list)
    unpin_paper_ids: list[str] = Field(default_factory=list)
    select_for_download: list[str] = Field(default_factory=list)
    deselect_for_download: list[str] = Field(default_factory=list)
    filters: dict[str, object] | None = None


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


class DownloadExecutionItem(BaseModel):
    paper_id: str
    action: str
    status: str
    target_path: str | None = None
    login_url: HttpUrl | None = None
    login_instructions: list[str] = Field(default_factory=list)
    error: str | None = None


class DownloadExecutionResult(BaseModel):
    items: list[DownloadExecutionItem]
    downloaded_count: int = 0
    requires_login_count: int = 0
    skipped_count: int = 0


class CitationRecord(BaseModel):
    paper_id: str
    citation_text: str
    access_url: HttpUrl
    link_status: LinkStatus = LinkStatus.UNCHECKED
    doi: str | None = None
    checked_url: HttpUrl | None = None
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


class ResearchRunResult(BaseModel):
    workspace: LiteratureWorkspace
    citation_audit: CitationAudit | None = None
    download_plan: DownloadPlan | None = None
    publisher_routes: object | None = None
    agent_interactions: object | None = None
    parse_report: dict[str, object] | None = None
    table_harness: dict[str, object] | None = None
    comparison_matrix: "ComparisonMatrixReport | None" = None
    storyline: list["StorylineClaim"] | None = None
    document_report: "ResearchDocumentReport | None" = None
    autonomous_loop_report: "AutonomousResearchLoopReport | None" = None


class ChatRequest(BaseModel):
    message: str
    live: bool | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    action: str
    session_id: str | None = None
    session_root: str | None = None
    workspace: LiteratureWorkspace | None = None
    research_result: ResearchRunResult | None = None
    citations: list[CitationRecord] = Field(default_factory=list)
    download_plan: DownloadPlan | None = None
    publisher_routes: object | None = None
    comparison_matrix: "ComparisonMatrixReport | None" = None
    warnings: list[str] = Field(default_factory=list)


class EvidenceSpan(BaseModel):
    paper_id: str
    section: str | None = None
    page: int | None = None
    table_id: str | None = None
    row_label: str | None = None
    column_label: str | None = None
    snippet: str | None = None
    parser: str | None = None
    confidence: float = 0.0


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
    evidence: EvidenceSpan


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
    evidence_count: int = 0
    quality_metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AgentCritique(BaseModel):
    reviewer: str
    severity: str = "info"
    finding: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    suggested_fix: str | None = None


class AgentDebateRound(BaseModel):
    round_index: int
    writer_draft: str
    critiques: list[AgentCritique] = Field(default_factory=list)
    revised_draft: str
    passed: bool
    score: float
    replan_actions: list[str] = Field(default_factory=list)


class AutonomousResearchLoopReport(BaseModel):
    objective: str
    final_answer: str
    rounds: list[AgentDebateRound] = Field(default_factory=list)
    passed: bool
    score: float
    replan_actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StorylineClaim(BaseModel):
    claim: str
    claim_type: str
    evidence: list[EvidenceSpan]
    confidence: float = 0.0
