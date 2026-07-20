from __future__ import annotations

from enum import StrEnum
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DownloadMode(StrEnum):
    ASK_EACH_TIME = "ask_each_time"
    DOWNLOAD_OPEN_ACCESS = "download_open_access"
    DOWNLOAD_SELECTED = "download_selected"
    DOWNLOAD_ALL_ALLOWED = "download_all_allowed"


class StorageConfig(BaseModel):
    paper_library_dir: Path = Path("./data/papers")
    metadata_dir: Path = Path("./data/metadata")
    cache_dir: Path = Path("./data/cache")
    sessions_dir: Path = Path("./sessions")
    workspace_snapshot_limit: int = 30


class CachePolicyConfig(BaseModel):
    default_ttl_seconds: int = 86_400
    allow_stale_on_source_failure: bool = True


class PublicationPolicyConfig(BaseModel):
    strict_all_claims: bool = True
    require_publishable_claim: bool = True


class PaperDownloadConfig(BaseModel):
    mode: DownloadMode = DownloadMode.ASK_EACH_TIME
    organize_by: str = "year_doi"
    filename_template: str = "{year}_{first_author}_{short_title}_{doi_hash}.pdf"
    save_metadata_even_if_pdf_skipped: bool = False
    allow_requires_login_download: bool = True


class DoclingParserConfig(BaseModel):
    export_markdown: bool = True
    extract_tables: bool = True
    extract_figures: bool = True


class PaddleOCRParserConfig(BaseModel):
    lang: str = "en"
    use_angle_cls: bool = True
    pdf_render_scale: float = 2.0
    max_pages: int | None = None
    ocr_batch_size: int = 4
    ocr_page_workers: int = 1
    cache_enabled: bool = True
    cache_dir: Path | None = None


class ParsingConfig(BaseModel):
    default_parser: str = "docling"
    parse_strategy: str = "auto"
    preferred_engines: list[str] = Field(
        default_factory=lambda: ["docling", "paddleocr", "marker", "grobid"]
    )
    docling: DoclingParserConfig = Field(default_factory=DoclingParserConfig)
    paddleocr: PaddleOCRParserConfig = Field(default_factory=PaddleOCRParserConfig)


class APIConfig(BaseModel):
    user_agent: str = "LitTrace/0.1"
    openalex_api_key: str | None = None
    unpaywall_email: str | None = None
    crossref_mailto: str | None = None
    request_timeout_seconds: float = 20.0
    enable_live_search: bool = False


class BrowserAutomationConfig(BaseModel):
    browser_act_path: str = "browser-act"
    required: bool = True
    default_browser_id: str | None = None
    default_browser_name: str = "littrace-publisher-auth"
    default_browser_type: str = "chrome"
    confirm_before_use: bool = True
    allow_confirm_browser_fallback: bool = True
    chrome_direct_open_retries: int = 3
    chrome_direct_retry_delay_seconds: float = 2.0


class CDPDownloaderConfig(BaseModel):
    cdp_url: str = "http://127.0.0.1:19222"
    default_output_dir: Path | None = None
    chrome_executable: Path | None = None
    chrome_user_data_dir: Path | None = None
    chrome_profile_name: str = "Default"
    remote_debugging_port: int = 19222
    auto_launch_chrome: bool = False
    cloudflare_wait_seconds: float = 60.0
    user_action_wait_seconds: float = 30.0
    command_timeout_seconds: float = 60.0
    websocket_reconnect_attempts: int = 3


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    fallback_models: list[str] = Field(default_factory=list)
    fallback_api_key: str | None = None
    fallback_base_url: str | None = None
    fallback_model: str | None = None
    request_timeout_seconds: float = 30.0
    temperature: float = 0.2
    enabled: bool = True
    intent_parser_enabled: bool = True


class LiteratureContextDefaults(BaseModel):
    visible_to_user: bool = True
    default_year_min: int = 2023
    default_recent_year_min: int = 2023
    active_context_limit: int = 15
    preferred_disciplines: list[str] = Field(default_factory=list)
    preferred_publishers: list[str] = Field(default_factory=list)
    preferred_journals: list[str] = Field(default_factory=list)


class EvalConfig(BaseModel):
    golden_set_dir: Path = Path("./eval/golden")
    traces_dir: Path = Path("./eval/traces")


class HarnessThresholdConfig(BaseModel):
    """Thresholds for harness quality checks — configurable via config.yaml."""

    performance_confidence: float = 0.65
    artifact_confidence: float = 0.6
    storyline_confidence: float = 0.7
    storyline_chain_min_evidence: int = 3


class RetryPolicyConfig(BaseModel):
    """Unified retry policy for all HTTP/LLM/external calls."""

    max_attempts: int = 3
    backoff_strategy: str = "exponential"
    base_delay_seconds: float = 0.8
    max_delay_seconds: float = 30.0
    # Health-gate thresholds for check_retry_health
    max_retry_rate: float = 0.5
    max_failure_rate: float = 0.2
    # Persistence: if set, retry traces are saved/loaded from this path
    persist_path: str | None = None
    # Max traces to retain in the persistent file (0 = unlimited)
    max_persist_entries: int = 2000


class CostBudgetConfig(BaseModel):
    """Token cost budget and tracking configuration.

    Set ``max_total_tokens`` to 0 to disable budget enforcement.
    """

    max_total_tokens: int = 0  # 0 = unlimited
    max_tokens_per_call: int = 0  # 0 = no per-call limit
    # Per-model price table (USD per 1K tokens) for cost estimation
    price_per_1k_input: dict[str, float] = Field(default_factory=lambda: {"deepseek-chat": 0.001})
    price_per_1k_output: dict[str, float] = Field(default_factory=lambda: {"deepseek-chat": 0.002})
    # Health-gate threshold: what fraction of budget has been consumed
    budget_warning_threshold: float = 0.8
    # Persistence: if set, cost tracker state is saved/loaded from this path
    persist_path: str | None = None
    # Max entries to retain in the persistent file (0 = unlimited).
    # When exceeded, oldest entries are pruned and file is rotated.
    max_persist_entries: int = 5000


class SchemaValidationConfig(BaseModel):
    """Schema validation configuration for LLM outputs."""

    enabled: bool = True
    # If True, schema validation failures produce ERROR findings;
    # if False, they produce WARNING findings
    strict: bool = True
    # If True, invalid items are dropped silently;
    # if False, they are kept with a finding (but not blocking)
    drop_invalid: bool = False


class RateLimitConfig(BaseModel):
    """Rate limiting for LLM API calls.

    Set ``max_concurrent`` or ``max_requests_per_minute`` to 0 to disable
    that layer of limiting. Both layers compose: concurrency limits parallel
    in-flight requests, while the sliding-window rate limit caps sustained
    throughput.
    """

    max_concurrent: int = 0  # 0 = unlimited
    max_requests_per_minute: int = 0  # 0 = unlimited


class CitationGuardConfig(BaseModel):
    """Configuration for the citation guard's claim-hint word list.

    ``claim_hints`` is a list of keywords that trigger citation checking.
    Sentences containing any of these words are considered "claim-bearing"
    and must have citation anchors. Extend this list for domain-specific
    vocabulary (e.g. add "gauge factor", "TFT", "bandgap" for sensor research).
    """

    claim_hints: list[str] = Field(default_factory=list)

    def effective_hints(self, defaults: list[str]) -> list[str]:
        """Return merged hints: configured hints if non-empty, else defaults."""
        return self.claim_hints if self.claim_hints else defaults


class LitTraceConfig(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    browser: BrowserAutomationConfig = Field(default_factory=BrowserAutomationConfig)
    cdp_downloader: CDPDownloaderConfig = Field(default_factory=CDPDownloaderConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    paper_download: PaperDownloadConfig = Field(default_factory=PaperDownloadConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    literature_context: LiteratureContextDefaults = Field(default_factory=LiteratureContextDefaults)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    harness: HarnessThresholdConfig = Field(default_factory=HarnessThresholdConfig)
    retry: RetryPolicyConfig = Field(default_factory=RetryPolicyConfig)
    cost_budget: CostBudgetConfig = Field(default_factory=CostBudgetConfig)
    schema_validation: SchemaValidationConfig = Field(default_factory=SchemaValidationConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    citation_guard: CitationGuardConfig = Field(default_factory=CitationGuardConfig)
    cache_policy: CachePolicyConfig = Field(default_factory=CachePolicyConfig)
    publication_policy: PublicationPolicyConfig = Field(default_factory=PublicationPolicyConfig)


def load_config(path: str | Path = "config.yaml") -> LitTraceConfig:
    _load_env_file(Path(".env.local"))
    config_path = Path(path)
    if not config_path.exists():
        return _with_env_overrides(LitTraceConfig())

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _with_env_overrides(LitTraceConfig.model_validate(raw))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")


def _with_env_overrides(config: LitTraceConfig) -> LitTraceConfig:
    config.llm.api_key = os.environ.get("DEEPSEEK_API_KEY") or config.llm.api_key
    config.llm.base_url = os.environ.get("DEEPSEEK_BASE_URL") or config.llm.base_url
    config.llm.model = os.environ.get("DEEPSEEK_MODEL") or config.llm.model
    config.browser.browser_act_path = (
        os.environ.get("LITTRACE_BROWSER_ACT_PATH") or config.browser.browser_act_path
    )
    config.browser.default_browser_id = (
        os.environ.get("LITTRACE_BROWSER_ID") or config.browser.default_browser_id
    )
    config.cdp_downloader.cdp_url = (
        os.environ.get("LITTRACE_CDP_URL") or config.cdp_downloader.cdp_url
    )
    chrome_executable = os.environ.get("LITTRACE_CHROME_EXECUTABLE")
    if chrome_executable:
        config.cdp_downloader.chrome_executable = Path(chrome_executable).expanduser()
    chrome_user_data_dir = os.environ.get("LITTRACE_CHROME_USER_DATA_DIR")
    if chrome_user_data_dir:
        config.cdp_downloader.chrome_user_data_dir = Path(chrome_user_data_dir).expanduser()
    config.cdp_downloader.chrome_profile_name = (
        os.environ.get("LITTRACE_CHROME_PROFILE_NAME") or config.cdp_downloader.chrome_profile_name
    )
    remote_port = os.environ.get("LITTRACE_REMOTE_DEBUGGING_PORT")
    if remote_port:
        try:
            config.cdp_downloader.remote_debugging_port = int(remote_port)
        except ValueError:
            pass
        else:
            config.cdp_downloader.cdp_url = (
                f"http://127.0.0.1:{config.cdp_downloader.remote_debugging_port}"
            )
    fallback_models = os.environ.get("LITTRACE_LLM_FALLBACK_MODELS")
    if fallback_models:
        config.llm.fallback_models = [
            model.strip() for model in fallback_models.split(",") if model.strip()
        ]
    config.llm.fallback_api_key = (
        os.environ.get("LITTRACE_FALLBACK_LLM_API_KEY") or config.llm.fallback_api_key
    )
    config.llm.fallback_base_url = (
        os.environ.get("LITTRACE_FALLBACK_LLM_BASE_URL") or config.llm.fallback_base_url
    )
    config.llm.fallback_model = (
        os.environ.get("LITTRACE_FALLBACK_LLM_MODEL") or config.llm.fallback_model
    )
    return config
