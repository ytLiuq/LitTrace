from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from typing import Literal


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


class ArtifactStorageConfig(BaseModel):
    backend: str = "local"
    local_root: Path = Path("./data/artifacts")
    bucket: str | None = None
    endpoint_url: str | None = None
    region: str | None = None
    path_prefix: str = ""


ObjectStoreConfig = ArtifactStorageConfig


class MetadataStoreConfig(BaseModel):
    backend: str = "postgres"
    postgres_dsn: str | None = "postgresql://littrace:littrace@localhost:5433/littrace"
    schema_name: str = "littrace"
    connect_timeout_seconds: int = 5
    # When True, state_store bootstrap will DROP legacy tables before creating
    # the consolidated layout. Default off so a typo in DSN can't nuke data.
    # Override with LITTRACE_ALLOW_SCHEMA_RESET=1 (demo stage only).
    allow_schema_reset: bool = False


class RagConfig(BaseModel):
    enabled: bool = False
    backend: str = "pgvector"
    postgres_dsn: str | None = None
    schema_name: str = "littrace_rag"
    collection_prefix: str = "littrace"
    embedding_provider: str = "openai-compatible"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    chunk_target_tokens: int = 700
    chunk_overlap_tokens: int = 120
    top_k: int = 12
    refresh_frequency: str = "daily"
    auto_refresh_enabled: bool = False
    # Daily-update auto-download: when on, the daily RAG job downloads
    # new OA PDFs automatically — but only into the artifact object store.
    # Bytes never land in the user's local paper_library_dir from this path.
    # For local PDFs the user must explicitly invoke a manual download.
    auto_download_open_access: bool = True
    login_required_policy: str = "queue_only"
    # --- Round 7: ANN index tuning --------------------------------------
    # ``index_kind`` selects the pgvector index family:
    #   * ``"hnsw"``  — default, recall-biased, good for up to ~1M rows.
    #   * ``"ivfflat"`` — train-once, faster build, slightly lower recall.
    #   * ``"none"``   — disable ANN; useful for tiny corpora (<1k rows)
    #                   where the planner prefers a sequential scan.
    # The HNSW parameters are the standard pgvector knobs: ``m`` is
    # the degree of each node, ``ef_construction`` trades build
    # time for graph quality, and ``ef_search`` is the per-query
    # search beam. Defaults are pgvector's; larger values lift
    # recall at the cost of latency. Benchmark script
    # (``scripts/benchmark_pgvector.py``) measures both axes.
    index_kind: Literal["hnsw", "ivfflat", "none"] = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    ivfflat_lists: int = 100


class FigureEnrichmentConfig(BaseModel):
    """Optional asynchronous multimodal enrichment for extracted figures."""

    enabled: bool = False
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    prompt: str = (
        "Analyze this scientific figure conservatively. Return JSON only with "
        "exactly these fields: figure_type (string), visual_summary (string), "
        "observations (array of strings), ocr_text (array of strings), "
        "confidence (number from 0 to 1). "
        "Only report details visible in the image or stated in the caption. "
        "Do not infer unshown experimental results."
    )
    max_figures_per_job: int = Field(default=20, ge=1, le=100)
    min_confidence: float = Field(default=0.65, ge=0.0, le=1.0)


class DownloadRetryConfig(BaseModel):
    enabled: bool = True
    background_worker_enabled: bool = False
    interval_seconds: float = 30.0
    batch_size: int = 10
    max_attempts: int = 3
    base_delay_seconds: float = 60.0


class CompactionConfig(BaseModel):
    """Round 5 settings for the Codex thread-compaction worker.

    ``threshold_turns`` and ``threshold_tokens`` are OR-ed when
    deciding whether a session is due for compaction. Once a
    successful ``thread/compact`` RPC runs, ``last_compacted_at``
    is stamped and the session is ignored for one hour
    regardless of how fast more turns pile up.
    """

    enabled: bool = False
    background_worker_enabled: bool = False
    interval_seconds: float = 60.0
    batch_size: int = 5
    max_attempts: int = 3
    base_delay_seconds: float = 30.0
    threshold_turns: int = 30
    threshold_tokens: int = 50_000
    request_timeout_seconds: float = 120.0


class PaperDownloadConfig(BaseModel):
    mode: DownloadMode = DownloadMode.ASK_EACH_TIME
    organize_by: str = "year_doi"
    filename_template: str = "{year}_{first_author}_{short_title}_{doi_hash}.pdf"
    save_metadata_even_if_pdf_skipped: bool = False
    # User-initiated paths (chat intent, /downloads/execute, attach-pdf) may
    # still pull gated PDFs through the local CDP login handoff.
    allow_requires_login_download: bool = True
    max_concurrent_downloads: int = Field(default=3, ge=1, le=8)


class DoclingParserConfig(BaseModel):
    export_markdown: bool = True
    extract_tables: bool = True
    extract_figures: bool = True
    describe_figures: bool = False


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
    docling_workers: int = Field(default=2, ge=1, le=4)
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
    core_api_key: str | None = None
    enable_europe_pmc: bool = True
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
    # LitTrace ships with a private Chrome profile under ./data/chrome-cdp
    # so it never collides with the user's day-to-day Chrome instance
    # (whose user-data-dir is locked while it is running). On the first
    # ``setup-browser`` launch LitTrace creates this directory and points
    # Chrome at it via --user-data-dir; the user must sign in to each
    # publisher they want full-text access to, once. Set to null to fall
    # back to the platform default Chrome user-data-dir (only safe when
    # your normal Chrome is not running).
    chrome_user_data_dir: Path | None = Path("./data/chrome-cdp")
    chrome_profile_name: str = "Default"
    remote_debugging_port: int = 19222
    # LitTrace launches its private Chrome on demand so the user does
    # not need to run ``setup-browser --launch`` before every session.
    # Disable here (or set LITTRACE_AUTO_LAUNCH_CHROME=false) to opt out.
    auto_launch_chrome: bool = True
    cloudflare_wait_seconds: float = 60.0
    user_action_wait_seconds: float = 30.0
    command_timeout_seconds: float = 60.0
    repository_download_timeout_seconds: float = 120.0
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
    request_timeout_seconds: float = 60.0
    metric_extraction_timeout_seconds: float = 150.0
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
    rag_golden_set_dir: Path = Path("./eval/rag_golden")
    traces_dir: Path = Path("./eval/traces")


class AgentRuntimeMode(StrEnum):
    LEGACY = "legacy"
    CODEX_APP_SERVER = "codex_app_server"


class CodexHomeMode(StrEnum):
    """How the App Server process resolves its Codex state directory."""

    ISOLATED = "isolated"
    SHARED = "shared"


class SandboxPolicy(StrEnum):
    """Three-tier sandbox policy aligned with codex-harness.

    Mirrors the App Server's ``sandbox`` parameter so operators can pick
    how much filesystem reach a Codex turn gets without hand-editing the
    thread-start overrides.
    """

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"


class AgentRuntimeConfig(BaseModel):
    """Execution runtime selected for interactive chat turns.

    ``legacy`` keeps the existing in-process coordinator.  The App Server
    runtime is deliberately opt-in while domain mutations move behind
    controlled, idempotent MCP commands.
    """

    mode: AgentRuntimeMode = AgentRuntimeMode.LEGACY
    codex_command: list[str] = Field(default_factory=lambda: ["codex", "app-server"])
    codex_config_overrides: dict[str, str] = Field(default_factory=dict)
    codex_home_mode: CodexHomeMode = CodexHomeMode.ISOLATED
    codex_home: Path = Path("./data/codex-home")
    scratch_root: Path = Path("./data/codex-runtime")
    startup_timeout_seconds: float = 20.0
    request_timeout_seconds: float = 60.0
    turn_timeout_seconds: float = 300.0
    fallback_to_legacy: bool = True
    mcp_server_name: str = "littrace"
    sandbox_policy: SandboxPolicy = SandboxPolicy.READ_ONLY
    writable_roots: list[Path] = Field(default_factory=list)
    interrupt_grace_seconds: float = 10.0
    # Rollout JSONL is opt-in. It is a side-channel for debugging
    # only — Postgres is still the source of truth.
    rollout_enabled: bool = False
    rollout_dir: Path | None = None


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
    artifact_storage: ArtifactStorageConfig = Field(default_factory=ArtifactStorageConfig)
    metadata_store: MetadataStoreConfig = Field(default_factory=MetadataStoreConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    figure_enrichment: FigureEnrichmentConfig = Field(default_factory=FigureEnrichmentConfig)
    download_retry: DownloadRetryConfig = Field(default_factory=DownloadRetryConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    browser: BrowserAutomationConfig = Field(default_factory=BrowserAutomationConfig)
    cdp_downloader: CDPDownloaderConfig = Field(default_factory=CDPDownloaderConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    paper_download: PaperDownloadConfig = Field(default_factory=PaperDownloadConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    literature_context: LiteratureContextDefaults = Field(default_factory=LiteratureContextDefaults)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    agent_runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)
    harness: HarnessThresholdConfig = Field(default_factory=HarnessThresholdConfig)
    retry: RetryPolicyConfig = Field(default_factory=RetryPolicyConfig)
    cost_budget: CostBudgetConfig = Field(default_factory=CostBudgetConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    citation_guard: CitationGuardConfig = Field(default_factory=CitationGuardConfig)
    # --- flat fields merged from collapsed sub-configs (5 deleted) ---
    # Cache policy (was CachePolicyConfig)
    cache_default_ttl_seconds: int = 86_400
    cache_allow_stale_on_source_failure: bool = True
    # Publication policy (was PublicationPolicyConfig)
    publication_strict_all_claims: bool = True
    publication_require_publishable_claim: bool = True
    # Sentinel (was SentinelConfig)
    sentinel_parse_on_daily: bool = True
    # Eval paths. The nested ``eval.golden_set_dir`` is the source of
    # truth — these top-level aliases are kept for backward compatibility
    # with serialized configs and external scripts. New code should read
    # ``config.eval.*`` instead.
    eval_golden_set_dir: Path = Path("./eval/golden")
    eval_traces_dir: Path = Path("./eval/traces")
    # Schema validation (was SchemaValidationConfig)
    schema_validation_enabled: bool = True
    schema_validation_strict: bool = True
    schema_validation_drop_invalid: bool = False

    @property
    def object_store(self) -> ArtifactStorageConfig:
        return self.artifact_storage

    @object_store.setter
    def object_store(self, value: ArtifactStorageConfig) -> None:
        self.artifact_storage = value


def load_config(path: str | Path | None = None) -> LitTraceConfig:
    """Load LitTrace configuration.

    Resolution order:
      1. ``LITTRACE_CONFIG`` environment variable (explicit override).
      2. ``path`` argument if provided.
      3. ``./config.yaml`` (relative to the current working directory).

    Raises ``FileNotFoundError`` when no config file is reachable instead
    of silently falling back to Pydantic defaults — the previous silent
    fallback made every CLI invocation run with the wrong ``mode``,
    ``rag.enabled`` and ``codex_command`` whenever the user launched
    littrace from outside the repository root (e.g. Finder, IDE).
    """
    _load_env_file(Path(".env.local"))
    env_override = os.environ.get("LITTRACE_CONFIG")
    candidates: list[Path] = []
    if env_override:
        candidates.append(Path(env_override))
    if path is not None:
        candidates.append(Path(path))
    candidates.append(Path("config.yaml"))

    for candidate in candidates:
        if candidate.exists():
            config_path = candidate
            break
    else:
        tried = ", ".join(str(c) for c in candidates)
        cwd = Path.cwd()
        raise FileNotFoundError(
            f"Could not locate LitTrace config (tried: {tried}). "
            f"cwd={cwd}. Run littrace commands from the repository root, "
            f"or set LITTRACE_CONFIG=/path/to/config.yaml."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    try:
        validated = LitTraceConfig.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError in practice
        raise _format_config_error(config_path, exc) from exc
    config = _with_env_overrides(validated)
    # Remember the config.yaml location so other components (notably
    # ``littrace doctor`` and the codex_home resolver) can interpret
    # relative paths against the same directory the user actually picked,
    # regardless of the cwd they happened to launch from.
    config._config_path = config_path
    # Re-anchor every relative Path declared in config.yaml to the config
    # directory. Without this, a user launching ``littrace-qt`` from
    # Finder or a parent directory ends up with paper_library_dir pointing
    # at ``./data/papers`` relative to the wrong cwd — and silent ``parsed=0``
    # failures in sentinel because there are no PDFs to parse.
    config = _reanchor_relative_paths(config, config_path.parent)
    return config


def _format_config_error(config_path: Path, exc: Exception) -> Exception:
    """Wrap a Pydantic ``ValidationError`` (or any other model
    validation failure) in a more user-friendly exception so the
    CLI / GUI exit message points at the offending field rather
    than dumping the raw error repr.

    Round 17: ``model_validate`` raises ``pydantic.ValidationError``
    with a ``loc`` tuple (e.g. ``("agent_runtime", "turn_timeout_seconds")``)
    and a ``msg`` string ("Input should be a valid number"). The
    wrapped exception re-emits those details as a one-line summary
    followed by a bulleted list of every error, so the user knows
    which key to fix in ``config.yaml`` without scrolling through
    pydantic's multi-line traceback.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        # Non-validation error — re-emit verbatim.
        return exc
    try:
        err_list = errors()
    except Exception:
        return exc
    lines: list[str] = [
        f"config.yaml 校验失败（{config_path}）：",
    ]
    for err in err_list[:10]:
        loc = ".".join(str(part) for part in err.get("loc", ())) or "<root>"
        msg = err.get("msg", "")
        # The ``input`` repr can be huge (e.g. a nested dict); cap
        # the inline value at 60 chars so the line stays
        # terminal-friendly.
        raw_input = err.get("input")
        if raw_input is not None:
            value_repr = repr(raw_input)
            if len(value_repr) > 60:
                value_repr = value_repr[:57] + "...'"
            msg = f"{msg}（值：{value_repr}）"
        lines.append(f"  • {loc}: {msg}")
    if len(err_list) > 10:
        lines.append(f"  … 还有 {len(err_list) - 10} 条错误未列出")
    return ValueError("\n".join(lines))


def _reanchor_relative_paths(
    config: LitTraceConfig, base_dir: Path
) -> LitTraceConfig:
    """Return a copy of ``config`` whose relative ``Path`` fields are
    resolved against ``base_dir`` instead of the process cwd.
    """
    storage = config.storage
    new_storage = storage.model_copy(
        update={
            "paper_library_dir": _reanchor(storage.paper_library_dir, base_dir),
            "metadata_dir": _reanchor(storage.metadata_dir, base_dir),
            "cache_dir": _reanchor(storage.cache_dir, base_dir),
            "sessions_dir": _reanchor(storage.sessions_dir, base_dir),
        }
    )
    artifact = config.artifact_storage
    new_artifact = artifact.model_copy(
        update={
            "local_root": _reanchor(artifact.local_root, base_dir),
        }
    )
    runtime = config.agent_runtime
    new_runtime = runtime.model_copy(
        update={
            "codex_home": _reanchor(runtime.codex_home, base_dir),
        }
    )
    return config.model_copy(
        update={
            "storage": new_storage,
            "artifact_storage": new_artifact,
            "agent_runtime": new_runtime,
        }
    )


def _reanchor(path: Path, base_dir: Path) -> Path:
    """Return ``path`` as an absolute Path anchored at ``base_dir`` when
    it is relative. Absolute paths pass through unchanged. ``Path(...)``
    constructors in LitTraceConfig default to strings like ``./data/papers``,
    so a typical call resolves to ``base_dir/data/papers``.
    """
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


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
    runtime_mode = os.environ.get("LITTRACE_AGENT_RUNTIME")
    if runtime_mode:
        config.agent_runtime.mode = AgentRuntimeMode(runtime_mode)
    codex_command = os.environ.get("LITTRACE_CODEX_COMMAND")
    if codex_command:
        config.agent_runtime.codex_command = [
            part for part in codex_command.split(" ") if part
        ]
    scratch_root = os.environ.get("LITTRACE_CODEX_SCRATCH_ROOT")
    if scratch_root:
        config.agent_runtime.scratch_root = Path(scratch_root).expanduser()
    codex_home_mode = os.environ.get("LITTRACE_CODEX_HOME_MODE")
    if codex_home_mode:
        config.agent_runtime.codex_home_mode = CodexHomeMode(codex_home_mode)
    codex_home = os.environ.get("LITTRACE_CODEX_HOME")
    if codex_home:
        config.agent_runtime.codex_home = Path(codex_home).expanduser()
    config.agent_runtime.fallback_to_legacy = _env_bool(
        "LITTRACE_CODEX_FALLBACK_TO_LEGACY",
        config.agent_runtime.fallback_to_legacy,
    )
    sandbox_env = os.environ.get("LITTRACE_CODEX_SANDBOX")
    if sandbox_env:
        config.agent_runtime.sandbox_policy = SandboxPolicy(sandbox_env)
    writable_roots_env = os.environ.get("LITTRACE_CODEX_WRITABLE_ROOTS")
    if writable_roots_env:
        config.agent_runtime.writable_roots = [
            Path(p).expanduser()
            for p in writable_roots_env.split(",")
            if p.strip()
        ]
    interrupt_grace_env = os.environ.get("LITTRACE_CODEX_INTERRUPT_GRACE_SECONDS")
    if interrupt_grace_env:
        try:
            config.agent_runtime.interrupt_grace_seconds = float(interrupt_grace_env)
        except ValueError:
            pass
    config.agent_runtime.rollout_enabled = _env_bool(
        "LITTRACE_CODEX_ROLLOUT_ENABLED",
        config.agent_runtime.rollout_enabled,
    )
    rollout_dir_env = os.environ.get("LITTRACE_CODEX_ROLLOUT_DIR")
    if rollout_dir_env:
        config.agent_runtime.rollout_dir = Path(rollout_dir_env).expanduser()
    config.llm.api_key = os.environ.get("DEEPSEEK_API_KEY") or config.llm.api_key
    config.llm.base_url = os.environ.get("DEEPSEEK_BASE_URL") or config.llm.base_url
    config.llm.model = os.environ.get("DEEPSEEK_MODEL") or config.llm.model
    config.browser.browser_act_path = (
        os.environ.get("LITTRACE_BROWSER_ACT_PATH") or config.browser.browser_act_path
    )
    api_timeout = os.environ.get("LITTRACE_API_REQUEST_TIMEOUT_SECONDS")
    if api_timeout:
        try:
            config.api.request_timeout_seconds = float(api_timeout)
        except ValueError:
            pass
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
    auto_launch = os.environ.get("LITTRACE_AUTO_LAUNCH_CHROME")
    if auto_launch is not None:
        config.cdp_downloader.auto_launch_chrome = auto_launch.lower() not in {
            "0", "false", "no", "off"
        }
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
    config.figure_enrichment.base_url = (
        os.environ.get("LITTRACE_FIGURE_ENRICHMENT_BASE_URL")
        or config.figure_enrichment.base_url
    )
    config.figure_enrichment.api_key = (
        os.environ.get("LITTRACE_FIGURE_ENRICHMENT_API_KEY")
        or config.figure_enrichment.api_key
    )
    config.figure_enrichment.model = (
        os.environ.get("LITTRACE_FIGURE_ENRICHMENT_MODEL")
        or config.figure_enrichment.model
    )
    config.artifact_storage.backend = (
        os.environ.get("LITTRACE_ARTIFACT_STORAGE_BACKEND") or config.artifact_storage.backend
    )
    artifact_root = os.environ.get("LITTRACE_ARTIFACT_LOCAL_ROOT")
    if artifact_root:
        config.artifact_storage.local_root = Path(artifact_root).expanduser()
    config.artifact_storage.bucket = (
        os.environ.get("LITTRACE_ARTIFACT_BUCKET") or config.artifact_storage.bucket
    )
    config.artifact_storage.endpoint_url = (
        os.environ.get("LITTRACE_ARTIFACT_ENDPOINT_URL")
        or config.artifact_storage.endpoint_url
    )
    config.artifact_storage.region = (
        os.environ.get("LITTRACE_ARTIFACT_REGION") or config.artifact_storage.region
    )
    config.artifact_storage.path_prefix = (
        os.environ.get("LITTRACE_ARTIFACT_PATH_PREFIX") or config.artifact_storage.path_prefix
    )
    config.metadata_store.backend = (
        os.environ.get("LITTRACE_METADATA_BACKEND") or config.metadata_store.backend
    )
    config.metadata_store.postgres_dsn = (
        os.environ.get("LITTRACE_POSTGRES_DSN") or config.metadata_store.postgres_dsn
    )
    config.metadata_store.schema_name = (
        os.environ.get("LITTRACE_POSTGRES_SCHEMA") or config.metadata_store.schema_name
    )
    postgres_connect_timeout = os.environ.get("LITTRACE_POSTGRES_CONNECT_TIMEOUT_SECONDS")
    if postgres_connect_timeout:
        try:
            config.metadata_store.connect_timeout_seconds = int(postgres_connect_timeout)
        except ValueError:
            pass
    config.metadata_store.allow_schema_reset = _env_bool(
        "LITTRACE_ALLOW_SCHEMA_RESET", config.metadata_store.allow_schema_reset
    )
    config.rag.enabled = _env_bool("LITTRACE_RAG_ENABLED", config.rag.enabled)
    config.rag.backend = os.environ.get("LITTRACE_RAG_BACKEND") or config.rag.backend
    config.rag.postgres_dsn = (
        os.environ.get("LITTRACE_RAG_POSTGRES_DSN") or config.rag.postgres_dsn
    )
    config.rag.schema_name = (
        os.environ.get("LITTRACE_RAG_SCHEMA") or config.rag.schema_name
    )
    config.rag.collection_prefix = (
        os.environ.get("LITTRACE_RAG_COLLECTION_PREFIX") or config.rag.collection_prefix
    )
    config.rag.embedding_provider = (
        os.environ.get("LITTRACE_RAG_EMBEDDING_PROVIDER") or config.rag.embedding_provider
    )
    config.rag.embedding_base_url = (
        os.environ.get("LITTRACE_RAG_EMBEDDING_BASE_URL") or config.rag.embedding_base_url
    )
    config.rag.embedding_api_key = (
        os.environ.get("LITTRACE_RAG_EMBEDDING_API_KEY") or config.rag.embedding_api_key
    )
    config.rag.embedding_model = (
        os.environ.get("LITTRACE_RAG_EMBEDDING_MODEL") or config.rag.embedding_model
    )
    embedding_dimension = os.environ.get("LITTRACE_RAG_EMBEDDING_DIMENSION")
    if embedding_dimension:
        try:
            config.rag.embedding_dimension = int(embedding_dimension)
        except ValueError:
            pass
    config.rag.auto_refresh_enabled = _env_bool(
        "LITTRACE_RAG_AUTO_REFRESH", config.rag.auto_refresh_enabled
    )
    config.download_retry.enabled = _env_bool(
        "LITTRACE_DOWNLOAD_RETRY_ENABLED", config.download_retry.enabled
    )
    config.download_retry.background_worker_enabled = _env_bool(
        "LITTRACE_DOWNLOAD_RETRY_WORKER", config.download_retry.background_worker_enabled
    )
    retry_interval = os.environ.get("LITTRACE_DOWNLOAD_RETRY_INTERVAL_SECONDS")
    if retry_interval:
        try:
            config.download_retry.interval_seconds = float(retry_interval)
        except ValueError:
            pass
    retry_attempts = os.environ.get("LITTRACE_DOWNLOAD_RETRY_MAX_ATTEMPTS")
    if retry_attempts:
        try:
            config.download_retry.max_attempts = int(retry_attempts)
        except ValueError:
            pass
    config.compaction.enabled = _env_bool(
        "LITTRACE_COMPACTION_ENABLED", config.compaction.enabled
    )
    config.compaction.background_worker_enabled = _env_bool(
        "LITTRACE_COMPACTION_WORKER", config.compaction.background_worker_enabled
    )
    compaction_interval = os.environ.get("LITTRACE_COMPACTION_INTERVAL_SECONDS")
    if compaction_interval:
        try:
            config.compaction.interval_seconds = float(compaction_interval)
        except ValueError:
            pass
    compaction_batch = os.environ.get("LITTRACE_COMPACTION_BATCH_SIZE")
    if compaction_batch:
        try:
            config.compaction.batch_size = int(compaction_batch)
        except ValueError:
            pass
    compaction_attempts = os.environ.get("LITTRACE_COMPACTION_MAX_ATTEMPTS")
    if compaction_attempts:
        try:
            config.compaction.max_attempts = int(compaction_attempts)
        except ValueError:
            pass
    compaction_turns = os.environ.get("LITTRACE_COMPACTION_THRESHOLD_TURNS")
    if compaction_turns:
        try:
            config.compaction.threshold_turns = int(compaction_turns)
        except ValueError:
            pass
    compaction_tokens = os.environ.get("LITTRACE_COMPACTION_THRESHOLD_TOKENS")
    if compaction_tokens:
        try:
            config.compaction.threshold_tokens = int(compaction_tokens)
        except ValueError:
            pass
    compaction_timeout = os.environ.get("LITTRACE_COMPACTION_REQUEST_TIMEOUT_SECONDS")
    if compaction_timeout:
        try:
            config.compaction.request_timeout_seconds = float(compaction_timeout)
        except ValueError:
            pass
    return config


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
