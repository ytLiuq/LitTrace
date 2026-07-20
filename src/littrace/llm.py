from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import httpx

from littrace.config import LitTraceConfig, RateLimitConfig
from littrace.log import get_logger, metrics, timed, cost_tracker
from littrace.models import LiteratureWorkspace
from littrace.rate_limit import RateLimitSlot, rate_limiter
from littrace.retry import retry_async, RetryConfig, BackoffStrategy, retry_tracker

logger = get_logger("llm")


@dataclass
class LLMReply:
    text: str
    used_llm: bool
    error: str | None = None


def _build_retry_config(config: LitTraceConfig) -> RetryConfig:
    """Build RetryConfig from LitTraceConfig.retry settings."""
    retry_cfg = config.retry
    return RetryConfig(
        max_attempts=retry_cfg.max_attempts,
        backoff_strategy=BackoffStrategy(retry_cfg.backoff_strategy),
        base_delay_seconds=retry_cfg.base_delay_seconds,
        max_delay_seconds=retry_cfg.max_delay_seconds,
        retry_status_codes=frozenset({429, 500, 502, 503, 504}),
        retry_on=(
            httpx.TimeoutException,
            httpx.TransportError,
            httpx.HTTPStatusError,
        ),
    )


def _check_budget(config: LitTraceConfig) -> str | None:
    """Check if token budget is exceeded. Returns error message if exceeded, None otherwise."""
    budget = config.cost_budget
    if budget.max_total_tokens <= 0:
        return None
    total = cost_tracker.total_tokens
    if total >= budget.max_total_tokens:
        return (
            f"Token budget exhausted: {total} / {budget.max_total_tokens} tokens used "
            f"({cost_tracker.total_cost_usd:.4f} USD)"
        )
    return None


_cost_tracker_initialized: bool = False


def _ensure_cost_tracker_persistence(config: LitTraceConfig) -> None:
    """One-time setup of cost tracker persistence path from config.

    If persist_path is configured, the tracker will auto-save after each record()
    and auto-load existing data on first call.
    Also sets max_entries for file rotation if configured.
    """
    global _cost_tracker_initialized
    if _cost_tracker_initialized:
        return
    persist_path = config.cost_budget.persist_path
    if persist_path:
        # Resolve relative to storage cache_dir if not absolute
        p = Path(persist_path)
        if not p.is_absolute():
            p = config.storage.cache_dir / p
        cost_tracker.set_persist_path(p)
        logger.info("cost_tracker_persistence_enabled", extra={"path": str(p)})
    # Set max_entries for rotation (0 = unlimited)
    max_entries = getattr(config.cost_budget, "max_persist_entries", 0)
    if max_entries > 0:
        cost_tracker.set_max_entries(max_entries)
        logger.info("cost_tracker_rotation_enabled", extra={"max_entries": max_entries})
    # Also sync price table from config
    for model, price in config.cost_budget.price_per_1k_input.items():
        output_price = config.cost_budget.price_per_1k_output.get(model, 0.002)
        cost_tracker.set_price(model, price, output_price)
    _cost_tracker_initialized = True


_retry_tracker_initialized: bool = False


def _ensure_retry_tracker_persistence(config: LitTraceConfig) -> None:
    """One-time setup of retry tracker persistence path from config."""
    global _retry_tracker_initialized
    if _retry_tracker_initialized:
        return
    persist_path = config.retry.persist_path
    if persist_path:
        p = Path(persist_path)
        if not p.is_absolute():
            p = config.storage.cache_dir / p
        retry_tracker.set_persist_path(p)
        logger.info("retry_tracker_persistence_enabled", extra={"path": str(p)})
    max_entries = getattr(config.retry, "max_persist_entries", 0)
    if max_entries > 0:
        retry_tracker.set_max_entries(max_entries)
    _retry_tracker_initialized = True


_rate_limiter_initialized: bool = False


def _ensure_rate_limiter(config: LitTraceConfig) -> None:
    """One-time setup of the LLM rate limiter from config.

    Configures concurrency and RPM limits. Both default to 0 (disabled),
    so the limiter is a no-op unless explicitly configured.
    """
    global _rate_limiter_initialized
    if _rate_limiter_initialized:
        return
    rl_cfg = config.rate_limit
    if rl_cfg.max_concurrent > 0 or rl_cfg.max_requests_per_minute > 0:
        rate_limiter.configure(
            RateLimitConfig(
                max_concurrent=rl_cfg.max_concurrent,
                max_requests_per_minute=rl_cfg.max_requests_per_minute,
            )
        )
        logger.info(
            "rate_limiter_enabled",
            extra={
                "max_concurrent": rl_cfg.max_concurrent,
                "max_rpm": rl_cfg.max_requests_per_minute,
            },
        )
    _rate_limiter_initialized = True


async def chat_completion(
    config: LitTraceConfig,
    system_prompt: str,
    user_message: str,
    workspace: LiteratureWorkspace | None = None,
    *,
    json_mode: bool = False,
) -> LLMReply:
    """Call the LLM with optional JSON mode.

    Args:
        json_mode: If True, adds ``response_format={"type": "json_object"}``
            to the request body, forcing the API to return valid JSON.
    """
    if not config.llm.enabled:
        return LLMReply(text="", used_llm=False, error="llm_disabled")
    if not config.llm.api_key:
        return LLMReply(text="", used_llm=False, error="missing_api_key")

    # Dimension 4: Cost budget pre-check
    # Initialize cost tracker persistence if configured
    _ensure_cost_tracker_persistence(config)
    # Initialize retry tracker persistence if configured
    _ensure_retry_tracker_persistence(config)
    # Initialize rate limiter if configured
    _ensure_rate_limiter(config)
    budget_error = _check_budget(config)
    if budget_error is not None:
        logger.error("budget_exceeded", extra={"error": budget_error})
        return LLMReply(text="", used_llm=False, error=budget_error)

    messages = [{"role": "system", "content": system_prompt}]
    if workspace is not None:
        messages.append({"role": "system", "content": _workspace_context_prompt(workspace)})
    messages.append({"role": "user", "content": user_message})

    errors: list[str] = []
    response = None
    used_endpoint_label = ""
    retry_config = _build_retry_config(config)
    for endpoint in _llm_endpoints(config):
        try:
            with timed("llm_request", model=endpoint.model, endpoint=endpoint.label):
                async with RateLimitSlot(rate_limiter):
                    response = await _post_chat_completion(
                        config,
                        messages,
                        endpoint,
                        retry_config,
                        json_mode=json_mode,
                    )
            used_endpoint_label = endpoint.label
            break
        except Exception as exc:
            errors.append(f"{endpoint.label}:{exc.__class__.__name__}: {exc}")
            logger.warning(
                "llm_endpoint_error",
                extra={
                    "endpoint": endpoint.label,
                    "model": endpoint.model,
                    "error": f"{exc.__class__.__name__}: {exc}",
                },
            )
    if response is None:
        return LLMReply(text="", used_llm=False, error="; ".join(errors) or "no_llm_endpoint")

    payload = response.json()
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    usage = payload.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens
    metrics.record(
        "llm_tokens",
        total_tokens,
        labels={
            "model": config.llm.model,
            "endpoint": used_endpoint_label,
        },
    )
    # Dimension 4: Record cost
    cost_tracker.record(
        model=config.llm.model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    logger.info(
        "llm_response",
        extra={
            "model": config.llm.model,
            "endpoint": used_endpoint_label,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "content_len": len(content),
            "cost_usd": cost_tracker.total_cost_usd,
        },
    )
    if not content:
        return LLMReply(text="", used_llm=False, error="empty_llm_response")
    return LLMReply(text=content, used_llm=True)


def _llm_endpoints(config: LitTraceConfig) -> list[LLMEndpoint]:
    endpoints = [
        LLMEndpoint(
            api_key=config.llm.api_key or "",
            base_url=config.llm.base_url,
            model=config.llm.model,
            label="primary",
        )
    ]
    for model in config.llm.fallback_models:
        if model != config.llm.model:
            endpoints.append(
                LLMEndpoint(
                    api_key=config.llm.api_key or "",
                    base_url=config.llm.base_url,
                    model=model,
                    label=f"primary_fallback_model:{model}",
                )
            )
    if config.llm.fallback_api_key and config.llm.fallback_base_url and config.llm.fallback_model:
        endpoints.append(
            LLMEndpoint(
                api_key=config.llm.fallback_api_key,
                base_url=config.llm.fallback_base_url,
                model=config.llm.fallback_model,
                label="fallback",
            )
        )
    return endpoints


def _build_request_body(
    config: LitTraceConfig,
    endpoint: LLMEndpoint,
    messages: list[dict[str, str]],
    *,
    json_mode: bool = False,
) -> dict[str, object]:
    """Build the JSON request body for the LLM API.

    Includes max_tokens from cost_budget config if configured (Dimension 4).
    Includes response_format=json_object when json_mode is True (Dimension 3).
    """
    body: dict[str, object] = {
        "model": endpoint.model,
        "messages": messages,
        "temperature": config.llm.temperature,
    }
    max_tokens = config.cost_budget.max_tokens_per_call
    if max_tokens > 0:
        body["max_tokens"] = max_tokens
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


async def _post_chat_completion(
    config: LitTraceConfig,
    messages: list[dict[str, str]],
    endpoint: LLMEndpoint,
    retry_config: RetryConfig | None = None,
    *,
    json_mode: bool = False,
) -> httpx.Response:
    """Post a chat completion request with unified retry.

    Uses @retry_async internally — the inner function _single_post does
    a single HTTP request, and the retry decorator handles retries.
    """
    cfg = retry_config or _build_retry_config(config)
    body = _build_request_body(config, endpoint, messages, json_mode=json_mode)

    @retry_async(cfg, operation=f"llm_post:{endpoint.label}", retry_on=cfg.retry_on)
    async def _single_post() -> httpx.Response:
        async with httpx.AsyncClient(timeout=config.llm.request_timeout_seconds) as client:
            response = await client.post(
                f"{endpoint.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {endpoint.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            # Let HTTPStatusError propagate — retry_async will retry based on config
            response.raise_for_status()
            return response

    return await _single_post()


def research_assistant_system_prompt() -> str:
    return (
        "You are LitTrace, a local academic research assistant for materials and chemistry. "
        "Answer in Chinese by default. Be concrete, evidence-aware, and conservative. "
        "For paper-specific claims, mention that citations and access links must be attached by "
        "the citation layer. Do not invent papers, metrics, or publisher access. "
        "If the current context is insufficient, say what should be searched, parsed, or verified next."
    )


def _workspace_context_prompt(workspace: LiteratureWorkspace) -> str:
    if not workspace.context.active_papers:
        return "Current literature context: empty."

    lines = ["Current literature context:"]
    for paper_id in workspace.context.active_papers[:12]:
        paper = workspace.papers[paper_id]
        lines.append(
            "- "
            f"id={paper.paper_id}; title={paper.title}; year={paper.year}; "
            f"source={paper.journal or paper.publisher}; doi={paper.doi}; "
            f"access={paper.access_type}"
        )
    if len(workspace.context.active_papers) > 12:
        lines.append(f"... {len(workspace.context.active_papers) - 12} more papers omitted.")
    return "\n".join(lines)


class LLMEndpoint(NamedTuple):
    api_key: str
    base_url: str
    model: str
    label: str
