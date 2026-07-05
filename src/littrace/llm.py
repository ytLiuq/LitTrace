from __future__ import annotations

from dataclasses import dataclass
import asyncio
from typing import NamedTuple

import httpx

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace


@dataclass
class LLMReply:
    text: str
    used_llm: bool
    error: str | None = None


async def chat_completion(
    config: LitTraceConfig,
    system_prompt: str,
    user_message: str,
    workspace: LiteratureWorkspace | None = None,
) -> LLMReply:
    if not config.llm.enabled:
        return LLMReply(text="", used_llm=False, error="llm_disabled")
    if not config.llm.api_key:
        return LLMReply(text="", used_llm=False, error="missing_api_key")

    messages = [{"role": "system", "content": system_prompt}]
    if workspace is not None:
        messages.append({"role": "system", "content": _workspace_context_prompt(workspace)})
    messages.append({"role": "user", "content": user_message})

    errors: list[str] = []
    response = None
    for endpoint in _llm_endpoints(config):
        try:
            response = await _post_chat_completion(config, messages, endpoint)
            break
        except Exception as exc:
            errors.append(f"{endpoint.label}:{exc.__class__.__name__}: {exc}")
    if response is None:
        return LLMReply(text="", used_llm=False, error="; ".join(errors) or "no_llm_endpoint")

    payload = response.json()
    content = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
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


async def _post_chat_completion(
    config: LitTraceConfig,
    messages: list[dict[str, str]],
    endpoint: LLMEndpoint,
) -> httpx.Response:
    retry_statuses = {429, 500, 502, 503, 504}
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=config.llm.request_timeout_seconds) as client:
        for attempt in range(1, 4):
            try:
                response = await client.post(
                    f"{endpoint.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {endpoint.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": endpoint.model,
                        "messages": messages,
                        "temperature": config.llm.temperature,
                    },
                )
                if response.status_code not in retry_statuses:
                    response.raise_for_status()
                    return response
                if attempt == 3:
                    response.raise_for_status()
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt == 3:
                    raise
            await asyncio.sleep(0.8 * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError("LLM retry loop exited unexpectedly")


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
