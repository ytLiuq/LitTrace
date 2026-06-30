from __future__ import annotations

from dataclasses import dataclass

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

    try:
        async with httpx.AsyncClient(timeout=config.llm.request_timeout_seconds) as client:
            response = await client.post(
                f"{config.llm.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.llm.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.llm.model,
                    "messages": messages,
                    "temperature": config.llm.temperature,
                },
            )
            response.raise_for_status()
    except Exception as exc:
        return LLMReply(text="", used_llm=False, error=f"{exc.__class__.__name__}: {exc}")

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
