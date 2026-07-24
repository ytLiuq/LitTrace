from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from littrace.config import LitTraceConfig
from littrace.retrieval.rag_profile import RagProfile


class EmbeddingClient(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class EmbeddingConfigurationError(ValueError):
    pass


@dataclass
class OpenAICompatibleEmbeddingClient:
    base_url: str
    api_key: str | None
    model: str
    timeout_seconds: float = 30.0

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        endpoint = self.base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                endpoint,
                headers=headers,
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Embedding response is missing a data list.")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        embeddings: list[list[float]] = []
        for item in ordered:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                raise ValueError("Embedding response item is missing an embedding list.")
            embeddings.append([float(value) for value in embedding])
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding response count mismatch: expected {len(texts)}, got {len(embeddings)}."
            )
        return embeddings


def embedding_client_from_config(
    config: LitTraceConfig,
    profile: RagProfile,
) -> EmbeddingClient:
    provider = profile.embedding_provider or config.rag.embedding_provider
    if provider != "openai-compatible":
        raise EmbeddingConfigurationError(
            f"Unsupported RAG embedding provider: {provider!r}. "
            "Use embedding_provider='openai-compatible'."
        )
    base_url = config.rag.embedding_base_url or config.llm.base_url
    api_key = config.rag.embedding_api_key or config.llm.api_key
    if not base_url:
        raise EmbeddingConfigurationError("RAG embedding_base_url or llm.base_url is required.")
    if not api_key:
        raise EmbeddingConfigurationError("RAG embedding_api_key or llm.api_key is required.")
    return OpenAICompatibleEmbeddingClient(
        base_url=base_url,
        api_key=api_key,
        model=profile.embedding_model,
        timeout_seconds=config.llm.request_timeout_seconds,
    )
