from __future__ import annotations

import asyncio
import json
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


_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class OpenAICompatibleEmbeddingClient:
    base_url: str
    api_key: str | None
    model: str
    timeout_seconds: float = 30.0
    batch_size: int = 4
    max_retries: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        endpoint = self.base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        embeddings: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for start in range(0, len(texts), max(self.batch_size, 1)):
                batch = texts[start : start + max(self.batch_size, 1)]
                batch_embeddings = await self._embed_with_retry(
                    client, endpoint, headers, batch
                )
                embeddings.extend(batch_embeddings)
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding response count mismatch: expected {len(texts)}, got {len(embeddings)}."
            )
        return embeddings

    async def _embed_with_retry(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: dict[str, str],
        batch: list[str],
    ) -> list[list[float]]:
        """Submit a single batch with exponential-backoff retry on 429 / 5xx.

        The OpenAI-compatible embedding API may return one error per item in
        a multi-item batch; on a true batch failure (HTTP-level error) we
        retry the whole batch. On success we trust the response counts
        (callers verify ``len(embeddings) == len(texts)``).
        """
        attempt = 0
        backoff = self.initial_backoff_seconds
        while True:
            try:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json={"model": self.model, "input": batch},
                )
                if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    attempt += 1
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.max_backoff_seconds)
                    continue
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = response.text[:500].replace("\n", " ")
                    raise httpx.HTTPStatusError(
                        f"{exc}; response={detail}",
                        request=exc.request,
                        response=exc.response,
                    ) from exc
                payload = response.json()
                return self._parse_response(payload, batch_size=len(batch))
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.max_backoff_seconds)
                    continue
                raise

    @staticmethod
    def _parse_response(payload: object, *, batch_size: int) -> list[list[float]]:
        if not isinstance(payload, dict):
            raise ValueError("Embedding response is not a JSON object.")
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Embedding response is missing a data list.")
        ordered = sorted(
            (
                item
                for item in data
                if isinstance(item, dict)
            ),
            key=lambda item: int(item.get("index", 0)),
        )
        result: list[list[float]] = []
        for item in ordered:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise ValueError(
                    f"Embedding response item is missing an embedding list (got {len(result)}/{batch_size})."
                )
            result.append([float(value) for value in embedding])
        if len(result) != batch_size:
            raise ValueError(
                f"Embedding response count mismatch: expected {batch_size}, got {len(result)}."
            )
        return result


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