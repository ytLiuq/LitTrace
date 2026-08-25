"""Tiny OpenAI-compatible embedding mock used for live/integration E2E.

Deterministic per-text hashing → float vector (length matches the
request's ``dimensions`` field, or ``MOCK_EMBED_DIM`` if the client
did not send one). Run with::

    uv run python -m scripts.mock_embedding_server --port 8765
    MOCK_EMBED_DIM=1024 uv run python -m scripts.mock_embedding_server

Then point ``LITTRACE_E2E_EMBEDDING_BASE_URL=http://127.0.0.1:8765/v1``
and use any non-empty ``LITTRACE_E2E_EMBEDDING_API_KEY``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import struct
from typing import Any

from fastapi import FastAPI


def _vector_for(text: str, dimensions: int) -> list[float]:
    """Deterministic unit vector derived from text bytes."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Stretch the digest to fill the requested dimensions via
    # round-robin expansion so cosine similarity is still meaningful
    # across calls (same text → same vector).
    pool: list[float] = []
    counter = 0
    while len(pool) < dimensions:
        h = hashlib.sha256(digest + struct.pack("<I", counter)).digest()
        for i in range(0, len(h), 4):
            pool.append(struct.unpack("<i", h[i:i + 4])[0] / 2**31)
            if len(pool) >= dimensions:
                break
        counter += 1
    norm = math.sqrt(sum(v * v for v in pool))
    if norm == 0:
        return [0.0] * dimensions
    return [v / norm for v in pool[:dimensions]]


# Default dimension is overridable via env var so the integration test
# can pin 1024 even though LitTrace's OpenAICompatibleEmbeddingClient
# does not currently forward ``dimensions`` in the payload.
_DEFAULT_DIM = int(os.environ.get("MOCK_EMBED_DIM", "1536"))


app = FastAPI(title="mock-embedding")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/embeddings")
async def embeddings(payload: dict[str, Any]) -> dict[str, Any]:
    texts: list[str] = payload.get("input") or []
    if isinstance(texts, str):
        texts = [texts]
    model: str = payload.get("model", "mock-embedding-v1")
    # Prefer an explicit per-request ``dimensions``, then module
    # default (driven by ``MOCK_EMBED_DIM``).
    requested = payload.get("dimensions")
    dimensions = int(requested) if isinstance(requested, int) and requested > 0 else _DEFAULT_DIM
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": idx,
                "embedding": _vector_for(text, dimensions),
            }
            for idx, text in enumerate(texts)
        ],
        "model": model,
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in texts),
            "total_tokens": sum(len(t.split()) for t in texts),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dim", type=int, default=None,
                        help="override MOCK_EMBED_DIM on the CLI.")
    args = parser.parse_args()
    if args.dim is not None:
        global _DEFAULT_DIM
        _DEFAULT_DIM = args.dim

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()