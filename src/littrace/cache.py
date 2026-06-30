from __future__ import annotations

import hashlib
from pathlib import Path

from littrace.config import LitTraceConfig


def cache_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def read_text_cache(config: LitTraceConfig, namespace: str, key: str) -> str | None:
    path = _cache_path(config, namespace, key)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def write_text_cache(config: LitTraceConfig, namespace: str, key: str, value: str) -> Path:
    path = _cache_path(config, namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _cache_path(config: LitTraceConfig, namespace: str, key: str) -> Path:
    return config.storage.cache_dir / namespace / f"{key}.txt"
