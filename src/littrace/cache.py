from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from littrace.config import LitTraceConfig


def cache_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def read_text_cache(config: LitTraceConfig, namespace: str, key: str) -> str | None:
    return read_cached_text(config, namespace, key).value


class CachedText:
    def __init__(self, value: str | None, *, stale: bool = False, created_at: str | None = None):
        self.value = value
        self.stale = stale
        self.created_at = created_at


def read_cached_text(
    config: LitTraceConfig,
    namespace: str,
    key: str,
    *,
    ttl_seconds: int | None = None,
    allow_stale: bool = False,
) -> CachedText:
    path = _cache_path(config, namespace, key)
    if not path.exists():
        return CachedText(None)
    created_at: str | None = None
    stored_ttl: int | None = None
    metadata_path = path.with_suffix(".meta.json")
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            created_at = metadata.get("created_at")
            raw_ttl = metadata.get("ttl_seconds")
            stored_ttl = raw_ttl if isinstance(raw_ttl, int) else None
        except (OSError, json.JSONDecodeError):
            created_at = None
    effective_ttl = (
        ttl_seconds
        if ttl_seconds is not None
        else stored_ttl
        if stored_ttl is not None
        else config.cache_policy.default_ttl_seconds
    )
    stale = False
    if effective_ttl < 0:
        stale = True
    elif effective_ttl > 0 and created_at:
        try:
            stale = (
                datetime.now(UTC) - datetime.fromisoformat(created_at)
            ).total_seconds() > effective_ttl
        except ValueError:
            stale = True
    if stale and not allow_stale:
        return CachedText(None, stale=True, created_at=created_at)
    return CachedText(path.read_text(encoding="utf-8"), stale=stale, created_at=created_at)


def write_text_cache(
    config: LitTraceConfig,
    namespace: str,
    key: str,
    value: str,
    *,
    ttl_seconds: int | None = None,
) -> Path:
    path = _cache_path(config, namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "ttl_seconds": ttl_seconds
                if ttl_seconds is not None
                else config.cache_policy.default_ttl_seconds,
            }
        ),
        encoding="utf-8",
    )
    return path


def _cache_path(config: LitTraceConfig, namespace: str, key: str) -> Path:
    return config.storage.cache_dir / namespace / f"{key}.txt"
