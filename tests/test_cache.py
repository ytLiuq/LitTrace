from littrace.cache import cache_key, read_cached_text, read_text_cache, write_text_cache
from littrace.config import LitTraceConfig, StorageConfig


def test_text_cache_roundtrip(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(cache_dir=tmp_path / "cache"))
    key = cache_key("https://example.org")

    write_text_cache(config, "publisher", key, "hello")

    assert read_text_cache(config, "publisher", key) == "hello"


def test_cache_tracks_ttl_and_can_return_stale_value(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(cache_dir=tmp_path / "cache"))
    key = cache_key("stale")
    write_text_cache(config, "publisher", key, "hello", ttl_seconds=0)

    stale = read_cached_text(config, "publisher", key, ttl_seconds=-1, allow_stale=True)

    assert stale.value == "hello"
    assert stale.stale


def test_cache_uses_the_ttl_recorded_with_the_cached_value(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(cache_dir=tmp_path / "cache"))
    key = cache_key("per-entry-ttl")
    write_text_cache(config, "publisher", key, "hello", ttl_seconds=-1)

    result = read_cached_text(config, "publisher", key)

    assert result.value is None
    assert result.stale
