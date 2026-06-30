from littrace.cache import cache_key, read_text_cache, write_text_cache
from littrace.config import LitTraceConfig, StorageConfig


def test_text_cache_roundtrip(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(cache_dir=tmp_path / "cache"))
    key = cache_key("https://example.org")

    write_text_cache(config, "publisher", key, "hello")

    assert read_text_cache(config, "publisher", key) == "hello"
