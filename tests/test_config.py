from pathlib import Path

from littrace.config import load_config


def test_load_config_reads_env_local_without_config_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    Path(".env.local").write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=test-key",
                "DEEPSEEK_BASE_URL=https://example.com",
                "DEEPSEEK_MODEL=deepseek-test",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config()

    assert config.llm.api_key == "test-key"
    assert config.llm.base_url == "https://example.com"
    assert config.llm.model == "deepseek-test"
