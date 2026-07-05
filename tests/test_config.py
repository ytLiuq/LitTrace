from pathlib import Path

from littrace.config import load_config
from littrace.config_wizard import write_config_template


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
                "LITTRACE_LLM_FALLBACK_MODELS=deepseek-v4-pro,deepseek-chat",
                "LITTRACE_FALLBACK_LLM_API_KEY=fallback-key",
                "LITTRACE_FALLBACK_LLM_BASE_URL=https://fallback.example.com",
                "LITTRACE_FALLBACK_LLM_MODEL=fallback-model",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config()

    assert config.llm.api_key == "test-key"
    assert config.llm.base_url == "https://example.com"
    assert config.llm.model == "deepseek-test"
    assert config.llm.fallback_models == ["deepseek-v4-pro", "deepseek-chat"]
    assert config.llm.fallback_api_key == "fallback-key"
    assert config.llm.fallback_base_url == "https://fallback.example.com"
    assert config.llm.fallback_model == "fallback-model"


def test_write_config_template_creates_yaml(tmp_path):
    target = tmp_path / "config.yaml"

    result = write_config_template(target)

    assert result.created
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "unpaywall_email" in text
    assert "browser_act_path" in text
