from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig


class ConfigWizardResult(BaseModel):
    path: str
    created: bool
    warnings: list[str] = Field(default_factory=list)


def write_config_template(path: str | Path = "config.yaml", overwrite: bool = False) -> ConfigWizardResult:
    target = Path(path)
    if target.exists() and not overwrite:
        return ConfigWizardResult(
            path=str(target),
            created=False,
            warnings=["Config already exists; pass overwrite=True to replace it."],
        )
    config = LitTraceConfig()
    raw = config.model_dump(mode="json")
    raw["api"]["enable_live_search"] = False
    raw["api"]["unpaywall_email"] = "you@example.com"
    raw["cdp_downloader"]["chrome_profile_name"] = "Default"
    raw["cdp_downloader"]["auto_launch_chrome"] = False
    raw["browser"]["browser_act_path"] = "browser-act"
    raw["browser"]["required"] = False
    raw["rag"]["enabled"] = False
    raw["rag"]["backend"] = "pgvector"
    raw["rag"]["postgres_dsn"] = "postgresql://littrace:littrace@localhost:5433/littrace"
    # Demo-stage defaults: explicit opt-in for any data acquisition. The
    # operator must set these to true to actually start pulling bytes.
    raw["rag"]["auto_download_open_access"] = False
    raw["rag"]["allow_requires_login_download"] = False
    raw["metadata_store"]["allow_schema_reset"] = False
    raw["cdp_downloader"]["auto_launch_chrome"] = False
    raw["rag"]["embedding_base_url"] = "https://api.openai.com/v1"
    # No placeholder API key — littrace enforces "real key required" on
    # startup. Ship an empty value and let the operator set their own.
    raw["rag"]["embedding_api_key"] = ""
    raw["parsing"]["default_parser"] = "docling"
    target.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return ConfigWizardResult(path=str(target), created=True)
