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
    raw["parsing"]["default_parser"] = "metadata_only"
    target.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return ConfigWizardResult(path=str(target), created=True)
