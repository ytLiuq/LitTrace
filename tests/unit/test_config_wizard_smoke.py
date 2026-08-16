"""Smoke test for littrace.config_wizard (write_config_template). No mocks —
the wizard seeds a real config.yaml on disk."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_write_config_template_creates_yaml_without_secrets(tmp_path: Path):
    from littrace.config_wizard import write_config_template

    target = tmp_path / "config.yaml"
    result = write_config_template(target)

    assert result.created
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    # No placeholder API keys leak into a fresh config.
    assert "your-openai-api-key" not in text
    # No silent Chrome auto-launch — default off.
    assert "auto_launch_chrome: false" in text
    # No silent auto-download — default off.
    assert "auto_download_open_access: false" in text
    # No accidental schema reset on every restart.
    assert "allow_schema_reset: false" in text


def test_write_config_template_does_not_overwrite_existing(tmp_path: Path):
    from littrace.config_wizard import write_config_template

    target = tmp_path / "config.yaml"
    target.write_text("preserved: true\n", encoding="utf-8")
    result = write_config_template(target, overwrite=False)
    assert result.created is False
    assert result.warnings
    assert target.read_text(encoding="utf-8") == "preserved: true\n"