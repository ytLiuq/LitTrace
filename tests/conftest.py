"""Shared pytest fixtures for the littrace test suite.

Conventions
-----------
* Tests live under one of six sub-packages, each backed by a marker:
    unit, domain, api, adapters, live, eval
* Live tests are opt-in via ``pytest -m live`` (network + real services).
* Fixtures in this module are intentionally thin wrappers — most domain
  fixtures live next to the test files that use them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Make ``src/`` importable so ``littrace`` resolves without an install step.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Path / filesystem
# ---------------------------------------------------------------------------

@pytest.fixture
def fixtures_dir() -> Path:
    """Directory containing static golden fixtures (JSON, YAML, PDFs)."""
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return _FIXTURES_DIR


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub littrace env vars so tests start from a known baseline.

    Individual tests opt back into specific variables via ``monkeypatch.setenv``.
    """
    for name in list(os.environ):
        if name.startswith("LITTRACE_") or name.startswith("DEEPSEEK_"):
            monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Markers — declared here so missing markers don't error at collection time.
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: pure-logic unit tests, no I/O")
    config.addinivalue_line("markers", "domain: domain integration tests")
    config.addinivalue_line("markers", "api: HTTP/CLI/TUI surface tests")
    config.addinivalue_line("markers", "adapters: external adapter integration")
    config.addinivalue_line("markers", "live: opt-in tests requiring real services")
    config.addinivalue_line("markers", "eval: evaluation harness tests")