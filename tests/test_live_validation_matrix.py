"""Real validation matrix for the five production paths.
These tests are opt-in. They never replace external services with mocks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _require_live() -> None:
    if os.environ.get("LITTRACE_LIVE_TESTS") != "1":
        pytest.skip("set LITTRACE_LIVE_TESTS=1 to run real validation tests")


def _run(script: str, *args: str, timeout: int) -> str:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    if completed.returncode:
        pytest.fail(f"{script} failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    return completed.stdout


def _json_output(output: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    pytest.fail(f"no JSON object in live command output: {output[-2000:]}")


@pytest.mark.live
def test_live_dual_path_30_paper_metrics() -> None:
    _require_live()
    output = _run("run_dual_path_30_e2e.py", "--path", "both", timeout=3600)
    payload = _json_output(output)
    assert len(payload["results"]) == 2
    for result in payload["results"]:
        assert result["object_records"] > 0
        assert result["objects_present"] == result["object_records"]
        assert result["metrics"]


@pytest.mark.live
def test_live_authenticated_publisher_downloads() -> None:
    _require_live()
    _run("run_seven_publisher_download_e2e.py", "--timeout", "300", "--user-wait-seconds", "90", timeout=2400)


@pytest.mark.live
def test_live_multi_topic_retrieval_policies() -> None:
    _require_live()
    output = _run("run_multi_topic_policy_e2e.py", timeout=1800)
    payload = _json_output(output)
    assert len(payload["topics"]) == 3
    assert all(item["gate"] == "accepted" or item.get("reason") for item in payload["topics"])
    accepted = [item for item in payload["topics"] if item["gate"] == "accepted"]
    assert accepted
    assert len({item["canonical_topic"] for item in accepted}) == len(accepted)
    assert all(item["accepted"] > 0 for item in accepted)


@pytest.mark.live
def test_live_embedding_failure_recovery() -> None:
    _require_live()
    _run("run_failure_recovery_e2e.py", timeout=900)
