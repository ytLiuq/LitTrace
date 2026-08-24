from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from littrace.codex_runtime.rollout import RolloutRecorder, rollout_path_for


def _read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_recorder_writes_jsonl_per_append(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    with RolloutRecorder(path) as recorder:
        recorder.append(type_="session_meta", session_id="s-1", codex_thread_id="thr-1")
        recorder.append(type_="turn_start", turn_id="t-1", user_text="hi")
        recorder.append(type_="event", method="item/agentMessage/delta", params={"delta": "x"})
        recorder.append(type_="turn_complete", turn_id="t-1", status="completed", reply="hi")

    lines = _read_lines(path)
    assert [line["type"] for line in lines] == [
        "session_meta", "turn_start", "event", "turn_complete",
    ]
    assert lines[0]["session_id"] == "s-1"
    assert lines[1]["turn_id"] == "t-1"
    assert lines[2]["method"] == "item/agentMessage/delta"
    assert lines[3]["status"] == "completed"
    # Every line carries an ISO timestamp.
    assert all("ts" in line for line in lines)


def test_recorder_swallows_open_oserror(tmp_path: Path) -> None:
    path = tmp_path / "blocked" / "rollout.jsonl"
    with mock.patch.object(Path, "open", side_effect=PermissionError("denied")):
        recorder = RolloutRecorder(path)
        recorder.open()
        # No exception; subsequent appends are no-ops.
        recorder.append(type_="turn_start", turn_id="t-1")
        assert recorder.is_open is False


def test_recorder_swallows_append_oserror(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    recorder = RolloutRecorder(path)
    recorder.open()
    with mock.patch.object(recorder._fh, "write", side_effect=OSError("disk full")):
        # No exception bubbles up.
        recorder.append(type_="turn_start", turn_id="t-1")


def test_recorder_close_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    recorder = RolloutRecorder(path)
    recorder.open()
    recorder.close()
    # Calling close() twice does not raise.
    recorder.close()
    assert recorder.is_open is False


def test_rollout_path_for_default_under_session_root(tmp_path: Path) -> None:
    # Minimal ChatSession stand-in — we only need .root and .session_id.
    class _FakeSession:
        session_id = "s-abc"
        root = tmp_path / "session-root"

    path = rollout_path_for(_FakeSession())  # type: ignore[arg-type]
    assert path == tmp_path / "session-root" / "rollouts" / "rollout-s-abc.jsonl"


def test_rollout_path_for_with_base_dir(tmp_path: Path) -> None:
    class _FakeSession:
        session_id = "s-abc"
        root = tmp_path / "session-root"

    base = tmp_path / "shared-rollouts"
    path = rollout_path_for(_FakeSession(), base_dir=base)  # type: ignore[arg-type]
    assert path == base / "rollout-s-abc.jsonl"


def test_rollout_recorder_appends_are_ordered(tmp_path: Path) -> None:
    """Mixed sequence of turns survives a single recorder cleanly."""
    path = tmp_path / "rollout.jsonl"
    with RolloutRecorder(path) as recorder:
        for turn_id in ("t-1", "t-2", "t-3"):
            recorder.append(type_="turn_start", turn_id=turn_id)
            recorder.append(type_="event", method="item/completed", turn_id=turn_id)
            recorder.append(type_="turn_complete", turn_id=turn_id, status="completed")

    lines = _read_lines(path)
    turn_ids = [line["turn_id"] for line in lines if line["type"] != "event" or "turn_id" in line]
    assert turn_ids == ["t-1", "t-1", "t-1", "t-2", "t-2", "t-2", "t-3", "t-3", "t-3"]