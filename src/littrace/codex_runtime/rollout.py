"""Per-turn rollout JSONL recorder for Codex App Server events.

The recorder is a side-channel log used for post-hoc debugging,
replay, and visualisation only. It does NOT participate in the
canonical session state — Postgres is still the source of truth
(round 1 decision). Every write is best-effort: ``OSError`` is
swallowed with a warning log so the rollout path can never block the
main turn flow.

The on-disk shape mirrors codex-harness's ``rollout-*.jsonl``
convention but the file is per LitTrace session, not per codex
thread, so the path stays stable across thread resumption and the
session-id emb is discoverable from the file name.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from littrace.session import ChatSession


log = logging.getLogger(__name__)


class RolloutRecorder:
    """Synchronous append-only writer for codex turn events.

    One LitTrace session maps to one JSONL file. Individual turns
    are tagged via the ``turn_id`` field on each event so an
    operator can ``jq 'select(.turn_id == "...")'`` to carve out a
    single turn without juggling multiple files.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: IO[str] | None = None

    def __enter__(self) -> "RolloutRecorder":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def open(self) -> None:
        """Open the file for append, creating parent directories."""
        if self._fh is not None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        except OSError:
            log.warning("rollout open failed: path=%s", self.path, exc_info=True)
            self._fh = None

    def append(self, *, type_: str, **payload: object) -> None:
        """Append one JSONL line. ``type_`` becomes the ``type`` field.

        Failures (full disk, permission denied, broken pipe) are
        swallowed after a warning log so the rollout path never
        raises back into the turn flow.
        """
        if self._fh is None:
            return
        record = {
            "type": type_,
            "ts": datetime.now(UTC).isoformat(),
            **payload,
        }
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        except OSError:
            log.warning(
                "rollout append failed: path=%s type=%s", self.path, type_,
                exc_info=True,
            )

    def close(self) -> None:
        """Close the underlying file. Idempotent."""
        if self._fh is None:
            return
        try:
            self._fh.close()
        except OSError:
            log.warning("rollout close failed: path=%s", self.path, exc_info=True)
        finally:
            self._fh = None

    @property
    def is_open(self) -> bool:
        return self._fh is not None


def rollout_path_for(
    session: "ChatSession",
    *,
    base_dir: Path | None = None,
) -> Path:
    """Resolve the per-session JSONL path.

    ``base_dir=None`` puts the file under ``<session.root>/rollouts/``
    so each LitTrace session owns its own rollout tree and the
    operator can ``rm -rf`` the directory when archiving a session.
    """
    root = base_dir if base_dir is not None else session.root / "rollouts"
    return root / f"rollout-{session.session_id}.jsonl"