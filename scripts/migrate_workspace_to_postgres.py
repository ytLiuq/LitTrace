"""One-shot backfill from disk workspace.json into Postgres.

Round 3 topic B makes Postgres the canonical workspace store.
Sessions that existed before the migration still have a
``<session.root>/workspace.json`` mirror on disk; this script
backfills each one into ``session_state.workspace_json`` and
captures the matching entry in ``session_state_snapshots``.

Run it once after upgrading. The disk workspace.json is left
in place as a cold backup so an operator can roll back by
restoring the directory and re-pointing the loader.

Usage::

    uv run python -m scripts.migrate_workspace_to_postgres --dry-run
    uv run python -m scripts.migrate_workspace_to_postgres
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

from littrace.config import LitTraceConfig, load_config
from littrace.models import LiteratureWorkspace
from littrace.state_db import (
    SessionStateRecord,
    SessionStateSnapshotRecord,
    state_store_from_config,
)


log = logging.getLogger("migrate_workspaces")


def _iter_workspace_files(sessions_dir: Path) -> list[Path]:
    if not sessions_dir.exists():
        return []
    return sorted(
        p
        for p in sessions_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def _load_workspace_json(session_dir: Path) -> dict[str, Any] | None:
    workspace_path = session_dir / "workspace.json"
    if not workspace_path.exists():
        return None
    try:
        return json.loads(workspace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("skip %s: %s", session_dir.name, exc)
        return None


def _load_manifest_json(session_dir: Path) -> dict[str, Any] | None:
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("manifest unreadable %s: %s", session_dir.name, exc)
        return None


def _migrate_one(
    session_dir: Path,
    store: Any,
    *,
    dry_run: bool,
) -> str:
    workspace_json = _load_workspace_json(session_dir)
    if workspace_json is None:
        return "skipped"

    session_id = session_dir.name
    manifest = _load_manifest_json(session_dir) or {}
    revision = int(manifest.get("revision") or workspace_json.get("context", {}).get("filters", {}).get("workspace_revision") or 1)
    workspace_sha256 = (
        manifest.get("workspace_sha256")
        or sha256(
            json.dumps(workspace_json, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
    )
    record = SessionStateRecord(
        session_id=session_id,
        workspace_sha256=workspace_sha256,
        workspace_json=workspace_json,
        manifest_json=manifest or {},
        artifact_index_json=manifest.get("artifact_index", {})
        if isinstance(manifest.get("artifact_index"), dict)
        else {},
        memory_view_json=workspace_json.get("context", {}).get("memory_view", {})
        if isinstance(workspace_json.get("context", {}).get("memory_view"), dict)
        else {},
        rag_profile_json=manifest.get("rag", {}) or {},
        revision=revision,
    )
    snapshot = SessionStateSnapshotRecord(
        session_id=session_id,
        revision=revision,
        workspace_sha256=workspace_sha256,
        workspace_json=workspace_json,
    )

    if dry_run:
        return "would-migrate"

    # Idempotent: if the row already exists (operator re-ran the
    # script), overwrite the workspace_json / manifest_json but
    # leave the revision untouched. upsert_session_state with
    # expected_revision=None falls through to the INSERT ON CONFLICT
    # branch and overwrites the payload.
    existing = store.get_session_state(session_id)
    if existing is not None:
        log.info("overwrite %s (existing revision %d)", session_id, existing.revision)
    else:
        log.info("insert %s (revision %d)", session_id, revision)
    store.upsert_session_state(record, expected_revision=None)
    store.upsert_session_snapshot(snapshot)
    return "migrated"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would happen without writing to Postgres",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="path to the littrace config file",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config: LitTraceConfig = load_config(path=args.config)
    sessions_dir = config.storage.sessions_dir
    log.info("scanning %s (dry_run=%s)", sessions_dir, args.dry_run)
    store = state_store_from_config(config)
    if not args.dry_run:
        store._ensure_schema()  # type: ignore[attr-defined]

    counts = {"migrated": 0, "skipped": 0, "would-migrate": 0}
    for session_dir in _iter_workspace_files(sessions_dir):
        result = _migrate_one(session_dir, store, dry_run=args.dry_run)
        counts[result] = counts.get(result, 0) + 1

    log.info(
        "done: migrated=%d skipped=%d would-migrate=%d",
        counts["migrated"],
        counts["skipped"],
        counts["would-migrate"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())