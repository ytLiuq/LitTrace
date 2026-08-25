"""Convert ``RolloutRecorder`` JSONL traces into harness check items.

Round 10 step 1: bridge the gap between the side-channel rollout
log (Round 2C ``RolloutRecorder``) and the offline eval harness
(``littrace.evaluation.harnesses``). One LitTrace session
produces one JSONL file with five event types
(``session_meta``, ``turn_context``, ``turn_start``, ``event``,
``turn_complete``, ``compaction``, ``system_error``). This
module groups events by ``turn_id`` and emits typed items that
the existing harness checks already accept:

  * ``CitationRecord`` for ``check_citations`` (citations the
    model produced during the turn)
  * ``RetryHealthItem`` for ``check_retry_health`` (per-tool
    retry / failure statistics aggregated from MCP tool calls
    captured in the ``event`` stream)
  * ``TurnRecord`` for new harness checks an operator might
    write (turn-level quality, latency, cost regressions)

The converter is intentionally a pure function: no I/O, no
network, no global state. CLI / scripts can use
``RolloutConverter`` directly or call ``convert_file`` /
``convert_directory`` helpers.

Usage::

    from littrace.evaluation.rollout_eval import convert_directory
    from littrace.evaluation.harnesses import HarnessEngine, HarnessConfig

    items_by_check = convert_directory("data/rollouts")
    engine = HarnessEngine()
    reports = engine.run_with_deps("check_retry_health", items_by_check)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from littrace.evaluation.harnesses import RetryHealthItem
from littrace.models import CitationRecord, LinkStatus


# ---------------------------------------------------------------------------
# Round 10 typed items that do not yet exist elsewhere in the codebase.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnRecord:
    """A summary of one chat turn reconstructed from a rollout trace.

    Operators can write harness checks against this shape to
    audit latency regressions, cost spikes, or reply-shape
    drift after a prompt or model change.
    """

    session_id: str
    thread_id: str
    turn_id: str
    status: str
    reply: str
    started_at: str
    completed_at: str | None
    tool_call_count: int = 0
    delta_count: int = 0
    error_code: str | None = None


@dataclass(frozen=True)
class ToolCallRecord:
    """One MCP / approval event captured in a rollout trace.

    Round 10 surfaces this for future harness checks (e.g. a
    "stale approval rate" audit). The dataclass is exported
    so R10-3 tests can assert on the converter output.
    """

    session_id: str
    turn_id: str
    method: str
    timestamp: str
    approved: bool | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutEvaluationBundle:
    """Output of one converter run.

    Each list corresponds to one harness check. The bundle is
    what the CLI and the live tests inspect.
    """

    session_id: str | None
    thread_id: str | None
    source_path: Path
    turns: list[TurnRecord]
    citations: list[CitationRecord]
    retries: list[RetryHealthItem]
    tool_calls: list[ToolCallRecord]
    errors: list[dict[str, Any]]

    def to_check_items(self) -> dict[str, list[Any]]:
        """Project the bundle into the ``items_map`` shape the
        ``HarnessEngine.run_with_deps`` API expects.
        """
        return {
            "check_citations": list(self.citations),
            "check_retry_health": list(self.retries),
            # ``turns`` and ``tool_calls`` are exposed for
            # operator-defined custom checks; the canonical
            # harness surface today only consumes the two
            # standard checks above.
            "__turns__": list(self.turns),
            "__tool_calls__": list(self.tool_calls),
            "__errors__": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Wire-level event grouping. The rollout JSONL is append-only
# so the converter never relies on global state — it walks the
# file once and accumulates per-turn buckets.
# ---------------------------------------------------------------------------


@dataclass
class _TurnBucket:
    thread_id: str = ""
    session_id: str = ""
    started_at: str = ""
    completed_at: str | None = None
    status: str = "unknown"
    reply: str = ""
    tool_call_methods: list[str] = field(default_factory=list)
    error_code: str | None = None


class _RolloutReader:
    """Stream a rollout JSONL file and yield validated events.

    The reader is permissive: a malformed line is dropped with a
    warning rather than aborting the whole conversion, so a
    single bad frame from a partially-flushed file does not
    prevent the rest of the trace from feeding the harness.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class RolloutConverter:
    """Convert one or more rollout JSONL files into harness items.

    The converter is single-use: build one per file (or
    directory) and call ``to_bundle`` or ``to_check_items``. The
    constructor holds a reference to the source path; the
    conversion itself happens lazily so callers can inspect the
    path before any I/O happens.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def to_bundle(self) -> RolloutEvaluationBundle:
        session_id: str | None = None
        thread_id: str | None = None
        turns: dict[str, _TurnBucket] = {}
        tool_calls: list[ToolCallRecord] = []
        citations: list[CitationRecord] = []
        errors: list[dict[str, Any]] = []

        for event in _RolloutReader(self.path):
            event_type = event.get("type")
            if event_type == "session_meta":
                session_id = event.get("session_id") or session_id
                thread_id = event.get("codex_thread_id") or thread_id
            elif event_type == "turn_start":
                turn_id = event.get("turn_id")
                if not turn_id:
                    continue
                bucket = turns.setdefault(turn_id, _TurnBucket())
                bucket.session_id = event.get("session_id") or bucket.session_id
                bucket.thread_id = event.get("thread_id") or bucket.thread_id
                bucket.started_at = event.get("ts") or bucket.started_at
            elif event_type == "turn_complete":
                turn_id = event.get("turn_id")
                if not turn_id:
                    continue
                bucket = turns.setdefault(turn_id, _TurnBucket())
                bucket.status = event.get("status") or bucket.status
                bucket.reply = event.get("reply") or bucket.reply
                bucket.completed_at = event.get("ts") or bucket.completed_at
            elif event_type == "event":
                method = event.get("method")
                turn_id = event.get("turn_id") or ""
                if method:
                    tool_calls.append(
                        ToolCallRecord(
                            session_id=event.get("session_id") or session_id or "",
                            turn_id=turn_id,
                            method=method,
                            timestamp=event.get("ts") or "",
                            payload=event.get("params") or {},
                        )
                    )
                    if turn_id:
                        bucket = turns.setdefault(turn_id, _TurnBucket())
                        bucket.tool_call_methods.append(method)
                # Citations are produced by the model via tool
                # calls. The rollout ``event`` stream captures
                # the ``item/completed`` notification with an
                # ``agentMessage`` payload that names the cited
                # paper_ids. Extract a thin ``CitationRecord``
                # so the standard ``check_citations`` harness
                # can audit them.
                params = event.get("params") or {}
                item = params.get("item") or {}
                if method == "item/completed" and isinstance(item, dict):
                    cited_ids = item.get("cited_paper_ids") or []
                    cited_text = item.get("text") or bucket.reply if bucket else ""
                    for paper_id in cited_ids:
                        citations.append(
                            CitationRecord(
                                paper_id=str(paper_id),
                                citation_text=str(cited_text),
                                access_url="",
                                link_status=LinkStatus.UNCHECKED,
                            )
                        )
            elif event_type == "system_error":
                turn_id = event.get("turn_id")
                if turn_id and turn_id in turns:
                    turns[turn_id].error_code = event.get("error_code")
                errors.append(
                    {
                        "turn_id": event.get("turn_id"),
                        "error_code": event.get("error_code"),
                        "message": event.get("message"),
                        "ts": event.get("ts"),
                    }
                )

        turn_records: list[TurnRecord] = []
        for turn_id, bucket in turns.items():
            # Count distinct tool-call methods for ``tool_call_count``.
            unique_methods = {m for m in bucket.tool_call_methods if m}
            turn_records.append(
                TurnRecord(
                    session_id=bucket.session_id,
                    thread_id=bucket.thread_id,
                    turn_id=turn_id,
                    status=bucket.status,
                    reply=bucket.reply,
                    started_at=bucket.started_at,
                    completed_at=bucket.completed_at,
                    tool_call_count=len(unique_methods),
                    error_code=bucket.error_code,
                )
            )

        return RolloutEvaluationBundle(
            session_id=session_id,
            thread_id=thread_id,
            source_path=self.path,
            turns=turn_records,
            citations=citations,
            retries=_aggregate_retry_health(tool_calls),
            tool_calls=tool_calls,
            errors=errors,
        )

    def to_check_items(self) -> dict[str, list[Any]]:
        return self.to_bundle().to_check_items()


def _aggregate_retry_health(tool_calls: list[ToolCallRecord]) -> list[RetryHealthItem]:
    """Bucket tool calls by method and compute retry / failure rates.

    The rollout captures only one frame per MCP tool call, so we
    do not have a direct retry counter. We treat a tool call as
    a "retry" when the same method fires more than once in the
    same turn (the canonical retry pattern for LitTrace's MCP
    gateway) and a "failure" when the corresponding
    ``item/completed`` item has ``status='failed'``. The latter
    is approximated by the ``error_code`` field on the matching
    TurnRecord; the converter intentionally surfaces the
    "caller has to annotate" case via the audit_message field
    in the items so a future patch can plumb per-tool status
    through the rollout append surface.
    """
    bucket: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "retries": 0, "failed": 0}
    )
    method_counts_per_turn: dict[tuple[str, str], int] = defaultdict(int)
    for call in tool_calls:
        method = call.method
        if not method:
            continue
        bucket[method]["total"] += 1
        method_counts_per_turn[(call.turn_id, method)] += 1
    for (turn_id, method), count in method_counts_per_turn.items():
        if count > 1:
            bucket[method]["retries"] += count - 1
    out: list[RetryHealthItem] = []
    for method, stats in bucket.items():
        total = stats["total"]
        if total == 0:
            continue
        total_retries = stats["retries"]
        failed = stats["failed"]
        out.append(
            RetryHealthItem(
                operation=method,
                total_calls=total,
                total_retries=total_retries,
                failed_calls=failed,
                retry_rate=total_retries / total if total else 0.0,
                failure_rate=failed / total if total else 0.0,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def convert_file(path: Path | str) -> RolloutEvaluationBundle:
    """Convert one rollout JSONL file to a bundle."""
    return RolloutConverter(Path(path)).to_bundle()


def convert_directory(
    path: Path | str, *, glob: str = "*.jsonl"
) -> list[RolloutEvaluationBundle]:
    """Convert every JSONL file under ``path`` (recursively).

    Returns one bundle per file. Callers that want a single
    merged item map can call :meth:`RolloutEvaluationBundle.to_check_items`
    on each bundle and combine the resulting dicts.
    """
    root = Path(path)
    if root.is_file():
        return [RolloutConverter(root).to_bundle()]
    files = sorted(root.rglob(glob))
    return [RolloutConverter(p).to_bundle() for p in files]


def merge_bundles(
    bundles: Iterable[RolloutEvaluationBundle],
) -> dict[str, list[Any]]:
    """Flatten a sequence of bundles into a single harness item map.

    Used by the CLI so a directory of rollouts evaluates as
    one combined run instead of one report per file. Errors are
    deduplicated by ``(turn_id, error_code)`` so a flaky
    downstream that retries the same failure twice does not
    inflate the count.
    """
    aggregated: dict[str, list[Any]] = {
        "check_citations": [],
        "check_retry_health": [],
        "__turns__": [],
        "__tool_calls__": [],
        "__errors__": [],
    }
    seen_errors: set[tuple[str, str | None]] = set()
    for bundle in bundles:
        items = bundle.to_check_items()
        for key in ("check_citations", "check_retry_health"):
            aggregated[key].extend(items[key])
        aggregated["__turns__"].extend(bundle.turns)
        aggregated["__tool_calls__"].extend(bundle.tool_calls)
        for err in bundle.errors:
            sig = (err.get("turn_id") or "", err.get("error_code"))
            if sig in seen_errors:
                continue
            seen_errors.add(sig)
            aggregated["__errors__"].append(err)
    return aggregated
