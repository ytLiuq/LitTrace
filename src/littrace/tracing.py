from __future__ import annotations

import json
from datetime import datetime

from littrace.config import LitTraceConfig


def append_trace(config: LitTraceConfig, event: str, payload: dict[str, object]) -> str:
    config.eval.traces_dir.mkdir(parents=True, exist_ok=True)
    path = config.eval.traces_dir / "littrace_events.jsonl"
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)
