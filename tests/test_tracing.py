from littrace.config import EvalConfig, LitTraceConfig
from littrace.tracing import append_trace


def test_append_trace_writes_jsonl(tmp_path):
    config = LitTraceConfig(eval=EvalConfig(traces_dir=tmp_path / "traces"))

    path = append_trace(config, "event", {"value": 1})

    assert "littrace_events.jsonl" in path
    assert '"event": "event"' in (tmp_path / "traces" / "littrace_events.jsonl").read_text(
        encoding="utf-8"
    )
