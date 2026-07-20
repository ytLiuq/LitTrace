from littrace.tool_contracts import (
    ToolExecutionLedger,
    ToolArtifactRef,
    ToolCallContext,
    ToolExecutionPolicy,
    list_tool_contracts,
    run_sync_tool,
    run_tool,
    tool_contract,
    tool_contract_summary,
)


def test_run_tool_wraps_successful_sync_call():
    result = __import__("asyncio").run(
        run_tool(tool_contract("build_comparison_matrices"), lambda value: {"value": value}, 3)
    )

    assert result.ok
    assert result.tool == "build_comparison_matrices"
    assert result.contract_id == "build_comparison_matrices:v1"
    assert result.output == {"value": 3}
    assert result.output_summary["type"] == "dict"
    assert result.elapsed_ms >= 0


def test_run_tool_wraps_errors():
    def fail(value):
        raise ValueError(f"bad {value}")

    result = __import__("asyncio").run(
        run_tool(tool_contract("build_comparison_matrices"), fail, 3)
    )

    assert not result.ok
    assert "ValueError" in (result.error or "")


def test_run_tool_records_context_metadata():
    result = __import__("asyncio").run(
        run_tool(
            tool_contract("build_research_plan"),
            lambda value: {"topic": value},
            "sensor",
            context=ToolCallContext(caller="planner", task_id="t1", metadata={"k": "v"}),
            metadata={"extra": True},
        )
    )

    assert result.metadata["k"] == "v"
    assert result.metadata["extra"] is True


def test_run_tool_can_redact_output_to_artifact_reference():
    ref = ToolArtifactRef(
        artifact_id="a1",
        kind="large_payload",
        producer="test",
        summary={"rows": 100},
    )
    result = __import__("asyncio").run(
        run_tool(
            tool_contract("build_comparison_matrices"),
            lambda value: {"rows": list(range(value))},
            100,
            output_mode="summary",
            output_ref=ref,
        )
    )

    assert result.ok
    assert result.output is None
    assert result.output_redacted
    assert result.output_ref == ref
    assert result.output_bytes_estimate > 0


def test_tool_contract_registry_includes_core_layers():
    names = {contract.name for contract in list_tool_contracts()}

    assert {
        "build_research_plan",
        "search_papers",
        "resolve_workspace_full_text",
        "parse_workspace_papers",
        "build_quality_metrics",
        "execute_downloads",
        "export_session_bundle",
    } <= names


def test_tool_contract_summary_reports_policy_relevant_groups():
    summary = tool_contract_summary()

    assert summary["count"] >= 5
    assert "retrieval" in summary["categories"]
    assert "search_papers" in summary["network_tools"]
    assert "parse_workspace_papers" in summary["workspace_mutation_tools"]


def test_run_tool_enforces_execution_policy():
    result = __import__("asyncio").run(
        run_tool(
            tool_contract("search_papers"),
            lambda value: value,
            "payload",
            policy=ToolExecutionPolicy(allow_network=False),
        )
    )

    assert not result.ok
    assert "ToolPolicyViolation" in (result.error or "")


def test_run_sync_tool_wraps_success_and_policy():
    result = run_sync_tool(
        tool_contract("build_comparison_matrices"),
        lambda value: {"value": value},
        3,
        context=ToolCallContext(caller="sync-test", metadata={"layer": "skill"}),
    )

    assert result.ok
    assert result.output == {"value": 3}
    assert result.metadata["layer"] == "skill"

    blocked = run_sync_tool(
        tool_contract("search_papers"),
        lambda value: value,
        "payload",
        policy=ToolExecutionPolicy(allow_network=False),
    )

    assert not blocked.ok
    assert "ToolPolicyViolation" in (blocked.error or "")


def test_reliability_contract_metadata_is_declared():
    report = tool_contract("build_research_document_report")
    search = tool_contract("search_papers")

    assert "ReleaseSnapshot" in report.provenance_outputs
    assert "publication_gate" in report.quality_requirements
    assert search.cache_policy == "source_ttl"
    assert "search_papers" in tool_contract_summary()["cached_tools"]


def test_tool_execution_ledger_enforces_budget_and_reuses_idempotent_result():
    contract = tool_contract("search_papers")
    ledger = ToolExecutionLedger(remaining_budget={"requests": 2.0})
    calls = 0

    def fetch(value):
        nonlocal calls
        calls += 1
        return {"value": value}

    first = __import__("asyncio").run(
        run_tool(contract, fetch, "MXene", ledger=ledger, idempotency_key="same-search")
    )
    second = __import__("asyncio").run(
        run_tool(contract, fetch, "MXene", ledger=ledger, idempotency_key="same-search")
    )

    assert first.ok
    assert second.ok
    assert second.metadata["idempotency_reused"]
    assert calls == 1
    assert ledger.remaining_budget["requests"] == 0.0

    blocked = __import__("asyncio").run(
        run_tool(contract, fetch, "CNT", ledger=ledger, idempotency_key="new-search")
    )
    assert not blocked.ok
    assert "budget" in (blocked.error or "")


def test_workspace_mutating_tool_never_reuses_ledger_result():
    contract = tool_contract("parse_workspace_papers")
    ledger = ToolExecutionLedger()
    calls = 0

    def parse(value):
        nonlocal calls
        calls += 1
        return {"value": value}

    first = __import__("asyncio").run(
        run_tool(contract, parse, "workspace", ledger=ledger, idempotency_key="same-workspace")
    )
    second = __import__("asyncio").run(
        run_tool(contract, parse, "workspace", ledger=ledger, idempotency_key="same-workspace")
    )

    assert first.ok and second.ok
    assert calls == 2
    assert not ledger.cached_results
