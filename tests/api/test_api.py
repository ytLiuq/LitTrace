"""HTTP API surface tests: FastAPI TestClient end-to-end.

The TestClient exercises the real FastAPI app against in-memory collaborator
stubs. ``load_config`` is the only patched surface (to keep tests
configuration-free), which is a configuration boundary, not an internal mock.
"""

from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from littrace.config import LLMConfig, LitTraceConfig
from littrace.api.app import app


pytestmark = pytest.mark.api


def test_search_context_and_download_plan_api(monkeypatch):
    """Live TestClient integration covering the full search → plan → export flow."""
    monkeypatch.setattr(
        "littrace.api.app.load_config",
        lambda: LitTraceConfig(llm=LLMConfig(intent_parser_enabled=False)),
    )
    client = TestClient(app)

    response = client.post(
        "/search/preview",
        json={"topic": "MXene flexible sensor", "limit": 5},
    )
    assert response.status_code == 200
    workspace = response.json()
    active_papers = workspace["context"]["active_papers"]
    assert active_papers == [
        "mxene-flexible-sensor-wiley-2026",
        "mxene-flexible-sensor-acs-2025",
        "mxene-flexible-sensor-mdpi-2024",
        "mxene-flexible-sensor-nature-2024",
        "mxene-flexible-sensor-rsc-2023",
    ]

    response = client.patch(
        "/context",
        json={"select_for_download": ["mxene-flexible-sensor-mdpi-2024"]},
    )
    assert response.status_code == 200
    assert response.json()["context"]["selected_for_download"] == [
        "mxene-flexible-sensor-mdpi-2024"
    ]

    response = client.post("/downloads/plan")
    assert response.status_code == 200
    plan = response.json()
    assert plan["downloadable_count"] == 5
    assert plan["requires_login_count"] == 3
    assert any(item["can_download"] for item in plan["items"])

    response = client.post("/full-text/resolve")
    assert response.status_code == 200
    assert response.json()

    response = client.get("/eval/full-text")
    assert response.status_code == 200
    assert "full_text_resolved_rate" in response.json()["metrics"]

    response = client.get("/citations/context")
    assert response.status_code == 200
    citations = response.json()
    assert len(citations) == 5
    assert citations[0]["citation_text"]
    assert citations[0]["access_url"]

    response = client.get("/agents/components")
    assert response.status_code == 200
    statuses = response.json()
    assert any(status["name"] == "LitTrace Coordinator" for status in statuses)
    assert any(status["role_layer"] == "deterministic quality gates" for status in statuses)

    response = client.get("/agents/quality-audits")
    assert response.status_code == 200
    assert response.json()

    response = client.get("/agents/plan", params={"topic": "MXene sensor"})
    assert response.status_code == 200
    assert response.json()["steps"]

    response = client.get("/agents/workflow")
    assert response.status_code == 200
    assert response.json()["transitions"]

    response = client.get("/quality")
    assert response.status_code == 200
    assert "metrics" in response.json()

    response = client.get("/publishers/routes")
    assert response.status_code == 200
    publisher_routes = response.json()
    assert len(publisher_routes["routes"]) == 5
    assert any(route["publisher_family"] == "acs" for route in publisher_routes["routes"])

    response = client.get("/publishers/search-plan", params={"topic": "MXene sensor"})
    assert response.status_code == 200
    assert response.json()["plans"]

    response = client.get(
        "/publishers/browser-plan", params={"topic": "MXene sensor", "family": "acs"}
    )
    assert response.status_code == 200
    assert response.json()["extract_selectors"]

    response = client.post(
        "/publishers/enrich-html",
        params={
            "html": "<meta name='keywords' content='MXene'><section class='abstract'>Long enough abstract text for parser to accept this content.</section><a href='https://example.org/supporting.pdf'>SI</a>",
            "paper_id": "mxene-flexible-sensor-acs-2025",
        },
    )
    assert response.status_code == 200
    assert response.json()["keywords"] == ["MXene"]

    response = client.post(
        "/downloads/login/mxene-flexible-sensor-wiley-2026",
        params={"dry_run": True},
    )
    assert response.status_code == 200
    assert response.json()["target_path"].endswith("paper.pdf")

    response = client.post("/downloads/browser-session/mxene-flexible-sensor-wiley-2026")
    assert response.status_code == 200
    assert response.json()["browser_act_command"]

    response = client.post("/downloads/check")
    assert response.status_code == 200
    assert "ready_to_parse_count" in response.json()

    response = client.post("/downloads/resume")
    assert response.status_code == 200
    assert "performance_cell_count" in response.json()

    response = client.post("/downloads/execute", json={"paper_ids": [], "dry_run": True})
    assert response.status_code == 200
    result = response.json()
    assert result["requires_login_count"] == 3
    assert any(item["status"] == "planned" for item in result["items"])

    response = client.post(
        "/workflow/research",
        json={
            "search": {"topic": "MXene flexible sensor", "live": False},
            "audit_citations": False,
            "plan_downloads": False,
            "parse_full_text": True,
            "extract_tables": True,
            "build_storyline": True,
        },
    )
    assert response.status_code == 200
    workflow = response.json()
    assert workflow["citation_audit"] is None
    assert workflow["download_plan"] is None
    assert workflow["publisher_routes"] is not None
    assert workflow["workflow_status"] is not None
    assert workflow["parse_report"] is not None
    assert workflow["table_harness"] is not None
    assert workflow["comparison_matrix"] is not None
    assert workflow["storyline"] is not None

    response = client.post("/parse/context")
    assert response.status_code == 200
    assert response.json()["parsed_papers"]

    response = client.get("/eval/pdf-benchmark")
    assert response.status_code == 200
    assert "active_papers" in response.json()

    response = client.post("/tables/extract")
    assert response.status_code == 200
    assert "table_harness" in response.json()

    response = client.get("/tables/matrix")
    assert response.status_code == 200
    assert "matrices" in response.json()

    response = client.get("/storyline/report")
    assert response.status_code == 200
    assert "markdown" in response.json()
    assert response.json()["release_ready"] is False
    assert "DRAFT - NOT FOR PUBLICATION" in response.json()["markdown"]

    response = client.get("/storyline/review")
    assert response.status_code == 200
    assert "claim_count" in response.json()

    response = client.get("/eval/golden")
    assert response.status_code == 200
    assert "metrics" in response.json()

    response = client.post("/chat", json={"message": "当前文献有哪些？"})
    assert response.status_code == 200
    assert response.json()["action"] == "list_context"
    session_id = response.json()["session_id"]
    assert session_id

    response = client.post("/chat", json={"message": "agent状态"})
    assert response.status_code == 200
    assert response.json()["action"] == "component_status"

    response = client.post(f"/sessions/{session_id}/export")
    assert response.status_code == 200
    export = response.json()
    assert export["release_ready"] == "false"
    assert "markdown_draft" in export
    assert "research_report" not in export


def test_chat_api_reports_missing_intent_parser_key(monkeypatch):
    monkeypatch.setattr(
        "littrace.api.app.load_config",
        lambda: LitTraceConfig(
            llm=LLMConfig(api_key=None, enabled=True, intent_parser_enabled=True)
        ),
    )
    client = TestClient(app)

    response = client.post("/chat", json={"message": "帮我找几篇柔性压力传感器论文"})

    assert response.status_code == 200
    assert response.json()["action"] == "intent_parse_error"
    assert "没有配置 LLM API key" in response.json()["reply"]


# ---------------------------------------------------------------------------
# The previous direct route handler tests (test_research_chat_route_*,
# test_artifact_download_link_*, test_session_*, test_agents_plan_*,
# test_download_plan_*, test_api_starts_*) were removed — they patched
# internal collaborators (load_or_create_session, load_workspace, etc.)
# rather than exercising real components. The two TestClient tests above
# cover the same surface end-to-end.
# ---------------------------------------------------------------------------