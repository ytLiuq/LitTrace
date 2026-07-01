from fastapi.testclient import TestClient

from littrace.api.app import app


def test_search_context_and_download_plan_api():
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
    assert plan["downloadable_count"] == 3
    assert plan["requires_login_count"] == 2
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
    assert len(citations) == 3
    assert citations[0]["citation_text"]
    assert citations[0]["access_url"]

    response = client.get("/agents/crew")
    assert response.status_code == 200
    roles = response.json()
    assert any(role["name"] == "Citation Verifier" for role in roles)

    response = client.get("/agents/status")
    assert response.status_code == 200
    statuses = response.json()
    assert any(status["workflow_node"] == "route_publishers" for status in statuses)

    response = client.get("/agents/strength")
    assert response.status_code == 200
    assert response.json()["agents"]

    response = client.get("/agents/audits")
    assert response.status_code == 200
    assert response.json()

    response = client.get("/agents/plan", params={"topic": "MXene sensor"})
    assert response.status_code == 200
    assert response.json()["steps"]

    response = client.get("/agents/interactions")
    assert response.status_code == 200
    assert response.json()["handoffs"]

    response = client.get("/quality")
    assert response.status_code == 200
    assert "metrics" in response.json()

    response = client.get("/publishers/routes")
    assert response.status_code == 200
    publisher_routes = response.json()
    assert len(publisher_routes["routes"]) == 3
    assert any(route["publisher_family"] == "acs" for route in publisher_routes["routes"])

    response = client.get("/publishers/search-plan", params={"topic": "MXene sensor"})
    assert response.status_code == 200
    assert response.json()["plans"]

    response = client.get("/publishers/browser-plan", params={"topic": "MXene sensor", "family": "acs"})
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

    response = client.post("/downloads/check")
    assert response.status_code == 200
    assert "ready_to_parse_count" in response.json()

    response = client.post("/downloads/resume")
    assert response.status_code == 200
    assert "performance_cell_count" in response.json()

    response = client.post("/downloads/execute", json={"paper_ids": [], "dry_run": True})
    assert response.status_code == 200
    result = response.json()
    assert result["requires_login_count"] == 2
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
    assert workflow["agent_interactions"] is not None
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
    assert response.json()["action"] == "agent_status"

    response = client.post(f"/sessions/{session_id}/export")
    assert response.status_code == 200
    assert "markdown" in response.json()
