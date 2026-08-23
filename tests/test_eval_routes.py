from fastapi.testclient import TestClient

from littrace.api.app import app
from littrace.config import EvalConfig, LitTraceConfig
from littrace.models import LiteratureWorkspace, PaperMetadata


def test_rag_golden_route_reports_missing_workspace_profile(monkeypatch, tmp_path):
    rag_golden = tmp_path / "rag-golden"
    rag_golden.mkdir()
    (rag_golden / "cases.jsonl").write_text(
        '{"case_id":"rag-1","question":"Find sensitivity evidence",'
        '"gold_evidence":[{"paper_id":"p1","required_terms":["sensitivity"]}]}\n',
        encoding="utf-8",
    )
    config = LitTraceConfig(eval=EvalConfig(rag_golden_set_dir=rag_golden))
    monkeypatch.setattr("littrace.api.app.load_config", lambda: config)
    monkeypatch.setattr("littrace.api.app.WORKSPACE", LiteratureWorkspace())

    response = TestClient(app).get("/eval/rag-golden", params={"top_k": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["case_count"] == 1
    assert payload["metrics"]["rag_recall_at_k"] == 0.0
    assert payload["cases"][0]["warnings"]


def test_task_golden_route_scores_current_workspace(monkeypatch, tmp_path):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "cases.jsonl").write_text(
        '{"case_id":"task-1","topic":"sensor",'
        '"expected_dois":["10.1000/sensor"]}\n',
        encoding="utf-8",
    )
    workspace = LiteratureWorkspace(
        papers={
            "p1": PaperMetadata(
                paper_id="p1",
                title="Sensor paper",
                doi="10.1000/sensor",
            )
        }
    )
    workspace.context.active_papers = ["p1"]
    config = LitTraceConfig(eval=EvalConfig(golden_set_dir=golden))
    monkeypatch.setattr("littrace.api.app.load_config", lambda: config)
    monkeypatch.setattr("littrace.api.app.WORKSPACE", workspace)

    response = TestClient(app).get("/eval/task-golden", params={"case_id": "task-1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["case_count"] == 1
    assert payload["evidence_grounded_task_success_rate"] == 1.0
