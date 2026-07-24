import asyncio

from littrace.config import LitTraceConfig, StorageConfig
from littrace.models import LiteratureWorkspace
from littrace.rag_jobs import (
    iter_sentinel_watchlist_ids,
    iter_workspace_session_ids,
    run_daily_rag_maintenance,
)
from littrace.session import create_chat_session, save_workspace


def test_iterators_find_sessions_and_watchlists(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    session = create_chat_session(config)
    sentinel_dir = tmp_path / "sentinel" / "mxene"
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    (sentinel_dir / "watchlist.yaml").write_text("watchlist_id: mxene\n", encoding="utf-8")

    assert iter_workspace_session_ids(tmp_path) == [session.session_id]
    assert iter_sentinel_watchlist_ids(tmp_path) == ["mxene"]


def test_daily_maintenance_runs_watchlists_and_sessions(monkeypatch, tmp_path):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    config.rag.enabled = True
    config.rag.auto_refresh_enabled = True
    session = create_chat_session(config)
    workspace = LiteratureWorkspace()
    save_workspace(session, workspace, config=config)
    sentinel_dir = tmp_path / "sentinel" / "mxene"
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    (sentinel_dir / "watchlist.yaml").write_text("watchlist_id: mxene\n", encoding="utf-8")

    calls: list[str] = []

    async def fake_run_sentinel(config, watchlist_id, topic=None):
        calls.append(f"sentinel:{watchlist_id}")

    async def fake_refresh(config, session_obj, workspace_obj=None):
        calls.append(f"refresh:{session_obj.session_id}")

        class DummyReport:
            warnings = []
            chunk_count = 0
            upserted_count = 0
            skipped = True
            skip_reason = "mock"

        return None, DummyReport()

    monkeypatch.setattr("littrace.rag_jobs.run_sentinel", fake_run_sentinel)
    monkeypatch.setattr("littrace.rag_jobs.refresh_session_rag_index", fake_refresh)

    report = asyncio.run(run_daily_rag_maintenance(config))

    assert report.sentinel_watchlists == 1
    assert report.sessions_refreshed == 1
    assert report.sessions_skipped == 0
    assert calls == ["sentinel:mxene", f"refresh:{session.session_id}"]


def test_daily_maintenance_skips_sessions_when_profile_auto_refresh_disabled(
    monkeypatch,
    tmp_path,
):
    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path))
    config.rag.enabled = True
    config.rag.auto_refresh_enabled = False
    create_chat_session(config)
    calls: list[str] = []

    async def fake_refresh(config, session_obj, workspace_obj=None):
        calls.append(f"refresh:{session_obj.session_id}")

    monkeypatch.setattr("littrace.rag_jobs.refresh_session_rag_index", fake_refresh)

    report = asyncio.run(run_daily_rag_maintenance(config))

    assert report.sessions_refreshed == 0
    assert report.sessions_skipped == 1
    assert calls == []
