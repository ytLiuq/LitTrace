from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from littrace.chrome_profiles import (
    build_chrome_launch_plan,
    cdp_uses_configured_profile,
)
from littrace.config import CDPDownloaderConfig, LitTraceConfig, load_config
from littrace.window_qt import (
    LitTraceQtWindow,
    _format_daily_preview_markdown,
    _render_message_html,
)
from littrace.downloads import execute_downloads
from littrace.models import AccessType, DownloadExecutionRequest, PaperMetadata


def test_daily_preview_is_markdown_not_nested_html() -> None:
    markdown = _format_daily_preview_markdown(
        {
            "topic": "MXene pressure sensor",
            "keywords": "MXene flexible",
            "year_min": 2023,
            "year_max": 2026,
            "target_papers": 10,
            "rounds_done": 2,
            "cumulative_candidates": 12,
            "new_candidates": 7,
            "seen_candidates": 5,
            "cumulative_downloaded": 4,
            "summary_lines": ["1. candidates_total: 12"],
            "warnings": ["one publisher needs login"],
        }
    )

    assert "检索文献：**12 篇**" in markdown
    assert "下载文献：**4 篇**" in markdown
    assert "<br>" not in markdown
    assert "&nbsp;" not in markdown
    rendered = _render_message_html(markdown)
    assert "<b>12 篇</b>" in rendered
    assert "&amp;nbsp;" not in rendered


def test_publisher_auth_plan_uses_visible_private_new_window(tmp_path: Path) -> None:
    executable = tmp_path / "chrome"
    executable.touch()
    user_data = tmp_path / "littrace-chrome"
    config = LitTraceConfig(
        cdp_downloader=CDPDownloaderConfig(
            chrome_executable=executable,
            chrome_user_data_dir=user_data,
            headless=True,
        )
    )

    plan = build_chrome_launch_plan(
        config,
        headless=False,
        new_window=True,
        initial_url="https://pubs.acs.org/action/showLogin",
    )

    assert plan is not None
    assert "--new-window" in plan.command
    assert "--headless=new" not in plan.command
    assert f"--user-data-dir={user_data}" in plan.command
    assert plan.command[-1] == "https://pubs.acs.org/action/showLogin"


def test_unknown_cdp_endpoint_is_not_reused(monkeypatch) -> None:
    config = LitTraceConfig()
    config.cdp_downloader.remote_debugging_port = 32100
    config.cdp_downloader.cdp_url = "http://127.0.0.1:32100"
    messages: list[str] = []
    window = SimpleNamespace(
        _external_chrome_proc=None,
        _controller=SimpleNamespace(config=config),
        _post_status=messages.append,
    )
    monkeypatch.setattr(
        "littrace.chrome_profiles.check_cdp_status",
        lambda _config: SimpleNamespace(available=True),
    )
    monkeypatch.setattr(
        "littrace.chrome_profiles.cdp_uses_configured_profile",
        lambda _config: False,
    )

    LitTraceQtWindow._isolate_private_cdp_port(window)

    assert config.cdp_downloader.remote_debugging_port != 32100
    assert config.cdp_downloader.cdp_url.endswith(
        str(config.cdp_downloader.remote_debugging_port)
    )
    assert any("其他 Chrome" in message for message in messages)


def test_cdp_profile_identity_uses_chrome_lock_owner(tmp_path: Path, monkeypatch) -> None:
    user_data = tmp_path / "littrace-chrome"
    user_data.mkdir()
    (user_data / "SingletonLock").symlink_to("host.local-12345")
    config = LitTraceConfig(
        cdp_downloader=CDPDownloaderConfig(chrome_user_data_dir=user_data)
    )
    monkeypatch.setattr(
        "littrace.chrome_profiles.check_cdp_status",
        lambda _config: SimpleNamespace(available=True),
    )
    monkeypatch.setattr(
        "littrace.chrome_profiles.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"Google Chrome --user-data-dir={user_data} --remote-debugging-port=19222",
        ),
    )

    assert cdp_uses_configured_profile(config)


def test_cookie_refresh_uses_browser_scope_without_creating_target(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("littrace.chrome_profiles.check_cdp_status", lambda _config: SimpleNamespace(
        available=True,
        cdp_url="http://127.0.0.1:19222",
        web_socket_debugger_url="ws://127.0.0.1:19222/devtools/browser/test",
    ))
    monkeypatch.setattr(
        "littrace.chrome_profiles._send_browser_cdp_command",
        lambda _status, method, params: calls.append(method) or {
            "result": {"cookies": [{"domain": ".onlinelibrary.wiley.com", "name": "session"}]}
        },
    )

    from littrace.chrome_profiles import read_cdp_cookies
    cookies = read_cdp_cookies(LitTraceConfig())

    assert cookies and calls == ["Storage.getCookies"]


def test_download_batch_closes_shared_target_on_success(monkeypatch, tmp_path) -> None:
    class FakeBrowser:
        def __init__(self):
            self.connected = False
            self.closed = False

        def connect_new_tab(self):
            self.connected = True

        def close_tab(self):
            self.closed = True

    browser = FakeBrowser()
    monkeypatch.setattr(
        "littrace.access_layer.cdp_core.CDPBrowser",
        lambda *a, **kw: browser,
    )
    monkeypatch.setattr("littrace.downloads.download_task_store_from_config", lambda _config: SimpleNamespace(
        upsert=lambda _task: None
    ))

    config = LitTraceConfig()
    config.storage.paper_library_dir = tmp_path / "papers"
    config.download_retry.enabled = False
    paper = PaperMetadata(
        paper_id="login-paper",
        title="Login paper",
        doi="10.1021/test.1",
        access_type=AccessType.REQUIRES_LOGIN,
    )
    async def fake_cdp(*args, **kwargs):
        task = args[3]
        return (
            __import__("littrace.models", fromlist=["DownloadExecutionItem"]).DownloadExecutionItem(
                paper_id=paper.paper_id,
                action="cdp_publisher_download",
                status="failed",
            ),
            task,
        )
    monkeypatch.setattr("littrace.downloads._execute_cdp_download_async", fake_cdp)
    result = __import__("asyncio").run(
        execute_downloads(
            config,
            [paper],
            DownloadExecutionRequest(paper_ids=[paper.paper_id]),
        )
    )

    assert result.items and browser.closed


def test_config_anchors_private_chrome_profile_to_config_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "cdp_downloader:\n  chrome_user_data_dir: ./private-chrome\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.cdp_downloader.chrome_user_data_dir == (
        tmp_path / "private-chrome"
    ).resolve()
