from pathlib import Path
import subprocess

from littrace.browser import (
    browser_act_command,
    browser_open_args,
    publisher_window_session_name,
    check_browser_act,
    prewarm_chrome_direct,
    is_browser_act_api_key_required,
    is_recoverable_browser_open_failed,
    is_recoverable_browser_window_closed,
    resolve_browser_act_command,
    run_browser_act,
)
from littrace.config import BrowserAutomationConfig, LitTraceConfig


def test_browser_act_command_uses_configured_path():
    config = LitTraceConfig(browser=BrowserAutomationConfig(browser_act_path="/tmp/browser-act"))

    assert browser_act_command(config, ["--version"]) == ["/tmp/browser-act", "--version"]


def test_browser_open_args_allows_restart_for_chrome_direct():
    config = LitTraceConfig(
        browser=BrowserAutomationConfig(default_browser_type="chrome-direct")
    )

    args = browser_open_args(config, "session", "browser-id", "https://example.org")

    assert args[-1] == "--allow-restart-chrome"


def test_browser_open_args_keeps_regular_chrome_lightweight():
    config = LitTraceConfig(browser=BrowserAutomationConfig(default_browser_type="chrome"))

    args = browser_open_args(
        config, "session", "browser-id", "https://example.org", headed=True
    )

    assert "--headed" in args
    assert "--allow-restart-chrome" not in args


def test_publisher_window_session_name_is_chat_scoped():
    assert publisher_window_session_name("20260704-abc_123") == "littrace-20260704-abc_123-publisher"
    assert publisher_window_session_name() == "littrace-publisher-window"


def test_check_browser_act_reports_missing_path():
    config = LitTraceConfig(browser=BrowserAutomationConfig(browser_act_path="/tmp/missing-browser-act"))

    status = check_browser_act(config)

    assert not status.available
    assert status.errors


def test_check_browser_act_reports_browser_and_session_state(monkeypatch):
    config = LitTraceConfig(
        browser=BrowserAutomationConfig(
            browser_act_path="/tmp/browser-act",
            default_browser_id="direct_local_1",
            default_browser_type="chrome-direct",
        )
    )

    def fake_run(args, **_kwargs):
        if args == ["/tmp/browser-act", "--version"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="v1.0.4", stderr="")
        if args == ["/tmp/browser-act", "browser", "list"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='id=direct_local_1 name="littrace" type=chrome-direct state=idle',
                stderr="",
            )
        if args == ["/tmp/browser-act", "session", "list"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="\n".join(
                    [
                        "session_name: littrace-chat-publisher",
                        "browser_type: chrome-direct",
                        "browser_id: direct_local_1",
                        "title: Example",
                    ]
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(subprocess, "run", fake_run)

    status = check_browser_act(config)

    assert status.available
    assert status.browser_found
    assert status.default_browser_type == "chrome-direct"
    assert status.active_sessions == ["littrace-chat-publisher"]


def test_resolve_browser_act_uses_uv_tool_location(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    tool = fake_home / ".local/share/uv/tools/browser-act-cli/bin/browser-act"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr("shutil.which", lambda _name: None)

    command = resolve_browser_act_command(LitTraceConfig())

    assert Path(command) == tool


def test_browser_window_closed_is_recoverable():
    assert is_recoverable_browser_window_closed(
        "Error: {'code': -32000, 'message': 'Browser window not found'}"
    )
    assert is_recoverable_browser_window_closed('Error: Session "littrace-acs-auth" not found')


def test_browser_open_unknown_error_is_recoverable():
    assert is_recoverable_browser_open_failed("Error 230404: Unknown error")
    assert is_recoverable_browser_open_failed("Error: Browser window not found")


def test_browser_act_api_key_required_is_detected():
    assert is_browser_act_api_key_required('Error 230103: API key required for "browser open". To use browser-act')


def test_run_browser_act_marks_window_closed_recoverable(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["browser-act"],
            returncode=1,
            stdout="",
            stderr="Error: Browser window not found",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_browser_act(LitTraceConfig(), ["session", "list"])

    assert result.recoverable_window_closed
    assert result.recoverable_browser_open_failed
    assert result.returncode == 1


def test_prewarm_chrome_direct_launches_google_chrome_on_macos(monkeypatch):
    calls = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    result = prewarm_chrome_direct(
        LitTraceConfig(browser=BrowserAutomationConfig(default_browser_type="chrome-direct"))
    )

    assert result.attempted
    assert result.ok
    assert calls == [["open", "-ga", "Google Chrome"]]
