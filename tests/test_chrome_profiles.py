import subprocess

from littrace.chrome_profiles import (
    build_browser_setup_report,
    build_chrome_launch_plan,
    discover_chrome_profiles,
    launch_chrome_for_cdp,
)
from littrace.cdp_downloader import CDPStatus
from littrace.config import CDPDownloaderConfig, LitTraceConfig


def test_discover_chrome_profiles_reads_local_state_and_cookie_markers(monkeypatch, tmp_path):
    chrome = tmp_path / "chrome"
    user_data = tmp_path / "user-data"
    profile = user_data / "Default"
    cookie_store = profile / "Network" / "Cookies"
    chrome.write_text("#!/bin/sh\n", encoding="utf-8")
    cookie_store.parent.mkdir(parents=True)
    cookie_store.write_bytes(b".sciencedirect.com\tcookie")
    (user_data / "Local State").write_text(
        '{"profile":{"info_cache":{"Default":{"name":"Research"}}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    config = LitTraceConfig(
        cdp_downloader=CDPDownloaderConfig(
            chrome_executable=chrome,
            chrome_user_data_dir=user_data,
        )
    )

    result = discover_chrome_profiles(config)

    assert result.executable == str(chrome)
    assert result.user_data_dir == str(user_data)
    assert result.profiles[0].name == "Default"
    assert result.profiles[0].display_name == "Research"
    assert result.profiles[0].publisher_cookie_domains == ["sciencedirect.com"]


def test_build_chrome_launch_plan_uses_cdp_url_port(monkeypatch, tmp_path):
    chrome = tmp_path / "chrome"
    user_data = tmp_path / "user-data"
    (user_data / "Profile 1").mkdir(parents=True)
    chrome.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    config = LitTraceConfig(
        cdp_downloader=CDPDownloaderConfig(
            cdp_url="http://127.0.0.1:9333",
            chrome_executable=chrome,
            chrome_user_data_dir=user_data,
            chrome_profile_name="Profile 1",
        )
    )

    plan = build_chrome_launch_plan(config)

    assert plan is not None
    assert "--remote-debugging-port=9333" in plan.command
    assert f"--user-data-dir={user_data}" in plan.command
    assert "--profile-directory=Profile 1" in plan.command


def test_build_browser_setup_report_suggests_start_command(monkeypatch, tmp_path):
    chrome = tmp_path / "chrome"
    user_data = tmp_path / "user-data"
    (user_data / "Default").mkdir(parents=True)
    chrome.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "littrace.chrome_profiles.check_cdp_status",
        lambda _config: CDPStatus(
            available=False,
            cdp_url="http://127.0.0.1:19222",
            error="ConnectError",
        ),
    )
    config = LitTraceConfig(
        cdp_downloader=CDPDownloaderConfig(
            chrome_executable=chrome,
            chrome_user_data_dir=user_data,
        )
    )

    report = build_browser_setup_report(config)

    assert report.launch_plan is not None
    assert report.instructions[0] == "Start Chrome for LitTrace CDP access:"
    assert "remote-debugging-port=19222" in report.instructions[1]


def test_launch_chrome_for_cdp_starts_process_when_missing(monkeypatch, tmp_path):
    chrome = tmp_path / "chrome"
    user_data = tmp_path / "user-data"
    (user_data / "Default").mkdir(parents=True)
    chrome.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    statuses = [
        CDPStatus(available=False, cdp_url="http://127.0.0.1:19222", error="missing"),
        CDPStatus(available=True, cdp_url="http://127.0.0.1:19222", browser="Chrome"),
    ]
    monkeypatch.setattr("littrace.chrome_profiles.check_cdp_status", lambda _config: statuses.pop(0))
    calls = []

    class FakePopen:
        def __init__(self, args, **_kwargs):
            calls.append(args)

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)
    config = LitTraceConfig(
        cdp_downloader=CDPDownloaderConfig(
            chrome_executable=chrome,
            chrome_user_data_dir=user_data,
        )
    )

    result = launch_chrome_for_cdp(config)

    assert result.launched
    assert calls
    assert "--profile-directory=Default" in calls[0]
