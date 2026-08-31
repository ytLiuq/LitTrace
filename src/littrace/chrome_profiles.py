from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from littrace.access_layer.cdp import CDPStatus, check_cdp_status
from littrace.config import LitTraceConfig


PUBLISHER_COOKIE_DOMAINS = [
    "wiley.com",
    "onlinelibrary.wiley.com",
    "acs.org",
    "pubs.acs.org",
    "sciencedirect.com",
    "elsevier.com",
    "nature.com",
    "springer.com",
    "rsc.org",
    "ieee.org",
]


class ChromeProfileInfo(BaseModel):
    name: str
    path: str
    display_name: str | None = None
    has_cookie_store: bool = False
    publisher_cookie_domains: list[str] = Field(default_factory=list)


class ChromeDiscoveryResult(BaseModel):
    platform: str
    executable: str | None = None
    user_data_dir: str | None = None
    profiles: list[ChromeProfileInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ChromeLaunchPlan(BaseModel):
    command: list[str]
    cdp_url: str
    profile_name: str
    user_data_dir: str


class ChromeLaunchResult(BaseModel):
    attempted: bool = False
    launched: bool = False
    already_available: bool = False
    cdp_status: CDPStatus | None = None
    command: list[str] = Field(default_factory=list)
    error: str | None = None


class BrowserSetupReport(BaseModel):
    cdp_status: CDPStatus
    discovery: ChromeDiscoveryResult
    launch_plan: ChromeLaunchPlan | None = None
    selected_profile: ChromeProfileInfo | None = None
    warnings: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)


def discover_chrome_profiles(config: LitTraceConfig) -> ChromeDiscoveryResult:
    system = platform.system().lower()
    warnings: list[str] = []
    executable = _resolve_chrome_executable(config)
    user_data_dir = _resolve_user_data_dir(config)
    profiles: list[ChromeProfileInfo] = []
    if user_data_dir is None:
        warnings.append("Could not infer Chrome user data directory for this platform.")
    else:
        # The default LitTrace chrome_user_data_dir lives under ./data and
        # is created lazily on first launch. Treat a missing directory as
        # "not yet provisioned" rather than a configuration error so the
        # first-time setup flow is not flagged as broken.
        if not user_data_dir.exists():
            user_data_dir.mkdir(parents=True, exist_ok=True)
            warnings.append(
                f"Initialized empty Chrome user-data-dir at {user_data_dir}. "
                "Open the LitTrace Chrome once and sign in to each publisher "
                "you want full-text access to."
            )
        profiles = _read_profiles(user_data_dir)
        if not profiles:
            warnings.append(f"No Chrome profiles were found under {user_data_dir}.")
    if executable is None:
        warnings.append("Google Chrome executable was not found.")
    return ChromeDiscoveryResult(
        platform=system,
        executable=str(executable) if executable else None,
        user_data_dir=str(user_data_dir) if user_data_dir else None,
        profiles=profiles,
        warnings=warnings,
    )


def build_chrome_launch_plan(
    config: LitTraceConfig,
    profile_name: str | None = None,
) -> ChromeLaunchPlan | None:
    discovery = discover_chrome_profiles(config)
    if not discovery.executable or not discovery.user_data_dir:
        return None
    selected = profile_name or config.cdp_downloader.chrome_profile_name or "Default"
    port = (
        _port_from_cdp_url(config.cdp_downloader.cdp_url)
        or config.cdp_downloader.remote_debugging_port
    )
    command = [
        discovery.executable,
        f"--remote-debugging-port={port}",
        # Chrome rejects WebSocket clients from the CDP origin unless this is
        # explicitly allowed. Keep the origin scoped to LitTrace's local port.
        f"--remote-allow-origins=http://127.0.0.1:{port}",
        f"--user-data-dir={discovery.user_data_dir}",
        f"--profile-directory={selected}",
    ]
    return ChromeLaunchPlan(
        command=command,
        cdp_url=f"http://127.0.0.1:{port}",
        profile_name=selected,
        user_data_dir=discovery.user_data_dir,
    )


def launch_chrome_for_cdp(
    config: LitTraceConfig,
    profile_name: str | None = None,
    wait_seconds: float = 4.0,
) -> ChromeLaunchResult:
    status = check_cdp_status(config)
    if status.available:
        return ChromeLaunchResult(
            attempted=False,
            launched=False,
            already_available=True,
            cdp_status=status,
        )
    plan = build_chrome_launch_plan(config, profile_name=profile_name)
    if plan is None:
        return ChromeLaunchResult(
            attempted=False,
            launched=False,
            cdp_status=status,
            error="Could not build a Chrome launch command.",
        )
    # LitTrace defaults chrome_user_data_dir to a private directory under
    # the repo so its Chrome process never collides with the user's
    # day-to-day browser. Ensure that directory exists before Chrome
    # starts — without it, Chrome refuses to launch with --user-data-dir.
    user_data_dir = Path(plan.user_data_dir)
    user_data_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.Popen(
            plan.command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        return ChromeLaunchResult(
            attempted=True,
            launched=False,
            cdp_status=status,
            command=plan.command,
            error=f"{exc.__class__.__name__}: {exc}",
        )
    # Chrome on macOS will exit immediately if its user-data-dir is already
    # held by another Chrome instance (it prints "正在现有的浏览器会话中打开。"
    # and quits). Without this poll the launcher just reports "CDP endpoint
    # did not become available" with no hint about the real cause — leaving
    # users to debug it themselves.
    time.sleep(0.4)
    if proc.poll() is not None:
        return ChromeLaunchResult(
            attempted=True,
            launched=False,
            cdp_status=status,
            command=plan.command,
            error=_explain_chrome_early_exit(plan, proc.returncode),
        )
    deadline = time.monotonic() + max(wait_seconds, 0.0)
    latest_status = status
    while time.monotonic() <= deadline:
        time.sleep(0.25)
        latest_status = check_cdp_status(config)
        if latest_status.available:
            return ChromeLaunchResult(
                attempted=True,
                launched=True,
                cdp_status=latest_status,
                command=plan.command,
            )
    return ChromeLaunchResult(
        attempted=True,
        launched=False,
        cdp_status=latest_status,
        command=plan.command,
        error="Chrome was launched, but the CDP endpoint did not become available in time.",
    )


def _explain_chrome_early_exit(plan: ChromeLaunchPlan, returncode: int | None) -> str:
    """Return a user-actionable error when the launched Chrome process exited
    before the CDP endpoint could come up. The common case on macOS is that
    another Chrome instance is already using the same user-data-dir — Chrome
    refuses to start a second one and exits immediately. Without this hint
    users see only the generic ``CDP endpoint did not become available``
    error and cannot diagnose the problem.
    """
    user_data_dir = Path(plan.user_data_dir)
    profile_dir = user_data_dir / plan.profile_name
    lock_candidates = [
        user_data_dir / "SingletonLock",
        user_data_dir / "Singleton",
        profile_dir / "LOCK",
    ]
    conflicting_lock = next(
        (lock for lock in lock_candidates if lock.exists()), None
    )
    if conflicting_lock is not None:
        return (
            f"Chrome exited immediately (returncode={returncode}). The user-data-dir "
            f"{user_data_dir!s} appears to be in use by another Chrome instance "
            f"(lock file: {conflicting_lock.name}). Quit the existing Google Chrome "
            f"window and re-run `littrace setup-browser --launch`, or set "
            f"cdp_downloader.chrome_user_data_dir in config.yaml to a fresh directory."
        )
    return (
        f"Chrome exited immediately (returncode={returncode}) before the CDP "
        f"endpoint could bind. No conflicting user-data-dir lock was detected; "
        f"check the launch command above for unsupported flags."
    )


def build_browser_setup_report(config: LitTraceConfig) -> BrowserSetupReport:
    status = check_cdp_status(config)
    discovery = discover_chrome_profiles(config)
    selected_profile = _select_profile(
        discovery.profiles, config.cdp_downloader.chrome_profile_name
    )
    launch_plan = build_chrome_launch_plan(config)
    warnings = list(discovery.warnings)
    if selected_profile is None and discovery.profiles:
        warnings.append(
            f"Configured Chrome profile {config.cdp_downloader.chrome_profile_name!r} was not found."
        )
    instructions: list[str] = []
    if not status.available:
        if launch_plan is not None:
            instructions.append("Start Chrome for LitTrace CDP access:")
            instructions.append(_shell_join(launch_plan.command))
        else:
            instructions.append(
                "Install Google Chrome or set cdp_downloader.chrome_executable and chrome_user_data_dir."
            )
    if selected_profile and not selected_profile.publisher_cookie_domains:
        instructions.append(
            "Open the selected Chrome profile and sign in to your institution or publisher once."
        )
    return BrowserSetupReport(
        cdp_status=status,
        discovery=discovery,
        launch_plan=launch_plan,
        selected_profile=selected_profile,
        warnings=warnings,
        instructions=instructions,
    )


def _resolve_chrome_executable(config: LitTraceConfig) -> Path | None:
    configured = config.cdp_downloader.chrome_executable
    if configured and configured.expanduser().exists():
        return configured.expanduser()
    candidates: list[Path] = []
    system = platform.system().lower()
    if system == "darwin":
        candidates.extend(
            [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            ]
        )
    elif system == "windows":
        for key in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            root = os.environ.get(key)
            if root:
                candidates.append(Path(root) / "Google/Chrome/Application/chrome.exe")
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            path = _which(name)
            if path:
                candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_user_data_dir(config: LitTraceConfig) -> Path | None:
    configured = config.cdp_downloader.chrome_user_data_dir
    if configured:
        return configured.expanduser()
    system = platform.system().lower()
    home = Path.home()
    if system == "darwin":
        return home / "Library/Application Support/Google/Chrome"
    if system == "windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "Google/Chrome/User Data"
        return None
    return home / ".config/google-chrome"


def _read_profiles(user_data_dir: Path) -> list[ChromeProfileInfo]:
    local_state = user_data_dir / "Local State"
    display_names = _read_profile_display_names(local_state)
    profile_names = set(display_names)
    for child in user_data_dir.iterdir():
        if child.is_dir() and (child.name == "Default" or child.name.startswith("Profile ")):
            profile_names.add(child.name)
    profiles: list[ChromeProfileInfo] = []
    for name in sorted(profile_names, key=_profile_sort_key):
        path = user_data_dir / name
        if not path.exists():
            continue
        profiles.append(
            ChromeProfileInfo(
                name=name,
                path=str(path),
                display_name=display_names.get(name),
                has_cookie_store=(path / "Network" / "Cookies").exists()
                or (path / "Cookies").exists(),
                publisher_cookie_domains=_detect_publisher_cookie_domains(path),
            )
        )
    return profiles


def _read_profile_display_names(local_state: Path) -> dict[str, str]:
    if not local_state.exists():
        return {}
    try:
        payload = json.loads(local_state.read_text(encoding="utf-8"))
    except Exception:
        return {}
    info_cache = payload.get("profile", {}).get("info_cache", {})
    if not isinstance(info_cache, dict):
        return {}
    display_names: dict[str, str] = {}
    for name, info in info_cache.items():
        if isinstance(name, str) and isinstance(info, dict):
            display = info.get("name")
            if isinstance(display, str):
                display_names[name] = display
    return display_names


def _detect_publisher_cookie_domains(profile_path: Path) -> list[str]:
    cookie_paths = [profile_path / "Network" / "Cookies", profile_path / "Cookies"]
    found: list[str] = []
    for cookie_path in cookie_paths:
        if not cookie_path.exists():
            continue
        try:
            data = cookie_path.read_bytes()
        except OSError:
            continue
        for domain in PUBLISHER_COOKIE_DOMAINS:
            if domain.encode("utf-8") in data and domain not in found:
                found.append(domain)
    return found


def _select_profile(
    profiles: list[ChromeProfileInfo],
    profile_name: str,
) -> ChromeProfileInfo | None:
    for profile in profiles:
        if profile.name == profile_name:
            return profile
    return None


def _which(name: str) -> Path | None:
    path = os.environ.get("PATH", "")
    for directory in path.split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.exists():
            return candidate
    return None


def _profile_sort_key(name: str) -> tuple[int, str]:
    if name == "Default":
        return (0, name)
    return (1, name)


def _shell_join(parts: list[str]) -> str:
    quoted: list[str] = []
    for part in parts:
        if not part:
            quoted.append("''")
        elif any(ch.isspace() for ch in part):
            quoted.append("'" + part.replace("'", "'\"'\"'") + "'")
        else:
            quoted.append(part)
    return " ".join(quoted)


def format_shell_command(parts: list[str]) -> str:
    return _shell_join(parts)


def _port_from_cdp_url(cdp_url: str) -> int | None:
    parsed = urlparse(cdp_url)
    return parsed.port
