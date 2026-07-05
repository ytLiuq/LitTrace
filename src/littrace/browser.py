from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig


class BrowserActStatus(BaseModel):
    available: bool
    command: str
    version: str | None = None
    browser_id: str | None = None
    browser_found: bool | None = None
    default_browser_type: str | None = None
    active_sessions: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    install_hint: str = "uv tool install browser-act-cli --python 3.12"


class BrowserActRunResult(BaseModel):
    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    recoverable_window_closed: bool = False
    recoverable_browser_open_failed: bool = False
    api_key_required: bool = False


class ChromeDirectPrewarmResult(BaseModel):
    attempted: bool = False
    ok: bool = False
    command: list[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


def resolve_browser_act_command(config: LitTraceConfig) -> str:
    configured = config.browser.browser_act_path
    if configured and configured != "browser-act" and Path(configured).expanduser().exists():
        return str(Path(configured).expanduser())
    if configured and configured != "browser-act":
        return configured
    env_path = os.environ.get("LITTRACE_BROWSER_ACT_PATH")
    if env_path and Path(env_path).expanduser().exists():
        return str(Path(env_path).expanduser())
    found = shutil.which("browser-act")
    if found:
        return found
    for candidate in _candidate_browser_act_paths():
        if candidate.exists():
            return str(candidate)
    return configured


def check_browser_act(config: LitTraceConfig) -> BrowserActStatus:
    command = resolve_browser_act_command(config)
    errors: list[str] = []
    diagnostics: list[str] = []
    version = None
    browser_found: bool | None = None
    active_sessions: list[str] = []
    try:
        result = subprocess.run(
            [command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
        else:
            errors.append(result.stderr.strip() or result.stdout.strip() or "browser-act --version failed")
    except Exception as exc:
        errors.append(f"{exc.__class__.__name__}: {exc}")
    if not errors:
        try:
            browser_result = subprocess.run(
                [command, "browser", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            browser_output = f"{browser_result.stdout or ''}\n{browser_result.stderr or ''}"
            if browser_result.returncode == 0:
                browser_found = bool(
                    config.browser.default_browser_id
                    and config.browser.default_browser_id in browser_output
                )
                if config.browser.default_browser_id and not browser_found:
                    diagnostics.append(
                        f"Configured browser id {config.browser.default_browser_id} was not found in browser-act browser list."
                    )
            else:
                diagnostics.append(browser_result.stderr.strip() or browser_result.stdout.strip() or "browser-act browser list failed")
        except Exception as exc:
            diagnostics.append(f"browser list {exc.__class__.__name__}: {exc}")
        try:
            session_result = subprocess.run(
                [command, "session", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            session_output = session_result.stdout or ""
            if session_result.returncode == 0:
                active_sessions = _parse_browser_act_sessions(session_output)
            else:
                diagnostics.append(session_result.stderr.strip() or session_result.stdout.strip() or "browser-act session list failed")
        except Exception as exc:
            diagnostics.append(f"session list {exc.__class__.__name__}: {exc}")
    return BrowserActStatus(
        available=not errors,
        command=command,
        version=version,
        browser_id=config.browser.default_browser_id,
        browser_found=browser_found,
        default_browser_type=config.browser.default_browser_type,
        active_sessions=active_sessions,
        diagnostics=diagnostics,
        errors=errors,
    )


def require_browser_act(config: LitTraceConfig) -> BrowserActStatus:
    status = check_browser_act(config)
    if config.browser.required and not status.available:
        raise RuntimeError(
            "browser-act is required for LitTrace full-text publisher access. "
            f"Install it with `{status.install_hint}` or set LITTRACE_BROWSER_ACT_PATH. "
            f"Errors: {'; '.join(status.errors)}"
        )
    return status


def browser_act_command(config: LitTraceConfig, args: list[str]) -> list[str]:
    return [resolve_browser_act_command(config), *args]


def browser_open_args(
    config: LitTraceConfig,
    session_name: str,
    browser_id: str,
    url: str,
    headed: bool = False,
) -> list[str]:
    args = [
        "--session",
        session_name,
        "browser",
        "open",
        browser_id,
        url,
    ]
    if headed:
        args.append("--headed")
    if config.browser.default_browser_type == "chrome-direct":
        args.append("--allow-restart-chrome")
    return args


def prewarm_chrome_direct(config: LitTraceConfig, wait_seconds: float = 1.5) -> ChromeDirectPrewarmResult:
    if config.browser.default_browser_type != "chrome-direct":
        return ChromeDirectPrewarmResult()
    if not sys.platform == "darwin":
        return ChromeDirectPrewarmResult(
            attempted=False,
            ok=True,
            error="Chrome direct prewarm is currently only implemented for macOS.",
        )
    command = ["open", "-ga", "Google Chrome"]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            time.sleep(max(wait_seconds, 0.0))
        return ChromeDirectPrewarmResult(
            attempted=True,
            ok=result.returncode == 0,
            command=command,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            error=None
            if result.returncode == 0
            else result.stderr or result.stdout or "Could not launch Google Chrome.",
        )
    except Exception as exc:
        return ChromeDirectPrewarmResult(
            attempted=True,
            ok=False,
            command=command,
            error=f"{exc.__class__.__name__}: {exc}",
        )


def run_browser_act(
    config: LitTraceConfig,
    args: list[str],
    timeout_seconds: float = 60.0,
) -> BrowserActRunResult:
    command = browser_act_command(config, args)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        return BrowserActRunResult(
            command=command,
            returncode=result.returncode,
            stdout=stdout,
            stderr=stderr,
            recoverable_window_closed=is_recoverable_browser_window_closed(
                f"{stdout}\n{stderr}"
            ),
            recoverable_browser_open_failed=is_recoverable_browser_open_failed(
                f"{stdout}\n{stderr}"
            ),
            api_key_required=is_browser_act_api_key_required(f"{stdout}\n{stderr}"),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return BrowserActRunResult(
            command=command,
            returncode=124,
            stdout=stdout,
            stderr=stderr or f"browser-act timed out after {timeout_seconds} seconds",
            recoverable_browser_open_failed=is_recoverable_browser_open_failed(
                f"{stdout}\n{stderr}"
            ),
            api_key_required=is_browser_act_api_key_required(f"{stdout}\n{stderr}"),
        )


def is_recoverable_browser_window_closed(output: str) -> bool:
    lowered = output.lower()
    return (
        "browser window not found" in lowered
        or "target closed" in lowered
        or "session" in lowered
        and ("not found" in lowered or "does not exist" in lowered)
    )


def is_recoverable_browser_open_failed(output: str) -> bool:
    lowered = output.lower()
    return (
        "error 230404" in lowered
        or "browser window not found" in lowered
        or "window not found" in lowered
        or ("unknown error" in lowered and "browser" in lowered)
    )


def is_browser_act_api_key_required(output: str) -> bool:
    lowered = output.lower()
    return "api key required" in lowered and "browser-act" in lowered


def browser_session_name_for_paper(paper_id: str, suffix: str = "auth") -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in paper_id)
    safe = safe.strip("-_") or "paper"
    return f"littrace-{safe[:24]}-{suffix}"


def publisher_window_session_name(scope_id: str | None = None) -> str:
    if not scope_id:
        return "littrace-publisher-window"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in scope_id)
    safe = safe.strip("-_") or "workspace"
    return f"littrace-{safe[:32]}-publisher"


def _parse_browser_act_sessions(output: str) -> list[str]:
    sessions: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("no active sessions"):
            continue
        if line.startswith("session_name:"):
            value = line.split(":", 1)[1].strip()
            if value and value not in sessions:
                sessions.append(value)
            continue
        if ":" in line:
            continue
        first = line.split()[0].strip(":")
        if first and first not in sessions:
            sessions.append(first)
    return sessions


def _default_uv_tool_browser_act_path() -> str | None:
    for candidate in _candidate_browser_act_paths():
        if candidate.exists():
            return str(candidate)
    return None


def _candidate_browser_act_paths() -> list[Path]:
    home = Path(os.path.expanduser("~"))
    executable = "browser-act.exe" if sys.platform.startswith("win") else "browser-act"
    paths = [
        Path(os.path.expanduser("~"))
        / ".local"
        / "share"
        / "uv"
        / "tools"
        / "browser-act-cli"
        / "bin"
        / executable,
        home / ".local" / "bin" / executable,
        home / ".cargo" / "bin" / executable,
    ]
    if sys.platform == "darwin":
        paths.extend(
            [
                Path("/opt/homebrew/bin") / executable,
                Path("/usr/local/bin") / executable,
            ]
        )
    if sys.platform.startswith("win"):
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            paths.append(Path(local_app) / "uv" / "tools" / "browser-act-cli" / "Scripts" / executable)
    return paths
