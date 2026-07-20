"""Public boundary for browser, publisher access, and local PDF retrieval.

Exports are loaded lazily so low-level compatibility modules can import a
specific implementation without pulling in the whole browser/login stack.
"""

from importlib import import_module


_EXPORTS = {
    "AuthorizedPdfArchiveResult": "archiving",
    "archive_authorized_pdf_response": "archiving",
    "AuthorizedPdfFetchResult": "browser_sessions",
    "BrowserAuthorizationWaitResult": "browser_sessions",
    "BrowserLoginSessionPlan": "browser_sessions",
    "LoginLaunchResult": "browser_sessions",
    "browser_login_session_for_paper": "browser_sessions",
    "fetch_authorized_pdf_after_user_auth": "browser_sessions",
    "launch_login_for_paper": "browser_sessions",
    "open_browser_login_session": "browser_sessions",
    "publisher_window_session_name_for_chat": "browser_sessions",
    "wait_for_browser_authorization": "browser_sessions",
    "CDPDownloadResult": "cdp",
    "CDPStatus": "cdp",
    "check_cdp_status": "cdp",
    "download_paper_via_cdp": "cdp",
    "build_download_plan": "download_planning",
    "execute_downloads": "download_planning",
    "plan_download": "download_planning",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
