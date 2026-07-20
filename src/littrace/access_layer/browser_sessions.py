from __future__ import annotations

from littrace.access_layer.browser import (
    BrowserActStatus,
    browser_act_command,
    check_browser_act,
    prewarm_chrome_direct,
    run_browser_act,
)
from littrace.access_layer.login_flow import (
    AuthorizedPdfFetchResult,
    BrowserAuthorizationWaitResult,
    BrowserLoginSessionPlan,
    LoginLaunchResult,
    browser_login_session_for_paper,
    browser_login_session_plans_for_workspace,
    detect_user_confirmation_required,
    discover_institutional_login_url_from_browser_session,
    discover_pdf_url_from_browser_session,
    fetch_authorized_pdf_after_user_auth,
    launch_login_for_paper,
    open_browser_login_session,
    publisher_window_session_name_for_chat,
    resume_browser_auth_after_user_close,
    wait_for_browser_authorization,
)

__all__ = [
    "AuthorizedPdfFetchResult",
    "BrowserActStatus",
    "BrowserAuthorizationWaitResult",
    "BrowserLoginSessionPlan",
    "LoginLaunchResult",
    "browser_act_command",
    "browser_login_session_for_paper",
    "browser_login_session_plans_for_workspace",
    "check_browser_act",
    "detect_user_confirmation_required",
    "discover_institutional_login_url_from_browser_session",
    "discover_pdf_url_from_browser_session",
    "fetch_authorized_pdf_after_user_auth",
    "launch_login_for_paper",
    "open_browser_login_session",
    "prewarm_chrome_direct",
    "publisher_window_session_name_for_chat",
    "resume_browser_auth_after_user_close",
    "run_browser_act",
    "wait_for_browser_authorization",
]
