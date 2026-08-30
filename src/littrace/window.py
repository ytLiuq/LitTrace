from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from littrace.agent_runtime import handle_agent_chat
from littrace.auto_resume import auto_resume_downloaded_pdfs
from littrace.config import load_config
from littrace.intent import parse_chat_intent
from littrace.access_layer import (
    browser_login_session_for_paper,
    fetch_authorized_pdf_after_user_auth,
    open_browser_login_session,
    publisher_window_session_name_for_chat,
    wait_for_browser_authorization,
)
from littrace.models import ChatRequest, DownloadPlan, LiteratureWorkspace, WorkflowTrace
from littrace.session import (
    ChatSession,
    append_message,
    create_chat_session,
    list_chat_sessions,
    load_or_create_session,
    load_workspace,
    save_workspace,
)
from littrace.state_db import state_store_from_config
from littrace.evidence.tables import decide_artifact_extraction_need
from littrace.tui import render_context_lines

# 后台同步 / RAG daily 集成
try:
    from littrace.rag_jobs import run_daily_rag_maintenance  # noqa: F401
except ImportError:  # pragma: no cover - defensive
    run_daily_rag_maintenance = None  # type: ignore[assignment]
try:
    from littrace.retrieval.rag_refresh import refresh_session_rag_index  # noqa: F401
except ImportError:  # pragma: no cover - defensive
    refresh_session_rag_index = None  # type: ignore[assignment]


# Round 28: ChatGPT Codex desktop-app visual language (Light theme).
# Warm-white canvas with a darker teal accent used sparingly; dark ink on
# light surfaces; hairline borders instead of heavy separators. Existing
# key names are kept (so call sites don't churn) — six new tokens were
# added for the flat-message avatars, inline tool cards, and the Stop pill.
DESIGN = {
    # Accent — darker teal so it reads cleanly on warm-white surfaces.
    "primary":             "#0f7a7b",
    "primary_hover":       "#0b6060",
    "primary_focus":       "#0b6060",
    "primary_on_dark":     "#0f7a7b",
    # Ink ladder.
    "ink":                 "#1f1f1f",
    "ink_muted":           "#6e6e6e",
    "ink_subtle":          "#a1a1a1",
    # Background ladder.
    "canvas":              "#faf9f5",   # Codex warm-white page bg
    "parchment":           "#f4f2ec",   # slightly warmer than the canvas
    "pearl":               "#ffffff",
    "surface_1":           "#ffffff",   # card surface
    "surface_2":           "#f7f6f1",   # input-area inner tint, tool card body
    "hairline":            "#e8e6e0",   # softer 1px border
    # Dark tokens (unchanged — used by the dark nav bar).
    "black":               "#0b0c0e",
    "dark_tile":           "#0f1011",
    "on_dark":             "#f7f8f8",
    # Body / accents.
    "body_muted":          "#a1a1a1",
    "accent_coral":        "#cc785c",   # kept for the trailing-dot pulse
    "accent_teal_subtle":  "#e6efee",   # kept for subtle highlight tints
    # Round 28: flat-message avatars + inline tool card + Stop pill.
    "avatar_bg":           "#1f1f1f",   # dark filled square for assistant
    "user_avatar_bg":      "#e6efee",   # soft teal square for user
    "pill_bg":             "#0f7a7b",   # Stop pill background
    "pill_fg":             "#ffffff",   # Stop pill text
    "tool_card_bg":        "#f7f6f1",   # tool card body background
    "tool_card_border":    "#e8e6e0",   # tool card hairline border
}


# Slash-command catalog used by the input-box autocomplete popup.
# Each entry is (name, description, context_only). `context_only` is
# recorded for a future workspace-aware filter pass; v1 shows all commands.
COMMAND_CATALOG: list[tuple[str, str, bool]] = [
    ("context", "显示 / 隐藏当前文献上下文", True),
    ("papers", "列出当前上下文文献", True),
    ("parse", "按当前解析模式处理 PDF", False),
    ("parse --ocr", "强制使用 OCR 解析", False),
    ("parse --text", "强制使用文本层解析", False),
    ("table", "抽取性能指标并生成对比表", True),
    ("storyline", "梳理论文回应关系", True),
    ("storyline-report", "导出 storyline 报告", True),
    ("storyline-review", "Reviewer 审阅 storyline", True),
    ("dashboard", "打开 RAG / Daily 仪表盘", True),
    ("quality", "运行质量门", True),
    ("agents", "列出可用 agents", False),
    ("workflow", "显示当前 workflow trace", True),
    ("quality-audits", "运行质量审计", True),
    ("plan", "显示当前执行计划", False),
    ("init-config", "运行 config wizard", False),
    ("login", "打开授权登录弹窗", True),
    ("attach", "手动附加本地 PDF", False),
    ("attach-si", "附加 SI / 补充材料", False),
    ("full-text", "构建 full-text context", True),
    ("backfill-dois", "回填 DOI", False),
    ("publisher-retrieve", "按 publisher 抓取", False),
    ("check-downloads", "检查当前下载计划", True),
    ("resume-downloads", "恢复下载（等待用户授权）", True),
    ("benchmark", "运行评测基准", False),
    ("golden-eval", "运行 golden set 评估", False),
    ("export", "导出当前 session", False),
    ("quit", "关闭窗口", False),
    ("全部下载", "选择当前上下文中全部待下载文献", True),
    ("选择第 N 篇下载", "选择第 N 篇进入下载计划", True),
    ("取消选择第 N 篇", "从下载计划中移除第 N 篇", True),
]


# Round 26: input-entry markdown tags. ``_render_input_markdown``
# strips every tag in this tuple before re-applying the regex matches,
# so anything not listed here is preserved across re-renders.
_INPUT_MD_TAGS: tuple[str, ...] = (
    "md_h1", "md_h2", "md_h3",
    "md_bold", "md_italic",
    "md_code", "md_codeblock",
    "md_list_bullet", "md_quote", "md_link",
)

# Round 27: same 10 tags, but bound to ``chat_text`` so assistant
# output also renders Markdown. Reusing the same names means the
# regex table and tag-lookup logic are shared.
_CHAT_MD_TAGS: tuple[str, ...] = _INPUT_MD_TAGS

# Round 27: ``_MD_PATTERNS`` ``kind`` -> the actual ``md_*`` tag to
# apply. ``md_heading`` is ``None`` because the tag name depends on
# the run length (``md_h1`` / ``md_h2`` / ``md_h3``); the renderer
# resolves it inline from ``m.group(1)``.
_KIND_TO_TAG: dict[str, str | None] = {
    "md_code":      "md_code",
    "md_codeblock": "md_codeblock",
    "md_heading":   None,
    "md_list":      "md_list_bullet",
    "md_quote":     "md_quote",
    "md_bold":      "md_bold",
    "md_italic":    "md_italic",
    "md_link":      "md_link",
}

# Round 26: regex catalog for the markdown live renderer.
# Order matters — fenced code blocks must be matched before the inline
# patterns so the ``**`` and ``*`` markers inside a code block don't
# get reinterpreted as bold / italic. Each entry yields ``(kind, m)``
# pairs where ``kind`` is one of the ``md_*`` tag names above.
_MD_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("md_codeblock", re.compile(r"```[\s\S]*?```")),
    ("md_code",      re.compile(r"`([^`\n]+?)`")),
    # ATX heading marker (we read the level from group(1) below).
    ("md_heading",   re.compile(r"^(#{1,3})\s+", re.MULTILINE)),
    ("md_list",      re.compile(r"^[-*]\s+", re.MULTILINE)),
    ("md_quote",     re.compile(r"^>\s?", re.MULTILINE)),
    ("md_bold",      re.compile(r"\*\*([^*\n]+?)\*\*")),
    # Italic must not flank ``**`` (bold) — the negative lookarounds
    # prevent the inner ``*x*`` of a ``**x**`` from being styled italic.
    ("md_italic",    re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")),
    ("md_link",      re.compile(r"\[([^\]\n]+?)\]\(([^)\n]+?)\)")),
)

# Round 26 → 27: status-bubble animation constants.
# Round 27 dropped the tick from 400ms to 200ms — the user feedback
# was that 400ms looked frozen on first impression. 200ms reads as
# "actively waiting" without becoming distracting.
_STATUS_ANIM_INTERVAL_MS = 200  # trailing-dot tick interval
# Phase labels whose bubble should pulse. Finalised labels
# (``Codex 已认证``, ``Codex 回复完成``) are NOT animated — they're
# replaced by ``_set_status_in_chat`` and the check below stops the
# cycle the moment a non-animatable label lands.
_STATUS_ANIMATED_PATTERNS: tuple[str, ...] = ("思考", "流式输出", "回复中")


class LitTraceWindow:
    def __init__(self) -> None:
        self.tk, self.ttk = _load_tk()
        # Enable Windows per-monitor DPI awareness so Tk doesn't snap to
        # 96 DPI on a high-DPI display — without this, ClearType font
        # rendering falls back to GDI grayscale and the GUI shows the
        # same "毛刺感" users see at native scaling.
        if sys.platform == "win32":
            try:
                import ctypes
                # Per-monitor v2 is the modern API; falls back to system
                # DPI if the host doesn't support it.
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass
        self.config = load_config()
        # Round 20: Window is a strict Codex surface. Same rules as the TUI:
        # an explicit ``LITTRACE_AGENT_RUNTIME=legacy`` is a hard error,
        # a legacy value inherited from an older config is overwritten
        # with a warning, and ``fallback_to_legacy`` is forced off so the
        # route layer can never silently call legacy chat.
        self.config = _resolve_window_config(self.config)
        self.session = create_chat_session(self.config)
        _scope_storage_to_session(self.config, self.session)
        self.workspace = LiteratureWorkspace()
        self.context_visible = True
        self.last_download_plan: DownloadPlan | None = None
        self.parse_strategy = "text_only"
        # Round 20: mid-turn elicitation modals. Each pending approval
        # is appended here and the Tk main thread drains it via
        # ``root.after`` callbacks.
        self._pending_approvals: list[dict[str, Any]] = []
        self._approval_roots: dict[int, Any] = {}
        self.context_popup = None
        self.ocr_popup = None
        self.login_popup = None
        self.rag_busy = False
        self.sync_session_btn: object | None = None
        self.full_daily_btn: object | None = None
        self.rag_text: object | None = None
        # Slash-command autocomplete popup state.
        self.command_popup: object | None = None
        self.command_popup_list: object | None = None
        self.command_popup_index: int = 0
        self.command_popup_query: str = ""

        self.root = self.tk.Tk()
        # Reset the module-level font cache so the new Tk root owns its
        # own tk.font.Font objects. Without this, a second window launched
        # in the same process would inherit fonts bound to the first root
        # and Tk would raise ``TclError: invalid command name ".!font"``
        # on the first paint.
        _reset_font_cache()
        # Round 26: install the CJK-friendly font chain as the Tk
        # default for every widget class. The Windows IME composition
        # window reads the widget-class default font when picking the
        # overlay face — without this, pinyin composition text and the
        # committed Chinese characters Tk renders from the YaHei/YaHei
        # UI fallback drift in size because the OS picks MS Shell Dlg
        # for the IME overlay. ``*Text*Font`` is the load-bearing entry
        # because ``input_entry`` is a ``tk.Text`` (not ``ttk.Entry``).
        _default_font = _font("body")
        self.root.option_add("*Font", _default_font)
        self.root.option_add("*TCombobox*Font", _default_font)
        self.root.option_add("*Text*Font", _default_font)
        self.root.option_add("*TEntry*Font", _default_font)
        # Round 26: status-bubble trailing-dots animation state. The
        # timers are managed via ``root.after`` ids so we can cancel a
        # pending tick if the user sends another turn or an error lands.
        self._status_anim_after_id: str | None = None
        self._status_dot_count: int = 0
        self._status_base_label: str = ""
        # Round 26: debounced markdown re-render of ``input_entry``.
        # Coalesces IME composition bursts into one re-render pass.
        self._input_md_after_id: str | None = None
        # Round 27: per-bubble action bookkeeping. Each entry is the
        # record pushed by ``_attach_bubble_actions`` and is rebuilt
        # from scratch on session switch (``_render_session_messages``).
        self._bubble_records: list[dict] = []
        # Round 27: last user-sent text. ``_send`` records it so the
        # 重新生成 button has a target to re-send.
        self._last_user_message: str | None = None
        # Round 27: stop flag for the active streaming turn. Set in
        # ``_handle_message_thread`` and consumed by the App Server
        # delta loop via ``stop_event=...``. Reset between turns.
        self._streaming_stop_event: "threading.Event | None" = None
        # Round 27: handles of the Stop button window anchored on the
        # streaming bubble (None while no bubble is streaming).
        self._stop_button_window: Any = None
        self._stop_button: Any = None
        # Round 28: the inline ``Pill.TButton`` stop pill (Phase 4).
        # ``_stop_pill`` is the Button widget; ``_stop_pill_index`` is
        # the Tk text index where the pill is anchored so we can delete
        # the surrounding window slot on finalize. ``_stop_button_*``
        # are kept around as legacy attributes for back-compat with the
        # Round 27 probe + tests.
        self._stop_pill: Any = None
        self._stop_pill_index: Any = None
        self._stop_button_frame: Any = None
        self.root.title("LitTrace")

        # Round 21: derive the window geometry from the *actual* screen
        # size rather than a hardcoded 1280x820. On a 4K monitor the
        # hardcoded size looks comically small; on a 1366x768 laptop
        # it overflows. The 0.78 / 0.82 factors leave room for the OS
        # taskbar and snap margins while still filling most of the
        # desktop. ``winfo_screen*`` returns physical pixels *before*
        # the DPI awareness call earlier in ``__init__``, so the
        # numbers are already in scaled-pixel units.
        screen_w = max(self.root.winfo_screenwidth(), 1024)
        screen_h = max(self.root.winfo_screenheight(), 720)
        target_w = int(screen_w * 0.78)
        target_h = int(screen_h * 0.82)
        # Center on the primary monitor.
        x = max(0, (screen_w - target_w) // 2)
        y = max(0, (screen_h - target_h) // 2)
        self.root.geometry(f"{target_w}x{target_h}+{x}+{y}")
        # Minsize is content-driven, not absolute — chat list + sidebar
        # need ~720px combined, but we leave headroom for the title bar
        # and a 600px-tall workspace preview.
        self.root.minsize(720, 560)
        self.root.configure(background=DESIGN["parchment"])

        self._configure_styles()
        self._build_layout()
        self._configure_copy_bindings()
        self._refresh_context()
        self._refresh_ocr_buttons()
        self._refresh_session_history()
        self._refresh_rag_panel()

    def run(self) -> None:
        # Round 22: hard-block on the Codex App Server handshake before
        # showing the Tk window. The user explicitly asked for "在
        # codex-cli没有准备好前，你都不应该打开GUI弹窗" — the GUI
        # must not flash up only to be replaced by an error modal two
        # seconds later. The shared TUI preflight drives
        # ``initialize`` + ``read_account`` against the App Server and
        # returns a structured ``CodexStartupError`` on any failure.
        try:
            asyncio.run(
                _codex_window_startup_preflight(self.config, self.session)
            )
        except Exception as exc:
            self._show_startup_error_modal(exc)
            return

        # Preflight passed — Tk window now visible.
        self.root.mainloop()

    def _configure_styles(self) -> None:
        self.style = self.ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except self.tk.TclError:
            pass
        self.style.configure(".", font=_font("body"), background=DESIGN["parchment"])
        self.style.configure("Canvas.TFrame", background=DESIGN["parchment"])
        self.style.configure("Nav.TFrame", background=DESIGN["black"])
        self.style.configure("Subnav.TFrame", background=DESIGN["parchment"])
        self.style.configure("Tile.TFrame", background=DESIGN["surface_1"])
        self.style.configure("Pearl.TFrame", background=DESIGN["pearl"])
        # Chat input area: outer host provides parchment border; inner pearl
        # uses a slightly tinted surface_2 to differentiate from the chat
        # output's white canvas above.
        self.style.configure("InputHost.TFrame", background=DESIGN["parchment"])
        self.style.configure("InputPearl.TFrame", background=DESIGN["surface_2"])
        self.style.configure(
            "DarkTile.TFrame",
            background=DESIGN["dark_tile"],
        )
        self.style.configure(
            "Nav.TLabel",
            background=DESIGN["black"],
            foreground=DESIGN["on_dark"],
            font=_font("nav"),
        )
        self.style.configure(
            "Brand.TLabel",
            background=DESIGN["parchment"],
            foreground=DESIGN["ink"],
            font=_font("tagline"),
        )
        self.style.configure(
            "Title.TLabel",
            background=DESIGN["canvas"],
            foreground=DESIGN["ink"],
            font=_font("display"),
        )
        # PaneTitle — Claude serif path for top-level pane headers only.
        # Used by 执行 Trace / 历史 Session / 当前文献上下文 / 后台同步 / Daily.
        self.style.configure(
            "PaneTitle.TLabel",
            background=DESIGN["canvas"],
            foreground=DESIGN["ink"],
            font=_font("title_serif"),
        )
        # Sub-header for short captions inside a pane (sans, body_strong).
        self.style.configure(
            "TileHeader.TLabel",
            background=DESIGN["canvas"],
            foreground=DESIGN["ink"],
            font=_font("body_strong"),
        )
        self.style.configure(
            "Caption.TLabel",
            background=DESIGN["canvas"],
            foreground=DESIGN["ink_muted"],
            font=_font("caption"),
        )
        self.style.configure(
            "Status.TLabel",
            background=DESIGN["parchment"],
            foreground=DESIGN["ink_muted"],
            font=_font("fine"),
        )
        # Hairline separator between right-pane sections.
        self.style.configure(
            "Horizontal.TSeparator",
            background=DESIGN["hairline"],
        )
        self.style.configure(
            "Primary.TButton",
            background=DESIGN["primary"],
            foreground=DESIGN["on_dark"],
            borderwidth=0,
            focusthickness=2,
            focuscolor=DESIGN["primary_focus"],
            padding=(16, 8),   # Round 28: tighter padding (was 18, 9)
            font=_font("button"),
        )
        self.style.map(
            "Primary.TButton",
            background=[("active", DESIGN["primary_focus"]), ("pressed", DESIGN["primary"])],
            foreground=[("disabled", DESIGN["body_muted"])],
        )
        # Secondary utility button — restrained: pearl background, ink foreground.
        # The teal accent stays reserved for one CTA per row.
        self.style.configure(
            "Secondary.TButton",
            background=DESIGN["pearl"],
            foreground=DESIGN["ink"],
            bordercolor=DESIGN["hairline"],
            lightcolor=DESIGN["pearl"],
            darkcolor=DESIGN["hairline"],
            padding=(14, 7),  # Round 28: tighter (was 15, 8)
            font=_font("button_utility"),
        )
        self.style.map(
            "Secondary.TButton",
            background=[("active", DESIGN["canvas"]), ("pressed", DESIGN["canvas"])],
            foreground=[("active", DESIGN["primary"])],
        )
        self.style.configure(
            "DarkUtility.TButton",
            background=DESIGN["ink"],
            foreground=DESIGN["on_dark"],
            borderwidth=0,
            padding=(14, 7),
            font=_font("button_utility"),
        )
        # Round 27: link-style button used for the per-bubble action
        # row (复制 / 重新生成 / 编辑). Reads as a tappable link,
        # not a chunky CTA — keeps the chat area visually quiet.
        self.style.configure(
            "Link.TButton",
            background=DESIGN["canvas"],
            foreground=DESIGN["primary"],
            borderwidth=0,
            relief=self.tk.FLAT,
            focusthickness=0,
            padding=(4, 2),     # Round 28: tighter (was 6, 2)
            font=_font("fine"),
        )
        self.style.map(
            "Link.TButton",
            foreground=[("active", DESIGN["primary_hover"])],
            background=[("active", DESIGN["accent_teal_subtle"])],
        )

        # ---- Round 28: Codex visual language additions ----

        # Flat-message avatars. Square (not circular — Tk on Windows does
        # not honour a border-radius on ttk widgets) sized at ~24px via
        # ``width=3`` at the 12pt ``avatar`` recipe. The dark-on-light
        # contrast lets the initial letter pop against the warm-white
        # canvas without needing a coloured ring.
        self.style.configure(
            "AvatarAssistant.TLabel",
            background=DESIGN["avatar_bg"],
            foreground=DESIGN["on_dark"],
            font=_font("avatar"),
            anchor="center",
            width=3,
            padding=(0, 2),
        )
        self.style.configure(
            "AvatarUser.TLabel",
            background=DESIGN["user_avatar_bg"],
            foreground=DESIGN["primary"],
            font=_font("avatar"),
            anchor="center",
            width=3,
            padding=(0, 2),
        )
        # Inline role label that sits next to the avatar on the same
        # line ("LitTrace" / "你"). Bold so the avatar row reads as a
        # single header rather than two unrelated pieces.
        self.style.configure(
            "RoleLabel.TLabel",
            background=DESIGN["canvas"],
            foreground=DESIGN["ink"],
            font=_font("body_strong"),
        )
        # Small grey timestamp label — used inside the sidebar session
        # list ("2h ago" / "Yesterday") and inside the chat header.
        self.style.configure(
            "TimestampLabel.TLabel",
            background=DESIGN["canvas"],
            foreground=DESIGN["ink_subtle"],
            font=_font("timestamp"),
        )

        # Tool card frame — inline collapsible card anchored in the chat
        # timeline. Hairline border + warm-white surface_2 background;
        # the chevron toggles the body in/out via ``_toggle_tool_card_body``.
        self.style.configure(
            "ToolCard.TFrame",
            background=DESIGN["tool_card_bg"],
            borderwidth=1,
            relief="solid",
            bordercolor=DESIGN["tool_card_border"],
        )
        self.style.configure(
            "ToolCardHeader.TFrame",
            background=DESIGN["tool_card_bg"],
        )
        self.style.configure(
            "ToolCard.TLabel",
            background=DESIGN["tool_card_bg"],
            foreground=DESIGN["ink"],
            font=_font("tool_name"),
        )
        self.style.configure(
            "ToolCardMeta.TLabel",
            background=DESIGN["tool_card_bg"],
            foreground=DESIGN["ink_muted"],
            font=_font("tool_meta"),
        )
        self.style.configure(
            "ToolCardStatus.TLabel",
            background=DESIGN["tool_card_bg"],
            foreground=DESIGN["primary"],
            font=_font("tool_meta"),
        )
        self.style.configure(
            "ToolCardStatusFailed.TLabel",
            background=DESIGN["tool_card_bg"],
            foreground=DESIGN["accent_coral"],
            font=_font("tool_meta"),
        )
        self.style.configure(
            "ToolCard.TButton",  # chevron toggle
            background=DESIGN["tool_card_bg"],
            foreground=DESIGN["ink_muted"],
            borderwidth=0,
            focusthickness=0,
            padding=(4, 0),
            font=_font("tool_meta"),
        )

        # Inline pill — anchored at the end of the latest assistant
        # message during streaming. Teal-filled so it reads as the
        # primary control in the chat timeline; ``focusthickness=0``
        # strips the dotted focus ring (Codex style).
        self.style.configure(
            "Pill.TButton",
            background=DESIGN["pill_bg"],
            foreground=DESIGN["pill_fg"],
            borderwidth=0,
            focusthickness=0,
            padding=(14, 6),
            font=_font("button"),
        )
        self.style.map(
            "Pill.TButton",
            background=[("active", DESIGN["primary_hover"])],
            foreground=[("disabled", DESIGN["body_muted"])],
        )

        # Send icon — small square button at the right edge of the
        # input column. The literal glyph is "↑" (Codex uses an arrow
        # icon); the style reads as a square CTA, not a wide pill.
        self.style.configure(
            "Send.TButton",
            background=DESIGN["primary"],
            foreground=DESIGN["on_dark"],
            borderwidth=0,
            focusthickness=0,
            width=3,
            padding=(0, 4),
            font=_font("button_utility"),
        )
        self.style.map(
            "Send.TButton",
            background=[("active", DESIGN["primary_hover"])],
        )

        # Sidebar session row — title + relative timestamp. Active row
        # gets a soft ``surface_2`` highlight (Phase 7 wiring).
        self.style.configure(
            "SessionRow.TLabel",
            background=DESIGN["canvas"],
            foreground=DESIGN["ink"],
            font=_font("body"),
        )
        self.style.configure(
            "SessionRowMeta.TLabel",
            background=DESIGN["canvas"],
            foreground=DESIGN["ink_subtle"],
            font=_font("timestamp"),
        )
        self.style.configure(
            "NewChat.TButton",
            background=DESIGN["canvas"],
            foreground=DESIGN["ink"],
            borderwidth=1,
            bordercolor=DESIGN["hairline"],
            focusthickness=0,
            padding=(10, 6),
            font=_font("button_utility"),
        )
        self.style.map(
            "NewChat.TButton",
            background=[("active", DESIGN["surface_2"])],
        )
        self.style.configure(
            "Slim.Vertical.TScrollbar",
            gripcount=0,
            background=DESIGN["ink_subtle"],
            darkcolor=DESIGN["ink_subtle"],
            lightcolor=DESIGN["ink_subtle"],
            troughcolor=DESIGN["canvas"],
            bordercolor=DESIGN["canvas"],
            arrowcolor=DESIGN["ink_subtle"],
            relief=self.tk.FLAT,
            width=7,
            arrowsize=0,
        )
        self.style.map(
            "Slim.Vertical.TScrollbar",
            background=[("active", DESIGN["ink_muted"]), ("pressed", DESIGN["ink_muted"])],
            arrowcolor=[("active", DESIGN["ink_muted"]), ("pressed", DESIGN["ink_muted"])],
        )

    def _build_subnav(self) -> None:
        """Build the top subnav bar (LitTrace brand + 4 action buttons)."""
        subnav = self.ttk.Frame(self.root, style="Subnav.TFrame", height=52)
        subnav.grid(row=0, column=0, sticky="ew")
        subnav.grid_propagate(False)
        subnav.columnconfigure(0, weight=1)
        self.ttk.Label(subnav, text="LitTrace", style="Brand.TLabel").grid(
            row=0, column=0, padx=32, sticky="w"
        )
        self.ttk.Button(
            subnav,
            text="文献上下文",
            style="Secondary.TButton",
            command=self._open_context_popup,
        ).grid(row=0, column=1, padx=(0, 8), pady=7)
        self.ttk.Button(
            subnav,
            text="隐藏上下文" if self.context_visible else "显示上下文",
            style="Secondary.TButton",
            command=self._toggle_context,
        ).grid(row=0, column=2, padx=(0, 8), pady=7)
        self.ttk.Button(
            subnav,
            text=self._parse_strategy_button_text(),
            style="Primary.TButton",
            command=self._toggle_parse_strategy,
        ).grid(row=0, column=3, padx=(0, 8), pady=7)
        self.ttk.Button(
            subnav,
            text="使用说明",
            style="Secondary.TButton",
            command=self._open_help_popup,
        ).grid(row=0, column=4, padx=(0, 32), pady=7)
        self.context_toggle_button = subnav.grid_slaves(row=0, column=2)[0]
        self.parse_strategy_button = subnav.grid_slaves(row=0, column=3)[0]

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self._build_subnav()

        main = self.ttk.Frame(self.root, style="Canvas.TFrame", padding=(28, 24, 28, 20))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.pane = self.tk.PanedWindow(
            main,
            orient=self.tk.HORIZONTAL,
            sashwidth=8,
            bd=0,
            bg=DESIGN["parchment"],
            sashrelief=self.tk.FLAT,
        )
        self.pane.grid(row=0, column=0, sticky="nsew")

        self.trace_frame = self.ttk.Frame(self.pane, style="Tile.TFrame", padding=(16, 18, 16, 16))
        self.trace_frame.columnconfigure(0, weight=1)
        self.trace_frame.rowconfigure(1, weight=3)
        # Round 28: the session history text moved to row=4 because the
        # ``+ 新对话`` button now occupies row=3. Update the weight to
        # match so the session list still stretches.
        self.trace_frame.rowconfigure(4, weight=2)
        self.pane.add(self.trace_frame, minsize=260)

        self.ttk.Label(self.trace_frame, text="执行 Trace", style="PaneTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )
        self.trace_text = self.tk.Text(
            self.trace_frame,
            wrap=self.tk.WORD,
            padx=0,
            pady=0,
            width=32,
            bd=0,
            relief=self.tk.FLAT,
            highlightthickness=0,
            bg=DESIGN["canvas"],
            fg=DESIGN["ink"],
            font=_font("caption"),
            spacing3=8,
        )
        self.trace_text.tag_configure(
            "trace_node", foreground=DESIGN["ink"], font=_font("body_strong")
        )
        self.trace_text.tag_configure(
            "trace_body", foreground=DESIGN["ink_muted"], font=_font("caption")
        )
        self.trace_text.grid(row=1, column=0, sticky="nsew")

        self.ttk.Label(self.trace_frame, text="历史 Session", style="PaneTitle.TLabel").grid(
            row=2, column=0, sticky="w", pady=(22, 10)
        )
        # Round 28: ``+ 新对话`` button above the session list. Ttk-styled
        # so it matches the rest of the sidebar.
        self.new_chat_btn = self.ttk.Button(
            self.trace_frame,
            text="+ 新对话",
            style="NewChat.TButton",
            command=self._new_chat,
        )
        self.new_chat_btn.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.session_history_text = self.tk.Text(
            self.trace_frame,
            wrap=self.tk.WORD,
            padx=0,
            pady=0,
            height=8,
            bd=0,
            relief=self.tk.FLAT,
            highlightthickness=0,
            bg=DESIGN["canvas"],
            fg=DESIGN["ink_muted"],
            font=_font("caption"),
            spacing3=6,
        )
        self.session_history_text.tag_configure("session_current", background=DESIGN["surface_2"])
        # Round 28: bumped to row=4 because the NewChat button now
        # occupies row=3.
        self.session_history_text.grid(row=4, column=0, sticky="nsew")

        chat_frame = self.ttk.Frame(self.pane, style="Tile.TFrame", padding=(20, 18, 20, 16))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        self.pane.add(chat_frame, minsize=560, stretch="always")

        self.chat_text = self.tk.Text(
            chat_frame,
            wrap=self.tk.WORD,
            padx=0,
            pady=0,
            bd=0,
            relief=self.tk.FLAT,
            highlightthickness=1,
            highlightbackground=DESIGN["hairline"],
            highlightcolor=DESIGN["hairline"],
            bg=DESIGN["canvas"],
            fg=DESIGN["ink"],
            insertbackground=DESIGN["primary"],
            font=_font("body"),
            spacing1=2,
            spacing3=12,
            state="disabled",
        )
        self.chat_text.tag_configure("role", foreground=DESIGN["ink"], font=_font("body_strong"))
        # Round 28: flat-message geometry. The avatar + role label sits
        # in the first 56 columns (a 24px avatar square + 6px pad + the
        # role text); the message body fills the remaining width up to
        # a 24px right margin. No more right-aligned user bubble or
        # 14px-vs-170px chat bubble — both messages read as full-width
        # text below their avatar, matching Codex's pattern.
        self.chat_text.tag_configure(
            "bubble_user",
            foreground=DESIGN["ink"],
            background=DESIGN["canvas"],
            font=_font("body"),
            justify=self.tk.LEFT,
            lmargin1=56,
            lmargin2=56,
            rmargin=24,
            spacing1=4,
            spacing3=12,
        )
        self.chat_text.tag_configure(
            "bubble_assistant",
            foreground=DESIGN["ink"],
            background=DESIGN["canvas"],
            font=_font("body"),
            justify=self.tk.LEFT,
            lmargin1=56,
            lmargin2=56,
            rmargin=24,
            spacing1=4,
            spacing3=12,
        )
        self.chat_text.tag_configure(
            "bubble_system",
            # Round 27: status pill style — left-aligned inline chip
            # in the assistant column, primary text on the teal accent.
            # Round 28: legacy tag kept for backward-compat tests; the
            # new inline status indicator (``_render_status_indicator``)
            # does NOT use this tag anymore — it draws a flat avatar +
            # label row directly on the bubble_assistant geometry.
            foreground=DESIGN["primary"],
            background=DESIGN["accent_teal_subtle"],
            font=_font("fine"),
            justify=self.tk.LEFT,
            lmargin1=14,
            lmargin2=14,
            rmargin=14,
            spacing1=8,
            spacing3=8,
        )
        # ``status_pulse`` overlays the trailing dots of an animatable
        # status bubble so they pop in the accent-coral — distinguishes
        # the live ticker from the static label.
        self.chat_text.tag_configure(
            "status_pulse",
            foreground=DESIGN["accent_coral"],
            font=_font("body_strong"),
        )
        self.chat_text.grid(row=0, column=0, sticky="nsew")
        chat_scroll = self.ttk.Scrollbar(
            chat_frame,
            orient=self.tk.VERTICAL,
            command=self.chat_text.yview,
            style="Slim.Vertical.TScrollbar",
        )
        chat_scroll.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        self.chat_text.configure(yscrollcommand=chat_scroll.set)

        # Round 28 (Phase 5): hairline separator above the input area.
        # Sits on its own row in chat_frame, full chat width.
        input_separator = self.ttk.Separator(
            chat_frame, orient="horizontal", style="Horizontal.TSeparator",
        )
        input_separator.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 0))

        # Wrap input in an outer host frame so the input area visually separates
        # from the chat output above (different background, hairline border).
        input_host = self.ttk.Frame(chat_frame, style="InputHost.TFrame")
        input_host.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 0))
        # Round 28 (Phase 5): 3-column grid centers the input column
        # within the chat pane. Left/right spacers (weight=1) absorb
        # slack so the input column stays centered as the window
        # resizes. The content column (weight=0) keeps the input
        # fixed-width — Codex's chat panel caps the input width to
        # keep the column visually anchored.
        input_host.columnconfigure(0, weight=1)
        input_host.columnconfigure(1, weight=0, minsize=720)
        input_host.columnconfigure(2, weight=1)
        input_frame = self.ttk.Frame(input_host, style="InputPearl.TFrame", padding=(14, 10))
        input_frame.grid(row=0, column=1, sticky="ew", padx=1, pady=(8, 12))
        input_frame.columnconfigure(0, weight=1)
        self.input_entry = self.tk.Text(
            input_frame,
            wrap=self.tk.WORD,
            height=3,
            bd=0,
            relief=self.tk.FLAT,
            highlightthickness=1,
            highlightbackground=DESIGN["hairline"],
            highlightcolor=DESIGN["primary"],
            bg=DESIGN["surface_2"],
            fg=DESIGN["ink"],
            insertbackground=DESIGN["primary"],
            font=_font("body"),
        )
        self.input_entry.grid(row=0, column=0, sticky="ew", ipady=4)
        self.input_entry.bind("<Return>", self._send_from_event)
        self.input_entry.bind("<Shift-Return>", lambda _event: None)
        self.input_entry.bind("<Key>", self._handle_input_keypress)
        self.input_entry.bind("<KeyRelease>", self._handle_input_keyrelease)
        # Round 28 (Phase 5): Send icon is the small teal square with
        # an up-arrow glyph (Codex pattern). ``Send.TButton`` is a
        # 24px-wide square (width=3 chars + padding=0,4).
        self.ttk.Button(input_frame, text="↑", style="Send.TButton", command=self._send).grid(
            row=0, column=1, padx=(10, 0)
        )

        self.context_frame = self.ttk.Frame(
            self.pane, style="Tile.TFrame", padding=(18, 18, 18, 16)
        )
        self.context_frame.columnconfigure(0, weight=1)
        self.context_frame.rowconfigure(1, weight=3)
        self.context_frame.rowconfigure(4, weight=1)
        self.pane.add(self.context_frame, minsize=340)

        self.ttk.Label(
            self.context_frame, text="当前文献上下文", style="PaneTitle.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))
        self.context_text = self.tk.Text(
            self.context_frame,
            wrap=self.tk.WORD,
            padx=0,
            pady=0,
            width=36,
            bd=0,
            relief=self.tk.FLAT,
            highlightthickness=0,
            bg=DESIGN["canvas"],
            fg=DESIGN["ink"],
            font=_font("caption"),
            spacing3=7,
        )
        self.context_text.grid(row=1, column=0, sticky="nsew")

        # 后台同步 / Daily 面板 — 把 daily_update 收编进窗口
        self.ttk.Separator(self.context_frame, orient="horizontal").grid(
            row=2, column=0, sticky="ew", pady=(14, 8)
        )
        self.ttk.Label(
            self.context_frame, text="后台同步 / Daily", style="PaneTitle.TLabel"
        ).grid(row=3, column=0, sticky="w", pady=(0, 8))
        self.rag_text = self.tk.Text(
            self.context_frame,
            wrap=self.tk.WORD,
            padx=0,
            pady=0,
            bd=0,
            relief=self.tk.FLAT,
            highlightthickness=0,
            bg=DESIGN["canvas"],
            fg=DESIGN["ink_muted"],
            font=_font("caption"),
            spacing3=4,
        )
        self.rag_text.grid(row=4, column=0, sticky="nsew")

        rag_btn_frame = self.ttk.Frame(self.context_frame, style="Tile.TFrame")
        rag_btn_frame.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        rag_btn_frame.columnconfigure(0, weight=1)
        rag_btn_frame.columnconfigure(1, weight=1)
        self.sync_session_btn = self.ttk.Button(
            rag_btn_frame,
            text="同步当前 Session",
            style="Secondary.TButton",
            command=self._trigger_session_sync,
        )
        self.sync_session_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.full_daily_btn = self.ttk.Button(
            rag_btn_frame,
            text="立即全量 Daily",
            style="Secondary.TButton",
            command=self._trigger_full_daily,
        )
        self.full_daily_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")
        self.status_var = self.tk.StringVar(value=f"Session: {self.session.session_id}")
        self.ttk.Label(
            self.root, textvariable=self.status_var, style="Status.TLabel", anchor="w"
        ).grid(row=2, column=0, sticky="ew", padx=32, pady=(0, 12))

        # Round 26: register the markdown tags on input_entry so the
        # live renderer (``_render_input_markdown``) can apply them.
        # Must run after ``input_entry`` exists — i.e. after the
        # ``input_frame`` block above. Idempotent; safe to call twice.
        self._configure_input_tags()
        # Round 27: same idea for the chat pane — bind the 10
        # ``md_*`` tags to ``chat_text`` so assistant bubbles can be
        # styled with ``_render_chat_markdown`` on finalise.
        self._configure_chat_markdown_tags()

    def _configure_copy_bindings(self) -> None:
        self.copy_menu = self.tk.Menu(self.root, tearoff=0)
        self.copy_menu.add_command(label="复制", command=self._copy_from_focused_text)
        self.copy_menu.add_command(label="全选", command=self._select_all_focused_text)
        for widget in [
            self.chat_text,
            self.trace_text,
            self.context_text,
            self.session_history_text,
        ]:
            widget.bind("<Command-c>", self._copy_event)
            widget.bind("<Control-c>", self._copy_event)
            widget.bind("<Command-a>", self._select_all_event)
            widget.bind("<Control-a>", self._select_all_event)
            widget.bind("<Button-3>", self._show_copy_menu)
            widget.bind("<Button-2>", self._show_copy_menu)
            widget.bind("<Control-Button-1>", self._show_copy_menu)
            widget.bind("<Button-1>", lambda event: event.widget.focus_set(), add="+")
            widget.bind("<Key>", self._readonly_text_key)
            widget.bind("<<Paste>>", self._break_event)
            widget.bind("<<Cut>>", self._break_event)

    def _copy_event(self, event) -> str:
        self._copy_text_selection(event.widget)
        return "break"

    def _break_event(self, _event) -> str:
        return "break"

    def _readonly_text_key(self, event) -> str | None:
        allowed = {"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"}
        if event.keysym in allowed:
            return None
        if event.state & 0x0004 and event.keysym.lower() in {"c", "a"}:
            return None
        if event.state & 0x0008 and event.keysym.lower() in {"c", "a"}:
            return None
        return "break"

    def _select_all_event(self, event) -> str:
        self._select_all_text(event.widget)
        return "break"

    def _show_copy_menu(self, event) -> str:
        event.widget.focus_set()
        self.copy_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _copy_from_focused_text(self) -> None:
        widget = self.root.focus_get()
        self._copy_text_selection(widget)

    def _select_all_focused_text(self) -> None:
        widget = self.root.focus_get()
        self._select_all_text(widget)

    def _copy_text_selection(self, widget) -> None:
        if not hasattr(widget, "get"):
            return
        try:
            text = widget.get(self.tk.SEL_FIRST, self.tk.SEL_LAST)
        except self.tk.TclError:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _select_all_text(self, widget) -> None:
        if not hasattr(widget, "tag_add"):
            return
        widget.tag_add(self.tk.SEL, "1.0", self.tk.END)
        widget.mark_set(self.tk.INSERT, "1.0")
        widget.see(self.tk.INSERT)

    def _send(self) -> None:
        message = self.input_entry.get("1.0", self.tk.END).strip()
        if not message:
            return
        if message in {"/quit", "/exit"}:
            self.input_entry.delete("1.0", self.tk.END)
            self.root.destroy()
            return
        # Round 27: delegate to ``_send_with_text`` so the 重新生成
        # button can re-fire the same logic. ``_send_with_text`` clears
        # the input, records ``_last_user_message``, and resets the
        # input box height — that ordering matters.
        self._send_with_text(message)

    def _send_with_text(self, text: str) -> None:
        """Round 27: shared send pipeline used by both the input box
        (Enter / Send button) and the 重新生成 action button.

        Records ``_last_user_message`` so 重新生成 can replay the
        original prompt, then clears and resets the input box, and
        finally spawns the worker thread that drives the Codex call.
        """
        message = text.strip()
        if not message:
            return
        self._last_user_message = message
        # Wipe the input box (only when called from the user-driven
        # path — 重新生成 is a no-op for the entry widget).
        try:
            self.input_entry.delete("1.0", self.tk.END)
            self.input_entry.configure(height=1)
        except self.tk.TclError:
            pass
        self._clear_status_in_chat()
        self._append_message("user", message)
        self._show_execution_path(message)
        threading.Thread(
            target=self._handle_message_thread, args=(message,), daemon=True,
        ).start()

    def _resize_input_to_content(self) -> None:
        """Round 27: grow ``input_entry`` to fit content up to 8 lines.

        Bounded so a runaway paste doesn't blow out the layout. The
        height resets to 1 once the user sends (``_send_with_text``
        calls ``configure(height=1)`` after deleting the buffer) or
        clears the entry manually.
        """
        try:
            raw = self.input_entry.get("1.0", self.tk.END)
        except self.tk.TclError:
            return
        if raw.strip():
            # Trailing newline means Tk's last empty line — count it as
            # one fewer so an entry like "abc\n" stays at 1 line.
            line_count = raw.count("\n") + (0 if raw.endswith("\n") else 1)
        else:
            line_count = 1
        new_height = max(1, min(line_count, 8))
        try:
            self.input_entry.configure(height=new_height)
        except self.tk.TclError:
            pass

    def _render_avatar(self, role: str) -> str:
        """Round 28: insert a flat-message avatar + role label header.

        Code pattern: an avatar square (24x24, dark-filled for the
        assistant, soft-teal for the user) followed by an inline role
        label ("LitTrace" or "你") on the same line. The avatar is
        embedded as a Tk ``window`` so it inherits the chat_text's
        vertical padding but does not consume a column of body text
        — the body text is offset by ``lmargin1=56`` on the
        ``bubble_*`` tags.

        Returns the Tk index where the body text should land
        (immediately after the newline that follows the role label).
        Caller is responsible for flipping ``chat_text`` between
        ``normal`` and ``disabled`` states around the call.
        """
        if role in {"assistant", "LitTrace"}:
            avatar_style = "AvatarAssistant.TLabel"
            initial = "L"
            role_text = "LitTrace"
        else:
            avatar_style = "AvatarUser.TLabel"
            initial = "你"
            role_text = "你"

        # Avatar square — the ttk.Frame wrapper is unused (a Toplevel
        # parent is required for window_create on chat_text). We pack
        # the label directly and let Tk draw the colored square.
        # ``window_create`` only accepts a single-pixel padx on Windows Tk,
        # so we put any inner spacing on the avatar_host Frame itself.
        avatar_host = self.tk.Frame(
            self.chat_text,
            background=DESIGN.get(
                "avatar_bg" if role in {"assistant", "LitTrace"} else "user_avatar_bg"
            ),
            padx=6,
            pady=2,
        )
        avatar_lbl = self.ttk.Label(avatar_host, text=initial, style=avatar_style)
        avatar_lbl.pack()
        self.chat_text.window_create(self.tk.END, window=avatar_host, padx=20)
        # Inline role label on the same visual line as the avatar.
        self.chat_text.insert(self.tk.END, f" {role_text}\n", ("role",))
        # Return the current INSERT index so the caller can stream the
        # message body starting from this position.
        return self.chat_text.index(self.tk.INSERT)

    def _attach_bubble_actions(
        self, role: str, text: str
    ) -> None:
        """Round 27: anchor a small action Frame under the just-inserted
        bubble.

        Assistant / LitTrace bubbles get 复制 + 重新生成; user bubbles
        get 复制 + 编辑. The frame is a Tk ``window`` child of
        ``chat_text``; we track the Frame widget + the line index of
        its embedded window slot so the session switcher can drop the
        frame and its empty line together (``Tk.window_create`` returns
        ``None`` on the Windows Tk build this app ships with, so we
        can't rely on the returned id for cleanup).

        Round 28: anchor at ``END - 1 lines`` (one line above the
        trailing blank line that separates this bubble from the next)
        so the action row lines up with the avatar column instead of
        the left edge of the chat_text.
        """
        try:
            # Capture the line on which the embedded window sits so
            # ``_render_session_messages`` can later delete the whole
            # line in one ``Text.delete`` call.
            line_index = self.chat_text.index(f"{self.tk.END} - 1 lines")
            frame = self.ttk.Frame(self.chat_text, style="Tile.TFrame")
            # Bind the local ``btn`` into the lambda so the 已复制
            # restore step can find it without a class-level mutable.
            def _make_copy(text_value: str):
                def _copy(btn) -> None:
                    self._copy_to_clipboard(text_value, btn)
                return _copy
            self.ttk.Button(
                frame,
                text="复制",
                style="Link.TButton",
                command=_make_copy(text),
            ).pack(side=self.tk.LEFT, padx=(0, 8))
            if role in {"assistant", "LitTrace"}:
                self.ttk.Button(
                    frame,
                    text="重新生成",
                    style="Link.TButton",
                    command=self._regenerate_last,
                ).pack(side=self.tk.LEFT, padx=(0, 8))
            elif role in {"user", "你"}:
                self.ttk.Button(
                    frame,
                    text="编辑",
                    style="Link.TButton",
                    command=lambda t=text: self._edit_user_bubble(t),
                ).pack(side=self.tk.LEFT)
            # Round 28: anchor one line above END so the action row
            # sits inside the message body line (aligned with the
            # avatar column), not on the trailing blank line. Tk's
            # ``window_create`` on Windows only accepts a single-pixel
            # padx — the avatar's lmargin1=56 already indents the body
            # text, so a 4px breathing room on the right is plenty.
            self.chat_text.window_create(
                f"{self.tk.END} - 1 lines", window=frame, padx=4,
            )
            self._bubble_records.append({
                "role": role,
                "text": text,
                "frame": frame,
                "line_index": line_index,
            })
        except self.tk.TclError:
            # Headless fallback — silently skip the action row.
            pass

    # ----- Round 28 (Phase 6): tool card subsystem -----

    def _render_tool_card(
        self,
        tool_name: str,
        status: str,
        duration: float | None,
        body: str,
    ) -> None:
        """Append an inline collapsible tool card to the chat timeline.

        The card sits below the assistant avatar header so the user
        can see which tool the model invoked during the turn. The
        header row shows the tool icon + name + duration + status
        pill + chevron; clicking the chevron toggles the body
        (collapsed by default — failed cards auto-expand so the
        user sees the error context without an extra click).

        Tk ``window_create`` does not support rounded corners or
        inline-floating widgets on Windows, so the card lives on its
        own line — visually compact, but always one block per tool.
        """
        try:
            self.chat_text.configure(state="normal")
            # Leading newline so the card sits on its own line below
            # any in-flight streamed text.
            self.chat_text.insert(self.tk.END, "\n")
            line_index = self.chat_text.index(self.tk.END)

            # Outer Frame with the hairline-bordered, surface_2 fill
            # look — Codex pattern. We use ``pack`` for the card
            # layout (instead of grid) so the body text can be
            # toggled with ``pack_forget``/``pack`` without mixing
            # geometry managers.
            card = self.ttk.Frame(
                self.chat_text, style="ToolCard.TFrame", padding=(10, 8),
            )

            # Header row: icon glyph + tool name + duration meta + status pill.
            header = self.ttk.Frame(card, style="ToolCardHeader.TFrame")
            header.pack(side=self.tk.TOP, fill=self.tk.X)

            icon_lbl = self.ttk.Label(
                header, text="⚙", style="ToolCard.TLabel",
            )
            icon_lbl.pack(side=self.tk.LEFT, padx=(0, 8))

            name_lbl = self.ttk.Label(
                header, text=tool_name, style="ToolCard.TLabel",
            )
            name_lbl.pack(side=self.tk.LEFT)

            duration_text = (
                f"{duration:.2f}s" if duration is not None else "—"
            )
            duration_lbl = self.ttk.Label(
                header, text=duration_text, style="ToolCardMeta.TLabel",
            )
            duration_lbl.pack(side=self.tk.RIGHT, padx=(8, 8))

            status_text = {
                "running": "运行中",
                "success": "成功",
                "failed": "失败",
            }.get(status, status)
            status_style = (
                "ToolCardStatusFailed.TLabel"
                if status == "failed"
                else "ToolCardStatus.TLabel"
            )
            status_lbl = self.ttk.Label(
                header, text=status_text, style=status_style,
            )
            status_lbl.pack(side=self.tk.RIGHT, padx=(0, 0))

            # Chevron toggle — collapsed by default; failed cards
            # auto-expand so the user sees the error.
            chevron = self.ttk.Button(
                header, text="▸", style="ToolCard.TButton",
                width=2,
            )
            chevron.pack(side=self.tk.RIGHT, padx=(0, 4))

            # Body: a tk.Text widget with the JSON payload. Wrapped
            # word-wise; ``pack_forget``/``pack`` toggles visibility.
            body_text = self.tk.Text(
                card,
                wrap=self.tk.WORD,
                height=4,
                bd=0,
                relief=self.tk.FLAT,
                background=DESIGN["surface_1"],
                foreground=DESIGN["ink_muted"],
                font=_font("mono"),
                highlightthickness=0,
                padx=6,
                pady=4,
            )
            body_text.insert(self.tk.END, body or "(无返回内容)")

            def toggle() -> None:
                if body_text.winfo_ismapped():
                    body_text.pack_forget()
                    chevron.configure(text="▸")
                else:
                    body_text.pack(
                        fill=self.tk.X, pady=(8, 0), padx=(24, 0),
                        side=self.tk.TOP,
                    )
                    chevron.configure(text="▾")

            chevron.configure(command=toggle)

            # Embed the card into chat_text BEFORE packing the
            # body so the card has a realized window when pack
            # runs (Tk's pack manager requires a realized parent).
            self.chat_text.window_create(
                self.tk.END, window=card, padx=56,
            )

            if status == "failed":
                # Auto-expand on failure so the user sees the error
                # without an extra click.
                body_text.pack(
                    fill=self.tk.X, pady=(8, 0), padx=(24, 0),
                    side=self.tk.TOP,
                )
                chevron.configure(text="▾")

            # Trailing newline so the next message doesn't run into
            # the card.
            self.chat_text.insert(self.tk.END, "\n")

            self._tool_card_records.append({
                "name": tool_name,
                "status": status,
                "frame": card,
                "line_index": line_index,
                "body_text": body_text,
            })
        except self.tk.TclError:
            # Headless fallback — skip silently.
            pass
        finally:
            try:
                self.chat_text.configure(state="disabled")
            except self.tk.TclError:
                pass
            self.chat_text.see(self.tk.END)

    def _copy_to_clipboard(self, text: str, btn=None) -> None:
        """Round 27: copy ``text`` to the clipboard; if ``btn`` is given,
        swap its label to ``已复制`` for 1.5s so the user has feedback.
        """
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            # Force the clipboard contents to persist after the GUI
            # closes — otherwise Tk's clipboard is wiped on exit on
            # some platforms.
            self.root.update_idletasks()
        except self.tk.TclError:
            return
        if btn is None:
            return
        try:
            original = btn.cget("text")
        except self.tk.TclError:
            return
        try:
            btn.configure(text="已复制")
        except self.tk.TclError:
            return
        def _restore() -> None:
            try:
                btn.configure(text=original)
            except self.tk.TclError:
                pass
        try:
            self.root.after(1500, _restore)
        except self.tk.TclError:
            pass

    def _regenerate_last(self) -> None:
        """Round 27: re-send ``_last_user_message``. Guarded against
        triggering while a turn is already running — the user would
        otherwise queue two concurrent Codex calls on the same session.
        """
        last = getattr(self, "_last_user_message", None)
        if not last:
            return
        stop_event = getattr(self, "_streaming_stop_event", None)
        if stop_event is not None and not stop_event.is_set():
            # A turn is mid-flight; ignore the click — user can wait.
            return
        self._send_with_text(last)

    def _edit_user_bubble(self, text: str) -> None:
        """Round 27: copy a previous user message back into the input
        box so the user can amend and resend.

        We do NOT delete the original bubble from the chat — the
        existing copy stays as context (Linear / Notion AI behave
        the same way). The user can manually delete via the chat
        log if they want a clean rewrite.
        """
        try:
            self.input_entry.delete("1.0", self.tk.END)
            self.input_entry.insert("1.0", text)
            self.input_entry.focus_set()
            self.input_entry.mark_set(self.tk.INSERT, self.tk.END)
            self._resize_input_to_content()
        except self.tk.TclError:
            pass

    def _handle_message_thread(self, message: str) -> None:
        # Round 23: stream deltas + phase events into the Tk main thread.
        # The Codex App Server reader loop invokes these callbacks on a
        # worker thread, so we MUST hop to the Tk thread via
        # ``root.after(0, ...)`` before touching any widget. We throttle
        # the status-bar updates to ~5 Hz so the user sees live counters
        # without flooding Tk with redraws.
        # Round 27: arm the Stop button. The ``threading.Event`` is the
        # bridge between the Tk click (GUI thread) and the asyncio
        # delta loop (worker thread). Reset at the start of every turn
        # so a stale flag from a previous turn cannot bleed in.
        self._streaming_stop_event = threading.Event()
        stream_state = {
            "first_delta_seen": False,
            "last_status_update": 0.0,
            "status_every": 0.2,  # seconds
            "streaming": False,
        }

        def on_phase(phase: str) -> None:
            label = self._PHASE_LABELS.get(phase)
            if not label:
                return
            # ``on_phase`` runs on the Codex reader thread; hop to Tk.
            # Round 25: status lives inside the chat area now, so
            # update the chat status bubble instead of the bottom
            # status bar.
            self.root.after(0, lambda: self._set_status_in_chat(label))

        def on_delta(delta: str) -> None:
            if not delta:
                return
            now = time.perf_counter()
            # Tk thread: append text immediately (cheap), update the
            # chat status bubble throttled. Tk redraws only fire when
            # the event loop runs, so a flurry of after() calls in the
            # same frame collapses into a single paint.
            def render():
                if not stream_state["streaming"]:
                    stream_state["streaming"] = True
                    self._begin_streaming_bubble()
                self._append_streaming_delta(delta)
                if now - stream_state["last_status_update"] >= stream_state["status_every"]:
                    stream_state["last_status_update"] = now
                    # Round 25: progress counter lives in the chat
                    # status bubble now.
                    self._set_status_in_chat(
                        f"Codex 流式输出中… {self._streaming_chars} chars"
                    )
            self.root.after(0, render)

        # Round 28 (Phase 6): tool events emitted by the Codex App
        # Server during the turn. The callback fires on the reader
        # thread; hop to Tk before touching widgets. The tool card
        # is appended inline in the chat timeline after the assistant
        # avatar header.
        def on_tool(name: str, status: str, duration: float | None, body: str) -> None:
            self.root.after(
                0,
                lambda: self._render_tool_card(name, status, duration, body),
            )

        try:
            response, workspace = asyncio.run(
                handle_agent_chat(
                    ChatRequest(message=message, session_id=self.session.session_id),
                    self.workspace,
                    self.config,
                    session=self.session,
                    elicitation_handler=self._make_window_elicitation_handler(),
                    on_delta=on_delta,
                    on_phase=on_phase,
                    on_tool=on_tool,
                    stop_event=self._streaming_stop_event,
                )
            )
        except Exception as exc:
            from littrace.codex_runtime.errors import AppServerError as _ASE

            def handle_error() -> None:
                # Drop the placeholder streaming bubble so the user
                # doesn't see a stranded "—" line on failure.
                self._finalize_streaming_bubble(None)
                # Round 25: failure clears the chat status bubble
                # instead of leaving a stale "Codex 思考中…" line.
                self._clear_status_in_chat()
                if isinstance(exc, _ASE):
                    # Round 20: Codex failures get a modal with remediation
                    # steps, not just a system-message bubble.
                    self._show_app_server_error_modal(exc)
                    return
                error_text = f"{exc.__class__.__name__}: {exc}"
                self._append_message("system", error_text)
                self.status_var.set("错误")

            self.root.after(0, handle_error)
            return

        def apply_response() -> None:
            self.workspace = workspace
            self.context_visible = workspace.context.visible_to_user
            save_workspace(self.session, self.workspace, config=self.config)
            append_message(self.session, "user", message)
            append_message(self.session, "assistant", response)
            # The streaming bubble already showed the live reply text;
            # replace its body with the canonical final text so the
            # session log matches the message we just persisted.
            if stream_state["streaming"]:
                self._finalize_streaming_bubble(response.reply or None)
            elif _is_user_effective_reply(response.reply):
                # Fast-path / early-response: no deltas were streamed,
                # so just append a normal bubble.
                self._append_message("assistant", response.reply)
            # Round 27: surface the truncated flag with a small system
            # annotation so the user knows the reply is partial.
            if getattr(response, "truncated", False):
                self._append_message(
                    "system", "（已被用户中止,部分内容未生成）"
                )
            # Round 25: replace the chat status bubble with the final
            # completion label so the user can see "Codex 回复完成"
            # without scanning the bottom status bar. Quiet replies
            # (e.g. "已切换解析模式") clear the bubble instead.
            if _is_user_effective_reply(response.reply):
                self._set_status_in_chat(
                    f"Codex 回复完成 · {response.action}"
                )
            else:
                self._clear_status_in_chat()
            if response.warnings:
                self._append_message("system", "；".join(response.warnings[:4]))
            self.last_download_plan = response.download_plan
            if response.research_result and response.research_result.workflow_trace:
                self._render_workflow_trace(response.research_result.workflow_trace)
            self.status_var.set(f"Action: {response.action} | Session: {self.session.session_id}")
            self._refresh_context()
            self._refresh_ocr_buttons()
            self._refresh_context_popup()
            self._refresh_session_history()
            self._refresh_rag_panel()

        self.root.after(0, apply_response)
    def _send_from_event(self, event) -> str:
        if event.state & 0x0001:
            return None
        self._send()
        return "break"

    # --- Slash-command autocomplete popup -------------------------------------

    def _handle_input_keypress(self, event) -> str | None:
        """Intercept Up/Down/Tab/Escape while the popup is visible."""
        if self.command_popup is None or not self.command_popup.winfo_exists():
            return None
        keysym = event.keysym
        if keysym == "Down":
            self._navigate_command_popup(+1)
            return "break"
        if keysym == "Up":
            self._navigate_command_popup(-1)
            return "break"
        if keysym == "Tab":
            self._commit_command_popup()
            return "break"
        if keysym == "Escape":
            self._hide_command_popup()
            return "break"
        if keysym in {"space", "Return"}:
            self._hide_command_popup()
        return None

    def _handle_input_keyrelease(self, event) -> None:
        """Re-evaluate popup visibility after each keystroke."""
        if event.keysym in {"Up", "Down", "Tab", "Escape"}:
            return
        self._refresh_command_popup()
        # Round 26: schedule a debounced markdown re-render so the user
        # sees headings / bold / inline code styled live while typing.
        # The popup refresh runs first (cheap, synchronous); the markdown
        # render coalesces IME bursts via ``_input_md_after_id``.
        self._schedule_input_markdown_render()
        # Round 27: grow the input box up to 8 lines so long prompts
        # stay readable instead of being scrolled off the bottom.
        self._resize_input_to_content()

    def _configure_input_tags(self) -> None:
        """Round 26: register markdown tags on input_entry for live
        re-rendering. Idempotent — safe to call after each rebuild of
        the widget tree (e.g. inside tests that re-init the stub).
        """
        widget = self.input_entry
        widget.tag_configure(
            "md_h1",
            font=_font("h1"),
            foreground=DESIGN["ink"],
            spacing1=8, spacing3=4,
        )
        widget.tag_configure(
            "md_h2",
            font=_font("h2"),
            foreground=DESIGN["ink"],
            spacing1=6, spacing3=2,
        )
        widget.tag_configure(
            "md_h3",
            font=_font("h3"),
            foreground=DESIGN["ink_muted"],
            spacing1=4, spacing3=2,
        )
        widget.tag_configure("md_bold", font=_font("body_strong"))
        # Italic needs an actual ``tk.font.Font`` because Tk's tuple
        # font spec does not honor ``slant``. Build one explicitly so
        # the italic glyphs lean as expected.
        try:
            from tkinter import font as _tkfont
            _italic_font = _tkfont.Font(
                family="Microsoft YaHei UI, Microsoft YaHei, Arial",
                size=13,
                slant="italic",
            )
        except Exception:
            # Headless fallback — degrade to the body font rather than
            # raise. Tests that don't load Tk still pass.
            _italic_font = _font("body")
        widget.tag_configure("md_italic", font=_italic_font)
        # Inline code — pill style on the teal accent background.
        widget.tag_configure(
            "md_code",
            font=_font("mono"),
            foreground=DESIGN["primary"],
            background=DESIGN["accent_teal_subtle"],
            spacing1=2, spacing3=2,
        )
        # Code block — monospace on the lighter surface, indented.
        widget.tag_configure(
            "md_codeblock",
            font=_font("mono"),
            background=DESIGN["surface_1"],
            foreground=DESIGN["ink"],
            lmargin1=8, lmargin2=8,
            spacing1=4, spacing3=4,
        )
        # Bullet lines: indent + tint the marker character. The renderer
        # tags the ``- `` / ``* `` prefix itself so only the bullet
        # glyph picks up the primary color.
        widget.tag_configure(
            "md_list_bullet",
            foreground=DESIGN["primary"],
            lmargin1=16, lmargin2=16,
        )
        # Blockquote — muted, deeply indented.
        widget.tag_configure(
            "md_quote",
            foreground=DESIGN["ink_muted"],
            lmargin1=24, lmargin2=24,
        )
        # Link — primary color + underline. Only the link text is tagged
        # (group 1 of the link regex), not the URL.
        widget.tag_configure(
            "md_link",
            foreground=DESIGN["primary"],
            underline=1,
        )

    def _configure_chat_markdown_tags(self) -> None:
        """Round 27: register the same 10 markdown tags on ``chat_text``
        so assistant output can be styled in place.

        Mirrors ``_configure_input_tags`` one-for-one. Idempotent — safe
        to call from test stubs the same way the input version is.
        """
        w = self.chat_text
        w.tag_configure(
            "md_h1",
            font=_font("h1"),
            foreground=DESIGN["ink"],
            spacing1=8, spacing3=4,
        )
        w.tag_configure(
            "md_h2",
            font=_font("h2"),
            foreground=DESIGN["ink"],
            spacing1=6, spacing3=2,
        )
        w.tag_configure(
            "md_h3",
            font=_font("h3"),
            foreground=DESIGN["ink_muted"],
            spacing1=4, spacing3=2,
        )
        w.tag_configure("md_bold", font=_font("body_strong"))
        try:
            from tkinter import font as _tkfont
            _italic_font = _tkfont.Font(
                family="Microsoft YaHei UI, Microsoft YaHei, Arial",
                size=13,
                slant="italic",
            )
        except Exception:
            _italic_font = _font("body")
        w.tag_configure("md_italic", font=_italic_font)
        w.tag_configure(
            "md_code",
            font=_font("mono"),
            foreground=DESIGN["primary"],
            background=DESIGN["accent_teal_subtle"],
            spacing1=2, spacing3=2,
        )
        w.tag_configure(
            "md_codeblock",
            font=_font("mono"),
            background=DESIGN["surface_1"],
            foreground=DESIGN["ink"],
            lmargin1=8, lmargin2=8, spacing1=4, spacing3=4,
        )
        w.tag_configure(
            "md_list_bullet",
            foreground=DESIGN["primary"],
            lmargin1=16, lmargin2=16,
        )
        w.tag_configure(
            "md_quote",
            foreground=DESIGN["ink_muted"],
            lmargin1=24, lmargin2=24,
        )
        w.tag_configure(
            "md_link",
            foreground=DESIGN["primary"],
            underline=1,
        )

    def _tk_index_to_char_offset(self, w, index_str: str) -> int:
        """Convert a Tk ``line.col`` index to its absolute char offset
        in widget ``w``.

        Round 27 helper. Tk's ``index`` accepts ``"1.0 + N chars"``
        but returns a ``line.col`` rather than the offset, so we
        reconstruct it by summing the per-line lengths above the
        target line. The +1 per line accounts for the implicit
        trailing ``\\n`` Tk uses to separate lines.
        """
        try:
            resolved = w.index(index_str)
        except self.tk.TclError:
            return 0
        line_str, _, col_str = resolved.partition(".")
        line = int(line_str)
        col = int(col_str)
        total = 0
        for prior in range(1, line):
            line_end = w.index(f"{prior}.end")
            _, _, lc = line_end.partition(".")
            total += int(lc) + 1   # +1 for the trailing newline
        return total + col

    def _render_chat_markdown(self, start_index_str: str, text: str) -> None:
        """Round 27: apply ``md_*`` tags to ``chat_text`` over the
        absolute character range beginning at ``start_index_str`` and
        spanning ``len(text)`` characters.

        Mirrors ``_render_input_markdown`` — strips every tag first so
        re-renders are deterministic, then walks ``_MD_PATTERNS`` and
        applies tags per match. Buffer text is NEVER mutated, so the
        model output stays verbatim.

        The streaming bubble always calls this once after finalisation;
        the per-turn ``_append_message`` path calls it for assistant
        messages before attaching the action buttons.
        """
        if not text:
            return
        w = self.chat_text
        start_offset = self._tk_index_to_char_offset(w, start_index_str)
        end_offset = start_offset + len(text)
        try:
            for tag in _CHAT_MD_TAGS:
                w.tag_remove(
                    tag,
                    f"1.0 + {start_offset} chars",
                    f"1.0 + {end_offset} chars",
                )
        except self.tk.TclError:
            return

        # line_starts maps each text line to its absolute char offset
        # within ``text`` (NOT within the widget). We use it to convert
        # match.start()/end() to Tk char offsets relative to the start.
        line_starts: list[int] = [0]
        for i, ch in enumerate(text):
            if ch == "\n" and i + 1 < len(text):
                line_starts.append(i + 1)

        def to_index(off: int) -> str:
            lo, hi = 0, len(line_starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if line_starts[mid] <= off:
                    lo = mid
                else:
                    hi = mid - 1
            return f"1.0 + {start_offset + off} chars"

        for kind, pat in _MD_PATTERNS:
            for m in pat.finditer(text):
                target_tag = _KIND_TO_TAG.get(kind)
                if kind == "md_heading":
                    target_tag = f"md_h{len(m.group(1))}"
                if target_tag is None:
                    continue
                try:
                    # Kinds that capture inner content in group(1): the
                    # tag should cover just the inner token, NOT the
                    # surrounding markdown markers — otherwise the user
                    # sees raw ``**`` / `` ` `` / ``[]`` text rendered
                    # in the bold/code/link style, which is jarring.
                    if kind in {"md_code", "md_bold", "md_italic", "md_link"}:
                        seg_start, seg_end = m.start(1), m.end(1)
                    else:
                        seg_start, seg_end = m.start(), m.end()
                    w.tag_add(
                        target_tag,
                        to_index(seg_start),
                        to_index(seg_end),
                    )
                except self.tk.TclError:
                    continue

    def _render_input_markdown(self) -> None:
        """Round 26: re-apply markdown tags over ``input_entry``.

        Strips every tag in ``_INPUT_MD_TAGS`` first so re-renders are
        deterministic, then walks ``_MD_PATTERNS`` and applies tags for
        every match.

        Does NOT mutate the buffer text — markdown markers (``#``,
        ``**``, ``\\` ``, etc.) stay in the widget so the user sees
        them while editing and the model still receives them verbatim
        via ``_send()``.
        """
        # Round 26 defensive guard: the scheduled render can fire after
        # the user closes the window — Tk delivers any pending ``after``
        # callbacks before tearing the widget tree down, but a stale id
        # can sneak in if the user types and immediately hits ``/quit``.
        # Swallow ``TclError`` so we never bring the GUI down with an
        # unhandled exception from a destroyed widget.
        try:
            widget = self.input_entry
            raw = widget.get("1.0", self.tk.END)
        except self.tk.TclError:
            return
        # Tk always appends a trailing newline; strip it for offset math.
        text = raw[:-1] if raw.endswith("\n") else raw

        try:
            for tag in _INPUT_MD_TAGS:
                widget.tag_remove(tag, "1.0", self.tk.END)
        except self.tk.TclError:
            return

        if not text:
            return

        # Build line-start offsets once so we can convert a buffer
        # char offset to a Tk ``line.col`` index without calling
        # ``index`` per match (which is slow across hundreds of
        # matches on a long buffer).
        line_starts: list[int] = [0]
        for i, ch in enumerate(text):
            if ch == "\n" and i + 1 < len(text):
                line_starts.append(i + 1)

        def to_index(offset: int) -> str:
            # Binary search the line whose start <= offset < next start.
            lo, hi = 0, len(line_starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if line_starts[mid] <= offset:
                    lo = mid
                else:
                    hi = mid - 1
            return f"{lo + 1}.{offset - line_starts[lo]}"

        for kind, pat in _MD_PATTERNS:
            for m in pat.finditer(text):
                try:
                    if kind == "md_code":
                        widget.tag_add(
                            "md_code",
                            to_index(m.start(1)),
                            to_index(m.end(1)),
                        )
                    elif kind == "md_codeblock":
                        widget.tag_add(
                            "md_codeblock",
                            to_index(m.start()),
                            to_index(m.end()),
                        )
                    elif kind == "md_heading":
                        level = len(m.group(1))  # 1..3
                        widget.tag_add(
                            f"md_h{level}",
                            to_index(m.start()),
                            to_index(m.end()),
                        )
                    elif kind == "md_list":
                        widget.tag_add(
                            "md_list_bullet",
                            to_index(m.start()),
                            to_index(m.end()),
                        )
                    elif kind == "md_quote":
                        widget.tag_add(
                            "md_quote",
                            to_index(m.start()),
                            to_index(m.end()),
                        )
                    elif kind == "md_bold":
                        widget.tag_add(
                            "md_bold",
                            to_index(m.start(1)),
                            to_index(m.end(1)),
                        )
                    elif kind == "md_italic":
                        widget.tag_add(
                            "md_italic",
                            to_index(m.start(1)),
                            to_index(m.end(1)),
                        )
                    elif kind == "md_link":
                        widget.tag_add(
                            "md_link",
                            to_index(m.start(1)),
                            to_index(m.end(1)),
                        )
                except self.tk.TclError:
                    # Empty / invalid range — harmless, skip.
                    continue

    def _schedule_input_markdown_render(self) -> None:
        """Round 26: debounce markdown re-render to ~150ms after the
        last keystroke. Coalesces IME composition bursts (where every
        keystroke fires ``<KeyRelease>``) into one render pass.
        """
        # Defensive: if the window is being torn down, ``self.root``
        # can fail to schedule a new ``after`` callback. Swallow it so
        # shutdown is never derailed by a stale KeyRelease event.
        try:
            if getattr(self, "_input_md_after_id", None) is not None:
                try:
                    self.root.after_cancel(self._input_md_after_id)
                except self.tk.TclError:
                    pass
            self._input_md_after_id = self.root.after(
                150, self._run_input_markdown_render,
            )
        except self.tk.TclError:
            self._input_md_after_id = None

    def _run_input_markdown_render(self) -> None:
        """Round 26: callback fired by ``after`` after the debounce
        window. Clears the pending id before re-rendering so the next
        keystroke can schedule a fresh tick without colliding.
        """
        self._input_md_after_id = None
        self._render_input_markdown()

    def _refresh_command_popup(self) -> None:
        text = self.input_entry.get("1.0", self.tk.END).rstrip("\n")
        slash_pos = self._find_active_slash(text)
        if slash_pos < 0:
            self._hide_command_popup()
            return
        query = text[slash_pos + 1 :]
        if not query or any(ch.isspace() for ch in query):
            # Multi-word or trailing space: no popup; let the user finish typing.
            self._hide_command_popup()
            return
        self._show_command_popup(query)

    def _find_active_slash(self, text: str) -> int:
        """Return the index of the most-recent `/` preceded by start-of-buffer /
        whitespace / newline, or -1 if none."""
        pos = -1
        for index, ch in enumerate(text):
            if ch != "/":
                continue
            if index == 0 or text[index - 1] in " \n\t":
                pos = index
        return pos

    def _show_command_popup(self, query: str) -> None:
        q = query.lower()
        matches = [
            (name, desc)
            for name, desc, _ctx in COMMAND_CATALOG
            if name.lower().startswith(q)
        ]
        if not matches:
            self._hide_command_popup()
            return
        if self.command_popup is None or not self.command_popup.winfo_exists():
            self._build_command_popup()
        listbox = self.command_popup_list
        listbox.delete(0, self.tk.END)
        for name, desc in matches:
            listbox.insert(self.tk.END, f"/{name}    {desc}")
        self.command_popup_index = 0
        self.command_popup_query = query
        listbox.selection_clear(0, self.tk.END)
        listbox.selection_set(0)
        listbox.activate(0)
        listbox.see(0)
        self._position_command_popup()
        self.command_popup.deiconify()
        self.command_popup.lift()

    def _build_command_popup(self) -> None:
        popup = self.tk.Toplevel(self.root)
        popup.overrideredirect(True)
        popup.configure(background=DESIGN["hairline"])
        frame = self.ttk.Frame(popup, style="Tile.TFrame", padding=2)
        frame.pack(fill=self.tk.BOTH, expand=True)
        listbox = self.tk.Listbox(
            frame,
            activestyle="none",
            highlightthickness=0,
            bd=0,
            bg=DESIGN["canvas"],
            fg=DESIGN["ink"],
            selectbackground=DESIGN["accent_teal_subtle"],
            selectforeground=DESIGN["ink"],
            font=_font("caption"),
            height=8,
            exportselection=False,
        )
        listbox.pack(fill=self.tk.BOTH, expand=True)
        listbox.bind("<ButtonRelease-1>", lambda _e: self._commit_command_popup())
        self.command_popup = popup
        self.command_popup_list = listbox

    def _position_command_popup(self) -> None:
        if self.command_popup is None:
            return
        x = self.input_entry.winfo_rootx()
        y = self.input_entry.winfo_rooty() - self.command_popup.winfo_reqheight() - 4
        self.command_popup.geometry(f"+{x}+{max(y, 0)}")

    def _navigate_command_popup(self, delta: int) -> None:
        if self.command_popup_list is None:
            return
        size = self.command_popup_list.size()
        if size == 0:
            return
        self.command_popup_index = (self.command_popup_index + delta) % size
        self.command_popup_list.selection_clear(0, self.tk.END)
        self.command_popup_list.selection_set(self.command_popup_index)
        self.command_popup_list.activate(self.command_popup_index)
        self.command_popup_list.see(self.command_popup_index)

    def _commit_command_popup(self) -> None:
        if self.command_popup_list is None:
            return
        sel = self.command_popup_list.curselection()
        if not sel:
            return
        line = self.command_popup_list.get(sel[0])
        command = line.split()[0]  # `/name`
        text = self.input_entry.get("1.0", self.tk.END).rstrip("\n")
        slash_pos = self._find_active_slash(text)
        if slash_pos < 0:
            self._hide_command_popup()
            return
        # Replace the entire `/<query>` span with the selected `/<command>`.
        end = slash_pos + 1 + len(self.command_popup_query)
        self.input_entry.delete(f"1.0 +{slash_pos} chars", f"1.0 +{end} chars")
        self.input_entry.insert(f"1.0 +{slash_pos} chars", command)
        self._hide_command_popup()

    def _hide_command_popup(self) -> None:
        if self.command_popup is not None and self.command_popup.winfo_exists():
            self.command_popup.withdraw()

    def _append_message(self, role: str, text: str) -> None:
        tag = _chat_bubble_tag(role)
        # chat_text is created in `state="disabled"` (read-only).) Flip to
        # normal to insert the new bubble, then back to disabled.
        self.chat_text.configure(state="normal")
        # Round 28: render the flat-message avatar header (avatar
        # square + role label + newline) at the current END. The avatar
        # occupies the first ~56 columns; the bubble tag's lmargin1=56
        # aligns the body text with the role label.
        body_anchor = self._render_avatar(role)
        # Round 27: capture the pre-insert end index so we can render
        # Markdown on exactly the bytes we just wrote (not the trailing
        # ``\n\n`` padding). ``END`` points one past the last char, which
        # is exactly where the new text will land.
        pre_end = self.chat_text.index(self.tk.END)
        self.chat_text.insert(self.tk.END, f"{text}\n\n", (tag,))
        # ``window_create`` must happen while the widget is in
        # ``normal`` state, so the action row anchors before we
        # flip back to disabled.
        if role in {"user", "assistant", "LitTrace", "你"}:
            self._attach_bubble_actions(role, text)
        self.chat_text.configure(state="disabled")
        self.chat_text.see(self.tk.END)
        if role in {"assistant", "LitTrace"}:
            # Render Markdown over the body bytes only — the avatar /
            # role-label line was written by ``_render_avatar`` and is
            # not part of the model output.
            self._render_chat_markdown(body_anchor, text)

    # ------------------------------------------------------------------
    # Round 23: streaming bubble + status bar live updates
    # ------------------------------------------------------------------

    # Phase ids emitted by ``CodexAppServerChatService`` mapped to a
    # localised status string. ``assistant`` overrides the normal
    # "Action: ..." status when a streaming reply is being assembled.
    _PHASE_LABELS = {
        "authenticated": "Codex 已认证",
        "thread_started": "Codex thread 已建立",
        "thread_resumed": "Codex thread 已恢复",
        "mcp_ready": "LitTrace MCP 服务已连接",
        "model_thinking": "Codex 思考中...",
        "turn_completed": "Codex 回复完成",
    }

    def _begin_streaming_bubble(self) -> None:
        """Insert a streaming assistant bubble placeholder.

        The bubble body is filled by ``_append_streaming_delta``. We
        track TWO marks so the finalizer can swap the streamed text
        for the canonical body without duplicating characters:

          * ``_streaming_start_mark`` — start of the streamed text
            region. Used by ``_finalize_streaming_bubble`` as the
            left edge of the deletion.
          * ``_streaming_mark`` — current end of the streamed text.
            ``_append_streaming_delta`` advances this after each
            insert so the next delta continues where the previous one
            ended (no ``-1c`` trickery, no off-by-one empty lines).

        No trailing padding is inserted here; ``_finalize_streaming_bubble``
        adds the ``\\n\\n`` after the canonical body lands.

        Round 28 (Phase 4): the Stop button is now a small teal
        ``Pill.TButton`` anchored on its own line DIRECTLY UNDER the
        streamed text — visually it appears as a compact pill below
        the partial reply. The pill is tracked via ``_stop_pill``
        (Button widget) and ``_stop_pill_index`` (Tk text index of
        the surrounding window slot) so the finalizer can destroy
        the widget AND delete the empty line cleanly.
        """
        self.chat_text.configure(state="normal")
        # Drop any stale Stop button from a previous turn first.
        self._destroy_stop_button_window()
        # Round 28: render the flat-message avatar header (avatar +
        # role label + newline) so the streaming bubble has the same
        # visual shape as a settled assistant message.
        self._render_avatar("assistant")
        # Leading newline so the streamed text lands on a fresh line
        # below the avatar header instead of running into the previous
        # bubble.
        self.chat_text.insert(self.tk.END, "\n")
        # Streaming region starts here. The Pill stop button will
        # land on the line AFTER the streamed text once it grows.
        self._streaming_start_mark = self.chat_text.index(self.tk.INSERT)
        self._streaming_mark = self._streaming_start_mark
        # Anchor the Stop pill below the streaming region. We insert
        # a leading newline now so the pill has a line of its own;
        # ``_finalize_streaming_bubble`` will destroy the pill widget
        # AND delete this line when the canonical reply lands.
        try:
            pill_btn = self.ttk.Button(
                self.chat_text,
                text="停止",
                style="Pill.TButton",
                command=self._request_stop_streaming,
            )
            self.chat_text.insert(self.tk.END, "\n")
            self._stop_pill_index = self.chat_text.index(self.tk.END)
            self.chat_text.window_create(self.tk.END, window=pill_btn, padx=4)
            self._stop_pill = pill_btn
        except self.tk.TclError:
            # Headless fallback — keep the streaming flow alive even
            # if the embedded widget tree fails to build.
            self._stop_pill = None
            self._stop_pill_index = None
        self.chat_text.configure(state="disabled")
        self.chat_text.see(self.tk.END)
        self._streaming_chars = 0

    def _destroy_stop_button_window(self) -> None:
        """Round 27 + Round 28: tear down the embedded Stop button (legacy
        ``Secondary.TButton`` on its own line) AND the new inline Stop
        pill (``Pill.TButton``).

        Called at the top of ``_finalize_streaming_bubble`` and from
        ``_begin_streaming_bubble`` to sweep stale state from a
        crashed previous turn. We destroy the Frame widget AND
        delete the line that held its window slot — ``destroy``
        alone leaves an empty placeholder line in the Text buffer
        that visually grows the bubble each turn.
        """
        # Legacy Round 27 Frame-based button (kept for back-compat).
        frame = getattr(self, "_stop_button_frame", None)
        line = getattr(self, "_stop_button_line", None)
        if frame is not None:
            try:
                frame.destroy()
            except self.tk.TclError:
                pass
        if line is not None:
            try:
                # ``line +1lines`` lands on the line AFTER the stop
                # button — that's the start of the streamed text we
                # want to keep.
                self.chat_text.delete(line, f"{line} + 1 lines")
            except self.tk.TclError:
                pass
        self._stop_button = None
        self._stop_button_frame = None
        self._stop_button_line = None
        # Round 28: the inline ``Pill.TButton`` stop pill. The pill is
        # a direct child of ``chat_text`` (not wrapped in a Frame),
        # so we destroy the Button widget and delete the surrounding
        # window slot in one shot.
        pill = getattr(self, "_stop_pill", None)
        pill_line = getattr(self, "_stop_pill_index", None)
        if pill is not None:
            try:
                pill.destroy()
            except self.tk.TclError:
                pass
        if pill_line is not None:
            try:
                self.chat_text.delete(pill_line, f"{pill_line} + 1 lines")
            except self.tk.TclError:
                pass
        self._stop_pill = None
        self._stop_pill_index = None

    def _request_stop_streaming(self) -> None:
        """Round 27: signal the running turn to abort streaming.

        The threading.Event lives on the GUI thread; the service loop
        checks ``is_set()`` between deltas and breaks out as soon as it
        sees the flag. The button is disabled and relabeled so the
        user sees the click took effect — the actual teardown lands
        on the next ``_finalize_streaming_bubble`` call.
        """
        stop_event = getattr(self, "_streaming_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        # Round 28: prefer the inline ``Pill.TButton`` stop pill; fall
        # back to the legacy ``_stop_button`` for back-compat with
        # Round 27 tests that still reference it.
        pill = getattr(self, "_stop_pill", None)
        if pill is None:
            pill = getattr(self, "_stop_button", None)
        if pill is not None:
            try:
                pill.configure(text="正在停止…", state="disabled")
            except self.tk.TclError:
                pass

    def _append_streaming_delta(self, delta: str) -> None:
        """Append one Codex delta to the streaming bubble (Tk thread only).

        Inserts ``delta`` at the current streaming mark and advances
        the mark so the next delta continues at the new end position.
        """
        if not delta:
            return
        self.chat_text.configure(state="normal")
        # Park the cursor at the current end of streamed text.
        self.chat_text.mark_set(self.tk.INSERT, self._streaming_mark)
        self.chat_text.insert(self.tk.INSERT, delta, ("bubble_assistant",))
        # Advance the mark so the next delta continues where this one ended.
        self._streaming_mark = self.chat_text.index(self.tk.INSERT)
        self.chat_text.configure(state="disabled")
        self.chat_text.see(self.tk.END)
        self._streaming_chars += len(delta)

    def _finalize_streaming_bubble(self, final_text: str | None) -> None:
        """Replace the streamed text with the canonical final reply.

        The bubble's leading newline and visual extent stay in place;
        we swap just the streamed body for ``response.reply`` (or a
        ``（无回复）`` placeholder) and append the trailing ``\\n\\n``
        that visually closes the bubble. Deleting from
        ``_streaming_start_mark`` to ``_streaming_mark`` removes ONLY
        the streamed text, so we don't double-print it.

        Round 27: also drop the Stop button (if any) and reset the
        ``_streaming_stop_event`` so the next turn starts clean.
        """
        if not hasattr(self, "_streaming_mark"):
            return
        # Drop the Stop button + its empty window slot BEFORE we
        # touch the streamed text — the stop button line sits just
        # above ``_streaming_start_mark``, so cleaning it up first
        # keeps the text indices of the streaming region stable.
        self._destroy_stop_button_window()
        # Reset the stop flag so the next turn starts with a fresh Event.
        if hasattr(self, "_streaming_stop_event"):
            self._streaming_stop_event = None
        body = final_text if final_text else "（无回复）"
        self.chat_text.configure(state="normal")
        # Delete whatever deltas were streamed between the start mark
        # and the current end mark. This is the bug fix vs the
        # Round 23 version, which deleted from ``_streaming_mark`` to
        # END — that left the streamed text in place and then
        # appended another copy of ``body`` after it.
        self.chat_text.delete(self._streaming_start_mark, self._streaming_mark)
        # Re-insert the canonical reply at the start position.
        self.chat_text.insert(
            self._streaming_start_mark, body, ("bubble_assistant",)
        )
        # Trailing padding that visually closes the bubble.
        self.chat_text.insert(self.tk.END, "\n\n", ("bubble_assistant",))
        # Round 27: render Markdown on the just-finalised body in place.
        # Streaming phase keeps the bubble as plain text for visual
        # stability; this is the single render-on-finalize moment.
        self._render_chat_markdown(self._streaming_start_mark, body)
        self.chat_text.configure(state="disabled")
        self.chat_text.see(self.tk.END)
        del self._streaming_start_mark
        del self._streaming_mark
        self._streaming_chars = 0

    # ------------------------------------------------------------------
    # Round 25: status bubble lives inside the chat area, not the
    # bottom-left status bar. Each call to ``_set_status_in_chat``
    # replaces the previous bubble so only the current Codex state
    # is visible. ``_clear_status_in_chat`` removes the bubble when
    # a turn ends in failure or the user switches session.
    # ------------------------------------------------------------------

    def _set_status_in_chat(self, label: str, _animate: bool = False) -> None:
        """Insert or replace the Codex status bubble inside the chat.

        Round 28: the status indicator now uses ``bubble_assistant``
        styling (flat-message geometry with the avatar column on the
        left) instead of the legacy ``bubble_system`` centered caption.
        It still updates in place — never duplicated — so the chat
        only ever shows the current phase / progress message, and the
        trailing-dot animation contract is unchanged.

        Note: ``Text.insert`` returns ``None`` on the Tk build this
        app ships with (the Tcl ``insert`` command itself returns
        empty), so we capture the bubble end by reading ``INSERT`` —
        which Tk advances to one position past the inserted text —
        immediately after the insert.

        Round 26: when ``_animate`` is False (the default for every
        external caller) we arm the trailing-dot cycle if ``label``
        matches an in-progress pattern. ``_animate_status_bubble_step``
        passes ``_animate=True`` so the cycle keeps ticking without
        restarting from dot 0 every tick.
        """
        self.chat_text.configure(state="normal")
        if hasattr(self, "_status_bubble_start") and hasattr(
            self, "_status_bubble_end"
        ):
            # Replace the previous bubble body in place.
            self.chat_text.delete(
                self._status_bubble_start, self._status_bubble_end
            )
            # Park the cursor at the bubble start BEFORE inserting so
            # Tk's ``INSERT`` mark advances past the new text
            # regardless of any cursor position left over from the
            # streaming bubble (Tk only auto-advances INSERT when the
            # insert is at or after the current cursor position).
            self.chat_text.mark_set(self.tk.INSERT, self._status_bubble_start)
            self.chat_text.insert(
                self._status_bubble_start,
                f"{label}\n\n",
                ("bubble_assistant",),
            )
            self._status_bubble_end = self.chat_text.index(self.tk.INSERT)
        else:
            # First time this turn — append at the end of the chat.
            self.chat_text.insert(self.tk.END, "\n")
            self._status_bubble_start = self.chat_text.index(self.tk.INSERT)
            # Park the cursor at the bubble start so the label
            # insert advances INSERT to the true end position.
            self.chat_text.mark_set(self.tk.INSERT, self._status_bubble_start)
            self.chat_text.insert(
                self._status_bubble_start,
                f"{label}\n\n",
                ("bubble_assistant",),
            )
            self._status_bubble_end = self.chat_text.index(self.tk.INSERT)
        self.chat_text.configure(state="disabled")
        self.chat_text.see(self.tk.END)
        # Round 27: highlight the trailing dots (if any) of the just
        # rendered status bubble so the live ticker pops visually.
        # Called after the see() so the Tk state has settled.
        self._tag_trailing_dots()

        # Round 26: arm / disarm trailing-dot animation. Only external
        # callers (``_animate=False``) make this decision; the
        # animation step itself passes ``_animate=True`` so the cycle
        # never restarts on every tick.
        if not _animate:
            if self._is_status_animatable(label):
                self._start_status_animation(label)
            else:
                self._stop_status_animation()

    def _tag_trailing_dots(self) -> None:
        """Round 27: highlight the trailing ``.`` of the current status
        bubble with the ``status_pulse`` tag so the live ticker dots
        pop visually against the static label.

        Called from the tail of ``_set_status_in_chat``. No-op when
        no bubble exists, the bubble body has no trailing dots, or the
        widget tree is being torn down (swallowed ``TclError``).
        """
        if not (
            hasattr(self, "_status_bubble_start")
            and hasattr(self, "_status_bubble_end")
        ):
            return
        try:
            self.chat_text.tag_remove(
                "status_pulse",
                self._status_bubble_start,
                self._status_bubble_end,
            )
            body = self.chat_text.get(
                self._status_bubble_start, self._status_bubble_end,
            )
        except self.tk.TclError:
            return
        body = body.rstrip("\n")
        n = len(body) - len(body.rstrip("."))
        if n <= 0:
            return
        try:
            start_idx = f"{self._status_bubble_start} + {len(body) - n} chars"
            end_idx = f"{self._status_bubble_start} + {len(body)} chars"
            self.chat_text.tag_add("status_pulse", start_idx, end_idx)
        except self.tk.TclError:
            pass

    def _clear_status_in_chat(self) -> None:
        """Remove the Codex status bubble if it currently exists.

        Round 26: also cancels any pending trailing-dot tick so the
        timer can never fire against a deleted bubble range.
        """
        # Always stop the timer first — even if no bubble exists yet,
        # a stale ``_status_base_label`` could leak across turns.
        self._stop_status_animation()
        if not (
            hasattr(self, "_status_bubble_start")
            and hasattr(self, "_status_bubble_end")
        ):
            return
        self.chat_text.configure(state="normal")
        self.chat_text.delete(
            self._status_bubble_start, self._status_bubble_end
        )
        self.chat_text.configure(state="disabled")
        del self._status_bubble_start
        del self._status_bubble_end

    # ------------------------------------------------------------------
    # Round 26: status-bubble trailing-dot animation
    #
    # When the Codex status bubble shows an in-progress label
    # (``Codex 思考中…``, ``Codex 流式输出中… N chars``) we cycle the
    # trailing dots every ``_STATUS_ANIM_INTERVAL_MS`` so the bubble
    # does not look frozen while waiting for the model. The cycle
    # stops the moment a non-animatable label lands
    # (``Codex 已认证``, ``Codex 回复完成``) or the bubble is cleared.
    # ------------------------------------------------------------------

    def _is_status_animatable(self, label: str) -> bool:
        """True if ``label`` should pulse trailing dots while visible."""
        return any(token in label for token in _STATUS_ANIMATED_PATTERNS)

    def _start_status_animation(self, base_label: str) -> None:
        """Begin (or restart) the trailing-dot cycle for ``base_label``.

        Stores the base label so each tick re-renders
        ``base_label + "." * count``. Cancels any in-flight tick first.
        """
        self._status_base_label = base_label
        self._status_dot_count = 0
        self._cancel_status_animation()
        self._schedule_status_animation_tick()

    def _schedule_status_animation_tick(self) -> None:
        """Queue the next 400ms tick that bumps the dot counter."""
        try:
            self._status_anim_after_id = self.root.after(
                _STATUS_ANIM_INTERVAL_MS, self._animate_status_bubble_step,
            )
        except self.tk.TclError:
            # Window is being torn down; no more ticks.
            self._status_anim_after_id = None

    def _animate_status_bubble_step(self) -> None:
        """Increment the dot counter and re-render the bubble.

        Called by ``root.after`` on the Tk main thread. Safe against
        mid-flight bubble deletion (``_status_bubble_*`` attrs gone):
        we just stop. Safe against re-entry: the inner call to
        ``_set_status_in_chat`` passes ``_animate=True`` so the cycle
        does not restart at dot 0. Safe against window shutdown
        (``chat_text`` destroyed): we swallow the ``TclError``.
        """
        self._status_anim_after_id = None
        if not (
            hasattr(self, "_status_bubble_start")
            and hasattr(self, "_status_bubble_end")
        ):
            # Bubble vanished (turn ended, error, session switch). Stop.
            self._cancel_status_animation()
            return
        # Cycle through 0..3 dots so the visible sequence is
        # ``""`` -> ``.`` -> ``..`` -> ``...`` then back to ``""``.
        self._status_dot_count = (self._status_dot_count + 1) % 4
        label = self._status_base_label + "." * self._status_dot_count
        try:
            self._set_status_in_chat(label, _animate=True)
        except self.tk.TclError:
            # ``chat_text`` was destroyed between the ``hasattr`` check
            # and the inner call — common during shutdown. Stop cleanly.
            self._cancel_status_animation()
            return
        if self._is_status_animatable(self._status_base_label):
            self._schedule_status_animation_tick()

    def _cancel_status_animation(self) -> None:
        """Cancel any pending tick. Safe to call when nothing is
        scheduled (e.g. before the first ``_set_status_in_chat`` of a
        turn) or after the timer has already fired (the cancel raises
        ``TclError`` on stale ids — we swallow it).
        """
        if self._status_anim_after_id is None:
            return
        try:
            self.root.after_cancel(self._status_anim_after_id)
        except self.tk.TclError:
            pass
        self._status_anim_after_id = None

    def _stop_status_animation(self) -> None:
        """Cancel the timer AND reset the cycle base so the next
        ``_set_status_in_chat`` of an animatable label starts fresh.
        """
        self._cancel_status_animation()
        self._status_base_label = ""
        self._status_dot_count = 0

    def _show_execution_path(self, message: str) -> None:
        steps = _execution_steps_for_message(message)
        self.status_var.set("正在执行: " + " -> ".join(steps))
        self._render_planned_trace(steps)

    def _render_planned_trace(self, steps: list[str]) -> None:
        self.trace_text.delete("1.0", self.tk.END)
        self.trace_text.insert(self.tk.END, "计划路径\n", ("trace_node",))
        for index, step in enumerate(steps, start=1):
            self.trace_text.insert(self.tk.END, f"{index}. {step}\n", ("trace_body",))

    def _render_workflow_trace(self, trace: WorkflowTrace) -> None:
        self.trace_text.delete("1.0", self.tk.END)
        if not trace.steps:
            self.trace_text.insert(self.tk.END, "暂无真实 trace。\n", ("trace_body",))
        for index, step in enumerate(trace.steps, start=1):
            self.trace_text.insert(self.tk.END, f"{index}. {step.node}\n", ("trace_node",))
            self.trace_text.insert(self.tk.END, f"状态: {step.status}\n", ("trace_body",))
            self.trace_text.insert(self.tk.END, f"原因: {step.reason}\n", ("trace_body",))
            if step.outputs:
                compact = ", ".join(f"{key}={value}" for key, value in step.outputs.items())
                self.trace_text.insert(self.tk.END, f"输出: {compact}\n", ("trace_body",))
            if step.next_node:
                self.trace_text.insert(self.tk.END, f"下一步: {step.next_node}\n", ("trace_body",))
            if step.next_reason:
                self.trace_text.insert(
                    self.tk.END, f"选择原因: {step.next_reason}\n", ("trace_body",)
                )
            self.trace_text.insert(self.tk.END, "\n", ("trace_body",))

    def _refresh_context(self) -> None:
        lines = render_context_lines(self.workspace)
        self.context_text.delete("1.0", self.tk.END)
        self.context_text.insert(self.tk.END, "\n".join(lines))

    def _refresh_rag_panel(self) -> None:
        if self.rag_text is None:
            return
        lines: list[str] = []
        filters = self.workspace.context.filters
        last_sync = getattr(filters, "research_background_last_sync_at", None)
        downloaded = getattr(filters, "research_background_last_downloaded_count", 0)
        parsed = getattr(filters, "research_background_last_parsed_count", 0)
        lines.append(f"Session: {self.session.session_id[:18]}…")
        lines.append(f"上次同步: {last_sync or '从未'}")
        lines.append(f"下载 / 解析: {downloaded} / {parsed}")

        try:
            store = state_store_from_config(self.config)
            report = store.async_tasks_queue_report(kind="embedding_job")
            lines.append("")
            lines.append("Embedding Jobs 队列:")
            lines.append(f"  queued={report.queued} running={report.running}")
            lines.append(f"  failed={report.failed} dead={report.dead}")
            lines.append(f"  total completed={report.completed}")
        except Exception as exc:
            lines.append("")
            lines.append(f"(队列查询失败: {exc.__class__.__name__}: {exc})")

        log_path = Path("logs/rag-daily.log")
        if log_path.exists():
            mtime = datetime.fromtimestamp(log_path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            lines.append("")
            lines.append(f"最近 launchd daily: {mtime}")
        else:
            lines.append("")
            lines.append("最近 launchd daily: (无 logs/rag-daily.log)")

        self.rag_text.delete("1.0", self.tk.END)
        self.rag_text.insert(self.tk.END, "\n".join(lines))

    def _set_rag_busy(self, busy: bool, label: str = "") -> None:
        def apply() -> None:
            self.rag_busy = busy
            state = "disabled" if busy else "normal"
            if self.sync_session_btn is not None:
                self.sync_session_btn.configure(state=state)
            if self.full_daily_btn is not None:
                self.full_daily_btn.configure(state=state)
            if label:
                self.status_var.set(label)

        self.root.after(0, apply)

    def _trigger_session_sync(self) -> None:
        if self.rag_busy:
            return
        if refresh_session_rag_index is None:
            self.status_var.set("RAG 不可用: refresh_session_rag_index 未导入")
            return
        self._set_rag_busy(True, "正在同步当前 Session…")
        threading.Thread(target=self._session_sync_thread, daemon=True).start()

    def _session_sync_thread(self) -> None:
        try:
            workspace = load_workspace(self.session)
            profile, report = asyncio.run(
                refresh_session_rag_index(self.config, self.session, workspace)
            )
            save_workspace(self.session, workspace, config=self.config)
            self.workspace = workspace
            summary = (
                f"同步完成: chunks={report.chunk_count} "
                f"upserted={report.upserted_count} papers={report.paper_count}"
            )
            if report.warnings:
                summary += f" warns={len(report.warnings)}"
            if report.skipped:
                summary += f" skipped({report.skip_reason or 'n/a'})"
        except Exception as exc:
            summary = f"同步失败: {exc.__class__.__name__}: {exc}"

        def done() -> None:
            self.status_var.set(summary)
            self._set_rag_busy(False)
            self._refresh_rag_panel()

        self.root.after(0, done)

    def _trigger_full_daily(self) -> None:
        if self.rag_busy:
            return
        if run_daily_rag_maintenance is None:
            self.status_var.set("RAG 不可用: run_daily_rag_maintenance 未导入")
            return
        self._set_rag_busy(True, "正在执行全量 Daily（自动下载到对象存储）…")
        threading.Thread(target=self._full_daily_thread, daemon=True).start()

    def _full_daily_thread(self) -> None:
        try:
            report = asyncio.run(run_daily_rag_maintenance(self.config))
            summary = (
                f"Daily 完成: refreshed={report.sessions_refreshed} "
                f"failed={report.sessions_failed} skipped={report.sessions_skipped} "
                f"jobs={report.embedding_jobs_processed}/{report.embedding_jobs_failed} "
                f"下载到对象存储={report.downloaded_count}"
            )
            if report.warnings:
                summary += f" warns={len(report.warnings)}"
        except Exception as exc:
            summary = f"Daily 失败: {exc.__class__.__name__}: {exc}"

        def done() -> None:
            self.status_var.set(summary)
            self._set_rag_busy(False)
            self._refresh_rag_panel()
            self._refresh_session_history()

        self.root.after(0, done)

    def _refresh_ocr_buttons(self) -> None:
        report = decide_artifact_extraction_need(self.workspace)
        self.status_var.set(
            f"OCR建议: {report.recommended_parse_strategy} | {self.session.session_id}"
        )

    def _toggle_context(self) -> None:
        if self.context_visible:
            self.pane.forget(self.context_frame)
            self.context_visible = False
            self.workspace.context.visible_to_user = False
        else:
            self.pane.add(self.context_frame, minsize=300)
            self.context_visible = True
            self.workspace.context.visible_to_user = True
        self.context_toggle_button.configure(
            text="隐藏上下文" if self.context_visible else "显示上下文"
        )

    def _toggle_parse_strategy(self) -> None:
        next_strategy = "ocr" if self.parse_strategy == "text_only" else "text_only"
        self._set_parse_strategy(next_strategy)

    def _set_parse_strategy(self, strategy: str) -> None:
        self.parse_strategy = strategy
        self.config.parsing.parse_strategy = strategy
        label = "只看文字层" if strategy == "text_only" else "使用 OCR"
        self.status_var.set(f"解析模式: {label} | Session: {self.session.session_id}")
        self._render_planned_trace([f"解析模式已切换为：{label}"])
        self.parse_strategy_button.configure(text=self._parse_strategy_button_text())
        if self.ocr_popup is not None:
            self.ocr_popup.destroy()
            self.ocr_popup = None

    def _parse_strategy_button_text(self) -> str:
        label = "文字层" if self.parse_strategy == "text_only" else "OCR"
        return f"文献解析模式：{label}"

    def _open_ocr_popup(self) -> None:
        if self.ocr_popup is not None and self.ocr_popup.winfo_exists():
            self.ocr_popup.lift()
            return
        self.ocr_popup = self.tk.Toplevel(self.root)
        self.ocr_popup.title("LitTrace 解析设置")
        self.ocr_popup.geometry("520x260")
        self.ocr_popup.configure(background=DESIGN["parchment"])
        self.ocr_popup.transient(self.root)
        self.ocr_popup.protocol("WM_DELETE_WINDOW", self._close_ocr_popup)

        report = decide_artifact_extraction_need(self.workspace)
        frame = self.ttk.Frame(self.ocr_popup, style="Tile.TFrame", padding=24)
        frame.pack(fill=self.tk.BOTH, expand=True)
        self.ttk.Label(frame, text="选择 PDF 解析方式", style="TileHeader.TLabel").pack(anchor="w")
        self.ttk.Label(
            frame, text=f"建议: {report.recommended_parse_strategy}", style="Caption.TLabel"
        ).pack(anchor="w", pady=(10, 0))
        self.ttk.Label(
            frame, text=report.reason, style="Caption.TLabel", wraplength=470, justify=self.tk.LEFT
        ).pack(anchor="w", pady=(4, 12))
        button_frame = self.ttk.Frame(frame, style="Tile.TFrame")
        button_frame.pack(fill=self.tk.X)
        self.ttk.Button(
            button_frame,
            text="只看文字层",
            style="Secondary.TButton",
            command=lambda: self._set_parse_strategy("text_only"),
        ).pack(side=self.tk.LEFT)
        self.ttk.Button(
            button_frame,
            text="使用 OCR",
            style="Primary.TButton",
            command=lambda: self._set_parse_strategy("ocr"),
        ).pack(side=self.tk.LEFT, padx=(8, 0))
        self.ttk.Button(
            button_frame, text="关闭", style="Secondary.TButton", command=self._close_ocr_popup
        ).pack(side=self.tk.RIGHT)

    def _close_ocr_popup(self) -> None:
        if self.ocr_popup is not None:
            self.ocr_popup.destroy()
            self.ocr_popup = None

    def _open_context_popup(self) -> None:
        if self.context_popup is not None and self.context_popup.winfo_exists():
            self.context_popup.lift()
            self._refresh_context_popup()
            return
        self.context_popup = self.tk.Toplevel(self.root)
        self.context_popup.title("LitTrace 当前文献上下文")
        self.context_popup.geometry("760x560")
        self.context_popup.configure(background=DESIGN["parchment"])
        self.context_popup.transient(self.root)
        self.context_popup.protocol("WM_DELETE_WINDOW", self._close_context_popup)
        self._refresh_context_popup()

    def _close_context_popup(self) -> None:
        if self.context_popup is not None:
            self.context_popup.destroy()
            self.context_popup = None

    def _refresh_context_popup(self) -> None:
        if self.context_popup is None or not self.context_popup.winfo_exists():
            return
        for child in self.context_popup.winfo_children():
            child.destroy()

        outer = self.ttk.Frame(self.context_popup, style="Tile.TFrame", padding=24)
        outer.pack(fill=self.tk.BOTH, expand=True)
        header = self.ttk.Frame(outer, style="Tile.TFrame")
        header.pack(fill=self.tk.X)
        self.ttk.Label(header, text="当前文献上下文", style="TileHeader.TLabel").pack(
            side=self.tk.LEFT
        )
        self.ttk.Button(
            header, text="全部选择下载", style="Primary.TButton", command=self._select_all_downloads
        ).pack(side=self.tk.RIGHT)
        self.ttk.Button(
            header, text="清空选择", style="Secondary.TButton", command=self._clear_downloads
        ).pack(side=self.tk.RIGHT, padx=(0, 8))

        if not self.workspace.context.active_papers:
            self.ttk.Label(
                outer, text="当前没有文献。先在主窗口输入检索任务。", style="Caption.TLabel"
            ).pack(anchor="w", pady=(16, 0))
            return

        list_frame = self.ttk.Frame(outer, style="Tile.TFrame")
        list_frame.pack(fill=self.tk.BOTH, expand=True, pady=(12, 0))
        canvas = self.tk.Canvas(list_frame, highlightthickness=0, bg=DESIGN["canvas"])
        scrollbar = self.ttk.Scrollbar(
            list_frame,
            orient=self.tk.VERTICAL,
            command=canvas.yview,
            style="Slim.Vertical.TScrollbar",
        )
        inner = self.ttk.Frame(canvas, style="Tile.TFrame")
        inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        selected = set(self.workspace.context.selected_for_download)
        for index, paper_id in enumerate(self.workspace.context.active_papers, start=1):
            paper = self.workspace.papers[paper_id]
            var = self.tk.BooleanVar(value=paper_id in selected)
            row = self.ttk.Frame(inner, style="Tile.TFrame", padding=(0, 6))
            row.pack(fill=self.tk.X, anchor="w")
            checkbox = self.ttk.Checkbutton(
                row,
                variable=var,
                command=lambda pid=paper_id, value=var: self._toggle_download_selection(pid, value),
            )
            checkbox.pack(side=self.tk.LEFT)
            year = paper.year or "n.d."
            source = paper.journal or paper.publisher or "unknown"
            text = f"{index}. {paper.title}\n{year} | {source}"
            self.ttk.Label(
                row, text=text, style="Caption.TLabel", wraplength=650, justify=self.tk.LEFT
            ).pack(side=self.tk.LEFT, fill=self.tk.X, expand=True)

    def _toggle_download_selection(self, paper_id: str, value) -> None:
        selected = list(self.workspace.context.selected_for_download)
        enabled = bool(value.get())
        if enabled and paper_id not in selected:
            selected.append(paper_id)
        if not enabled and paper_id in selected:
            selected.remove(paper_id)
        self.workspace.context.selected_for_download = selected
        save_workspace(self.session, self.workspace, config=self.config)
        self._refresh_context()
        self.status_var.set(f"已选择下载 {len(selected)} 篇 | Session: {self.session.session_id}")

    def _select_all_downloads(self) -> None:
        self.workspace.context.selected_for_download = list(self.workspace.context.active_papers)
        save_workspace(self.session, self.workspace, config=self.config)
        self._refresh_context()
        self._refresh_context_popup()

    def _clear_downloads(self) -> None:
        self.workspace.context.selected_for_download = []
        save_workspace(self.session, self.workspace, config=self.config)
        self._refresh_context()
        self._refresh_context_popup()

    def _open_login_popup(self, download_plan: DownloadPlan | None = None) -> None:
        plan = download_plan or self.last_download_plan
        if plan is None:
            self._append_message(
                "assistant", "当前还没有下载计划。请先说“选择第 N 篇下载”或“生成下载计划”。"
            )
            return
        login_items = [item for item in plan.items if item.requires_login]
        if not login_items:
            self._append_message("assistant", "当前下载计划里没有需要登录的文献。")
            return
        if self.login_popup is not None and self.login_popup.winfo_exists():
            self.login_popup.lift()
            return
        self.login_popup = self.tk.Toplevel(self.root)
        self.login_popup.title("LitTrace 登录下载")
        self.login_popup.geometry("760x460")
        self.login_popup.configure(background=DESIGN["parchment"])
        self.login_popup.transient(self.root)
        self.login_popup.protocol("WM_DELETE_WINDOW", self._close_login_popup)

        outer = self.ttk.Frame(self.login_popup, style="Tile.TFrame", padding=24)
        outer.pack(fill=self.tk.BOTH, expand=True)
        self.ttk.Label(outer, text="需要你授权登录后下载的文献", style="TileHeader.TLabel").pack(
            anchor="w"
        )
        self.ttk.Label(
            outer,
            text="LitTrace 只打开授权页面，不绕过登录。认证完成后请保持授权窗口打开；PDF 归档完成后主窗口会继续解析，并尝试关闭后台会话。",
            style="Caption.TLabel",
            wraplength=720,
            justify=self.tk.LEFT,
        ).pack(anchor="w", pady=(8, 12))
        for item in login_items:
            paper = self.workspace.papers.get(item.paper_id)
            row = self.ttk.Frame(outer, style="Tile.TFrame", padding=(0, 6))
            row.pack(fill=self.tk.X, anchor="w")
            title = paper.title if paper else item.title
            self.ttk.Label(
                row, text=title, style="Caption.TLabel", wraplength=560, justify=self.tk.LEFT
            ).pack(side=self.tk.LEFT, fill=self.tk.X, expand=True)
            self.ttk.Button(
                row,
                text="打开登录页",
                style="Primary.TButton",
                command=lambda pid=item.paper_id: self._launch_login(pid),
            ).pack(side=self.tk.RIGHT)

    def _close_login_popup(self) -> None:
        if self.login_popup is not None:
            self.login_popup.destroy()
            self.login_popup = None

    def _launch_login(self, paper_id: str) -> None:
        paper = self.workspace.papers.get(paper_id)
        if paper is None:
            self._append_message("system", "没有找到这篇文献。")
            return
        plan = browser_login_session_for_paper(
            self.config,
            paper,
            self.workspace.full_text_reports.get(paper_id),
            browser_session_name=publisher_window_session_name_for_chat(self.session.session_id),
        )
        self._render_planned_trace(
            ["打开授权窗口", "等待用户完成授权", "后台获取 PDF", "归档并解析"]
        )
        if plan.error or not plan.browser_act_command:
            self._append_message("system", plan.error or "无法打开授权浏览器。")
            return
        self.status_var.set("等待授权完成，LitTrace 会自动继续")
        threading.Thread(
            target=self._wait_for_login_and_resume_thread,
            args=(paper_id, plan.session_name or ""),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._open_login_browser_thread,
            args=(paper_id,),
            daemon=True,
        ).start()

    def _open_login_browser_thread(self, paper_id: str) -> None:
        paper = self.workspace.papers.get(paper_id)
        if paper is None:
            return
        result = open_browser_login_session(
            self.config,
            paper,
            self.workspace.full_text_reports.get(paper_id),
            timeout_seconds=60.0,
            browser_session_name=publisher_window_session_name_for_chat(self.session.session_id),
        )
        if not result.opened:
            self.root.after(
                0,
                lambda: self._append_message(
                    "system",
                    f"授权浏览器打开失败：{result.error or result.stderr or result.stdout}",
                ),
            )
            return
        if result.fallback_used:
            self.root.after(
                0,
                lambda: self.status_var.set("已切换备用浏览器，等待授权完成"),
            )

    def _wait_for_login_and_resume_thread(self, paper_id: str, session_name: str) -> None:
        paper = self.workspace.papers.get(paper_id)
        if paper is None:
            return
        report = self.workspace.full_text_reports.get(paper_id)
        auth = wait_for_browser_authorization(
            self.config,
            session_name,
            timeout_seconds=180.0,
            poll_interval_seconds=2.0,
        )
        if not auth.authorized:
            if auth.requires_user_confirmation:
                self.root.after(0, self._show_user_confirmation_popup)
            self.root.after(
                0,
                lambda: self.status_var.set("授权等待超时，请重新打开授权窗口"),
            )
            return
        fetch = fetch_authorized_pdf_after_user_auth(
            self.config,
            paper,
            report,
            timeout_seconds=60.0,
            auth_wait_result=auth,
            browser_session_name=session_name,
        )
        self.workspace, resume = auto_resume_downloaded_pdfs(
            self.config,
            self.workspace,
            self.session,
        )

        def apply_result() -> None:
            save_workspace(self.session, self.workspace, config=self.config)
            self._refresh_context()
            self._refresh_context_popup()
            if fetch.error:
                self._append_message("system", f"授权已完成，但后台 PDF 获取失败：{fetch.error}")
                self.status_var.set("PDF 获取失败")
                return
            if resume.parsed_count or resume.ready_to_parse_count or resume.auto_archived_count:
                self._append_message("assistant", "授权已完成，PDF 已后台归档并开始解析。")
                self.status_var.set("授权完成，PDF 已归档")
            else:
                self.status_var.set("授权完成，正在等待 PDF 下载落盘")

        self.root.after(0, apply_result)
        # Keep the chat-scoped publisher browser alive for the next paper.

    def _show_user_confirmation_popup(self) -> None:
        popup = self.tk.Toplevel(self.root)
        popup.title("需要真人验证")
        popup.geometry("420x180")
        popup.configure(background=DESIGN["parchment"])
        popup.transient(self.root)
        popup.lift()
        outer = self.ttk.Frame(popup, style="Tile.TFrame", padding=20)
        outer.pack(fill=self.tk.BOTH, expand=True)
        self.ttk.Label(outer, text="请完成浏览器中的真人验证", style="TileHeader.TLabel").pack(
            anchor="w"
        )
        self.ttk.Label(
            outer,
            text="ACS/Cloudflare 需要你确认是真人。请在已打开的授权浏览器窗口中完成验证，并保持窗口打开，直到 LitTrace 显示 PDF 已归档。",
            style="Caption.TLabel",
            wraplength=360,
            justify=self.tk.LEFT,
        ).pack(anchor="w", pady=(8, 14))
        self.ttk.Button(outer, text="我知道了", command=popup.destroy).pack(anchor="e")

    def _close_browser_session_silently(self, session_name: str | None) -> None:
        if not session_name:
            return
        from littrace.access_layer.browser_sessions import run_browser_act

        run_browser_act(
            self.config,
            ["session", "close", session_name],
            timeout_seconds=10.0,
        )

    def _format_relative_time(self, iso: str) -> str:
        """Round 28: render an ISO timestamp as a human-readable relative
        string for the sidebar session list.

        Output ladder (matching the Codex / Linear / Notion convention):
          * <1 minute ago:     "Just now"
          * <60 minutes ago:   "Xm ago"
          * same day:          "Xh ago"
          * yesterday:         "Yesterday"
          * <7 days ago:       weekday short ("Mon", "Tue" ...)
          * else:              "MMM D" (e.g. "Aug 23")

        Tolerates naive and timezone-aware ISO strings; falls back to
        the raw input if parsing fails so a malformed row never crashes
        the sidebar refresh.
        """
        if not iso:
            return ""
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return iso
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "Just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24 and dt.date() == now.date():
            return f"{hours}h ago"
        # Day boundary — compare date().
        if dt.date() == (now.date() - __import__("datetime").timedelta(days=1)):
            return "Yesterday"
        if delta.days < 7:
            return dt.strftime("%a")  # "Mon", "Tue", ...
        return dt.strftime("%b %d")  # "Aug 23"

    def _new_chat(self) -> None:
        """Round 28: start a fresh session and switch to it.

        Mirrors the ``+ 新对话`` button on the sidebar. ``create_chat_session``
        allocates a new session id; we then route through
        ``_switch_session`` so the workspace re-load, RAG panel refresh,
        and chat timeline rehydrate all run consistently.
        """
        new_session = create_chat_session(self.config)
        self._switch_session(new_session.session_id)

    def _refresh_session_history(self) -> None:
        self.session_history_text.delete("1.0", self.tk.END)
        summaries = list_chat_sessions(self.config, limit=8)
        if not summaries:
            self.session_history_text.insert(self.tk.END, "暂无历史 session。")
        for index, item in enumerate(summaries):
            current = "当前 " if item.session_id == self.session.session_id else ""
            relative = self._format_relative_time(item.updated_at)
            tag = f"session_{index}"
            start = self.session_history_text.index(self.tk.INSERT)
            self.session_history_text.insert(
                self.tk.END,
                # Round 28: relative timestamp replaces the raw ISO.
                # Falls back to the raw value if the formatter returned
                # the input unchanged (parse failure path).
                f"{current}{item.topic}\n"
                f"{relative}  ·  消息 {item.message_count}  ·  文献 {item.paper_count}\n"
                f"{item.session_id}\n\n",
            )
            end = self.session_history_text.index(self.tk.INSERT)
            self.session_history_text.tag_add(tag, start, end)
            self.session_history_text.tag_configure(
                tag,
                foreground=DESIGN["primary"]
                if item.session_id != self.session.session_id
                else DESIGN["ink"],
                # Round 28: softer highlight (surface_2 instead of
                # accent_teal_subtle) so the current-session row reads
                # as a subtle background tint rather than a loud chip.
                background=DESIGN["surface_2"]
                if item.session_id == self.session.session_id
                else DESIGN["canvas"],
                lmargin1=3,
                lmargin2=3,
                rmargin=3,
                spacing1=2,
                spacing3=8,
            )
            self.session_history_text.tag_bind(
                tag,
                "<Button-1>",
                lambda _event, sid=item.session_id: self._switch_session(sid),
            )
            self.session_history_text.tag_bind(
                tag,
                "<Enter>",
                lambda _event: self.session_history_text.configure(cursor="hand2"),
            )
            self.session_history_text.tag_bind(
                tag,
                "<Leave>",
                lambda _event: self.session_history_text.configure(cursor=""),
            )

    def _switch_session(self, session_id: str) -> None:
        if session_id == self.session.session_id:
            return
        self.session = load_or_create_session(self.config, session_id)
        _scope_storage_to_session(self.config, self.session)
        self.workspace = load_workspace(self.session)
        self.last_download_plan = None
        # Round 25: drop any in-progress Codex status bubble so the
        # switched session doesn't inherit the previous turn's state.
        self._clear_status_in_chat()
        self._render_session_messages()
        self._refresh_context()
        self._refresh_context_popup()
        self._refresh_session_history()
        self._render_planned_trace([f"已切换到历史 session：{session_id}"])
        self.status_var.set(f"Session: {self.session.session_id}")

    def _render_session_messages(self) -> None:
        # Round 27: drop the embedded action frames from the previous
        # session BEFORE wiping the text buffer so Tk can garbage-
        # collect them. Each frame is a Tk child of ``chat_text``;
        # we destroy the widget AND delete the line that held its
        # embedded window slot so the new session starts with a
        # clean canvas.
        self.chat_text.configure(state="normal")
        # Round 28 (Phase 6): tool-card cleanup BEFORE the bubble
        # records so the line indices line up. Each tool card lives
        # on its own line with a single embedded Tk Frame; destroying
        # the Frame + deleting the surrounding line is the same
        # pattern as ``_bubble_records``.
        for record in getattr(self, "_tool_card_records", []):
            frame = record.get("frame")
            line_index = record.get("line_index")
            if frame is not None:
                try:
                    frame.destroy()
                except self.tk.TclError:
                    pass
            if line_index is not None:
                try:
                    self.chat_text.delete(
                        line_index, f"{line_index} + 1 lines"
                    )
                except self.tk.TclError:
                    pass
        self._tool_card_records = []
        for record in getattr(self, "_bubble_records", []):
            frame = record.get("frame")
            line_index = record.get("line_index")
            if frame is not None:
                try:
                    frame.destroy()
                except self.tk.TclError:
                    pass
            if line_index is not None:
                try:
                    self.chat_text.delete(
                        line_index, f"{line_index} + 1 lines"
                    )
                except self.tk.TclError:
                    pass
        self._bubble_records = []
        self.chat_text.delete("1.0", self.tk.END)
        self.chat_text.configure(state="disabled")
        for record in state_store_from_config(self.config).list_chat_messages(self.session.session_id):
            role = "你" if record.get("role") == "user" else "LitTrace"
            content = record.get("content_json") or record.get("content_text")
            text = _message_text(content)
            if text and (role == "你" or _is_user_effective_reply(text)):
                bubble_role = "user" if role == "你" else "assistant"
                # Round 27: route through ``_append_message`` so the
                # bubble gets the standard MD render + per-bubble
                # action row. ``_append_message`` handles the
                # ``state="normal"/"disabled"`` discipline.
                self._append_message(bubble_role, text)
        self.chat_text.see(self.tk.END)

    def _open_help_popup(self) -> None:
        popup = self.tk.Toplevel(self.root)
        popup.title("LitTrace 使用说明")
        popup.geometry("560x360")
        popup.configure(background=DESIGN["parchment"])
        popup.transient(self.root)
        frame = self.ttk.Frame(popup, style="Tile.TFrame", padding=24)
        frame.pack(fill=self.tk.BOTH, expand=True)
        self.ttk.Label(frame, text="使用说明", style="TileHeader.TLabel").pack(anchor="w")
        text = self.tk.Text(
            frame,
            wrap=self.tk.WORD,
            bd=0,
            relief=self.tk.FLAT,
            highlightthickness=0,
            bg=DESIGN["canvas"],
            fg=DESIGN["ink"],
            font=_font("body"),
            height=10,
        )
        text.pack(fill=self.tk.BOTH, expand=True, pady=(12, 12))
        text.insert(
            self.tk.END,
            "直接输入研究任务，例如：我想了解薄膜压敏传感阵列的相关文献。\n\n"
            "左侧显示真实 Workflow Trace 和历史 Session；点击历史 Session 可以切换上下文。\n\n"
            "右侧显示当前文献上下文。需要选择下载文献时，点击“文献上下文”打开选择窗口。\n\n"
            "顶部“文献解析模式”按钮会在文字层和 OCR 之间切换。",
        )
        text.configure(state=self.tk.DISABLED)
        self.ttk.Button(frame, text="关闭", style="Secondary.TButton", command=popup.destroy).pack(
            anchor="e"
        )


def main() -> None:
    LitTraceWindow().run()


def _execution_steps_for_message(message: str) -> list[str]:
    intent = parse_chat_intent(message)
    steps = ["识别任务意图"]
    if "search" in intent.actions:
        steps.extend(["提取研究主题", "检索候选文献", "更新当前文献上下文", "弹出文献选择"])
    if "download" in intent.actions or "select_downloads" in intent.actions:
        steps.append("准备下载选择")
    if "parse" in intent.actions:
        steps.append("按当前解析模式处理 PDF")
    if "table" in intent.actions:
        steps.append("抽取性能指标并生成对比表")
    if "storyline" in intent.actions:
        steps.append("梳理论文回应关系与发展脉络")
    if "document" in intent.actions:
        steps.append("组织学术化报告")
    if "autonomous_review" in intent.actions:
        steps.append("运行质量门与可选 Reviewer 审查")
    if len(steps) == 1:
        steps.append("基于当前上下文直接回答")
    return steps


def _is_user_effective_reply(reply: str) -> bool:
    quiet_prefixes = [
        "已切换解析模式",
        "已隐藏当前文献上下文",
        "已显示当前文献上下文",
    ]
    return bool(reply.strip()) and not any(reply.startswith(prefix) for prefix in quiet_prefixes)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ["message", "reply"]:
            value = content.get(key)
            if isinstance(value, str):
                return value
    return ""


def _chat_bubble_tag(role: str) -> str:
    if role in {"user", "你"}:
        return "bubble_user"
    if role in {"assistant", "LitTrace"}:
        return "bubble_assistant"
    return "bubble_system"


def _font(token: str):
    """Return a Tk font object with anti-aliasing enabled.

    Why this is not a plain tuple
    -----------------------------
    Returning a bare ``("Arial", 13, "normal")`` tuple tells Tk to
    render the text through GDI's default path, which on Windows uses
    *grayscale* anti-aliasing at best and often falls back to bitmapped
    glyphs for non-system sizes — that's the "毛刺感" (jagged edges)
    users see.

    Tk 8.6+ exposes an ``antialiased=True`` flag on ``tkinter.font.Font``
    that switches the renderer to ClearType (sub-pixel AA) when the
    underlying Tk build supports it. We:

      1. Resolve the (family, size, weight) tuple for ``token``.
      2. Build a cached ``tk.font.Font`` per token (memoized on the
         function attribute so we don't allocate N font objects per
         redraw).
      3. Try ``antialiased=True`` first; fall back to the plain Font
         constructor if the running Tk build rejects the kwarg.

    Tk accepts either a Font object or a tuple wherever a ``font=``
    kwarg is required, so existing call sites do not change.
    """
    # CJK-friendly family chain. ``Microsoft YaHei UI`` ships with every
    # modern Windows install and exposes the CJK glyph set Tk needs for
    # Chinese chat content + IME composition text. ``PingFang SC`` covers
    # macOS dev paths. ``Arial`` stays at the tail so Western text keeps
    # its metrics. Tk walks the comma-separated chain to the first family
    # it can resolve at construction time.
    sans = "Microsoft YaHei UI, Microsoft YaHei, PingFang SC, Arial"
    sans_fallback = "Helvetica, Liberation Sans, sans-serif"
    serif = "Microsoft YaHei UI, Microsoft YaHei, PingFang SC, Arial"
    serif_fallback = "Helvetica, Liberation Sans, serif"
    recipes = {
        "display":      (f"{serif}, {serif_fallback}", 22, "normal"),
        "title_serif":  (f"{serif}, {serif_fallback}", 18, "normal"),
        "tagline":      (f"{sans}, {sans_fallback}",   14, "bold"),    # 13 -> 14
        # Round 28: body bumped to 15pt to match the Codex desktop body
        # size; matches a long-form reading distance rather than a dense
        # IDE/Linear-style 13pt.
        "body":         (f"{sans}, {sans_fallback}",   15, "normal"),  # 13 -> 15
        "body_strong":  (f"{sans}, {sans_fallback}",   15, "bold"),
        # Round 26: italic variant. Used by ``md_italic`` tag (Tk tag
        # tuples can't express ``slant``, so the tag wraps a real
        # ``tk.font.Font(slant='italic')`` instead — see
        # ``_configure_input_tags``).
        "body_italic":  (f"{sans}, {sans_fallback}",   15, "normal"),  # 13 -> 15
        # Round 26: monospace recipe for inline code / code blocks.
        # ``Microsoft YaHei UI`` at the tail covers CJK inside a code
        # block (rare but possible).
        "mono":         ("Consolas, Menlo, Courier New, Microsoft YaHei UI", 13, "normal"),  # 12 -> 13
        # Round 26: heading sizes for markdown live-rendering.
        "h1":           (f"{sans}, {sans_fallback}",   20, "bold"),    # 18 -> 20
        "h2":           (f"{sans}, {sans_fallback}",   17, "bold"),    # 15 -> 17
        "h3":           (f"{sans}, {sans_fallback}",   15, "normal"),
        "caption":      (f"{sans}, {sans_fallback}",   12, "normal"),  # 11 -> 12
        "button":       (f"{sans}, {sans_fallback}",   13, "normal"),  # 12 -> 13
        "button_utility":(f"{sans}, {sans_fallback}",  12, "normal"),  # 11 -> 12
        "fine":         (f"{sans}, {sans_fallback}",   11, "normal"),
        "nav":          (f"{sans}, {sans_fallback}",   11, "normal"),
        # Round 28: new recipes for the Codex visual language.
        # ``avatar`` is the bold initial letter inside the flat-message
        # avatar square (assistant "L" / user "你"). ``tool_name`` /
        # ``tool_meta`` style the inline tool card header + duration.
        # ``timestamp`` is the small relative-time label in the sidebar.
        "avatar":       (f"{sans}, {sans_fallback}",   12, "bold"),
        "tool_name":    (f"{sans}, {sans_fallback}",   13, "bold"),
        "tool_meta":    (f"{sans}, {sans_fallback}",   12, "normal"),
        "timestamp":    (f"{sans}, {sans_fallback}",   11, "normal"),
    }
    family, size, weight = recipes.get(token, recipes["body"])

    cache = getattr(_font, "_cache", None)
    if cache is None:
        cache = {}
        _font._cache = cache  # type: ignore[attr-defined]
    key = (token, family, size, weight)
    cached = cache.get(key)
    if cached is not None:
        return cached

    # Lazy import — `_font()` is called from both __main__ (the running
    # GUI) and from unit tests that don't necessarily have Tk loaded.
    try:
        from tkinter import font as tkfont
        from tkinter import TclError as _TkTclError
    except Exception:
        # Headless fallback: return the tuple verbatim so callers can
        # still treat the result as font spec data.
        return (family, size, weight)

    # Tk 8.6+ on Windows supports ``antialiased=True`` which switches
    # rendering from GDI grayscale to ClearType (sub-pixel AA). Older
    # builds — and the Tk shipped with some Conda Python distributions —
    # reject the kwarg with a ``TclError`` at the ``font create`` layer.
    # We try AA first; on rejection we fall back to plain ``Font`` and
    # on a missing family we fall back to Tk's default face (which it
    # always has installed).
    try:
        f = tkfont.Font(family=family, size=size, weight=weight, antialiased=True)
    except (TypeError, _TkTclError):
        try:
            f = tkfont.Font(family=family, size=size, weight=weight)
        except _TkTclError:
            f = tkfont.Font(size=size, weight=weight)
    cache[key] = f
    return f


# Reset the cache between Tk application instances. Without this, fonts
# created against a destroyed Tk root get reused for the new root and
# Tk raises ``TclError: invalid command name ".!font"`` on first paint.
def _reset_font_cache() -> None:
    _font._cache = {}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Round 28: shared helpers
# ---------------------------------------------------------------------------


def _card(window, parent, padding=(16, 14), style="Tile.TFrame"):
    """Round 28: factory for the 22+ ``ttk.Frame(parent, style="Tile.TFrame",
    padding=...)`` literals scattered across the GUI.

    Returns the inner Frame; callers chain ``.grid(...)`` / ``.pack(...)``
    as before. Default style reproduces the prior behaviour (white
    surface). Pass ``style="Canvas.TFrame"`` etc. for variant cards.
    """
    return window.ttk.Frame(parent, style=style, padding=padding)


def _modal(
    window,
    title: str,
    body_factory,
    footer_factory=None,
    geometry: str = "640x420",
    transient: bool = True,
    grab: bool = True,
    on_close=None,
):
    """Round 28: standard Codex-style modal shell.

    Creates a ``tk.Toplevel``, configures it with the new ``DESIGN``
    tokens, and lays out three rows directly on the Toplevel:

      1. Title bar — ttk-styled Frame containing a TileHeader.TLabel
         with the ``title`` text. Spans the top with horizontal
         padding so the title sits inside a warm-white tile.
      2. Body — the caller-supplied ``body_factory(modal)`` callable
         creates and returns the body widget as a DIRECT child of the
         modal (not nested in an intermediate Frame). This keeps
         ``modal.winfo_children()`` returns simple — tests iterate
         direct children to find the Text widget rendering the
         arguments. The returned widget is gridded on row 1
         (sticky=nsew).
      3. Footer — optional caller-supplied ``footer_factory(modal)``
         callable that builds the button bar (also a direct child of
         modal). If omitted, a single "关闭" Secondary.TButton is
         rendered right-aligned.

    The factory pattern (vs. passing already-created widgets) avoids
    creating throwaway Toplevels just to satisfy Tk's parent
    requirement at construction time. Each factory closure can capture
    any state it needs from the caller (e.g. an exception message,
    keyboard-binding callbacks).

    Returns the ``tk.Toplevel`` so callers can wire ``WM_DELETE_WINDOW``
    or attach keyboard bindings (Y/N/Esc for elicitation modals).
    """
    modal = window.tk.Toplevel(window.root)
    modal.title(title)
    modal.geometry(geometry)
    modal.configure(background=DESIGN["parchment"])
    if transient:
        modal.transient(window.root)
    if grab:
        modal.grab_set()

    # Row 0 — title bar (ttk-styled Frame + TileHeader label).
    title_bar = window.ttk.Frame(modal, style="Tile.TFrame", padding=(24, 16, 24, 8))
    title_bar.grid(row=0, column=0, sticky="ew")
    window.ttk.Label(
        title_bar, text=title, style="TileHeader.TLabel",
    ).pack(anchor="w")

    modal.columnconfigure(0, weight=1)
    modal.rowconfigure(1, weight=1)  # body row stretches with the Toplevel

    # Row 1 — body (direct child of modal).
    body = body_factory(modal)
    body.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 8))

    # Row 2 — footer.
    if footer_factory is not None:
        footer = footer_factory(modal)
        footer.grid(row=2, column=0, sticky="e", padx=24, pady=(8, 16))
    else:
        # Default: single 关闭 button, right-aligned.
        bar = window.ttk.Frame(modal, style="Tile.TFrame")
        window.ttk.Button(
            bar, text="关闭", style="Secondary.TButton", command=modal.destroy,
        ).pack(side="right")
        bar.grid(row=2, column=0, sticky="e", padx=24, pady=(8, 16))

    if on_close is not None:
        modal.protocol("WM_DELETE_WINDOW", lambda: (on_close(), modal.destroy()))

    return modal


def _scope_storage_to_session(config, session: ChatSession) -> None:
    session_storage = session.root / "papers"
    config.storage.paper_library_dir = session_storage
    config.storage.metadata_dir = session.root / "metadata"
    config.storage.cache_dir = session.root / "cache"
    config.storage.paper_library_dir.mkdir(parents=True, exist_ok=True)
    config.storage.metadata_dir.mkdir(parents=True, exist_ok=True)
    config.storage.cache_dir.mkdir(parents=True, exist_ok=True)
    if config.parsing.paddleocr.cache_dir is None:
        config.parsing.paddleocr.cache_dir = session.root / "ocr-cache"


def _load_tk():
    try:
        import tkinter as tk
        from tkinter import ttk
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Tkinter is not available in this Python environment. "
            "Install/use a Python build with Tk support, then run littrace-window again."
        ) from exc
    return tk, ttk


# ---------------------------------------------------------------------------
# Round 20: strict Codex surface helpers for littrace-window.
# ---------------------------------------------------------------------------


def _resolve_window_config(config):
    """Round 20: mirror of ``tui._resolve_tui_config``.

    The Window is now a strict Codex App Server surface — same rules
    as the TUI: explicit ``LITTRACE_AGENT_RUNTIME=legacy`` is a hard
    error, inherited LEGACY mode is overwritten with a warning, and
    ``fallback_to_legacy`` is forced off so the route layer cannot
    silently call legacy chat.
    """
    explicit_env = os.environ.get("LITTRACE_AGENT_RUNTIME", "").strip().lower()
    if explicit_env == "legacy":
        raise SystemExit(
            "littrace-window requires the Codex App Server. Unset "
            "LITTRACE_AGENT_RUNTIME or set it to 'codex_app_server'."
        )
    from littrace.config import AgentRuntimeMode

    if config.agent_runtime.mode == AgentRuntimeMode.LEGACY:
        config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    config.agent_runtime.fallback_to_legacy = False
    return config


async def _codex_window_startup_preflight(config, session) -> None:
    """Round 20: probe Codex reachability before showing the Tk window.

    Awaits the shared :func:`littrace.tui._codex_startup_preflight`
    (which already handles the timeout override and the
    ``AppServerError`` → ``CodexStartupError`` translation) and adds
    a Window-specific surface for the missing-binary case so the
    operator sees ``littrace-window`` in the remediation steps
    instead of ``littrace-tui``.

    Raises :class:`littrace.tui.CodexStartupError` on missing binary /
    failed handshake. The caller renders the error as a Tk modal.
    """
    import shutil as _shutil
    from littrace.tui import (
        CodexStartupError,
        _codex_startup_preflight,
        _remediation_for_app_server_error,
    )
    from littrace.codex_runtime.errors import AppServerError

    command = list(config.agent_runtime.codex_command or ["codex", "app-server"])
    if _shutil.which(command[0]) is None:
        raise CodexStartupError(
            f"Codex CLI 未找到: '{command[0]}' 不在 PATH。",
            remediation=[
                "安装 Codex CLI: `npm install -g @openai/codex`",
                "或者把 codex 可执行文件加到 PATH",
                "或者在 config.yaml 里把 agent_runtime.codex_command 改成绝对路径",
                "然后重新运行 littrace-window",
            ],
        )
    try:
        # The TUI preflight already owns the
        # ``LITTRACE_CODEX_STARTUP_TIMEOUT_SECONDS`` override and the
        # ``AppServerError`` → ``CodexStartupError`` translation. The
        # previous implementation called it without ``await`` so the
        # coroutine was silently dropped on the floor and the
        # handshake never ran.
        await _codex_startup_preflight(config, session)
    except CodexStartupError:
        raise
    except AppServerError as exc:
        # Defence in depth: if a future caller bypasses the TUI
        # preflight (e.g. a custom ``client_factory`` short-circuits
        # it), the Tk modal still gets a structured error envelope
        # instead of a raw ``AppServerError``.
        raise CodexStartupError(
            f"Codex App Server 握手失败: {exc}",
            remediation=_remediation_for_app_server_error(str(exc)),
        ) from exc


def _make_window_elicitation_handler(window):
    """Round 20: install a Tk-thread-aware elicitation handler.

    Mirrors ``tui._make_elicitation_handler`` but schedules UI updates
    via ``root.after`` because the handler runs on the Codex reader
    loop's asyncio loop, not the Tk thread.
    """
    import asyncio as _asyncio

    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        loop = _asyncio.get_running_loop()
        future: _asyncio.Future[dict[str, Any]] = loop.create_future()
        request_id = (
            params.get("serverRequestId")
            or params.get("requestId")
            or id(future)
        )
        window.root.after(
            0,
            lambda: window._open_elicitation_modal(int(request_id) if isinstance(request_id, int) else id(future), params, future),
        )
        return await future

    return handler


def _open_elicitation_modal(window, request_key: int, params: dict[str, Any], future) -> None:
    """Round 20: render a Tk modal for an MCP elicitation request.

    The Codex App Server reader loop awaits the future; the button
    callbacks resolve it via ``call_soon_threadsafe`` so cross-loop
    safety matches ``tui._resolve_approval_future``.
    """
    from littrace.tui import (
        _DECLINE_RESPONSE,
        _ACCEPT_RESPONSE,
        _CANCEL_RESPONSE,
        _parse_elicitation_payload,
    )

    server, tool, arguments = _parse_elicitation_payload(params)
    title = f"Codex 请求: {server}.{tool}"
    body = json.dumps(arguments, indent=2, ensure_ascii=False)

    def resolve(payload: dict[str, Any]) -> None:
        # Look up the modal via the registry — the buttons resolve
        # before ``_modal`` returns, so we can't capture ``modal`` in
        # this closure. ``_approval_roots`` is set immediately after
        # ``_modal`` returns and is idempotent on ``pop``.
        modal_ref = window._approval_roots.pop(request_key, None)
        if modal_ref is not None:
            modal_ref.destroy()
        fut_loop = getattr(future, "_loop", None)
        if fut_loop is not None:
            # The Tk main thread has no running asyncio loop, so
            # ``asyncio.get_running_loop()`` raises ``RuntimeError``
            # here. The future is owned by the Codex reader loop on a
            # worker thread — the only safe way to wake it is
            # ``call_soon_threadsafe``. We still guard with a try
            # block for the case where the future is on the same loop
            # (e.g. tests that drive the handler from the Tk thread
            # directly).
            try:
                import asyncio as _asyncio
                running_loop = _asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is not fut_loop:
                fut_loop.call_soon_threadsafe(future.set_result, payload)
                return
        future.set_result(payload)

    # Round 28: body + footer are factory closures so ``_modal`` can
    # build the outer frame once and pass it as the parent. Avoids
    # throwaway Toplevels.
    def make_body(parent):
        widget = window.tk.Text(
            parent, wrap="word", height=20, width=80, font=_font("body"),
        )
        widget.insert("1.0", body)
        widget.configure(state="disabled")
        return widget

    def make_footer(parent):
        bar = window.ttk.Frame(parent, style="Tile.TFrame")
        window.ttk.Button(
            bar, text="批准 (Y)", style="Primary.TButton",
            command=lambda: resolve(_ACCEPT_RESPONSE),
        ).pack(side="right", padx=(4, 0))
        window.ttk.Button(
            bar, text="拒绝 (N)", style="Secondary.TButton",
            command=lambda: resolve(_DECLINE_RESPONSE),
        ).pack(side="right", padx=(4, 0))
        window.ttk.Button(
            bar, text="取消 (Esc)", style="Secondary.TButton",
            command=lambda: resolve(_CANCEL_RESPONSE),
        ).pack(side="right", padx=(4, 0))
        return bar

    modal = _modal(
        window, title, make_body,
        footer_factory=make_footer,
        geometry="720x520",
    )

    # Keyboard shortcuts — bind on the modal (focus is grabbed).
    modal.bind("<Escape>", lambda _event: resolve(_CANCEL_RESPONSE))
    modal.bind("<y>", lambda _event: resolve(_ACCEPT_RESPONSE))
    modal.bind("<Y>", lambda _event: resolve(_ACCEPT_RESPONSE))
    modal.bind("<n>", lambda _event: resolve(_DECLINE_RESPONSE))
    modal.bind("<N>", lambda _event: resolve(_DECLINE_RESPONSE))
    window._approval_roots[request_key] = modal


def _show_app_server_error_modal(window, exc) -> None:
    """Round 20 + 28: modal that surfaces Codex App Server failure + remediation.

    Round 28: rendered through the shared ``_modal`` helper so the
    shell uses the new ``DESIGN`` tokens (TileHeader title, ttk-styled
    outer) instead of a raw ``tk.Label``/``tk.Frame`` gray window.
    """
    from littrace.tui import _remediation_for_app_server_error

    body_text = (
        f"{exc}\n\n修复步骤:\n"
        + "\n".join(f"- {step}" for step in _remediation_for_app_server_error(str(exc)))
    )

    def make_body(parent):
        widget = window.tk.Text(parent, wrap="word", height=18, width=70, font=_font("body"))
        widget.insert("1.0", body_text)
        widget.configure(state="disabled")
        return widget

    _modal(window, "Codex App Server 不可用", make_body, geometry="640x420")


def _show_startup_error_modal(window, exc) -> None:
    """Round 20 + 28: full-screen modal for Codex App Server startup failures.

    Surfaces the exception and any ``remediation`` list attached to the
    ``CodexStartupError`` so the operator can act without consulting the
    README. Round 28: rendered through ``_modal`` so the shell matches
    the rest of the popup surfaces.
    """
    message = str(exc)
    remediation = getattr(exc, "remediation", None) or []
    body_lines = [f"原因: {message}", ""]
    if remediation:
        body_lines.append("修复步骤:")
        body_lines.extend(f"- {step}" for step in remediation)
    body_text = "\n".join(body_lines) + "\n"

    def make_body(parent):
        widget = window.tk.Text(parent, wrap="word", height=20, width=80, font=_font("body"))
        widget.insert("1.0", body_text)
        widget.configure(state="disabled")
        return widget

    # The OK button must also tear down the root window — the modal is
    # shown because the GUI cannot start, so closing it without
    # destroying root leaves a hung Tk process.
    _modal(
        window,
        "LitTrace Window 无法连接到 Codex App Server",
        make_body,
        geometry="720x500",
        on_close=window.root.destroy,
    )


# Patch helpers onto the LitTraceWindow class so existing call sites
# stay unchanged. Round 20 only adds methods; existing __init__/run
# code paths now invoke these.
LitTraceWindow._open_elicitation_modal = lambda self, request_key, params, future: _open_elicitation_modal(
    self, request_key, params, future
)
LitTraceWindow._make_window_elicitation_handler = lambda self: _make_window_elicitation_handler(self)
LitTraceWindow._show_app_server_error_modal = lambda self, exc: _show_app_server_error_modal(self, exc)
LitTraceWindow._show_startup_error_modal = lambda self, exc: _show_startup_error_modal(self, exc)


if __name__ == "__main__":
    main()
