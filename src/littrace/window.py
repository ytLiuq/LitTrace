from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from pathlib import Path

from littrace.agent_runtime import handle_agent_chat
from littrace.auto_resume import auto_resume_downloaded_pdfs
from littrace.config import load_config
from littrace.intent import parse_chat_intent
from littrace.shell_controller import ShellController, ShellEvent
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


# Linear-dominant palette with materials-chemistry muted-teal accent.
# Tones map to Linear's neutral ladder; primary is `#3a8a8c` (muted teal,
# echoes solution-chemistry, not playful). Serif path is opt-in for
# paper titles via `PaneTitle.TLabel`.
DESIGN = {
    "primary":             "#3a8a8c",
    "primary_hover":       "#4ea3a5",
    "primary_focus":       "#4ea3a5",
    "primary_on_dark":     "#3a8a8c",
    "ink":                 "#0b0c0e",
    "ink_muted":           "#5c6068",
    "ink_subtle":          "#a4a7ad",
    "canvas":              "#fbfbfc",
    "parchment":           "#f5f6f6",
    "pearl":               "#ffffff",
    "surface_1":           "#ffffff",
    "surface_2":           "#f7f8f8",
    "hairline":            "#d8d9dc",
    "black":               "#0b0c0e",
    "dark_tile":           "#0f1011",
    "on_dark":             "#f7f8f8",
    "body_muted":          "#a4a7ad",
    "accent_coral":        "#cc785c",
    "accent_teal_subtle":  "#e6efee",
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


class LitTraceWindow:
    def __init__(self) -> None:
        self.tk, self.ttk = _load_tk()
        self.config = load_config()
        # Route chat through ``ShellController`` so service reuse and
        # the same event payload that powers ``littrace-qt`` work here
        # too. The previous direct ``handle_agent_chat`` background
        # thread called ``self.root.after(0, apply_response)`` from the
        # worker thread, which Tk rejects with ``RuntimeError: main
        # thread is not in main loop`` -- so the assistant reply never
        # made it to the chat widget. ``ShellController`` keeps the
        # asyncio worker on its own thread, and the bus handlers below
        # re-post every event onto the Tk main loop via ``root.after``.

        self.controller = ShellController(self.config)
        self.controller.start()
        self._wire_controller_to_widgets()
        self.session = self.controller.session
        self.workspace = self.controller.workspace
        self.context_visible = True
        self.last_download_plan: DownloadPlan | None = None
        self.parse_strategy = "text_only"
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
        self.root.title("LitTrace")
        self.root.geometry("1280x820")
        self.root.minsize(900, 600)
        self.root.configure(background=DESIGN["parchment"])
        # Start the cross-thread event pump now that ``self.root`` exists.
        self._start_controller_pump()

        self._configure_styles()
        self._build_layout()
        self._configure_copy_bindings()
        self._refresh_context()
        self._refresh_ocr_buttons()
        self._refresh_session_history()
        self._refresh_rag_panel()

    def run(self) -> None:
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
            padding=(18, 9),
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
            padding=(15, 8),
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
        self.trace_frame.rowconfigure(3, weight=2)
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
        self.session_history_text.tag_configure("session_current", background=DESIGN["accent_teal_subtle"])
        self.session_history_text.grid(row=3, column=0, sticky="nsew")

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
        self.chat_text.tag_configure(
            "bubble_user",
            foreground=DESIGN["ink"],
            background=DESIGN["accent_teal_subtle"],
            font=_font("body"),
            justify=self.tk.RIGHT,
            lmargin1=170,
            lmargin2=170,
            rmargin=14,
            spacing1=6,
            spacing3=12,
        )
        self.chat_text.tag_configure(
            "bubble_assistant",
            foreground=DESIGN["ink"],
            background=DESIGN["pearl"],
            font=_font("body"),
            justify=self.tk.LEFT,
            lmargin1=14,
            lmargin2=14,
            rmargin=170,
            spacing1=6,
            spacing3=12,
        )
        self.chat_text.tag_configure(
            "bubble_system",
            foreground=DESIGN["ink_muted"],
            background=DESIGN["canvas"],
            font=_font("caption"),
            justify=self.tk.CENTER,
            lmargin1=80,
            lmargin2=80,
            rmargin=80,
            spacing1=4,
            spacing3=10,
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

        # Wrap input in an outer host frame so the input area visually separates
        # from the chat output above (different background, hairline border).
        input_host = self.ttk.Frame(chat_frame, style="InputHost.TFrame")
        input_host.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        input_host.columnconfigure(0, weight=1)
        input_frame = self.ttk.Frame(input_host, style="InputPearl.TFrame", padding=(14, 10))
        input_frame.grid(row=0, column=0, sticky="ew", padx=1, pady=(1, 1))
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
        self.ttk.Button(input_frame, text="发送", style="Primary.TButton", command=self._send).grid(
            row=0, column=1, padx=(12, 0)
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
        self.input_entry.delete("1.0", self.tk.END)
        if message in {"/quit", "/exit"}:
            self.root.destroy()
            return
        # Hand off to ``ShellController``; the reply comes back through
        # ``ShellController`` emits ``message_appended`` for the user
        # role as well; the bus handler below is what actually writes
        # it into the chat widget. Doing it here too would duplicate
        # the message; using the bus as the single source of truth
        # keeps the writer in exactly one place.
        self.controller.submit_user_message(message)

    def _wire_controller_to_widgets(self) -> None:
        # Bridge every controller event onto the Tk main loop.
        # ``ShellEventBus`` emits on the controller's asyncio worker
        # thread. ``self.root.after(0, callable)`` from a non-main
        # thread turns out to be unreliable — the idle handler is
        # dropped during the cross-thread call in this test setup
        # (``update()`` does not always flush after-events posted from
        # another thread). Use a Python ``queue.Queue`` and have the
        # Tk main thread ``after()`` itself poll it: the call to
        # ``self.root.after(100, _drain)`` *is* made from the Tk
        # constructor so it lands on the right thread, and each
        # iteration reads whatever the worker posted in the meantime.
        import queue as _queue

        self._event_queue: _queue.Queue = _queue.Queue()

        def _drain() -> None:
            while True:
                try:
                    callable_, args = self._event_queue.get_nowait()
                except Exception:
                    break
                try:
                    callable_(*args)
                except Exception:
                    # Single event failures must not stop the pump.
                    pass
            self.root.after(100, _drain)

        def post(callable_):
            def _wrapper(event: ShellEvent) -> None:
                self._event_queue.put((callable_, (event,)))

            return _wrapper

        controller = self.controller

        def on_message(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_MESSAGE_APPENDED:
                role = event.payload.get("role", "system")
                text = event.payload.get("text", "")
                self._append_message(role, text)
                if role == "assistant":
                    action = event.payload.get("action", "")
                    if action:
                        self.status_var.set(
                            f"Action: {action} | Session: {self.session.session_id}"
                        )

        def on_status(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_STATUS_CHANGED:
                self.status_var.set(event.payload.get("text", ""))

        def on_workspace(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_WORKSPACE_REFRESHED:
                try:
                    self._refresh_context()
                    self._refresh_ocr_buttons()
                    self._refresh_context_popup()
                    self._refresh_session_history()
                    self._refresh_rag_panel()
                except Exception:
                    pass

        def on_error(event: ShellEvent) -> None:
            if event.kind == controller.EVENT_ERROR:
                self._append_message("system", f"⚠️ {event.payload.get('message', 'error')}")
                self.status_var.set("错误")

        controller.bus.subscribe(post(on_message))
        controller.bus.subscribe(post(on_status))
        controller.bus.subscribe(post(on_workspace))
        controller.bus.subscribe(post(on_error))

        # The drain loop itself is started by ``_start_controller_pump``
        # once ``self.root`` exists.

    def _start_controller_pump(self) -> None:
        # Called from ``__init__`` after ``self.root = self.tk.Tk()`` so
        # the idle task is scheduled on the Tk main thread.
        def _drain() -> None:
            while True:
                try:
                    callable_, args = self._event_queue.get_nowait()
                except Exception:
                    break
                try:
                    callable_(*args)
                except Exception:
                    pass
            self.root.after(100, _drain)

        self.root.after(100, _drain)

    def _handle_message_thread(self, message: str) -> None:
        # Legacy background-thread chat handler kept only for source
        # compatibility with subclasses that override ``_send`` and
        # call into here directly. ``_send`` itself no longer uses
        # this path — it hands the message to ``ShellController``.
        return None

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
        self.chat_text.insert(self.tk.END, f"{text}\n\n", (tag,))
        self.chat_text.configure(state="disabled")
        self.chat_text.see(self.tk.END)

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

    def _refresh_session_history(self) -> None:
        self.session_history_text.delete("1.0", self.tk.END)
        summaries = list_chat_sessions(self.config, limit=8)
        if not summaries:
            self.session_history_text.insert(self.tk.END, "暂无历史 session。")
        for index, item in enumerate(summaries):
            current = "当前 " if item.session_id == self.session.session_id else ""
            tag = f"session_{index}"
            start = self.session_history_text.index(self.tk.INSERT)
            self.session_history_text.insert(
                self.tk.END,
                f"{current}{item.topic}\n{item.updated_at} | 消息 {item.message_count} | 文献 {item.paper_count}\n{item.session_id}\n\n",
            )
            end = self.session_history_text.index(self.tk.INSERT)
            self.session_history_text.tag_add(tag, start, end)
            self.session_history_text.tag_configure(
                tag,
                foreground=DESIGN["primary"]
                if item.session_id != self.session.session_id
                else DESIGN["ink"],
                background=DESIGN["accent_teal_subtle"]
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
        self._render_session_messages()
        self._refresh_context()
        self._refresh_context_popup()
        self._refresh_session_history()
        self._render_planned_trace([f"已切换到历史 session：{session_id}"])
        self.status_var.set(f"Session: {self.session.session_id}")

    def _render_session_messages(self) -> None:
        self.chat_text.delete("1.0", self.tk.END)
        for record in state_store_from_config(self.config).list_chat_messages(self.session.session_id):
            role = "你" if record.get("role") == "user" else "LitTrace"
            content = record.get("content_json") or record.get("content_text")
            text = _message_text(content)
            if text and (role == "你" or _is_user_effective_reply(text)):
                bubble_role = "user" if role == "你" else "assistant"
                self.chat_text.insert(
                    self.tk.END, f"{text}\n\n", (_chat_bubble_tag(bubble_role),)
                )
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


def _font(token: str) -> tuple[str, int, str]:
    # Inter is the body face (Linear / Notion / Stripe family). Helvetica Neue
    # is the platform fallback. Cormorant Garamond is the Claude-derived serif
    # used ONLY for `display` and `title_serif` (pane headers + paper rows).
    # Tk falls through the comma-joined family string when the first face is
    # not installed.
    sans = "Inter"
    sans_fallback = "Helvetica Neue"
    serif = "Cormorant Garamond"
    serif_fallback = "Iowan Old Style"
    fonts = {
        "display":      (f"{serif}, {serif_fallback}", 22, "normal"),
        "title_serif":  (f"{serif}, {serif_fallback}", 18, "normal"),
        "tagline":      (f"{sans}, {sans_fallback}",   13, "bold"),
        "body":         (f"{sans}, {sans_fallback}",   13, "normal"),
        "body_strong":  (f"{sans}, {sans_fallback}",   13, "bold"),
        "caption":      (f"{sans}, {sans_fallback}",   11, "normal"),
        "button":       (f"{sans}, {sans_fallback}",   12, "normal"),
        "button_utility":(f"{sans}, {sans_fallback}",  11, "normal"),
        "fine":         (f"{sans}, {sans_fallback}",   10, "normal"),
        "nav":          (f"{sans}, {sans_fallback}",   10, "normal"),
    }
    return fonts.get(token, fonts["body"])


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


if __name__ == "__main__":
    main()
