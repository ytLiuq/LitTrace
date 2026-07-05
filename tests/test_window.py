import inspect

from littrace import window


def test_window_entrypoint_exists():
    assert callable(window.main)
    assert hasattr(window, "LitTraceWindow")


def test_window_execution_steps_for_research_request():
    steps = window._execution_steps_for_message("我想了解一下薄膜压敏传感阵列的相关文献，请帮我调研一下")

    assert "检索候选文献" in steps
    assert "弹出文献选择" in steps


def test_window_filters_non_user_effective_replies():
    assert not window._is_user_effective_reply("已切换解析模式：使用 OCR。")
    assert window._is_user_effective_reply("我先按主题做了一轮文献检索。")


def test_window_scopes_storage_to_session(tmp_path):
    from littrace.config import LitTraceConfig, StorageConfig
    from littrace.session import create_chat_session

    config = LitTraceConfig(storage=StorageConfig(sessions_dir=tmp_path / "sessions"))
    session = create_chat_session(config)

    window._scope_storage_to_session(config, session)

    assert config.storage.paper_library_dir == session.root / "papers"
    assert config.storage.metadata_dir == session.root / "metadata"
    assert config.storage.cache_dir == session.root / "cache"


def test_window_parse_strategy_button_text_is_specific():
    class Dummy:
        parse_strategy = "text_only"

    assert window.LitTraceWindow._parse_strategy_button_text(Dummy()) == "文献解析模式：文字层"


def test_window_input_uses_multiline_text_widget():
    source = window.LitTraceWindow._build_layout.__code__.co_names

    assert "Text" in source
    assert "Entry" not in source


def test_window_chat_uses_bubble_tags_without_role_labels():
    assert window._chat_bubble_tag("user") == "bubble_user"
    assert window._chat_bubble_tag("assistant") == "bubble_assistant"
    assert window._chat_bubble_tag("system") == "bubble_system"
    assert "role" not in window.LitTraceWindow._append_message.__code__.co_consts


def test_window_session_history_is_clickable():
    source = inspect.getsource(window.LitTraceWindow._refresh_session_history)

    assert "tag_bind" in source
    assert "_switch_session" in source
    assert "hand2" in source


def test_window_output_text_widgets_have_copy_bindings():
    build_source = inspect.getsource(window.LitTraceWindow._build_layout)
    source = inspect.getsource(window.LitTraceWindow._configure_copy_bindings)
    class_source = inspect.getsource(window.LitTraceWindow)

    assert "state=self.tk.DISABLED" not in build_source
    assert "<Command-c>" in source
    assert "<Control-c>" in source
    assert "_copy_event" in source
    assert "_copy_text_selection" in class_source
    assert "复制" in source
    assert "_readonly_text_key" in class_source


def test_window_has_user_confirmation_popup_for_cloudflare():
    class_source = inspect.getsource(window.LitTraceWindow)

    assert "_show_user_confirmation_popup" in class_source
    assert "请完成浏览器中的真人验证" in class_source
    assert "requires_user_confirmation" in class_source
