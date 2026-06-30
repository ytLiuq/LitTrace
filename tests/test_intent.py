from littrace.intent import parse_chat_intent


def test_parse_chat_intent_handles_composite_instruction():
    intent = parse_chat_intent("检索 2024 年后的 AFM 和 ACS Nano，先别下载，生成性能对比表")

    assert "search" in intent.actions
    assert "table" in intent.actions
    assert "download" not in intent.actions
    assert intent.skip_download
    assert intent.year_min == 2024
    assert "Advanced Functional Materials" in intent.journals
    assert "ACS Nano" in intent.journals
