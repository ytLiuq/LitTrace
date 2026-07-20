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


def test_parse_chat_intent_handles_auto_replan():
    intent = parse_chat_intent("请自动重规划并多轮反驳修订")

    assert "autonomous_review" in intent.actions
    assert intent.auto_replan


def test_parse_chat_intent_handles_text_only_parse_strategy():
    intent = parse_chat_intent("请只看文字层解析PDF，不要OCR")

    assert "parse" in intent.actions
    assert intent.parse_strategy == "text_only"


def test_parse_chat_intent_treats_research_request_as_search():
    intent = parse_chat_intent("我想了解一下薄膜压敏传感阵列的相关文献，请帮我调研一下")

    assert "search" in intent.actions
    assert intent.topic == "薄膜压敏传感阵列"


def test_parse_chat_intent_removes_research_noise_words():
    intent = parse_chat_intent("我想了解一下碳基PDMS柔性薄膜传感器长时间受压漂移的研究，请帮我详细调研一下")

    assert "search" in intent.actions
    assert intent.topic == "碳基PDMS柔性薄膜传感器长时间受压漂移"


def test_parse_chat_intent_understands_natural_storyline_request():
    intent = parse_chat_intent("这些文章放在一起到底走了哪几条路线？")

    assert "storyline" in intent.actions


def test_parse_chat_intent_understands_literature_survey_phrase():
    intent = parse_chat_intent("这个方向有什么文献和相关工作？")

    assert "search" in intent.actions


def test_parse_chat_intent_marks_ambiguous_short_command():
    intent = parse_chat_intent("继续")

    assert intent.ambiguous
    assert intent.confidence < 0.72
    assert intent.clarification_questions
