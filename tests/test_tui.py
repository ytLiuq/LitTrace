from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.tui import render_context_lines, render_ocr_choice_lines, wrap_text


def test_tui_renders_context_and_ocr_choice():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Traceable Paper", year=2026)],
    )

    context_lines = render_context_lines(workspace)
    choice_lines = render_ocr_choice_lines(workspace)

    assert any("Traceable Paper" in line for line in context_lines)
    assert any("只看文字层" in line for line in choice_lines)
    assert any("使用 OCR" in line for line in choice_lines)


def test_tui_wrap_text_preserves_content():
    assert wrap_text("abcdef", 3) == ["abc", "def"]
