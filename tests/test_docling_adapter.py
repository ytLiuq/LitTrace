from littrace.ocr.docling_adapter import markdown_to_sections


def test_markdown_to_sections_preserves_heading_evidence():
    sections = markdown_to_sections(
        "# Title\n\nIntro text.\n\n## Methods\nFabrication method details.",
        "p1",
    )

    assert [section["name"] for section in sections] == ["Title", "Methods"]
    assert sections[1]["evidence"]["parser"] == "docling"
    assert "Fabrication" in sections[1]["text"]
