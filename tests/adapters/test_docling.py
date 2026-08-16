"""Docling table-extraction test (real parser function, no mocking)."""

from __future__ import annotations

import pytest

from littrace.ocr.docling_adapter import _tables_from_docling_dict


pytestmark = pytest.mark.adapters


def test_docling_tables_extract_structured_cells():
    tables = _tables_from_docling_dict(
        {
            "tables": [
                {
                    "caption": "Performance",
                    "data": {
                        "grid": [
                            [{"text": "Material"}, {"text": "Gauge factor"}],
                            [{"text": "PDMS/CNT"}, {"text": "12.5"}],
                        ]
                    },
                }
            ]
        },
        "p1",
    )

    assert tables[0].caption == "Performance"
    assert tables[0].cells[0]["text"] == "Material"
    assert tables[0].cells[3]["text"] == "12.5"
