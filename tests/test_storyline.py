from littrace.models import PaperMetadata
from littrace.models import LiteratureWorkspace
from littrace.storyline import (
    build_storyline_from_workspace,
    build_storyline_preview,
    verify_storyline_preview,
)


def test_storyline_preview_is_conservative_with_metadata_only():
    claims = build_storyline_preview(
        [
            PaperMetadata(paper_id="p1", title="Paper 1", year=2025, journal="ACS Nano"),
            PaperMetadata(
                paper_id="p2",
                title="Paper 2",
                year=2026,
                journal="Advanced Functional Materials",
            ),
        ]
    )

    assert claims
    assert "full-text parsing is required" in claims[0].claim
    assert verify_storyline_preview(claims).passed


def test_storyline_from_parsed_text_builds_solution_limit_response_claims():
    workspace = LiteratureWorkspace(
        parsed_papers={
            "p1": {
                "sections": [
                    {
                        "name": "Methods",
                        "text": "The fabrication method improves sensing performance.",
                        "evidence": {"paper_id": "p1", "page": 2, "parser": "docling"},
                    },
                    {
                        "name": "Limitations",
                        "text": "A key limitation is long-term stability.",
                        "evidence": {"paper_id": "p1", "page": 8, "parser": "docling"},
                    },
                    {
                        "name": "Discussion",
                        "text": "The new design can overcome stability challenges.",
                        "evidence": {"paper_id": "p1", "page": 9, "parser": "docling"},
                    },
                ]
            }
        }
    )

    claims = build_storyline_from_workspace(workspace)

    assert {claim.claim_type for claim in claims} >= {
        "prior_solution",
        "remaining_limitation",
        "later_response",
    }
