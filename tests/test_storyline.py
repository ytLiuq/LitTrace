from littrace.context import add_papers
from littrace.harnesses import check_storyline_claims
from littrace.models import PaperMetadata
from littrace.models import LiteratureWorkspace
from littrace.storyline import (
    build_storyline_from_workspace,
    build_storyline_preview,
    render_structured_storyline_report,
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
    workspace = add_papers(
        LiteratureWorkspace(
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
        ),
        [PaperMetadata(paper_id="p1", title="One Paper", year=2026)],
    )

    claims = build_storyline_from_workspace(workspace)

    assert {claim.claim_type for claim in claims} >= {
        "prior_solution",
        "remaining_limitation",
        "later_response",
    }


def test_storyline_builds_conservative_chain_across_parsed_papers():
    workspace = add_papers(
        LiteratureWorkspace(
            parsed_papers={
                "p1": {
                    "sections": [
                        {
                            "name": "Methods",
                            "text": "The fabrication method defines the sensing film.",
                            "evidence": {"page": 2, "parser": "docling"},
                        }
                    ]
                },
                "p2": {
                    "sections": [
                        {
                            "name": "Challenges",
                            "text": "A remaining limitation is drift during cycling.",
                            "evidence": {"page": 7, "parser": "docling"},
                        }
                    ]
                },
                "p3": {
                    "sections": [
                        {
                            "name": "Discussion",
                            "text": "The encapsulated structure can address drift and improve stability.",
                            "evidence": {"page": 9, "parser": "docling"},
                        }
                    ]
                },
            }
        ),
        [
            PaperMetadata(paper_id="p1", title="Earlier Method", year=2023),
            PaperMetadata(paper_id="p2", title="Observed Limitation", year=2024),
            PaperMetadata(paper_id="p3", title="Later Response", year=2026),
        ],
    )

    claims = build_storyline_from_workspace(workspace)
    chain = [claim for claim in claims if claim.claim_type == "solution_limit_response_chain"]

    assert chain
    assert "不应扩展为未验证的领域共识" in chain[0].claim
    assert check_storyline_claims(chain).passed


def test_structured_storyline_report_includes_references():
    workspace = add_papers(
        LiteratureWorkspace(
            parsed_papers={
                "p1": {
                    "sections": [
                        {
                            "name": "Methods",
                            "text": "The fabrication method defines the sensing film.",
                            "evidence": {"page": 2, "parser": "docling"},
                        },
                        {
                            "name": "Limitations",
                            "text": "A limitation is cycling drift.",
                            "evidence": {"page": 7, "parser": "docling"},
                        },
                    ]
                }
            }
        ),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026, doi="10.1000/story")],
    )

    report = render_structured_storyline_report(workspace)

    assert "前人解决了什么" in report
    assert "引用与访问链接" in report
    assert "https://doi.org/10.1000/story" in report
