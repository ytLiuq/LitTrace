from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.storyline_review import review_storyline


def test_storyline_review_reports_missing_claims():
    report = review_storyline(LiteratureWorkspace())

    assert not report.passed
    assert report.warnings


def test_storyline_review_accepts_grounded_claims():
    workspace = add_papers(
        LiteratureWorkspace(
            parsed_papers={
                "p1": {
                    "sections": [
                        {
                            "name": "Methods",
                            "text": "The fabrication method defines the sensing film.",
                            "evidence": {"page": 2},
                        }
                    ]
                }
            }
        ),
        [PaperMetadata(paper_id="p1", title="Paper", year=2026)],
    )

    report = review_storyline(workspace)

    assert report.claim_count >= 1
