from __future__ import annotations

import re

from littrace.models import PaperMetadata
from littrace.sentinel.state import Watchlist


def _year_in_range(paper: PaperMetadata, watchlist: Watchlist) -> bool:
    """Round 17: honour ``year_min`` AND ``year_max`` when scoring.
    ``year_max`` is optional (older watchlists don't have it) so we
    fall back to "no upper bound" when it's ``None``.
    """
    if not paper.year:
        return False
    if paper.year < watchlist.year_min:
        return False
    if watchlist.year_max is not None and paper.year > watchlist.year_max:
        return False
    return True


def score_novelty(paper: PaperMetadata, watchlist: Watchlist) -> float:
    score = 0.2
    topic_words = set(re.findall(r"[A-Za-z0-9]+", watchlist.topic.lower()))
    text = " ".join(
        [paper.title or "", paper.abstract or "", " ".join(paper.authors or [])]
    ).lower()
    matches = sum(1 for word in topic_words if word and word in text)
    score += min(0.5, 0.1 * matches)
    if _year_in_range(paper, watchlist):
        score += 0.2
    if paper.citation_count:
        score += min(0.1, paper.citation_count / 1000.0)
    return round(min(1.0, score), 3)


def score_relevance(paper: PaperMetadata, watchlist: Watchlist) -> float:
    text = " ".join([paper.title or "", paper.abstract or ""]).lower()
    query_terms = watchlist.query_variants or [watchlist.topic]
    hits = 0
    for query in query_terms:
        for token in re.findall(r"[A-Za-z0-9]+", query.lower()):
            if token and token in text:
                hits += 1
    base = 0.25 + min(0.55, hits * 0.08)
    if _year_in_range(paper, watchlist):
        base += 0.1
    return round(min(1.0, base), 3)
