from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRoute:
    name: str
    purpose: str
    priority: int


MATERIALS_CHEMISTRY_ROUTES = [
    SourceRoute("crossref", "DOI, journal, publisher, and metadata lookup", 10),
    SourceRoute("openalex", "Open scholarly graph, OA status, metadata enrichment", 20),
    SourceRoute("semantic_scholar", "Citation graph, relevance, related papers", 30),
    SourceRoute("unpaywall", "Open-access PDF discovery by DOI", 40),
    SourceRoute("arxiv", "Recent preprints for computational materials and chemistry", 50),
    SourceRoute("chemrxiv", "Recent chemistry preprints", 60),
    SourceRoute("publisher:wiley", "Wiley publisher landing pages and authorized PDFs", 70),
    SourceRoute("publisher:acs", "ACS publisher landing pages and authorized PDFs", 80),
    SourceRoute("publisher:springer_nature", "Springer Nature and Nature Portfolio links", 90),
    SourceRoute("publisher:rsc", "Royal Society of Chemistry links", 100),
    SourceRoute("publisher:elsevier", "ScienceDirect links", 110),
    SourceRoute("publisher:mdpi", "MDPI open-access articles and PDFs", 120),
]


def route_sources(discipline: str, wants_recent: bool = True) -> list[SourceRoute]:
    normalized = discipline.lower()
    if "material" in normalized or "chem" in normalized:
        routes = MATERIALS_CHEMISTRY_ROUTES.copy()
    else:
        routes = MATERIALS_CHEMISTRY_ROUTES[:5]

    if wants_recent:
        recent_names = {"arxiv", "chemrxiv", "openalex", "semantic_scholar"}
        routes.sort(key=lambda route: (route.name not in recent_names, route.priority))
    else:
        routes.sort(key=lambda route: route.priority)
    return routes
