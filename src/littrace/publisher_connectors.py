from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl

from littrace.citations import best_access_url
from littrace.models import AccessType, LiteratureWorkspace, PaperMetadata


class PublisherAccessRoute(BaseModel):
    paper_id: str
    publisher_family: str
    landing_url: HttpUrl
    pdf_url: HttpUrl | None = None
    requires_login: bool = False
    confidence: float = 0.0
    notes: list[str] = Field(default_factory=list)


class PublisherRouteReport(BaseModel):
    routes: list[PublisherAccessRoute]
    warnings: list[str] = Field(default_factory=list)


PUBLISHER_ALIASES = {
    "acs": ["american chemical society", "acs", "acs nano", "nano letters"],
    "wiley": ["wiley", "advanced materials", "advanced functional materials"],
    "nature": ["nature", "springer nature", "nature portfolio"],
    "mdpi": ["mdpi"],
    "rsc": ["royal society of chemistry", "rsc"],
    "elsevier": ["elsevier", "science direct", "sciencedirect"],
}


def build_publisher_route_report(papers: list[PaperMetadata]) -> PublisherRouteReport:
    routes = [build_publisher_route(paper) for paper in papers]
    warnings = [
        f"{route.paper_id}: publisher family inferred as unknown"
        for route in routes
        if route.publisher_family == "unknown"
    ]
    return PublisherRouteReport(routes=routes, warnings=warnings)


def publisher_routes_for_workspace(workspace: LiteratureWorkspace) -> PublisherRouteReport:
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    return build_publisher_route_report(papers)


def build_publisher_route(paper: PaperMetadata) -> PublisherAccessRoute:
    family = infer_publisher_family(paper)
    landing_url = best_access_url(paper)
    notes: list[str] = []
    confidence = 0.5

    if paper.doi:
        notes.append("DOI landing page is preferred for authorized access.")
        confidence += 0.2
    if paper.source_urls:
        notes.append("Source URLs were supplied by the retrieval layer.")
        confidence += 0.1
    if paper.pdf_url and paper.access_type == AccessType.OPEN_ACCESS:
        notes.append("Open-access PDF URL is available.")
        confidence += 0.2
    if paper.access_type == AccessType.REQUIRES_LOGIN:
        notes.append("Publisher access likely requires user authentication.")

    return PublisherAccessRoute(
        paper_id=paper.paper_id,
        publisher_family=family,
        landing_url=landing_url,
        pdf_url=paper.pdf_url,
        requires_login=paper.access_type == AccessType.REQUIRES_LOGIN,
        confidence=min(confidence, 0.95),
        notes=notes,
    )


def infer_publisher_family(paper: PaperMetadata) -> str:
    haystack = " ".join(
        value
        for value in [paper.publisher, paper.journal, paper.title]
        if value
    ).lower()
    for family, aliases in PUBLISHER_ALIASES.items():
        if any(alias in haystack for alias in aliases):
            return family
    return "unknown"
