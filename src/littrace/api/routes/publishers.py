from __future__ import annotations

from fastapi import APIRouter

from littrace.publisher_connectors import PublisherRouteReport, PublisherSearchPlanReport, build_publisher_search_plan, publisher_routes_for_workspace
from littrace.publisher_retrieval import BrowserRetrievalPlan, PublisherEnrichment, PublisherRetrievalResult, build_browser_retrieval_plan, fetch_publisher_search_results, merge_retrieval_result_into_workspace, parse_publisher_article_html
from littrace.supplementary import register_supplementary_links


class _AppProxy:
    def __getattr__(self, name: str):
        from littrace.api import app as api_app

        return getattr(api_app, name)


api_app = _AppProxy()

router = APIRouter()


@router.get("/publishers/routes", response_model=PublisherRouteReport)
def publisher_routes() -> PublisherRouteReport:
    return publisher_routes_for_workspace(api_app.WORKSPACE)


@router.get("/publishers/search-plan", response_model=PublisherSearchPlanReport)
def publisher_search_plan(topic: str) -> PublisherSearchPlanReport:
    return build_publisher_search_plan(topic)


@router.post("/publishers/retrieve", response_model=PublisherRetrievalResult)
async def publisher_retrieve(topic: str, family: str = "acs", merge: bool = False) -> PublisherRetrievalResult:
    plan_report = build_publisher_search_plan(topic, families=[family])
    if not plan_report.plans:
        raise KeyError(f"No publisher search plan for {family}")
    result = await fetch_publisher_search_results(api_app.load_config(), plan_report.plans[0])
    if merge:
        api_app._set_workspace(merge_retrieval_result_into_workspace(api_app.WORKSPACE, result))
    return result


@router.get("/publishers/browser-plan", response_model=BrowserRetrievalPlan)
def publisher_browser_plan(topic: str, family: str = "acs") -> BrowserRetrievalPlan:
    plan_report = build_publisher_search_plan(topic, families=[family])
    if not plan_report.plans:
        raise KeyError(f"No publisher search plan for {family}")
    return build_browser_retrieval_plan(plan_report.plans[0])


@router.post("/publishers/enrich-html", response_model=PublisherEnrichment)
def publisher_enrich_html(html: str, paper_id: str | None = None) -> PublisherEnrichment:
    enrichment = parse_publisher_article_html(html)
    if paper_id and enrichment.supplementary_links:
        api_app._set_workspace(
            register_supplementary_links(
                api_app.WORKSPACE,
                paper_id,
                [str(link) for link in enrichment.supplementary_links],
            )
        )
    return enrichment
