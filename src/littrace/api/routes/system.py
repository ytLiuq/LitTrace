from __future__ import annotations

from fastapi import APIRouter

from littrace.access_layer.browser_sessions import BrowserActStatus, check_browser_act
from littrace.config import load_config
from littrace.config_wizard import ConfigWizardResult, write_config_template
from littrace.log import metrics
from littrace.retrieval.source_router import route_sources

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics")
def get_metrics() -> dict[str, object]:
    return metrics.snapshot()


@router.get("/doctor/browser", response_model=BrowserActStatus)
def doctor_browser() -> BrowserActStatus:
    return check_browser_act(load_config())


@router.post("/config/init", response_model=ConfigWizardResult)
def config_init(path: str = "config.yaml", overwrite: bool = False) -> ConfigWizardResult:
    return write_config_template(path, overwrite=overwrite)


@router.get("/sources/materials-chemistry")
def materials_chemistry_sources(wants_recent: bool = True) -> list[dict[str, object]]:
    return [route.__dict__ for route in route_sources("materials chemistry", wants_recent)]
