from __future__ import annotations

from fastapi import APIRouter

from littrace.access_layer.browser_sessions import BrowserActStatus, check_browser_act
from littrace.config import load_config
from littrace.config_wizard import ConfigWizardResult, write_config_template
from littrace.log import metrics
from littrace.rag_jobs import run_pending_embedding_jobs
from littrace.artifact_ops import reconcile_session_artifacts
from littrace.rag_ops import (
    RagDoctorReport,
    RagJobsStatusReport,
    build_rag_jobs_status_report,
    requeue_dead_rag_jobs,
    run_rag_doctor,
)
from littrace.retrieval.source_router import route_sources

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics")
def get_metrics() -> dict[str, object]:
    return metrics.snapshot()


@router.get("/doctor/browser", response_model=BrowserActStatus)
def doctor_browser() -> BrowserActStatus:
    return check_browser_act(load_config())


@router.get("/doctor/rag", response_model=RagDoctorReport)
def doctor_rag() -> RagDoctorReport:
    return run_rag_doctor(load_config())


@router.get("/rag/jobs", response_model=RagJobsStatusReport)
def rag_jobs_status(
    status: str | None = None,
    session_id: str | None = None,
    limit: int = 20,
) -> RagJobsStatusReport:
    return build_rag_jobs_status_report(
        load_config(),
        status=status,
        session_id=session_id,
        limit=limit,
    )


@router.post("/rag/jobs/run", response_model=dict[str, object])
async def rag_jobs_run(limit: int = 20) -> dict[str, object]:
    report = await run_pending_embedding_jobs(load_config(), limit=limit)
    return report.model_dump(mode="json")


@router.post("/rag/reconcile/{session_id}", response_model=dict[str, object])
def rag_reconcile(session_id: str, limit: int = 200) -> dict[str, object]:
    return reconcile_session_artifacts(load_config(), session_id, limit=limit).model_dump(mode="json")


@router.post("/rag/jobs/requeue-dead", response_model=dict[str, object])
def rag_jobs_requeue_dead(
    session_id: str | None = None,
    limit: int = 20,
) -> dict[str, object]:
    return {
        "requeued": requeue_dead_rag_jobs(
            load_config(),
            session_id=session_id,
            limit=limit,
        )
    }


@router.post("/config/init", response_model=ConfigWizardResult)
def config_init(path: str = "config.yaml", overwrite: bool = False) -> ConfigWizardResult:
    return write_config_template(path, overwrite=overwrite)


@router.get("/sources/materials-chemistry")
def materials_chemistry_sources(wants_recent: bool = True) -> list[dict[str, object]]:
    return [route.__dict__ for route in route_sources("materials chemistry", wants_recent)]
