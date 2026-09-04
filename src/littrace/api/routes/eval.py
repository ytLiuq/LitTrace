from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header

from littrace.api.backend import api_app
from littrace.eval_api import (
    EvalMetricReport,
    full_text_metrics_from_workspace,
    parsing_metrics,
    retrieval_metrics,
    storyline_metrics,
)
from littrace.evaluation.golden_eval import GoldenEvalReport, _load_cases, run_golden_eval
from littrace.evaluation.pdf_benchmark import (
    LivePDFBenchmarkReport,
    PDFBenchmarkReport,
    benchmark_pdf_parsing,
    benchmark_single_pdf,
)
from littrace.evaluation.quality_report import QualityReport
from littrace.evaluation.rag_eval import RagEvalReport, run_rag_golden_eval
from littrace.evaluation.retrieval_eval import RetrievalEvalReport, run_retrieval_golden_eval
from littrace.evaluation.task_eval import TaskEvalReport, evaluate_task_runs
from littrace.rerank_learning import RerankLearningReport, learn_rerank_policy_from_golden
from littrace.skill_runner import build_quality_report_skill



router = APIRouter(tags=["evaluation"])


@router.post("/eval/retrieval", response_model=EvalMetricReport)
def eval_retrieval(topic: str | None = None) -> EvalMetricReport:
    config = api_app.load_config()
    return EvalMetricReport(
        run_id="preview",
        topic=topic,
        metrics=retrieval_metrics(workspace=api_app.WORKSPACE, config=config),
    )


@router.post("/eval/pdf-parsing", response_model=EvalMetricReport)
def eval_pdf_parsing(topic: str | None = None) -> EvalMetricReport:
    config = api_app.load_config()
    return EvalMetricReport(
        run_id="preview",
        topic=topic,
        metrics=parsing_metrics(workspace=api_app.WORKSPACE, config=config),
    )


@router.get("/eval/pdf-benchmark", response_model=PDFBenchmarkReport)
def eval_pdf_benchmark(
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> PDFBenchmarkReport:
    config = api_app.load_config()
    from littrace.api.auth import resolve_request_session
    auth = resolve_request_session(config, header_session_id=x_littrace_session_id)
    return benchmark_pdf_parsing(
        api_app.WORKSPACE, config, session_id=auth.session_id
    )


@router.post("/eval/pdf-benchmark/file", response_model=LivePDFBenchmarkReport)
def eval_pdf_benchmark_file(path: str, parse_strategy: str | None = None) -> LivePDFBenchmarkReport:
    return benchmark_single_pdf(Path(path), api_app.load_config(), parse_strategy=parse_strategy)


@router.get("/eval/full-text", response_model=EvalMetricReport)
def eval_full_text() -> EvalMetricReport:
    return EvalMetricReport(
        run_id="preview", metrics=full_text_metrics_from_workspace(api_app.WORKSPACE)
    )


@router.post("/eval/storyline", response_model=EvalMetricReport)
def eval_storyline(topic: str | None = None) -> EvalMetricReport:
    return EvalMetricReport(
        run_id="preview", topic=topic, metrics=storyline_metrics(workspace=api_app.WORKSPACE)
    )


@router.post("/eval/end-to-end", response_model=EvalMetricReport)
def eval_end_to_end(topic: str | None = None) -> EvalMetricReport:
    config = api_app.load_config()
    metrics = {}
    metrics.update(retrieval_metrics(workspace=api_app.WORKSPACE, config=config))
    metrics.update(parsing_metrics(workspace=api_app.WORKSPACE, config=config))
    metrics.update(storyline_metrics(workspace=api_app.WORKSPACE))
    metrics.update(full_text_metrics_from_workspace(api_app.WORKSPACE))
    return EvalMetricReport(run_id="preview", topic=topic, metrics=metrics)


@router.get("/eval/golden", response_model=GoldenEvalReport)
def eval_golden() -> GoldenEvalReport:
    return run_golden_eval(api_app.load_config(), api_app.WORKSPACE)


@router.get("/eval/retrieval-golden", response_model=RetrievalEvalReport)
async def eval_retrieval_golden(live: bool = True) -> RetrievalEvalReport:
    return await run_retrieval_golden_eval(api_app.load_config(), live=live)


@router.get("/eval/rag-golden", response_model=RagEvalReport)
async def eval_rag_golden(top_k: int = 10) -> RagEvalReport:
    return await run_rag_golden_eval(
        api_app.load_config(),
        api_app.WORKSPACE,
        top_k=top_k,
    )


@router.get("/eval/task-golden", response_model=TaskEvalReport)
def eval_task_golden(case_id: str) -> TaskEvalReport:
    config = api_app.load_config()
    cases = [
        case
        for case in _load_cases(config.eval.golden_set_dir)
        if str(case.get("case_id") or case.get("topic") or "") == case_id
    ]
    if not cases:
        return TaskEvalReport(
            case_count=0,
            evidence_grounded_task_success_rate=0.0,
            warnings=[f"Golden task case not found: {case_id}"],
        )
    return evaluate_task_runs(cases, {case_id: api_app.WORKSPACE})


@router.get("/eval/rerank-learn", response_model=RerankLearningReport)
async def eval_rerank_learn(live: bool = True) -> RerankLearningReport:
    return await learn_rerank_policy_from_golden(api_app.load_config(), live=live)


@router.get("/quality", response_model=QualityReport)
def quality(
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> QualityReport:
    config = api_app.load_config()
    from littrace.api.auth import resolve_request_session
    auth = resolve_request_session(config, header_session_id=x_littrace_session_id)
    return build_quality_report_skill(
        config, api_app.WORKSPACE, session_id=auth.session_id
    )
