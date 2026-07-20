from littrace.models import LiteratureWorkspace, PaperMetadata, PaperSearchRequest
from littrace.retrieval.adapters import (
    SourceFailureClass,
    SourceHealth,
    classify_source_exception,
    record_search_provenance,
)


def test_source_adapter_classifies_recoverable_and_invalid_failures():
    assert classify_source_exception(TimeoutError()) == SourceFailureClass.TRANSIENT
    assert classify_source_exception(ValueError()) == SourceFailureClass.INVALID_INPUT


def test_source_provenance_records_request_result_and_health():
    workspace = LiteratureWorkspace()
    events = record_search_provenance(
        workspace,
        PaperSearchRequest(topic="MXene sensor"),
        [PaperMetadata(paper_id="p1", title="Paper", doi="10.1000/example")],
        {"openalex": SourceHealth(source_name="openalex", healthy=True, request_count=1)},
    )

    assert len(events) == 1
    assert events[0].response_hash
    assert events[0].request_fingerprint
    assert workspace.source_records
    assert workspace.context.filters.source_health["openalex"]["healthy"] is True
