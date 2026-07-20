"""Evidence extraction, synthesis, and report composition."""
from littrace.evidence.claims import (
    record_claim_verification,
    register_evidence,
    verify_structured_claim,
    workspace_evidence_registry,
)

__all__ = [
    "record_claim_verification",
    "register_evidence",
    "verify_structured_claim",
    "workspace_evidence_registry",
]
