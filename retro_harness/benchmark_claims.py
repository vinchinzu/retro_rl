"""Compatibility shim for :mod:`retro_harness.benchmark.claims`."""

from retro_harness.benchmark.claims import *  # noqa: F403
from retro_harness.benchmark.claims import (
    ClaimValidationError,
    EvaluationContract,
    PolicyIdentity,
    StartIdentity,
    _audit_from_record,
    _canonicalize_metadata,
    _canonicalize_metadata_value,
    _contract_from_record,
    _normalize_assist_mode,
    _policy_name,
    _record_identity_digest,
    _validate_identity_digest,
    _validate_seed_report_record,
    _validate_serialized_claim_fields,
    policy_identity_for,
    validate_claim,
)

__all__ = [
    "ClaimValidationError",
    "EvaluationContract",
    "PolicyIdentity",
    "StartIdentity",
    "policy_identity_for",
    "validate_claim",
]
