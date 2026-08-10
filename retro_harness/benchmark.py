"""Compatibility facade for benchmark claims, runners, and reports.

Canonical claim ownership lives in :mod:`retro_harness.benchmark_claims`;
execution and report serialization live in
:mod:`retro_harness.benchmark_runner`.
"""

from retro_harness.benchmark_claims import (
    ClaimValidationError,
    EvaluationContract,
    PolicyIdentity,
    StartIdentity,
    policy_identity_for,
    validate_claim,
)
from retro_harness.benchmark_runner import (
    BenchmarkAttemptResult,
    BenchmarkCase,
    BenchmarkPolicy,
    BenchmarkRunResult,
    BenchmarkTier,
    IdlePolicy,
    RandomPolicy,
    SEED_ROBUSTNESS_SCHEMA_VERSION,
    SeedAttemptResult,
    SeedRobustnessConfig,
    SeedRobustnessReport,
    SeedValue,
    run_benchmark,
    run_seed_robustness,
    write_seed_robustness_report,
    zero_action_for_env,
)

__all__ = [
    "BenchmarkAttemptResult",
    "BenchmarkCase",
    "BenchmarkPolicy",
    "BenchmarkRunResult",
    "BenchmarkTier",
    "ClaimValidationError",
    "EvaluationContract",
    "IdlePolicy",
    "PolicyIdentity",
    "RandomPolicy",
    "SEED_ROBUSTNESS_SCHEMA_VERSION",
    "SeedAttemptResult",
    "SeedRobustnessConfig",
    "SeedRobustnessReport",
    "SeedValue",
    "StartIdentity",
    "policy_identity_for",
    "run_benchmark",
    "run_seed_robustness",
    "validate_claim",
    "write_seed_robustness_report",
    "zero_action_for_env",
]
