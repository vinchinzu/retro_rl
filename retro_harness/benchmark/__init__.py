"""Benchmark claims, runners, and seed-robustness reports.

Compatibility shims keep the historical import paths working:

* :mod:`retro_harness.benchmark_claims`
* :mod:`retro_harness.benchmark_runner`
"""

from __future__ import annotations

from typing import Any

from retro_harness.benchmark.claims import (
    ClaimValidationError,
    EvaluationContract,
    PolicyIdentity,
    StartIdentity,
    policy_identity_for,
    validate_claim,
)
from retro_harness.benchmark.runner import (
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
    "SEED_CAMPAIGN_SCHEMA_VERSION",
    "SEED_ROBUSTNESS_SCHEMA_VERSION",
    "SeedAttemptResult",
    "SeedCampaignContractMismatch",
    "SeedCampaignError",
    "SeedCampaignInfraError",
    "SeedCampaignLedger",
    "SeedCampaignResult",
    "SeedCampaignRunner",
    "SeedExecutionRow",
    "SeedExecutionStatus",
    "SeedRobustnessConfig",
    "SeedRobustnessReport",
    "SeedValue",
    "StartIdentity",
    "atomic_write_json",
    "atomic_write_text",
    "config_contract_digest",
    "policy_identity_for",
    "run_benchmark",
    "run_seed_campaign",
    "run_seed_robustness",
    "validate_claim",
    "write_seed_robustness_report",
    "zero_action_for_env",
]

_CAMPAIGN_EXPORTS = {
    "SEED_CAMPAIGN_SCHEMA_VERSION",
    "SeedCampaignContractMismatch",
    "SeedCampaignError",
    "SeedCampaignInfraError",
    "SeedCampaignLedger",
    "SeedCampaignResult",
    "SeedCampaignRunner",
    "SeedExecutionRow",
    "SeedExecutionStatus",
    "atomic_write_json",
    "atomic_write_text",
    "config_contract_digest",
    "run_seed_campaign",
}


def __getattr__(name: str) -> Any:
    if name in _CAMPAIGN_EXPORTS:
        import retro_harness.seed_campaign as _seed_campaign

        return getattr(_seed_campaign, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
