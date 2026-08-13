"""Compatibility shim for :mod:`retro_harness.benchmark.runner`.

Canonical implementations live under :mod:`retro_harness.benchmark`.
This module preserves historical import paths used by games and scripts.
"""

from __future__ import annotations

from typing import Any

from retro_harness.benchmark.runner import *  # noqa: F403
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
    _contract_for_seed_case,
    _contract_for_seed_result,
    _is_unbound_policy_identity,
    _policy_identity_is_verifiable,
    _policy_name,
    _same_policy_identity,
    _validate_contract_for_case,
    run_benchmark,
    run_seed_robustness,
    write_seed_robustness_report,
    zero_action_for_env,
)
from retro_harness.benchmark.seed_robustness import (
    _audit_with_contract_identity,
    _to_jsonable,
    _validate_seed_result_budget,
    _validate_seed_value,
)

__all__ = [
    "BenchmarkAttemptResult",
    "BenchmarkCase",
    "BenchmarkPolicy",
    "BenchmarkRunResult",
    "BenchmarkTier",
    "IdlePolicy",
    "RandomPolicy",
    "SEED_ROBUSTNESS_SCHEMA_VERSION",
    "SeedAttemptResult",
    "SeedRobustnessConfig",
    "SeedRobustnessReport",
    "SeedValue",
    "run_benchmark",
    "run_seed_robustness",
    "write_seed_robustness_report",
    "zero_action_for_env",
]


# Campaign runner re-exports (canonical home: retro_harness.seed_campaign).
def __getattr__(name: str) -> Any:
    if name in {
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
    }:
        import retro_harness.seed_campaign as _seed_campaign

        return getattr(_seed_campaign, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
