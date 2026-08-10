"""Compatibility checks for the decomposed benchmark public API."""

import retro_harness
from retro_harness import benchmark, benchmark_claims, benchmark_runner


CLAIM_SYMBOLS = (
    "StartIdentity",
    "PolicyIdentity",
    "policy_identity_for",
    "EvaluationContract",
    "ClaimValidationError",
    "validate_claim",
)

RUNNER_SYMBOLS = (
    "BenchmarkTier",
    "BenchmarkCase",
    "BenchmarkAttemptResult",
    "BenchmarkRunResult",
    "SeedValue",
    "SeedRobustnessConfig",
    "SeedAttemptResult",
    "SEED_ROBUSTNESS_SCHEMA_VERSION",
    "SeedRobustnessReport",
    "BenchmarkPolicy",
    "IdlePolicy",
    "RandomPolicy",
    "zero_action_for_env",
    "run_seed_robustness",
    "run_benchmark",
    "write_seed_robustness_report",
)

ROOT_FACADE_SYMBOLS = CLAIM_SYMBOLS + (
    "BenchmarkTier",
    "BenchmarkCase",
    "BenchmarkAttemptResult",
    "BenchmarkRunResult",
    "IdlePolicy",
    "RandomPolicy",
    "run_benchmark",
    "zero_action_for_env",
)


def test_benchmark_claim_facade_preserves_canonical_objects() -> None:
    for name in CLAIM_SYMBOLS:
        assert getattr(benchmark, name) is getattr(benchmark_claims, name)


def test_benchmark_runner_facade_preserves_canonical_objects() -> None:
    for name in RUNNER_SYMBOLS:
        assert getattr(benchmark, name) is getattr(benchmark_runner, name)


def test_root_facade_preserves_benchmark_objects() -> None:
    for name in ROOT_FACADE_SYMBOLS:
        assert getattr(retro_harness, name) is getattr(benchmark, name)
