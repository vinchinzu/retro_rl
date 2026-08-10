"""Tests for retro_harness.benchmark module."""

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from retro_harness.audit import (
    AuditCapabilities,
    AuditedEnv,
    AttemptAudit,
    InterventionClass,
    RuntimeObservationClass,
)
from retro_harness.model_artifacts import PolicyArtifact, PolicyArtifactError
from retro_harness.benchmark import (
    BenchmarkCase,
    BenchmarkTier,
    ClaimValidationError,
    EvaluationContract,
    IdlePolicy,
    PolicyIdentity,
    policy_identity_for,
    RandomPolicy,
    SeedAttemptResult,
    SeedRobustnessConfig,
    SeedRobustnessReport,
    StartIdentity,
    run_benchmark,
    run_seed_robustness,
    validate_claim,
    write_seed_robustness_report,
    zero_action_for_env,
)
from retro_harness.recordings import iter_jsonl


class FakeDiscreteActionSpace:
    n = 4

    def sample(self):
        return 3


class FakeArrayActionSpace:
    shape = (12,)
    dtype = np.int8

    def sample(self):
        return np.ones(self.shape, dtype=self.dtype)


class FakeEnv:
    def __init__(
        self,
        *,
        success_after=2,
        truncated_after=None,
        array_actions=False,
        info_extra=None,
        audited=True,
    ):
        self.success_after = success_after
        self.truncated_after = truncated_after
        self.info_extra = dict(info_extra or {})
        self.audit_info = (
            {
                "ram_writes": 0,
                "mid_run_loads": 0,
                "assists": {},
                "audit_capabilities": AuditCapabilities.all("fake-env").to_record(),
            }
            if audited
            else {}
        )
        self.action_space = FakeArrayActionSpace() if array_actions else FakeDiscreteActionSpace()
        self.closed = False
        self.reset_count = 0
        self.step_count = 0

    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        return np.zeros((2, 2), dtype=np.uint8), {
            "count": 0,
            "flag": np.int64(1),
            **self.audit_info,
            **self.info_extra,
        }

    def step(self, action):
        self.step_count += 1
        info = {
            "count": self.step_count,
            "array": np.array([1, 2, 3], dtype=np.int64),
            **self.audit_info,
            **self.info_extra,
        }
        terminated = self.success_after is not None and self.step_count >= self.success_after
        truncated = self.truncated_after is not None and self.step_count >= self.truncated_after
        reward = 1.5
        return np.zeros((2, 2), dtype=np.uint8), reward, terminated, truncated, info

    def close(self):
        self.closed = True


class CountingPolicy:
    name = "counting"

    def __init__(self):
        self.reset_calls = 0
        self.act_calls = 0

    def reset(self, env, case):
        self.reset_calls += 1

    def act(self, obs, info, env, case):
        self.act_calls += 1
        return zero_action_for_env(env)


class ClaimedPolicyA:
    name = "claimed"

    def reset(self, env, case):
        return None

    def act(self, obs, info, env, case):
        return zero_action_for_env(env)


class ClaimedPolicyB:
    name = "claimed"

    def reset(self, env, case):
        return None

    def act(self, obs, info, env, case):
        return zero_action_for_env(env)


def _success(info, terminated, truncated):
    return info.get("count", 0) >= 2


def _seed_config(**kwargs):
    values = {
        "generator": "fixture-generator",
        "generator_version": "1.0",
        "logic": "standard",
        "goal": "reach house chest",
        "seeds": ("alpha", "beta"),
        "budget": 3,
        "success_threshold": 1,
        "runtime_observation_class": "Bronze",
        "intervention_class": "Clean",
    }
    values.update(kwargs)
    return SeedRobustnessConfig(**values)


def _instrumented_audit(**values):
    defaults = {
        "ram_writes": 0,
        "mid_run_loads": 0,
        "assists": {},
        "capabilities": AuditCapabilities.all("fixture-audit"),
    }
    defaults.update(values)
    return AttemptAudit(**defaults)


def test_zero_action_for_discrete_env():
    env = FakeEnv()
    assert zero_action_for_env(env) == 0


def test_zero_action_for_array_env():
    env = FakeEnv(array_actions=True)
    action = zero_action_for_env(env)
    assert action.shape == (12,)
    assert np.all(action == 0)


def test_idle_policy_uses_zero_action():
    env = FakeEnv(array_actions=True)
    action = IdlePolicy().act(None, {}, env, None)
    assert np.all(action == 0)


def test_random_policy_delegates_to_action_space():
    env = FakeEnv()
    assert RandomPolicy().act(None, {}, env, None) == 3


def test_run_benchmark_success_records_attempts():
    case = BenchmarkCase(
        benchmark_id="fake_success",
        display_name="Fake Success",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 2",
        max_steps=5,
        build_env=lambda: FakeEnv(success_after=2),
        is_success=_success,
    )
    policy = CountingPolicy()
    result = run_benchmark(case, policy, attempts=2)

    assert len(result.attempts) == 2
    assert result.successes == 2
    assert result.success_rate == 1.0
    assert policy.reset_calls == 2
    assert policy.act_calls == 4


def test_uninstrumented_env_cannot_produce_clean_claim():
    case = BenchmarkCase(
        benchmark_id="missing-audit",
        display_name="Missing audit",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 2",
        max_steps=2,
        build_env=lambda: FakeEnv(success_after=2, audited=False),
        is_success=_success,
    )

    with pytest.raises(ClaimValidationError, match="audit instrumentation"):
        run_benchmark(case, IdlePolicy())


def test_missing_audit_fields_remain_unknown_and_fail_closed():
    contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.CLEAN,
        start_identity=StartIdentity("Start"),
        policy_identity=PolicyIdentity("fixture"),
    )
    audit = AttemptAudit.from_info({})
    assert audit.ram_writes is None
    assert audit.mid_run_loads is None
    assert audit.assists is None

    with pytest.raises(ClaimValidationError, match="audit instrumentation"):
        validate_claim(contract, audit)


def test_audited_env_emits_complete_dry_run_trail():
    env = AuditedEnv(
        FakeEnv(success_after=1, audited=False),
        capabilities=AuditCapabilities.all("fixture-wrapper"),
    )
    _, info = env.reset()
    assert AttemptAudit.from_info(info).has_complete_instrumentation
    env.record_ram_write()
    env.record_state_load()
    env.record_assist("health")
    _, _, _, _, info = env.step(0)
    audit = AttemptAudit.from_info(info)
    assert audit.ram_writes == 1
    assert audit.mid_run_loads == 1
    assert audit.assists == {"health": 1}
    assert audit.capabilities.provider == "fixture-wrapper"
    env.close()


def test_audited_env_owns_backend_write_and_state_load_boundaries():
    class Data:
        def set_value(self, key, value):
            self.last_write = (key, value)

    class Emulator:
        def set_state(self, state):
            self.state = state

    backend = FakeEnv(success_after=1, audited=False)
    backend.data = Data()
    backend.em = Emulator()
    env = AuditedEnv(
        backend,
        capabilities=AuditCapabilities.all("owned-boundary"),
    )
    env.reset()
    env.data.set_value("health", 99)
    env.em.set_state(b"mid-run")
    assert env.audit().ram_writes == 1
    assert env.audit().mid_run_loads == 1
    assert env.audit().assists == {"data.set_value": 1}

    env.load_start_state(b"next-start")
    assert env.audit().ram_writes == 0
    assert env.audit().mid_run_loads == 0
    assert env.audit().assists == {}


def test_policy_artifact_round_trip_rejects_weight_and_schema_mismatch(tmp_path):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"weights-v1")
    lock = tmp_path / "uv.lock"
    lock.write_text("locked", encoding="utf-8")
    artifact = PolicyArtifact.from_checkpoint(
        checkpoint,
        dependency_lock_path=lock,
        algorithm="PPO",
        hyperparameters={"learning_rate": 0.0003, "batch_size": 64},
        training_seed=7,
        observation_schema_digest="obs-v1",
        action_schema_digest="act-v1",
        reward_schema_digest="reward-v1",
        wrapper_schema_digest="wrappers-v1",
        rom_identity_digest="rom-sha",
        state_identity_digest="state-sha",
        core_identity_digest="core-sha",
        source_commit="deadbeef",
    )
    manifest = artifact.write(tmp_path / "policy.artifact.json")

    loaded = PolicyArtifact.load(
        manifest,
        checkpoint_path=checkpoint,
        expected_schema_digests={
            "observation": "obs-v1",
            "action": "act-v1",
            "reward": "reward-v1",
            "wrapper": "wrappers-v1",
        },
    )
    assert loaded == artifact
    assert loaded.to_policy_identity("fixture").identity_digest == artifact.identity_digest

    with pytest.raises(PolicyArtifactError, match="observation schema"):
        PolicyArtifact.load(
            manifest,
            checkpoint_path=checkpoint,
            expected_schema_digests={"observation": "obs-v2"},
        )

    with pytest.raises(PolicyArtifactError, match="rom identity"):
        PolicyArtifact.load(
            manifest,
            checkpoint_path=checkpoint,
            expected_environment_identity_digests={"rom": "different-rom"},
        )

    checkpoint.write_bytes(b"weights-v2")
    with pytest.raises(PolicyArtifactError, match="checkpoint digest"):
        PolicyArtifact.load(manifest, checkpoint_path=checkpoint)


def test_learned_policy_without_artifact_identity_is_rejected():
    class LearnedFixture:
        name = "learned"

        def predict(self, observation):
            return 0

    with pytest.raises(PolicyArtifactError, match="PolicyArtifact"):
        policy_identity_for(LearnedFixture())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("benchmark_id", "foreign-benchmark", "benchmark ID"),
        ("objective", "foreign objective", "objective"),
        ("runtime_observation_class", RuntimeObservationClass.GOLD, "observation class"),
        ("start_identity", StartIdentity("foreign-start"), "start identity"),
    ],
)
def test_run_benchmark_rejects_contract_foreign_to_case(field, value, message):
    case = BenchmarkCase(
        benchmark_id="case-id",
        display_name="Case",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 1",
        max_steps=1,
        build_env=lambda: FakeEnv(success_after=1),
        is_success=_success,
    )
    contract_values = {
        "runtime_observation_class": RuntimeObservationClass.BRONZE,
        "intervention_class": InterventionClass.CLEAN,
        "start_identity": StartIdentity(case.start_state),
        "policy_identity": PolicyIdentity("idle"),
        "benchmark_id": case.benchmark_id,
        "objective": case.objective,
    }
    contract_values[field] = value
    foreign_contract = EvaluationContract(**contract_values)

    with pytest.raises(ValueError, match=message):
        run_benchmark(case, IdlePolicy(), contract=foreign_contract)


def test_run_benchmark_rejects_foreign_policy_identity_before_execution():
    build_calls = 0

    def build_env():
        nonlocal build_calls
        build_calls += 1
        return FakeEnv(success_after=1)

    case = BenchmarkCase(
        benchmark_id="case-id",
        display_name="Case",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 1",
        max_steps=1,
        build_env=build_env,
        is_success=_success,
    )
    foreign_contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.CLEAN,
        start_identity=StartIdentity(case.start_state),
        policy_identity=PolicyIdentity("foreign-policy"),
        benchmark_id=case.benchmark_id,
        objective=case.objective,
    )

    with pytest.raises(ClaimValidationError, match="policy identity"):
        run_benchmark(case, IdlePolicy(), contract=foreign_contract)

    assert build_calls == 0


def test_run_benchmark_rejects_resource_contract_for_clean_case():
    policy = IdlePolicy()
    case = BenchmarkCase(
        benchmark_id="clean-case",
        display_name="Clean case",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 1",
        max_steps=1,
        build_env=lambda: FakeEnv(success_after=1),
        is_success=_success,
    )
    resource_contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.RESOURCE_ASSISTED,
        start_identity=StartIdentity(case.start_state),
        policy_identity=policy_identity_for(policy),
        benchmark_id=case.benchmark_id,
        objective=case.objective,
        assist_contract_path="docs/ASSIST_CONTRACT.md",
        assist_contract_digest="fixture-assist-digest",
    )

    with pytest.raises(ValueError, match="intervention class"):
        run_benchmark(case, policy, contract=resource_contract)


def test_case_contract_binds_assist_mode():
    policy = IdlePolicy()
    case = BenchmarkCase(
        benchmark_id="assist-mode-case",
        display_name="Assist mode case",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 1",
        max_steps=1,
        build_env=lambda: FakeEnv(success_after=1),
        is_success=lambda info, terminated, truncated: info.get("count", 0) >= 1,
        contract=EvaluationContract(
            runtime_observation_class=RuntimeObservationClass.BRONZE,
            intervention_class=InterventionClass.RESOURCE_ASSISTED,
            start_identity=StartIdentity("Start"),
            policy_identity=PolicyIdentity("unbound-policy"),
            benchmark_id="assist-mode-case",
            objective="Reach count 1",
            assist_contract_path="docs/ASSIST_CONTRACT.md",
            assist_contract_digest="fixture-assist-digest",
            assist_mode="resources",
        ),
    )
    contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.RESOURCE_ASSISTED,
        start_identity=StartIdentity(case.start_state),
        policy_identity=policy_identity_for(policy),
        benchmark_id=case.benchmark_id,
        objective=case.objective,
        assist_contract_path="docs/ASSIST_CONTRACT.md",
        assist_contract_digest="fixture-assist-digest",
        assist_mode="ammo",
    )

    with pytest.raises(ValueError, match="assist_mode"):
        run_benchmark(case, policy, contract=contract)


def test_two_distinct_policy_classes_cannot_share_one_contract_identity():
    policy_a = ClaimedPolicyA()
    policy_b = ClaimedPolicyB()
    identity_a = policy_identity_for(policy_a)
    identity_b = policy_identity_for(policy_b)
    assert identity_a.identity_digest != identity_b.identity_digest

    case = BenchmarkCase(
        benchmark_id="policy-identity",
        display_name="Policy identity",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 1",
        max_steps=1,
        build_env=lambda: FakeEnv(success_after=1),
        is_success=lambda info, terminated, truncated: info.get("count", 0) >= 1,
    )
    contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.CLEAN,
        start_identity=StartIdentity(case.start_state),
        policy_identity=identity_a,
        benchmark_id=case.benchmark_id,
        objective=case.objective,
    )

    with pytest.raises(ClaimValidationError, match="policy identity"):
        run_benchmark(
            case,
            policy_a,
            contract=contract.with_policy(PolicyIdentity("claimed")),
        )
    result = run_benchmark(case, policy_a, contract=contract)
    assert result.successes == 1
    assert validate_claim(result.attempts[0].to_record(case, result.policy_name)) is True
    with pytest.raises(ClaimValidationError, match="policy identity"):
        run_benchmark(case, policy_b, contract=contract)


def test_policy_identity_uses_deterministic_bytecode_fallback_without_source():
    namespace = {}
    exec(
        compile(
            "def reset(self, env, case):\n    return None\n"
            "def act(self, obs, info, env, case):\n    return 0\n",
            "<legacy-policy>",
            "exec",
        ),
        namespace,
    )
    legacy_type = type(
        "LegacyPolicy",
        (),
        {
            "__module__": "legacy_fixture",
            "name": "legacy",
            "reset": namespace["reset"],
            "act": namespace["act"],
        },
    )
    policy = legacy_type()

    first = policy_identity_for(policy)
    second = policy_identity_for(legacy_type())
    assert first == second
    assert first.metadata["fingerprint_kind"] == "bytecode"

    case = BenchmarkCase(
        benchmark_id="legacy-policy",
        display_name="Legacy policy",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 1",
        max_steps=1,
        build_env=lambda: FakeEnv(success_after=1),
        is_success=lambda info, terminated, truncated: info.get("count", 0) >= 1,
    )
    contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.CLEAN,
        start_identity=StartIdentity(case.start_state),
        policy_identity=first,
        benchmark_id=case.benchmark_id,
        objective=case.objective,
    )
    assert run_benchmark(case, policy, contract=contract).successes == 1


def test_dynamic_same_module_same_qualname_opaque_classes_are_unverifiable():
    def make_opaque_policy():
        return type(
            "OpaquePolicy",
            (),
            {
                "__module__": "opaque_fixture",
                "__qualname__": "OpaquePolicy",
                "__call__": staticmethod(print),
            },
        )()

    policy_a = make_opaque_policy()
    policy_b = make_opaque_policy()
    identity_a = policy_identity_for(policy_a)
    identity_b = policy_identity_for(policy_b)
    assert identity_a == identity_b
    assert identity_a.metadata["fingerprint_kind"] == "module-qualified-name"

    case = BenchmarkCase(
        benchmark_id="opaque-policy-identity",
        display_name="Opaque policy identity",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 1",
        max_steps=1,
        build_env=lambda: FakeEnv(success_after=1),
        is_success=lambda info, terminated, truncated: info.get("count", 0) >= 1,
    )
    assert run_benchmark(case, policy_a).successes == 1
    for policy, identity in ((policy_a, identity_a), (policy_b, identity_b)):
        contract = EvaluationContract(
            runtime_observation_class=RuntimeObservationClass.BRONZE,
            intervention_class=InterventionClass.CLEAN,
            start_identity=StartIdentity(case.start_state),
            policy_identity=identity,
            benchmark_id=case.benchmark_id,
            objective=case.objective,
        )
        with pytest.raises(ClaimValidationError, match="unverifiable"):
            run_benchmark(case, policy, contract=contract)


def test_run_benchmark_timeout_sets_failure_reason():
    case = BenchmarkCase(
        benchmark_id="fake_timeout",
        display_name="Fake Timeout",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Never succeeds",
        max_steps=3,
        build_env=lambda: FakeEnv(success_after=None),
        is_success=lambda info, terminated, truncated: False,
    )
    result = run_benchmark(case, IdlePolicy())

    attempt = result.attempts[0]
    assert attempt.success is False
    assert attempt.failure_reason == "max_steps"
    assert attempt.steps == 3


def test_run_benchmark_writes_jsonl_log():
    case = BenchmarkCase(
        benchmark_id="fake_log",
        display_name="Fake Log",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 2",
        max_steps=5,
        build_env=lambda: FakeEnv(success_after=2),
        is_success=_success,
    )
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "benchmarks.jsonl"
        result = run_benchmark(case, IdlePolicy(), log_path=log_path)

        entries = iter_jsonl(log_path)
        assert len(entries) == 2
        assert entries[0]["event"] == "benchmark_attempt"
        assert entries[1]["event"] == "benchmark_summary"
        assert entries[1]["success_rate"] == 1.0
        assert result.log_path == log_path


def test_attempt_log_is_json_safe():
    case = BenchmarkCase(
        benchmark_id="fake_json",
        display_name="Fake Json",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 2",
        max_steps=5,
        build_env=lambda: FakeEnv(success_after=2),
        is_success=_success,
    )
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "benchmarks.jsonl"
        run_benchmark(case, IdlePolicy(), log_path=log_path)
        entries = iter_jsonl(log_path)
        assert entries[0]["final_info"]["array"] == [1, 2, 3]


def test_run_seed_robustness_writes_deterministic_st_fixture_report():
    config = SeedRobustnessConfig(
        generator="fixture-generator",
        generator_version="1.0",
        logic="standard",
        goal="reach house chest",
        seeds=("alpha", "beta", "gamma"),
        budget=3,
        success_threshold=2,
        runtime_observation_class="Bronze",
        intervention_class=InterventionClass.RESOURCE_ASSISTED,
        assist_contract_path="fixtures/assist-contract.md",
        assist_contract_digest="fixture-assist-digest",
    )
    outcomes = {
        "alpha": (2, {"terminal_milestone": "house_chest", "assists": {"missile": 1}}),
        "beta": (
            None,
            {
                "terminal_milestone": "red_door",
                "failure_mode": "stalled",
                "assists": {"missile": 2},
            },
        ),
        "gamma": (2, {"terminal_milestone": "house_chest", "assists": {}}),
    }
    seen_seeds = []

    def build_case(seed):
        seen_seeds.append(seed)
        success_after, info_extra = outcomes[seed]
        return BenchmarkCase(
            benchmark_id=f"fixture_{seed}",
            display_name="Seed fixture",
            game="FakeGame",
            start_state=f"power_on_{seed}",
            tier=BenchmarkTier.BRONZE,
            objective=config.goal,
            max_steps=config.budget,
            build_env=lambda: FakeEnv(
                success_after=success_after,
                info_extra=info_extra,
            ),
            is_success=lambda info, terminated, truncated: success_after is not None
            and _success(info, terminated, truncated),
            contract=EvaluationContract(
                runtime_observation_class=RuntimeObservationClass.BRONZE,
                intervention_class=config.intervention_class,
                start_identity=StartIdentity(f"power_on_{seed}"),
                policy_identity=PolicyIdentity("unbound-policy"),
                benchmark_id=f"fixture_{seed}",
                objective=config.goal,
                assist_contract_path=config.assist_contract_path,
                assist_contract_digest=config.assist_contract_digest,
            ),
        )

    with tempfile.TemporaryDirectory() as td:
        report_path = Path(td) / "seed_report.json"
        report = run_seed_robustness(
            config,
            build_case,
            IdlePolicy(),
            report_path=report_path,
        )

        first_bytes = report_path.read_bytes()
        write_seed_robustness_report(report_path, report)
        assert report_path.read_bytes() == first_bytes
        record = json.loads(first_bytes)

    assert seen_seeds == ["alpha", "beta", "gamma"]
    assert report.successes == 2
    assert report.threshold_met is True
    assert record["config"]["seed_count"] == 3
    assert record["config"]["success_threshold"] == 2
    assert record["summary"] == {
        "required_successes": 2,
        "seeds_successful": 2,
        "seeds_total": 3,
        "success_rate": 2 / 3,
        "threshold_met": True,
    }
    assert record["seed_results"][0]["frames"] == 2
    assert record["seed_results"][0]["terminal_milestone"] == "house_chest"
    assert record["seed_results"][0]["assists"] == {"missile": 1}
    assert record["seed_results"][1]["outcome"] == "failure"
    assert record["seed_results"][1]["failure_mode"] == "stalled"
    assert record["seed_results"][1]["terminal_milestone"] == "red_door"
    assert validate_claim(record) is True
    for seed_record in record["seed_results"]:
        assert seed_record["runtime_observation_class"] == "Bronze"
        assert seed_record["intervention_class"] == "Resource-assisted"
        assert seed_record["start_identity_digest"]
        assert seed_record["policy_identity_digest"]


@pytest.mark.parametrize("case_steps", [2, 4])
def test_run_seed_robustness_requires_case_budget_to_match(case_steps):
    config = _seed_config(seeds=("alpha",), success_threshold=1)
    case = BenchmarkCase(
        benchmark_id="budget_mismatch",
        display_name="Budget mismatch",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective=config.goal,
        max_steps=case_steps,
        build_env=lambda: FakeEnv(success_after=1),
        is_success=_success,
    )

    with pytest.raises(ValueError, match="must use exactly"):
        run_seed_robustness(config, lambda seed: case, IdlePolicy())


def test_seed_robustness_report_rejects_over_budget_frames():
    config = _seed_config()
    results = tuple(
        SeedAttemptResult(
            seed=seed,
            success=False,
            frames=config.budget + 1 if seed == config.seeds[0] else config.budget,
        )
        for seed in config.seeds
    )

    with pytest.raises(ValueError, match="exceed the published frame budget"):
        SeedRobustnessReport(config, "idle", results)


@pytest.mark.parametrize(
    ("identity_field", "foreign_identity", "message"),
    [
        ("start_identity", StartIdentity("foreign-start"), "start identity"),
        ("policy_identity", PolicyIdentity("foreign-policy"), "policy identity"),
    ],
)
def test_seed_report_rejects_shared_config_identity_contradiction(
    identity_field,
    foreign_identity,
    message,
):
    config = _seed_config(
        start_identity=StartIdentity("published-start"),
        policy_identity=PolicyIdentity("published-policy"),
    )
    results = []
    for seed in config.seeds:
        start_identity_digest = config.start_identity.identity_digest
        policy_identity_digest = config.policy_identity.identity_digest
        if identity_field == "start_identity":
            start_identity_digest = foreign_identity.identity_digest
        else:
            policy_identity_digest = foreign_identity.identity_digest
        results.append(
            SeedAttemptResult(
                seed=seed,
                success=False,
                frames=0,
                runtime_observation_class=config.runtime_observation_class,
                intervention_class=config.intervention_class,
                start_identity_digest=start_identity_digest,
                policy_identity_digest=policy_identity_digest,
            )
        )

    with pytest.raises(ValueError, match=message):
        SeedRobustnessReport(config, "published-policy", tuple(results))


def test_validate_claim_rejects_config_contract_identity_contradiction():
    shared_contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.CLEAN,
        start_identity=StartIdentity("published-start"),
        policy_identity=PolicyIdentity("published-policy"),
        benchmark_id="seed-set",
        objective="reach house chest",
    )
    config = _seed_config(contract=shared_contract)
    results = tuple(
        SeedAttemptResult(
            seed=seed,
            success=False,
            frames=0,
                runtime_observation_class=config.runtime_observation_class,
                intervention_class=config.intervention_class,
                attempt_audit=_instrumented_audit(),
                contract=shared_contract,
        )
        for seed in config.seeds
    )
    record = SeedRobustnessReport(config, "published-policy", results).to_record()
    record["config"]["contract"]["policy_identity_digest"] = PolicyIdentity(
        "foreign-policy"
    ).identity_digest

    with pytest.raises(ClaimValidationError, match="policy_identity"):
        validate_claim(record)


def test_validate_claim_rejects_nested_config_contract_assist_mode_tampering():
    shared_contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.RESOURCE_ASSISTED,
        start_identity=StartIdentity("published-start"),
        policy_identity=PolicyIdentity("published-policy"),
        benchmark_id="seed-set",
        objective="reach house chest",
        assist_contract_path="docs/ASSIST_CONTRACT.md",
        assist_contract_digest="fixture-assist-digest",
        assist_mode="resources",
    )
    config = _seed_config(
        contract=shared_contract,
        intervention_class=InterventionClass.RESOURCE_ASSISTED,
    )
    results = tuple(
        SeedAttemptResult(
            seed=seed,
            success=False,
            frames=0,
                runtime_observation_class=config.runtime_observation_class,
                intervention_class=config.intervention_class,
                attempt_audit=_instrumented_audit(),
                contract=shared_contract,
        )
        for seed in config.seeds
    )
    record = SeedRobustnessReport(config, "published-policy", results).to_record()
    assert record["config"]["assist_mode"] == "resources"
    record["config"]["contract"]["assist_mode"] = "tampered"

    with pytest.raises(ClaimValidationError, match="assist_mode"):
        validate_claim(record)


def test_run_seed_robustness_rejects_over_budget_extracted_frames():
    config = _seed_config(seeds=("alpha",), success_threshold=1)
    case = BenchmarkCase(
        benchmark_id="extracted_frames",
        display_name="Extracted frames",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective=config.goal,
        max_steps=config.budget,
        build_env=lambda: FakeEnv(success_after=1),
        is_success=_success,
    )

    def extract(seed, attempt):
        return SeedAttemptResult(
            seed=seed,
            success=attempt.success,
            frames=config.budget + 1,
            attempt_audit=_instrumented_audit(),
        )

    with pytest.raises(ValueError, match="exceed the published frame budget"):
        run_seed_robustness(config, lambda seed: case, IdlePolicy(), result_extractor=extract)


def test_write_seed_robustness_report_rejects_nonfinite_metadata(tmp_path):
    config = _seed_config(metadata={"score": float("nan")})
    results = tuple(
        SeedAttemptResult(
            seed=seed,
            success=False,
            frames=0,
            attempt_audit=_instrumented_audit(),
        )
        for seed in config.seeds
    )
    report = SeedRobustnessReport(config, "idle", results)
    report_path = tmp_path / "nested" / "seed_report.json"

    with pytest.raises(ValueError, match="finite JSON numbers"):
        write_seed_robustness_report(report_path, report)

    assert not report_path.exists()
    assert not report_path.parent.exists()


@pytest.mark.parametrize(
    "metadata",
    [{"bad": object()}, {1: "non-string key"}],
)
def test_write_seed_robustness_report_rejects_non_json_metadata(tmp_path, metadata):
    config = _seed_config(metadata=metadata)
    results = tuple(
        SeedAttemptResult(
            seed=seed,
            success=False,
            frames=0,
            attempt_audit=_instrumented_audit(),
        )
        for seed in config.seeds
    )
    report = SeedRobustnessReport(config, "idle", results)

    with pytest.raises(TypeError, match="JSON"):
        write_seed_robustness_report(tmp_path / "seed_report.json", report)


def _contract(
    *,
    observation=RuntimeObservationClass.GOLD,
    intervention=InterventionClass.CLEAN,
    **kwargs,
):
    return EvaluationContract(
        runtime_observation_class=observation,
        intervention_class=intervention,
        start_identity=StartIdentity("power_on"),
        policy_identity=PolicyIdentity("fixture-policy"),
        **kwargs,
    )


def test_typed_contract_and_audit_validate_and_emit_identity_digests():
    contract = _contract()
    audit = _instrumented_audit(
        start_identity_digest=contract.start_identity.identity_digest,
        policy_identity_digest=contract.policy_identity.identity_digest,
    )

    assert validate_claim(contract, audit) is True
    record = contract.to_record()
    assert record["runtime_observation_class"] == "Gold"
    assert record["intervention_class"] == "Clean"
    assert record["start_identity_digest"] == contract.start_identity.identity_digest
    assert record["policy_identity_digest"] == contract.policy_identity.identity_digest


def test_serialized_attempt_record_can_be_validated():
    contract = _contract()
    audit = _instrumented_audit(
        start_identity_digest=contract.start_identity.identity_digest,
        policy_identity_digest=contract.policy_identity.identity_digest,
    )
    record = {
        "contract": contract.to_record(),
        "attempt_audit": audit.to_record(),
    }

    assert validate_claim(record) is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ram_writes", 1, "RAM writes"),
        ("mid_run_loads", 1, "mid-run loads"),
        ("assists", {"health": 1}, "assists"),
    ],
)
def test_clean_claim_rejects_interventions(field, value, message):
    contract = _contract()
    audit = _instrumented_audit(
        start_identity_digest=contract.start_identity.identity_digest,
        policy_identity_digest=contract.policy_identity.identity_digest,
        **{field: value},
    )

    with pytest.raises(ClaimValidationError, match=message):
        validate_claim(contract, audit)


def test_assisted_contract_requires_path_and_digest():
    with pytest.raises(ValueError, match="both assist_contract_path"):
        _contract(intervention=InterventionClass.RESOURCE_ASSISTED)

    contract = _contract(
        intervention=InterventionClass.RESOURCE_ASSISTED,
        assist_contract_path="docs/ASSIST_CONTRACT.md",
        assist_contract_digest="sha256:fixture",
    )
    audit = _instrumented_audit(
        assists={"ammo": 1},
        start_identity_digest=contract.start_identity.identity_digest,
        policy_identity_digest=contract.policy_identity.identity_digest,
    )
    assert validate_claim(contract, audit) is True


@pytest.mark.parametrize(
    "field,value",
    [("runtime_observation_class", "platinum"), ("intervention_class", "magic")],
)
def test_invalid_class_strings_rejected(field, value):
    values = {
        "generator": "fixture-generator",
        "generator_version": "1.0",
        "logic": "standard",
        "goal": "goal",
        "seeds": ("alpha",),
        "budget": 1,
        "success_threshold": 1,
        "runtime_observation_class": "Bronze",
        "intervention_class": "Clean",
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match="invalid|expected"):
        SeedRobustnessConfig(**values)


def test_legacy_benchmark_tier_is_adapted_on_every_attempt_record():
    case = BenchmarkCase(
        benchmark_id="legacy_contract",
        display_name="Legacy contract",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.SILVER,
        objective="Reach count 1",
        max_steps=1,
        build_env=lambda: FakeEnv(success_after=1),
        is_success=lambda info, terminated, truncated: _success(
            info, terminated, truncated
        ),
    )
    result = run_benchmark(case, IdlePolicy())
    record = result.attempts[0].to_record(case, result.policy_name)

    assert record["runtime_observation_class"] == "Silver"
    assert record["intervention_class"] == "Clean"
    assert record["start_identity_digest"]
    assert record["policy_identity_digest"]
