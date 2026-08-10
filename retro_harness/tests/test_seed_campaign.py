"""Tests for fail-closed resumable SeedCampaignRunner (rr-gbd.33)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from retro_harness.audit import (
    AuditCapabilities,
    InterventionClass,
    RuntimeObservationClass,
)
from retro_harness.benchmark import (
    BenchmarkCase,
    BenchmarkTier,
    EvaluationContract,
    IdlePolicy,
    PolicyIdentity,
    SeedCampaignContractMismatch,
    SeedCampaignRunner,
    SeedExecutionStatus,
    SeedRobustnessConfig,
    StartIdentity,
    atomic_write_text,
    config_contract_digest,
    run_seed_campaign,
    write_seed_robustness_report,
)
from retro_harness.seed_campaign import SeedCampaignLedger


class FakeDiscreteActionSpace:
    n = 4

    def sample(self):
        return 3


class FakeEnv:
    def __init__(self, *, success_after=2, info_extra=None, audited=True):
        self.success_after = success_after
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
        self.action_space = FakeDiscreteActionSpace()
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.zeros((2, 2), dtype=np.uint8), {
            "count": 0,
            **self.audit_info,
            **self.info_extra,
        }

    def step(self, action):
        self.step_count += 1
        info = {
            "count": self.step_count,
            **self.audit_info,
            **self.info_extra,
        }
        terminated = self.success_after is not None and self.step_count >= self.success_after
        return np.zeros((2, 2), dtype=np.uint8), 1.0, terminated, False, info

    def close(self):
        return None


def _success(info, terminated, truncated):
    return info.get("count", 0) >= 2


def _config(**kwargs):
    values = {
        "generator": "campaign-fixture",
        "generator_version": "1.0",
        "logic": "standard",
        "goal": "reach house",
        "seeds": ("alpha", "beta", "gamma"),
        "budget": 3,
        "success_threshold": 2,
        "runtime_observation_class": "Bronze",
        "intervention_class": "Clean",
    }
    values.update(kwargs)
    return SeedRobustnessConfig(**values)


def _build_case_factory(outcomes):
    def build_case(seed):
        success_after, info_extra = outcomes[seed]
        return BenchmarkCase(
            benchmark_id=f"campaign_{seed}",
            display_name="Campaign fixture",
            game="FakeGame",
            start_state=f"power_on_{seed}",
            tier=BenchmarkTier.BRONZE,
            objective="reach house",
            max_steps=3,
            build_env=lambda: FakeEnv(
                success_after=success_after,
                info_extra=info_extra,
            ),
            is_success=lambda info, terminated, truncated: success_after is not None
            and _success(info, terminated, truncated),
            contract=EvaluationContract(
                runtime_observation_class=RuntimeObservationClass.BRONZE,
                intervention_class=InterventionClass.CLEAN,
                start_identity=StartIdentity(f"power_on_{seed}"),
                policy_identity=PolicyIdentity("unbound-policy"),
                benchmark_id=f"campaign_{seed}",
                objective="reach house",
            ),
        )

    return build_case


def _happy_outcomes():
    return {
        "alpha": (2, {"terminal_milestone": "house"}),
        "beta": (None, {"terminal_milestone": "door", "failure_mode": "stalled"}),
        "gamma": (2, {"terminal_milestone": "house"}),
    }


def test_atomic_write_text_replaces_destination(tmp_path):
    path = tmp_path / "out.json"
    path.write_text("old\n", encoding="utf-8")
    atomic_write_text(path, "new\n")
    assert path.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.glob(".out.json.*.tmp")) == []


def test_campaign_happy_path_claimable_and_ordered(tmp_path):
    config = _config()
    result = run_seed_campaign(
        config,
        _build_case_factory(_happy_outcomes()),
        policy_factory=lambda seed: IdlePolicy(),
        ledger_path=tmp_path / "ledger.json",
        report_path=tmp_path / "campaign.json",
    )
    assert result.claimable is True
    assert result.successes == 2
    assert result.threshold_met is True
    assert [row.status for row in result.rows] == [
        SeedExecutionStatus.SUCCESS,
        SeedExecutionStatus.FAILURE,
        SeedExecutionStatus.SUCCESS,
    ]
    assert [row.seed for row in result.rows] == ["alpha", "beta", "gamma"]
    report = json.loads((tmp_path / "campaign.json").read_text(encoding="utf-8"))
    assert report["claimable"] is True
    assert report["summary"]["infra_errors"] == 0
    assert report["seed_results"][0]["execution_status"] == "success"
    assert report["seed_results"][1]["execution_status"] == "failure"
    # Publishable projection validates.
    classic = result.to_seed_robustness_report()
    write_seed_robustness_report(tmp_path / "classic.json", classic)
    assert (tmp_path / "classic.json").exists()


def test_infra_error_ordered_row_and_non_claimable(tmp_path):
    outcomes = _happy_outcomes()

    def build_case(seed):
        if seed == "beta":
            raise RuntimeError("rom missing")
        return _build_case_factory(outcomes)(seed)

    result = run_seed_campaign(
        _config(),
        build_case,
        policy_factory=lambda seed: IdlePolicy(),
        ledger_path=tmp_path / "ledger.json",
        report_path=tmp_path / "campaign.json",
    )
    assert result.claimable is False
    assert result.infra_error_count == 1
    assert result.rows[1].status is SeedExecutionStatus.INFRA_ERROR
    assert result.rows[1].error and "rom missing" in result.rows[1].error
    assert result.rows[0].status is SeedExecutionStatus.SUCCESS
    assert result.rows[2].status is SeedExecutionStatus.SUCCESS
    record = result.to_record()
    assert record["seed_results"][1]["failure_mode"] == "INFRA_ERROR"
    assert record["seed_results"][1]["execution_status"] == "infra_error"
    assert record["seed_results"][1]["ram_writes"] is None
    with pytest.raises(Exception, match="non-claimable|INFRA_ERROR"):
        result.to_seed_robustness_report()


def test_missing_audit_is_infra_not_clean_success(tmp_path):
    def build_case(seed):
        return BenchmarkCase(
            benchmark_id=f"unaudited_{seed}",
            display_name="Unaudited",
            game="FakeGame",
            start_state=f"power_on_{seed}",
            tier=BenchmarkTier.BRONZE,
            objective="reach house",
            max_steps=3,
            build_env=lambda: FakeEnv(success_after=2, audited=False),
            is_success=_success,
            contract=EvaluationContract(
                runtime_observation_class=RuntimeObservationClass.BRONZE,
                intervention_class=InterventionClass.CLEAN,
                start_identity=StartIdentity(f"power_on_{seed}"),
                policy_identity=PolicyIdentity("unbound-policy"),
                benchmark_id=f"unaudited_{seed}",
                objective="reach house",
            ),
        )

    result = run_seed_campaign(
        _config(seeds=("alpha",), success_threshold=1),
        build_case,
        policy_factory=lambda seed: IdlePolicy(),
        ledger_path=tmp_path / "ledger.json",
    )
    assert result.claimable is False
    assert result.rows[0].status is SeedExecutionStatus.INFRA_ERROR
    assert result.rows[0].to_record()["ram_writes"] is None


def test_resume_is_byte_identical(tmp_path):
    config = _config()
    outcomes = _happy_outcomes()
    continuous_dir = tmp_path / "continuous"
    resumed_dir = tmp_path / "resumed"
    continuous_dir.mkdir()
    resumed_dir.mkdir()

    continuous = run_seed_campaign(
        config,
        _build_case_factory(outcomes),
        policy_factory=lambda seed: IdlePolicy(),
        ledger_path=continuous_dir / "ledger.json",
        report_path=continuous_dir / "campaign.json",
    )

    # Partial run: stop after first seed by raising on beta, with stop_on_infra.
    calls = {"n": 0}

    def build_partial(seed):
        calls["n"] += 1
        if seed == "beta":
            raise RuntimeError("interrupt")
        return _build_case_factory(outcomes)(seed)

    partial = SeedCampaignRunner(
        config=config,
        build_case=build_partial,
        policy_factory=lambda seed: IdlePolicy(),
        ledger_path=resumed_dir / "ledger.json",
        stop_on_infra_error=True,
    ).run()
    assert partial.rows[0].status is SeedExecutionStatus.SUCCESS
    assert partial.infra_error_count >= 1

    # Rewrite ledger to only the successful first seed so resume continues cleanly.
    ledger = SeedCampaignLedger.load(resumed_dir / "ledger.json", expected_config=config)
    first_only = SeedCampaignLedger(
        config=config,
        policy_name=ledger.policy_name,
        rows=(ledger.rows[0],),
        contract_digest=ledger.contract_digest,
    )
    first_only.write(resumed_dir / "ledger.json")

    resumed = run_seed_campaign(
        config,
        _build_case_factory(outcomes),
        policy_factory=lambda seed: IdlePolicy(),
        ledger_path=resumed_dir / "ledger.json",
        report_path=resumed_dir / "campaign.json",
    )
    assert resumed.claimable is True
    continuous_bytes = (continuous_dir / "campaign.json").read_bytes()
    resumed_bytes = (resumed_dir / "campaign.json").read_bytes()
    assert resumed_bytes == continuous_bytes
    assert continuous.to_record() == resumed.to_record()


def test_contract_mismatch_cannot_resume(tmp_path):
    config = _config()
    run_seed_campaign(
        config,
        _build_case_factory(_happy_outcomes()),
        policy_factory=lambda seed: IdlePolicy(),
        ledger_path=tmp_path / "ledger.json",
    )
    foreign = _config(goal="different goal")
    assert config_contract_digest(config) != config_contract_digest(foreign)
    with pytest.raises(SeedCampaignContractMismatch, match="cannot resume"):
        run_seed_campaign(
            foreign,
            _build_case_factory(_happy_outcomes()),
            policy_factory=lambda seed: IdlePolicy(),
            ledger_path=tmp_path / "ledger.json",
        )


def test_policy_factory_called_per_seed(tmp_path):
    seen = []

    def policy_factory(seed):
        seen.append(seed)
        return IdlePolicy()

    run_seed_campaign(
        _config(seeds=("alpha", "beta"), success_threshold=1),
        _build_case_factory(
            {
                "alpha": (2, {}),
                "beta": (2, {}),
            }
        ),
        policy_factory=policy_factory,
        ledger_path=tmp_path / "ledger.json",
    )
    assert seen == ["alpha", "alpha", "beta"]
    # First call peeks policy name on empty ledger; then one call per seed.


def test_stop_on_infra_error_pads_remaining(tmp_path):
    def build_case(seed):
        if seed == "alpha":
            raise OSError("backend down")
        return _build_case_factory(_happy_outcomes())(seed)

    result = SeedCampaignRunner(
        config=_config(),
        build_case=build_case,
        policy_factory=lambda seed: IdlePolicy(),
        ledger_path=tmp_path / "ledger.json",
        stop_on_infra_error=True,
    ).run()
    assert len(result.rows) == 3
    assert all(row.status is SeedExecutionStatus.INFRA_ERROR for row in result.rows)
    assert result.claimable is False
    assert "backend down" in (result.rows[0].error or "")
