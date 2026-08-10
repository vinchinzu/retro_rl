"""Multi-seed early tip S/T campaign (rr-gbd.25)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retro_harness.benchmark import (
    SeedCampaignContractMismatch,
    SeedExecutionStatus,
)
from retro_harness.benchmark_claims import validate_claim
from sm_rando.early_tip_campaign import (
    DEFAULT_SEEDS,
    POLICY_NAME,
    ShipToMorphCampaignPolicy,
    build_campaign_config,
    campaign_summary,
    default_spec,
    ensure_campaign_seeds,
    morph_reachable_without_spoiler,
    run_early_tip_campaign,
)
from sm_rando.seed import SeedPackage, write_fixture_seed


def test_fixture_seeds_have_vanilla_morph(tmp_path: Path) -> None:
    packages = ensure_campaign_seeds(("1337", "1338"), seeds_root=tmp_path)
    assert len(packages) == 2
    for pkg in packages:
        assert morph_reachable_without_spoiler(pkg) is True
        assert (pkg.directory / "meta.json").is_file()


def test_dry_campaign_meets_st_threshold_and_is_claimable(tmp_path: Path) -> None:
    seeds_root = tmp_path / "seeds"
    ledger = tmp_path / "ledger.json"
    report = tmp_path / "campaign.json"
    classic = tmp_path / "classic.json"

    result = run_early_tip_campaign(
        mode="dry",
        seeds=("1337", "1338", "1339"),
        success_threshold=2,
        budget=8,
        seeds_root=seeds_root,
        ledger_path=ledger,
        report_path=report,
        classic_report_path=classic,
    )

    assert result.claimable is True
    assert result.successes == 3
    assert result.threshold_met is True
    assert result.infra_error_count == 0
    assert [row.status for row in result.rows] == [
        SeedExecutionStatus.SUCCESS,
        SeedExecutionStatus.SUCCESS,
        SeedExecutionStatus.SUCCESS,
    ]
    assert [row.seed for row in result.rows] == ["1337", "1338", "1339"]

    record = json.loads(report.read_text(encoding="utf-8"))
    assert record["event"] == "seed_campaign_report"
    assert record["claimable"] is True
    assert record["summary"]["threshold_met"] is True
    assert record["summary"]["infra_errors"] == 0
    assert record["config"]["goal"] == "ship_to_morph"
    assert record["config"]["metadata"]["spoiler_oracle"] is False
    assert record["config"]["metadata"]["substrate"] == "vanilla"
    assert record["policy"] == POLICY_NAME
    for seed_row in record["seed_results"]:
        assert seed_row["execution_status"] == "success"
        assert seed_row["terminal_milestone"] == "morph_ball"
        assert seed_row["outcome"] == "success"
        assert seed_row["ram_writes"] == 0
        assert seed_row["mid_run_loads"] == 0

    assert classic.is_file()
    classic_record = json.loads(classic.read_text(encoding="utf-8"))
    assert classic_record["event"] == "seed_robustness_report"
    assert classic_record["summary"]["threshold_met"] is True
    assert validate_claim(classic_record) is True

    summary = campaign_summary(result)
    assert summary["claimable"] is True
    assert summary["threshold_met"] is True


def test_dry_campaign_infra_error_is_fail_closed_non_claimable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sm_rando import early_tip_campaign as campaign

    seeds_root = tmp_path / "seeds"
    ensure_campaign_seeds(("a", "b", "c"), seeds_root=seeds_root)

    original = campaign.build_case_for_seed

    def flaky_build(seed, *, spec, config):
        if seed == "b":
            raise RuntimeError("rom missing")
        return original(seed, spec=spec, config=config)

    monkeypatch.setattr(campaign, "build_case_for_seed", flaky_build)

    result = campaign.run_early_tip_campaign(
        mode="dry",
        seeds=("a", "b", "c"),
        success_threshold=2,
        budget=8,
        seeds_root=seeds_root,
        ledger_path=tmp_path / "ledger.json",
        report_path=tmp_path / "campaign.json",
        write_classic=False,
    )

    assert result.claimable is False
    assert result.infra_error_count == 1
    assert result.rows[1].status is SeedExecutionStatus.INFRA_ERROR
    assert result.rows[1].error and "rom missing" in result.rows[1].error
    assert result.threshold_met is False
    record = result.to_record()
    assert record["seed_results"][1]["failure_mode"] == "INFRA_ERROR"
    assert record["seed_results"][1]["execution_status"] == "infra_error"
    with pytest.raises(Exception, match="non-claimable|INFRA_ERROR"):
        result.to_seed_robustness_report()


def test_contract_mismatch_refuses_resume(tmp_path: Path) -> None:
    seeds_root = tmp_path / "seeds"
    ledger = tmp_path / "ledger.json"
    report = tmp_path / "campaign.json"

    run_early_tip_campaign(
        mode="dry",
        seeds=("1337", "1338"),
        success_threshold=1,
        budget=8,
        seeds_root=seeds_root,
        ledger_path=ledger,
        report_path=report,
        write_classic=False,
    )

    with pytest.raises(SeedCampaignContractMismatch):
        run_early_tip_campaign(
            mode="dry",
            seeds=("1337", "1338"),
            success_threshold=2,  # contract change
            budget=8,
            seeds_root=seeds_root,
            ledger_path=ledger,
            report_path=report,
            write_classic=False,
        )


def test_non_vanilla_morph_fixture_records_failure(tmp_path: Path) -> None:
    seeds_root = tmp_path / "seeds"
    # Custom package: Morph replaced so seed-agnostic tip cannot claim clear.
    directory = seeds_root / "fixture_x"
    pkg = write_fixture_seed(seed_number="x", name="fixture_x", directory=directory)
    pkg.locations = [
        {"location": "Morphing Ball", "item": "Bombs", "area": "Crateria"},
        {"location": "Bombs", "item": "Morphing Ball", "area": "Crateria"},
    ]
    pkg.write()
    assert morph_reachable_without_spoiler(SeedPackage.load(directory)) is False

    # Only one seed in campaign — threshold 1; expect failure not infra.
    result = run_early_tip_campaign(
        mode="dry",
        seeds=("x",),
        success_threshold=1,
        budget=8,
        seeds_root=seeds_root,
        ledger_path=tmp_path / "ledger.json",
        report_path=tmp_path / "campaign.json",
        write_classic=False,
    )
    assert result.claimable is True  # game failure, not infra
    assert result.successes == 0
    assert result.threshold_met is False
    assert result.rows[0].status is SeedExecutionStatus.FAILURE
    assert result.rows[0].result is not None
    assert result.rows[0].result.failure_mode == "morph_not_at_vanilla_location"


def test_policy_name_and_config_metadata() -> None:
    policy = ShipToMorphCampaignPolicy()
    assert policy.name == POLICY_NAME
    spec = default_spec(mode="dry", seeds=DEFAULT_SEEDS)
    config = build_campaign_config(spec)
    assert config.goal == "ship_to_morph"
    assert config.metadata["spoiler_oracle"] is False
    assert config.metadata["edge"] == "ship_to_morph"
    assert config.success_threshold == 2
    assert config.seed_count == 3
