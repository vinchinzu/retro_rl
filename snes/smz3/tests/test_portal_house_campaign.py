"""Multi-seed portal→house S/T campaign (rr-gbd.13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retro_harness.benchmark import (
    SeedCampaignContractMismatch,
    SeedExecutionStatus,
)
from retro_harness.benchmark_claims import validate_claim
from smz3.portal_house_campaign import (
    DEFAULT_SEEDS,
    POLICY_NAME,
    PortalToHouseCampaignPolicy,
    build_campaign_config,
    campaign_summary,
    default_spec,
    ensure_campaign_seeds,
    portal_house_clearable_without_spoiler,
    run_portal_house_campaign,
)
from smz3.seed import SeedPackage, write_fixture_seed


def test_fixture_seeds_clearable_without_spoiler(tmp_path: Path) -> None:
    packages = ensure_campaign_seeds(("1337", "1338"), seeds_root=tmp_path)
    assert len(packages) == 2
    for pkg in packages:
        assert portal_house_clearable_without_spoiler(pkg) is True
        assert (pkg.directory / "meta.json").is_file()
        assert pkg.settings.get("morphlocation") == "original"


def test_dry_campaign_meets_st_threshold_and_is_claimable(tmp_path: Path) -> None:
    seeds_root = tmp_path / "seeds"
    ledger = tmp_path / "ledger.json"
    report = tmp_path / "campaign.json"
    classic = tmp_path / "classic.json"

    result = run_portal_house_campaign(
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
    assert record["config"]["goal"] == "portal_to_house"
    assert record["config"]["metadata"]["spoiler_oracle"] is False
    assert record["config"]["metadata"]["substrate"] == "fixture"
    assert record["config"]["metadata"]["seed_source"] == "fixture"
    assert record["policy"] == POLICY_NAME
    for seed_row in record["seed_results"]:
        assert seed_row["execution_status"] == "success"
        assert seed_row["terminal_milestone"] == "links_house_chest"
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
    assert summary["substrate"] == "fixture"


def test_dry_campaign_infra_error_is_fail_closed_non_claimable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smz3 import portal_house_campaign as campaign

    seeds_root = tmp_path / "seeds"
    ensure_campaign_seeds(("a", "b", "c"), seeds_root=seeds_root)

    original = campaign.build_case_for_seed

    def flaky_build(seed, *, spec, config):
        if seed == "b":
            raise RuntimeError("rom missing")
        return original(seed, spec=spec, config=config)

    monkeypatch.setattr(campaign, "build_case_for_seed", flaky_build)

    result = campaign.run_portal_house_campaign(
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

    run_portal_house_campaign(
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
        run_portal_house_campaign(
            mode="dry",
            seeds=("1337", "1338"),
            success_threshold=2,  # contract change
            budget=8,
            seeds_root=seeds_root,
            ledger_path=ledger,
            report_path=report,
            write_classic=False,
        )


def test_non_original_morph_fixture_records_failure(tmp_path: Path) -> None:
    seeds_root = tmp_path / "seeds"
    directory = seeds_root / "fixture_x"
    pkg = write_fixture_seed(
        seed_number="x",
        name="fixture_x",
        directory=directory,
        settings={"morphlocation": "randomized"},
    )
    assert portal_house_clearable_without_spoiler(SeedPackage.load(directory)) is False
    del pkg

    result = run_portal_house_campaign(
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
    assert result.rows[0].result.failure_mode == "portal_house_settings_not_seed_agnostic"


def test_policy_name_and_config_metadata() -> None:
    policy = PortalToHouseCampaignPolicy()
    assert policy.name == POLICY_NAME
    spec = default_spec(mode="dry", seeds=DEFAULT_SEEDS)
    config = build_campaign_config(spec)
    assert config.goal == "portal_to_house"
    assert config.metadata["spoiler_oracle"] is False
    assert config.metadata["edge"] == "portal_to_house"
    assert config.metadata["substrate"] == "fixture"
    assert config.success_threshold == 2
    assert config.seed_count == 3
