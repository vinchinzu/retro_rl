"""The fuzz harness: a seed replays exactly, every profile produces what it promises, a failure is named."""
from __future__ import annotations

import sys
import dataclasses
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from cow_test_helpers import make_barn_ram, set_cow, write_u16

from harvest.core.animal_status import COW_DAILY_BRUSHED_FLAG, COW_DAILY_MILKED_FLAG, COW_DAILY_TALKED_FLAG
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.tools.barn_fuzz import PROFILES, Receipt, judge, make_scenario

DONE = COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG | COW_DAILY_MILKED_FLAG


def test_a_seed_is_a_full_replay():
    assert make_scenario(7, "all") == make_scenario(7, "all")
    assert make_scenario(7, "herd") != make_scenario(8, "herd")


@pytest.mark.parametrize("profile", [name for name in PROFILES if name != "all"])
def test_each_profile_stays_in_its_lane(profile):
    for seed in range(40):
        scenario = make_scenario(seed, profile)
        assert scenario.profile == profile
        assert 1 <= len(scenario.cows) <= 12
        assert len({cow.slot for cow in scenario.cows}) == len(scenario.cows)
        if profile in ("twelve", "wander", "knock", "morning"):
            assert len(scenario.cows) == 12
        if profile == "knock":
            assert scenario.kick_rate > 0
        if profile == "outside":
            assert scenario.outside is not None


def test_all_reaches_every_profile():
    assert {make_scenario(seed, "all").profile for seed in range(120)} == set(PROFILES) - {"all"}


def scenario_for(cows: int = 2, **changes):
    base = make_scenario(1, "twelve")
    fields = dict(base.__dict__)
    fields["cows"] = base.cows[:cows]
    fields.update(changes)
    return type(base)(**fields)


def finished_ram(cows: int = 2):
    ram = make_barn_ram(cows=cows, fed_flags=(1 << cows) - 1)
    ram[ADDR_TILEMAP] = 0x00
    for slot in range(cows):
        set_cow(ram, slot, flags=DONE)
    return ram


def verdict(ram, scenario=None, **receipt):
    row = Receipt(plan="success", **receipt)
    task = SimpleNamespace(phase_results=())
    return judge(ram, scenario or scenario_for(), task, row)


def test_a_finished_morning_is_clean():
    assert verdict(finished_ram()).ok


def test_each_missing_receipt_is_named():
    ram = finished_ram()
    set_cow(ram, 1, flags=COW_DAILY_MILKED_FLAG)
    problems = verdict(ram).problems
    assert any(text.startswith("untalked") for text in problems)
    assert any(text.startswith("unbrushed") for text in problems)


def test_a_cow_left_outside_fails_the_trial():
    from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE

    ram = finished_ram()
    ram[COW_SLOT_BASE + COW_SLOT_SIZE + 2] = 1
    assert any(text.startswith("outside") for text in verdict(ram).problems)


def test_short_hay_excuses_the_unfed_but_nothing_else():
    ram = finished_ram()
    ram2 = make_barn_ram(cows=2, hay=0)
    ram2[ADDR_TILEMAP] = 0x00
    for slot in range(2):
        set_cow(ram2, slot, flags=DONE)
    assert verdict(ram2, scenario_for(hay=0)).ok
    assert any(text.startswith("unfed") for text in verdict(ram2, scenario_for(hay=900)).problems)
    assert verdict(ram, scenario_for(hay=0)).ok


def test_a_tired_farmer_is_excused_the_brush_only():
    ram = finished_ram()
    set_cow(ram, 0, flags=COW_DAILY_TALKED_FLAG | COW_DAILY_MILKED_FLAG)
    assert verdict(ram, scenario_for(stamina=0)).ok
    assert not verdict(ram, scenario_for(stamina=50)).ok


def test_a_missing_cow_cannot_make_the_receipt_clean():
    from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE

    ram = finished_ram()
    ram[COW_SLOT_BASE + COW_SLOT_SIZE] = 0
    assert "missing:[1]" in verdict(ram).problems


@pytest.mark.parametrize("hay,fed,clean", [(1, 0, False), (1, 1, True), (1, 3, True), (0, 0, True)])
def test_a_hay_shortage_still_requires_using_the_available_hay(hay, fed, clean):
    from harvest.core.animal_status import ADDR_FED_COWS_FLAGS

    ram = finished_ram()
    write_u16(ram, ADDR_FED_COWS_FLAGS, fed)
    assert verdict(ram, scenario_for(hay=hay)).ok == clean


def test_prefed_adults_do_not_excuse_an_unfed_cow():
    from harvest.core.animal_status import ADDR_FED_COWS_FLAGS

    ram = finished_ram()
    write_u16(ram, ADDR_FED_COWS_FLAGS, 1)
    assert not verdict(ram, scenario_for(hay=1, fed_flags=1)).ok


def test_pregnant_cows_share_one_fodder_requirement():
    from harvest.core.animal_status import ADDR_FED_COWS_FLAGS
    from harvest.tools.barn_fuzz import STATUS_PREGNANT

    scenario = scenario_for(hay=1)
    scenario = dataclasses.replace(scenario, cows=tuple(dataclasses.replace(cow, status=STATUS_PREGNANT) for cow in scenario.cows))
    ram = finished_ram()
    for cow in scenario.cows:
        set_cow(ram, cow.slot, status=STATUS_PREGNANT, flags=DONE)
    write_u16(ram, ADDR_FED_COWS_FLAGS, 0)
    assert not verdict(ram, scenario).ok
    write_u16(ram, ADDR_FED_COWS_FLAGS, 0x1000)
    assert verdict(ram, scenario).ok


@pytest.mark.parametrize("field,value,problem", [
    ("plan", "timeout", "plan:"),
    ("recoveries", 1, "recoveries:"),
    ("escaped", [0], "escaped:"),
])
def test_terminal_failures_are_never_excused_by_complete_care(field, value, problem):
    row = Receipt(plan="success")
    setattr(row, field, value)
    result = judge(finished_ram(), scenario_for(), SimpleNamespace(phase_results=()), row)
    assert not result.ok
    assert any(text.startswith(problem) for text in result.problems)


def test_an_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="profile"):
        make_scenario(1, "typo")


@pytest.mark.parametrize("field", ["ram_writes", "mid_run_loads"])
def test_a_runtime_intervention_fails_the_trial(field):
    assert any(text.startswith("interventions:") for text in verdict(finished_ram(), **{field: 1}).problems)


def test_empty_hands_and_milk_flags_do_not_prove_payment():
    from harvest.core.ram_catalog import field_spec
    from cow_test_helpers import write_u24

    ram = finished_ram()
    write_u24(ram, field_spec("shipping_money").address, 20)
    assert "unpaid milk:0/300" in verdict(ram, shipping_before=200).problems
    write_u24(ram, field_spec("shipping_money").address, 50)
    result = verdict(ram, shipping_before=200)
    assert result.ok
    assert result.shipping_after == 500 and result.milk_value == 300


def test_already_milked_cows_owe_no_second_payment():
    scenario = scenario_for()
    scenario = dataclasses.replace(scenario, cows=tuple(dataclasses.replace(cow, flags=DONE) for cow in scenario.cows))
    assert verdict(finished_ram(), scenario, shipping_before=0).ok


def test_reports_fail_for_red_incomplete_or_mixed_batches(tmp_path, monkeypatch, capsys):
    from harvest.tools import barn_fuzz as fuzz

    monkeypatch.setattr(fuzz, "OUT", tmp_path)
    folder = tmp_path / "batch"
    folder.mkdir()
    manifest = {"seeds": [1, 2], "source_digest": "same"}
    (folder / "batch.json").write_text(json.dumps(manifest))
    row = {"seed": 1, "profile": "twelve", "source_digest": "same", "receipt": {"ok": True}}
    (folder / "1.json").write_text(json.dumps(row))
    args = SimpleNamespace(tag="batch")
    assert fuzz.command_report(args) == 1
    assert "incomplete batch" in capsys.readouterr().out
    row["seed"] = 2
    (folder / "2.json").write_text(json.dumps(row))
    assert fuzz.command_report(args) == 0
    row["source_digest"] = "different"
    (folder / "2.json").write_text(json.dumps(row))
    assert fuzz.command_report(args) == 1
    assert "controller changed" in capsys.readouterr().out
    row["source_digest"] = "same"
    row["receipt"] = {"ok": False, "problems": ["unfed:[0]"]}
    (folder / "2.json").write_text(json.dumps(row))
    assert fuzz.command_report(args) == 1


@pytest.mark.parametrize("mode", ["crash", "timeout"])
@pytest.mark.parametrize("profile,seed,expected", [("twelve", 1, "twelve"), ("all", 1028, "outside")])
def test_worker_records_infrastructure_failures_and_discards_stale_success(tmp_path, monkeypatch, mode, profile, seed, expected):
    from harvest.tools import barn_fuzz as fuzz

    monkeypatch.setattr(fuzz, "OUT", tmp_path)
    folder = tmp_path / "batch"
    folder.mkdir()
    out = folder / f"{seed}.json"
    out.write_text(json.dumps({"seed": 1, "receipt": {"ok": True}}))

    def fail(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(args[0], 600)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(fuzz.subprocess, "run", fail)
    row = fuzz._spawn_trial(("batch", profile, seed, 1000))
    assert row["profile"] == expected
    assert not row["receipt"]["ok"]
    assert row["receipt"]["problems"][0].startswith("crash:")
    assert json.loads(out.read_text()) == json.loads(json.dumps(row))
    assert (folder / f"{seed}.log").exists()


@pytest.mark.parametrize("option", ["--trials", "--workers", "--max-frames"])
def test_zero_budgets_are_rejected(option):
    from harvest.tools.barn_fuzz import main

    with pytest.raises(SystemExit) as error:
        main(["run", "unused", option, "0"])
    assert error.value.code == 2


def local_scenario(seed, profile="all"):
    from harvest.tools.barn_fuzz import PINS

    scenario = make_scenario(seed, profile)
    if not (PINS / f"{scenario.base_pin}.state").exists():
        pytest.skip(f"missing local pin {scenario.base_pin}")
    return scenario


@pytest.mark.rom
@pytest.mark.parametrize("seed", [1066, 1105, 1110, 1165, 1167])
def test_barn_fuzz_replays_regressions(seed):
    from harvest.tools.barn_fuzz import run_trial

    receipt = run_trial(local_scenario(seed))
    assert receipt.ok, dataclasses.asdict(receipt)
    assert receipt.frames <= 90_000
    assert 0 < receipt.barn_frames <= receipt.frames
    assert receipt.ram_writes == receipt.mid_run_loads == 0


@pytest.mark.rom
def test_trial_frame_budget_includes_injected_idle():
    from harvest.tools.barn_fuzz import run_trial

    scenario = local_scenario(1, "twelve")
    scenario = dataclasses.replace(scenario, cows=scenario.cows[:1], pair=(15, 14), idle_out=0, idle_in=5000)
    receipt = run_trial(scenario, max_frames=2000)
    assert receipt.plan == "timeout"
    assert receipt.frames == 2000
    assert 0 < receipt.barn_frames <= receipt.frames


@pytest.mark.rom
@pytest.mark.parametrize("point", [(56, 347), (56, 348), (56, 349), (59, 348)])
def test_near_bin_stand_pays_on_the_rom(point, monkeypatch):
    from harvest.tasks.cow_task import CowChoresTask, SHIP
    from harvest.tools.barn_fuzz import run_trial

    scenario = local_scenario(1, "twelve")
    scenario = dataclasses.replace(scenario, cows=scenario.cows[:1], pair=(15, 14), idle_out=0, idle_in=0)
    original = CowChoresTask._job_stand
    monkeypatch.setattr(CowChoresTask, "_job_stand", lambda self, ram, job:
                        (point, 0) if job.kind == SHIP else original(self, ram, job))
    receipt = run_trial(scenario)
    assert receipt.ok, dataclasses.asdict(receipt)
    assert receipt.milk_value > 0
    assert receipt.shipping_after - receipt.shipping_before == receipt.milk_value
