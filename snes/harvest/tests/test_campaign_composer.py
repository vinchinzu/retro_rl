"""Backward milestone solver and season-pin lineage. No emulator."""

from __future__ import annotations

import numpy as np
import yaml

from harvest.planner.campaign_composer import (
    BLOCK_ORDER,
    CROP_TARGET,
    CROP_WINDOW,
    DESIGNED_BERRIES,
    DUPLICATION_BERRIES,
    LAST_FEEDABLE_LATEST,
    MILESTONES_PATH,
    POWER_BERRY_TARGET,
    SCORE_TARGET,
    SeasonPinSession,
    apply_block_order,
    arbitration_journal,
    calendar_ymd,
    day_intents_for_morning,
    milestones_by_id,
    morning_index,
    parent_run_hash,
    render_milestones,
    run_whatif_day,
    season_pin_name,
    solve_milestones,
    write_milestones,
)
from harvest.planner.day_phase_types import PhaseSpec
from harvest.runtime.save_state_io import parse_save_state, resolve_state_path
from harvest.scripts.run_to_day2 import _parse_args


class _Planner:
    def __init__(self) -> None:
        self._phase = "plan_day"
        self._current_task = None


# Farm worktree CROP_QUOTA_MILESTONES: three slices of 168 land on 504.
# Potato and turnip close Y3 Spring D30. Tomato and corn close Y3 Summer D29.
_QUOTA_SLICES: dict[str, tuple[tuple[int, str, int, int], ...]] = {
    "potato": ((1, "Spring", 30, 168), (2, "Spring", 30, 168), (3, "Spring", 30, 168)),
    "turnip": ((1, "Spring", 30, 168), (2, "Spring", 30, 168), (3, "Spring", 30, 168)),
    "tomato": ((1, "Summer", 30, 168), (2, "Summer", 30, 168), (3, "Summer", 29, 168)),
    "corn": ((1, "Summer", 30, 168), (2, "Summer", 30, 168), (3, "Summer", 29, 168)),
}


def _quota_dicts(crop: str) -> list[dict]:
    return [
        {"year": year, "season": season, "day": day, "shipped": shipped}
        for year, season, day, shipped in _QUOTA_SLICES[crop]
    ]


def _reject_overshoot_target(target: int, quotas: list[dict]) -> None:
    """Refuse a crop target, or a running ship total, past the high side."""
    if int(target) > CROP_WINDOW[1]:
        raise ValueError(f"overshoot target {target}")
    running = 0
    for row in quotas:
        running += int(row["shipped"])
        if running > CROP_WINDOW[1]:
            raise ValueError(f"overshoot target {running}")
    if int(target) != running or not (CROP_WINDOW[0] <= running <= CROP_WINDOW[1]):
        raise ValueError(f"crop target {target} shipped {running} outside {CROP_WINDOW[0]}-{CROP_WINDOW[1]}")


def _jsonish(value: object) -> object:
    if isinstance(value, dict):
        return {key: _jsonish(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(item) for item in value]
    return value


def _disk_milestones() -> dict[str, dict]:
    loaded = yaml.safe_load(MILESTONES_PATH.read_text(encoding="utf-8"))
    assert loaded["target"]["ranch_mastery"] == SCORE_TARGET
    assert loaded["target"]["power_berry_n"] == POWER_BERRY_TARGET
    return {row["id"]: row for row in loaded["milestones"]}


def test_ten_day_milestone_move_changes_slack() -> None:
    base = milestones_by_id()
    moved = milestones_by_id({"wife_bit": 10})
    assert base["wife_bit"].slack_days == 0
    assert moved["wife_bit"].slack_days == base["wife_bit"].slack_days - 10
    assert moved["wife_bit"].committed_index == base["wife_bit"].committed_index + 10
    assert moved["child_2"].slack_days == base["child_2"].slack_days - 10
    assert moved["child_1"].slack_days == base["child_1"].slack_days - 10
    assert moved["child_2"].latest_index == base["child_2"].latest_index
    potato = milestones_by_id({"crop_potato": 10})
    assert potato["crop_potato"].slack_days == base["crop_potato"].slack_days - 10
    assert potato["crop_turnip"].slack_days == base["crop_turnip"].slack_days
    assert len([row for row in solve_milestones({"wife_bit": 10}) if row.data.get("eat")]) == 10


def test_power_berry_count_is_exactly_ten() -> None:
    rows = solve_milestones()
    eats = [row for row in rows if row.data.get("eat") is True]
    assert len(eats) == POWER_BERRY_TARGET
    assert sum(row.data.get("edge") == "designed" for row in eats) == DESIGNED_BERRIES
    assert sum(row.data.get("edge") == "duplication" for row in eats) == DUPLICATION_BERRIES
    summary = milestones_by_id()["power_berry_n"]
    assert summary.data["target"] == 10
    assert summary.data["designed"] == 7
    assert summary.data["duplication"] == 3
    text = render_milestones(rows)
    assert "ranch_mastery: 999" in text
    assert "power_berry_n: 10" in text
    assert "ranch_mastery: 984" not in text
    assert ": 984\n" not in text
    assert SCORE_TARGET == 999


def test_crop_quotas_and_f_gate_deadlines() -> None:
    found = milestones_by_id()
    assert found["wife_bit"].latest_index == 128
    assert found["wife_bit"].data["children_present_latest_index"] == 188
    assert found["child_2"].committed_index == 238
    assert found["child_2_age60"].committed_index == 298
    assert found["cow_slot0"].data["feedable"] is True
    assert found["cow_slot0"].latest_index < found["cow_feedable_01"].latest_index
    assert "cover" not in found["cow_slot0"].depends_on
    assert "namings" not in found["happiness"].data
    assert found["cow_feedable_11"].latest_index == LAST_FEEDABLE_LATEST == 249
    assert found["cow_feedable_11"].slack_days == 249 - found["cow_feedable_11"].committed_index
    assert found["cow_feedable_11"].slack_days > 0
    for crop in ("potato", "turnip"):
        row = found[f"crop_{crop}"]
        assert row.latest_index == 269
        assert row.data["target"] == CROP_TARGET
        assert sum(item["shipped"] for item in row.data["quotas"]) == CROP_TARGET
        assert CROP_WINDOW[0] <= CROP_TARGET <= CROP_WINDOW[1]
        assert all(item["shipped"] <= CROP_WINDOW[1] for item in row.data["quotas"])
    for crop in ("tomato", "corn"):
        row = found[f"crop_{crop}"]
        assert row.latest_index == 298
        assert row.data["quotas"] == _quota_dicts(crop)
        assert row.slack_days == 298 - row.committed_index
    assert found["cover"].data["minimum_tiles"] == 2950
    assert found["money_hold"].data["end_g"] == 620_000
    assert found["hen"].data["target"] == 12


def test_overshoot_crop_target_is_rejected() -> None:
    disk = _disk_milestones()
    for crop in _QUOTA_SLICES:
        row = disk[f"crop_{crop}"]
        close = "Y3 Spring D30" if crop in ("potato", "turnip") else "Y3 Summer D29"
        assert row["latest"] == close
        assert row["window_lo"] == CROP_WINDOW[0]
        assert row["window_hi"] == CROP_WINDOW[1]
        assert row["quotas"] == _quota_dicts(crop)
        _reject_overshoot_target(int(row["target"]), row["quotas"])
    for refused in (
        (512, [{"shipped": 512}]),
        (504, [{"shipped": 168}, {"shipped": 168}, {"shipped": 176}]),
        (520, _quota_dicts("potato")),
    ):
        try:
            _reject_overshoot_target(refused[0], refused[1])
        except ValueError as exc:
            assert str(exc).startswith("overshoot target")
        else:
            raise AssertionError(f"overshoot target {refused[0]} was accepted")


def test_milestones_yaml_matches_the_solver(tmp_path) -> None:
    # Crop quotas on disk are the farm table. Do not rewrite that file.
    copied = write_milestones(tmp_path / "milestones.yaml")
    assert copied.read_text(encoding="utf-8") == render_milestones()
    disk = _disk_milestones()
    solved = milestones_by_id()
    assert list(disk) == [row.id for row in solve_milestones()]
    for ident, row in disk.items():
        solved_row = dict(_jsonish(solved[ident].as_dict()))
        if not ident.startswith("crop_"):
            assert row == solved_row
            continue
        quotas = row["quotas"]
        solved_quotas = solved_row.pop("quotas")
        disk_row = dict(row)
        disk_row.pop("quotas")
        assert disk_row == solved_row
        assert quotas == _quota_dicts(ident.removeprefix("crop_"))
        assert sum(item["shipped"] for item in solved_quotas) == CROP_TARGET
        assert all(int(item["shipped"]) <= CROP_WINDOW[1] for item in solved_quotas)
    assert disk["crop_potato"]["quotas"] == _quota_dicts("potato")


def test_day_intents_are_the_yaml_commitments_for_that_morning() -> None:
    disk = _disk_milestones()
    for index in range(300):
        year, season, day = calendar_ymd(index)
        assert morning_index(year - 1, season, day) == index
    expected = {
        index: [row["id"] for row in disk.values() if int(row["committed_index"]) == index]
        for index in range(5)
    }
    assert expected[0] == ["berry_field_1", "berry_dup_transition"]
    assert expected[1] == ["berry_field_2"]
    assert expected[2] == ["berry_stump"]
    assert expected[3] == ["berry_statue"]
    assert expected[4] == ["berry_cave", "berry_dup_cave"]
    for index, ids in expected.items():
        _year, season, day = calendar_ymd(index)
        picked = day_intents_for_morning(_year - 1, season, day)
        assert [row["id"] for row in picked] == ids
        for intent, ident in zip(picked, ids):
            assert intent["committed_index"] == index
            assert intent["date"] == disk[ident]["date"]
            assert intent["term"] == disk[ident]["term"]
            assert intent.get("eat") is True
            assert intent["edge"] == disk[ident]["edge"]
    assert day_intents_for_morning(0, 0, 7) == []
    assert disk["power_berry_n"]["target"] == 10
    assert sum(1 for row in disk.values() if row.get("eat") is True) == 10


def test_planner_journal_records_milestone_intents() -> None:
    import sys
    from pathlib import Path

    tests = Path(__file__).resolve().parent
    if str(tests) not in sys.path:
        sys.path.insert(0, str(tests))
    from day_plan_test_helpers import make_date_world
    from harvest.planner.multi_day_planner import MultiDayPlannerTask

    planner = MultiDayPlannerTask(until_season=0, until_day=4)
    world = make_date_world(0x15, season=0, day=2)
    planner.reset(world)
    planner._plan_year = 0
    planner._active_day = (0, 2)
    planner._journal_day_complete(world, sleep_reason="test")
    row = planner.day_journal[-1]
    assert row["intent_source"] == "milestones.yaml"
    assert [item["id"] for item in row["intents"]] == ["berry_field_2"]
    assert row["intents"][0]["edge"] == "designed"
    planner._active_day = (0, 4)
    planner._journal_day_complete(world, sleep_reason="test")
    assert [item["id"] for item in planner.day_journal[-1]["intents"]] == ["berry_statue"]


def test_perfect_planner_defers_unsupported_intent_before_selecting_tasks(monkeypatch) -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from day_plan_test_helpers import make_date_world
    from harvest.planner.multi_day_planner import MultiDayPlannerTask
    import harvest.planner.multi_day_planner as module

    intent = {"id": "unimplemented_prize", "term": "clock", "committed_index": 1}
    monkeypatch.setattr(module, "day_intents_for_morning", lambda *args: [intent])
    planner = MultiDayPlannerTask()
    planner.perfect_rancher = True
    world = make_date_world(0x15, season=0, day=2)
    planner.reset(world)
    planner._build_day_task(world)
    deferred = planner.last_day_decision.deferred
    assert any(row.params.get("intent_id") == intent["id"] for row in deferred)
    assert any("native_skill_missing" in row.reason for row in deferred)


def test_save_season_pins_parses_without_starting_a_run() -> None:
    args = _parse_args(["--save-season-pins", "--state", "PR_Y1_Summer_D1", "--trials", "1"])
    assert args.save_season_pins is True
    assert args.trials == 1
    assert _parse_args([]).save_season_pins is False


def test_resume_trace_matches_the_parent_next_day(tmp_path) -> None:
    ram = np.frombuffer(parse_save_state(resolve_state_path("Y1_Summer_D1_Pin")).ram, dtype=np.uint8).copy()
    saved: list[Path] = []

    def _save(path):
        path.write_bytes(b"pin")
        saved.append(path)
        return path

    parent = SeasonPinSession(
        save=True,
        power_on=False,
        source_sha256="abc",
        start_key=(0, 0, 30),
        state_name="Y1_D30_Morning_Pin",
        save_dir=tmp_path,
    )
    parent.before_step(_Planner(), ram, frame=12880, save_state=_save)
    assert saved and saved[0].name == "PR_Y1_Summer_D1.state"
    assert season_pin_name(0, 1) == "PR_Y1_Summer_D1"
    parent.note([0, 1, 0], ram)
    parent.note([0, 0, 0], ram)
    first = parent.finish()
    assert first["pins"][0]["parent_run_hash"] == parent_run_hash(
        power_on=False, source_sha256="abc", start_key=(0, 0, 30)
    )
    assert first["next_day"]["frames"] == 2
    resume = SeasonPinSession(
        save=False,
        power_on=False,
        source_sha256="def",
        start_key=(0, 1, 1),
        state_name="PR_Y1_Summer_D1",
        save_dir=tmp_path,
    )
    assert resume.loaded["parent_run_hash"] == first["parent_run_hash"]
    resume.before_step(_Planner(), ram, frame=0, save_state=_save)
    resume.note([0, 1, 0], ram)
    resume.note([0, 0, 0], ram)
    second = resume.finish()
    assert second["next_day_match"] is True
    assert second["next_day"]["frames"] == 2
    drift = SeasonPinSession(
        save=False,
        power_on=False,
        source_sha256="def",
        start_key=(0, 1, 1),
        state_name="PR_Y1_Summer_D1",
        save_dir=tmp_path,
    )
    drift.before_step(_Planner(), ram, frame=0, save_state=_save)
    other = ram.copy()
    other[0] ^= 1
    drift.note([0, 1, 0], other)
    drift.note([0, 0, 0], ram)
    assert drift.finish()["next_day_match"] is False


def _phase(name: str, kind: str, frames: int = 0) -> PhaseSpec:
    return PhaseSpec(name, kind, estimated_frames=frames or None)


def test_arbiter_orders_deadline_then_animals_crops_social_cover() -> None:
    phases = [
        _phase("EXIT_TO_FARM", "exit"),
        _phase("COW_CHORES", "cow_chores"),
        _phase("NAV_TO_BARN", "multi_nav"),
        _phase("CROP_WATER", "crop"),
        _phase("MOUNTAIN_BERRY", "mountain_berry"),
        _phase("EVE_TALK_LOOP", "eve_talk_loop"),
        _phase("PLANT_PASTURE", "grass"),
        _phase("CLEAR_FIELD", "clear_field"),
        _phase("RETURN_HOME", "return_home"),
        _phase("GO_TO_SLEEP", "sleep"),
    ]
    # Barn nav is listed after chores on purpose. It stays with the animals.
    ordered = [phase.phase for phase in apply_block_order(phases)]
    assert ordered == [
        "EXIT_TO_FARM",
        "MOUNTAIN_BERRY",
        "COW_CHORES",
        "NAV_TO_BARN",
        "CROP_WATER",
        "EVE_TALK_LOOP",
        "PLANT_PASTURE",
        "CLEAR_FIELD",
        "RETURN_HOME",
        "GO_TO_SLEEP",
    ]
    assert apply_block_order([], storm=True) == []
    assert [phase.phase for phase in apply_block_order([_phase("GO_TO_SLEEP", "sleep")], storm=True)] == [
        "GO_TO_SLEEP"
    ]


def test_arbitration_journal_has_a_frame_account_per_block() -> None:
    phases = [
        _phase("MOUNTAIN_BERRY", "mountain_berry", 100),
        _phase("COW_CHORES", "cow_chores", 80),
        _phase("CROP_WATER", "crop", 50),
        _phase("EVE_TALK_LOOP", "eve_talk_loop", 40),
        _phase("PLANT_PASTURE", "grass", 30),
        _phase("CLEAR_FIELD", "clear_field", 20),
    ]
    results = []
    step = 0
    for phase, frames in (
        ("MOUNTAIN_BERRY", 100),
        ("COW_CHORES", 80),
        ("CROP_WATER", 50),
        ("EVE_TALK_LOOP", 40),
        ("PLANT_PASTURE", 30),
        ("CLEAR_FIELD", 20),
    ):
        step += frames
        results.append({"phase": phase, "status": "success", "step": step})
    journal = arbitration_journal(phases, results)
    assert journal["order"] == list(BLOCK_ORDER)
    assert [row["block"] for row in journal["blocks"]] == list(BLOCK_ORDER)
    by_name = {row["block"]: row for row in journal["blocks"]}
    assert by_name["deadline"]["lane"] == "day"
    assert by_name["cover"]["lane"] == "post_18"
    assert by_name["hoeing"]["lane"] == "post_18"
    assert by_name["cover"]["phases"] == ["PLANT_PASTURE"]
    assert by_name["hoeing"]["phases"] == ["CLEAR_FIELD"]
    for row in journal["blocks"]:
        assert row["account"] > 0
        assert row["frames"] == row["account"]
        assert "frames" in row and "account" in row


def test_full_scale_whatif_day_runs_every_block() -> None:
    journal = run_whatif_day()
    assert journal["evidence_kind"] == "model"
    assert journal["emulator_frames"] == 0
    assert "ranch_mastery" not in journal and "power_berry_n" not in journal
    assert journal["order"] == list(BLOCK_ORDER)
    assert journal["within_envelope"] is True
    assert journal["day_frames"] <= journal["day_budget"]
    assert [row["block"] for row in journal["blocks"]] == [
        "deadline",
        "animals",
        "crops",
        "social",
        "cover",
        "hoeing",
    ]
    post = [row for row in journal["blocks"] if row["lane"] == "post_18"]
    assert [row["block"] for row in post] == ["cover", "hoeing"]
    assert sum(row["hot_spring_refills"] for row in post) >= 1
    for row in journal["blocks"]:
        assert row["account"] > 0
        assert row["frames"] > 0
        assert row["frames"] <= row["account"]
        assert row["done"] == row["units"]
        assert row["units"] > 0
    # A tight clock still runs deadline before animals, and still hoes after 18:00.
    tight = run_whatif_day({"day_budget": 4000})
    tight_by = {row["block"]: row for row in tight["blocks"]}
    assert tight_by["deadline"]["done"] == tight_by["deadline"]["units"]
    assert tight_by["animals"]["done"] < tight_by["animals"]["units"]
    assert tight_by["crops"]["frames"] == 0
    assert tight_by["social"]["frames"] == 0
    assert tight_by["cover"]["done"] == tight_by["cover"]["units"]
    assert tight_by["hoeing"]["done"] == tight_by["hoeing"]["units"]
    assert tight_by["cover"]["hot_spring_refills"] + tight_by["hoeing"]["hot_spring_refills"] >= 1


def test_planner_journal_records_per_block_frames() -> None:
    import sys
    from pathlib import Path

    tests = Path(__file__).resolve().parent
    if str(tests) not in sys.path:
        sys.path.insert(0, str(tests))
    from day_plan_test_helpers import make_date_world
    from harvest.planner.day_plan_decision import DayPlanDecision, PlanningFacts
    from harvest.planner.multi_day_planner import MultiDayPlannerTask

    planner = MultiDayPlannerTask(until_season=0, until_day=1)
    world = make_date_world(0x15, season=0, day=2)
    planner.reset(world)
    planner._plan_year = 0
    planner._active_day = (0, 2)
    planner._last_day_decision = DayPlanDecision(
        phases=(
            _phase("CROP_WATER", "crop", 50),
            _phase("MOUNTAIN_BERRY", "mountain_berry", 100),
            _phase("COW_CHORES", "cow_chores", 80),
            _phase("EVE_TALK_LOOP", "eve_talk_loop", 40),
            _phase("CLEAR_FIELD", "clear_field", 20),
            _phase("PLANT_PASTURE", "grass", 30),
        ),
        facts=PlanningFacts(source="test", weekday=1, hour=6),
    )
    step = 0
    planner._last_day_phase_results = []
    for name, kind, frames in (
        ("MOUNTAIN_BERRY", "mountain_berry", 100),
        ("COW_CHORES", "cow_chores", 80),
        ("CROP_WATER", "crop", 50),
        ("EVE_TALK_LOOP", "eve_talk_loop", 40),
        ("PLANT_PASTURE", "grass", 30),
        ("CLEAR_FIELD", "clear_field", 20),
    ):
        step += frames
        planner._last_day_phase_results.append(
            {"phase": name, "kind": kind, "status": "success", "step": step}
        )
    planner._journal_day_complete(world, sleep_reason="test")
    journal = planner.day_journal[-1]
    arbitration = journal["arbitration"]
    assert arbitration["order"] == list(BLOCK_ORDER)
    by_name = {row["block"]: row for row in arbitration["blocks"]}
    assert [row["block"] for row in arbitration["blocks"]][:5] == [
        "deadline",
        "animals",
        "crops",
        "social",
        "cover",
    ]
    for row in arbitration["blocks"]:
        assert row["frames"] == row["account"] > 0
    assert by_name["cover"]["lane"] == "post_18"
    assert by_name["hoeing"]["lane"] == "post_18"
    assert journal["intent_source"] == "milestones.yaml"
    whatif = run_whatif_day()
    planner._day_journal.append(whatif)
    recorded = planner.day_journal[-1]
    assert recorded["evidence_kind"] == "model"
    assert recorded["emulator_frames"] == 0
    assert all(row["frames"] > 0 for row in recorded["blocks"])


def test_native_intent_is_selected_before_advisor_and_has_registered_skill() -> None:
    from day_plan_test_helpers import make_date_world
    from harvest.planner.day_plan_decision import auto_day_plan_decision
    from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
    from harvest.planner.tasks.chicken_sale import CowPurchaseTask

    seen = []
    class Advisor:
        def advise_day_plan(self, decision):
            seen.extend(decision.phases)
            return decision
    world = make_date_world(0x15, season=0, day=3)
    decision = auto_day_plan_decision(ram=world.ram, advisor=Advisor(),
                                     milestone_intents=[{"id": "hen"}])
    phase = next(row for row in seen if row.params.get("intent_id") == "hen")
    assert phase in decision.phases
    task = build_phase_task(TaskBuildContext(), phase, world)
    assert isinstance(task, CowPurchaseTask)
    assert task.species == "chicken"


def test_cow_count_without_an_initialized_slot_is_not_a_receipt() -> None:
    from harvest.planner.campaign_composer import verify_intent_receipt
    spec = PhaseSpec("BUY", "cow_purchase", {"intent_id": "cow_slot0"})
    before = {"num_cows": 0, "cow_slots": ()}
    after = {"num_cows": 1, "cow_slots": ()}
    assert not verify_intent_receipt(spec, before, after)
    assert verify_intent_receipt(spec, before, {**after, "cow_slots": (0,)})


def test_generic_day_keeps_rule_order_and_has_no_campaign_dispatch(monkeypatch) -> None:
    from day_plan_test_helpers import make_date_world
    from harvest.planner.day_plan_decision import DayPlanDecision, PlanningFacts
    from harvest.planner.multi_day_planner import MultiDayPlannerTask
    import harvest.planner.multi_day_planner as module
    phases = (_phase("CROP_WATER", "crop"), _phase("MOUNTAIN_BERRY", "mountain_berry"))
    decision = DayPlanDecision(phases=phases, facts=PlanningFacts(source="test", hour=6, weekday=3))
    calls = []
    def select(**kwargs):
        calls.append(kwargs)
        return decision
    monkeypatch.setattr(module, "auto_day_plan_decision", select)
    planner = MultiDayPlannerTask()
    world = make_date_world(0x15, season=0, day=3)
    planner.reset(world)
    task = planner._build_day_task(world)
    assert task.phases == phases
    assert task.arbitrate is False
    assert calls[0]["milestone_intents"] == ()


def test_intent_success_requires_rom_receipt_and_frame_clock(monkeypatch) -> None:
    from dataclasses import replace
    from types import SimpleNamespace
    from day_plan_test_helpers import make_date_world, set_live_u16
    from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseKind
    from harvest.planner.day_phase_registry import PHASE_TASK_BUILDERS
    from harvest.planner.day_plan_orchestrator import DayPlanTask
    from retro_harness import TaskResult, TaskStatus

    fake = SimpleNamespace(reset=lambda world: None,
                           step=lambda world: TaskResult(status=TaskStatus.SUCCESS))
    monkeypatch.setitem(PHASE_TASK_BUILDERS, PhaseKind.COW_PURCHASE,
                        lambda *args: fake)
    phase = PhaseSpec("MILESTONE_HEN", "cow_purchase", {"intent_id": "hen"},
                      failure_policy="opportunistic")
    policy = replace(DayPlannerPolicy(), include_end_day=False)
    world = make_date_world(0x00, season=0, day=3)
    world.frame = 100
    plan = DayPlanTask(phase_sequence=[phase], policy=policy)
    plan.reset(world)
    plan.step(world)
    assert plan.phase_results[0]["status"] == "skipped"
    assert plan.phase_results[0]["receipt"]["verified"] is False
    assert plan.phase_results[0]["frames"] == 0

    # Unit seam: mutate synthetic RAM between actions. This is not ROM evidence.
    calls = []
    def step(world):
        calls.append(1)
        return TaskResult(status=TaskStatus.RUNNING if len(calls) == 1 else TaskStatus.SUCCESS)
    fake.step = step
    plan.reset(world)
    plan.step(world)
    from harvest.core.ram_catalog import field_spec
    world.ram[field_spec("num_chickens").address + 0x4000] = 1
    world.frame = 130
    plan.step(world)
    receipt = plan.phase_results[0]
    assert receipt["status"] == "success"
    assert receipt["receipt"]["verified"] is True
    assert receipt["receipt"]["before"]["num_chickens"] == 0
    assert receipt["receipt"]["after"]["num_chickens"] == 1
    assert receipt["frames"] == 30
