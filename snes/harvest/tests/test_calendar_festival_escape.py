"""Festival roam-and-talk escape (Star Night / New Year forced festivals)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pytest

from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.planner.multi_day_planner import MultiDayPlannerTask
from harvest.planner.tasks.festival_escape import (
    FESTIVAL_MAPS,
    FestivalEscapeTask,
    day_plan_owns_festival,
)
from harvest.tasks.nav import make_action
from retro_harness import ActionResult, TaskResult, TaskStatus


def _world(tilemap: int):
    ram = np.zeros(0x24000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    return SimpleNamespace(ram=ram)


@pytest.mark.rom
def test_natural_new_year_finishes_without_an_evening_wait_or_day_floor():
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.campaign_composer import season_pin_session
    from harvest.scripts.run_to_day2 import _parse_args
    from retro_harness import AuditedEnv, AuditCapabilities, WorldState

    env = AuditedEnv(make_harvest_env("PR_Y2_Spring_D1", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.new_year_test"))
    try:
        env.reset()
        planner = MultiDayPlannerTask(until_year=2)
        world = WorldState(frame=0, ram=env.get_ram(), info={}, obs=None)
        planner.reset(world)
        pins = season_pin_session(_parse_args(["--state", "PR_Y2_Spring_D1"]), (1, 0, 1), None)
        pins.before_step(planner, world.ram, 0, lambda path: path, env)
        entered = None
        for frame in range(30000):
            world = WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)
            tilemap = read_ram_value(world.ram, "tilemap")
            if tilemap == 0x3A and entered is None:
                entered = frame
            if entered is not None and tilemap != 0x3A:
                planner._sync_festival_phase(world)
                break
            result = planner.step(world)
            assert result.status == TaskStatus.RUNNING, result.reason
            env.step(result.action.action if result.action else make_action())
        else:
            pytest.fail("New Year did not finish within 30,000 frames")
        assert entered is not None
        assert frame - entered < 10000
        assert read_ram_value(world.ram, "hour") < 18
        print(f"[NEW_YEAR] {frame - entered} frames, hour={read_ram_value(world.ram, 'hour')}, no floor")
        assert not planner.floor._tripped
        assert not planner._floor_abandoned
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_unplanned_year_two_flower_festival_finishes_the_day():
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.day_plan_status import is_house_tilemap
    from retro_harness import AuditedEnv, AuditCapabilities, WorldState

    env = AuditedEnv(make_harvest_env("LN_y23_flower_failure", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.flower_escape_test"))
    try:
        env.reset()
        assert read_ram_value(env.get_ram(), "year") == 1
        assert read_ram_value(env.get_ram(), "day") == 23
        task = FestivalEscapeTask()
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(20000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert read_ram_value(env.get_ram(), "day") == 24
        assert is_house_tilemap(read_ram_value(env.get_ram(), "tilemap"))
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


class FestivalEscapeTests(unittest.TestCase):
    def test_forced_festival_time_is_excluded_from_the_work_budget(self) -> None:
        task = MultiDayPlannerTask()
        task._step_count = 100
        task.floor.begin_day(50)
        task.floor._late_since = 80
        task._sync_festival_phase(_world(0x3A))
        task._step_count = 50100
        task._sync_festival_phase(_world(0x15))
        self.assertEqual(task.floor._started_at, 50050)
        self.assertEqual(task.floor._late_since, 50080)
        self.assertIsNone(task.floor.check(50100, 18))

    def test_new_year_uses_native_talk_flags_without_waiting_for_evening(self) -> None:
        from harvest.core.ram_catalog import field_spec

        world = _world(0x3A)
        def put(name, value):
            spec = field_spec(name)
            world.ram[spec.address:spec.address + spec.width] = list(value.to_bytes(spec.width, "little"))
        put("hour", 7)
        put("input_lock", 1)
        task = FestivalEscapeTask()
        task.reset(world)
        task._npcs = lambda ram: {0: (136, 296), 1: (128, 440), 2: (120, 344)}
        task._new_year_walk = lambda pos, goal: make_action(right=True)
        task._steps = 1000
        task._clock, task._clock_frame = (7, 0), 999
        # Prayer is acknowledged by the ROM, and villager slot 2 is still due.
        world.ram[0x11F7A + 0x4000] = 1
        action = task._scripted_step(world.ram).action
        self.assertTrue(np.any(action))
        self.assertEqual(task._target, 2)
        self.assertNotIn(2, task._visited)
        world.ram[0x11F7A + 0x4000] |= 1 << 3
        task._scripted_step(world.ram)
        self.assertIn(2, task._visited)
        self.assertEqual(task._target, 1)

    def test_unplanned_flower_festival_uses_the_dance_sequence(self) -> None:
        planner = MultiDayPlannerTask()
        result = planner._step_inner(_world(0x08))
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "festival_settle")
        self.assertEqual(planner._phase, "festival")

    def test_scripted_phase_then_logged_random_fallback(self) -> None:
        import contextlib
        import io

        t = FestivalEscapeTask(scripted_budget=50, seed=1)
        w = _world(0x0A)
        t.reset(w)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for _ in range(120):
                self.assertEqual(t.step(w).status, TaskStatus.RUNNING)
        self.assertIn("random roam fallback", buf.getvalue())
        self.assertEqual(buf.getvalue().count("fallback"), 1)

    def test_runs_and_is_deterministic(self) -> None:
        a, b = FestivalEscapeTask(seed=3), FestivalEscapeTask(seed=3)
        w = _world(0x0A)
        a.reset(w)
        b.reset(w)
        for _ in range(500):
            ra, rb = a.step(w), b.step(w)
            self.assertEqual(ra.status, TaskStatus.RUNNING)
            self.assertTrue((ra.action.action == rb.action.action).all())

    def test_succeeds_once_the_map_changes(self) -> None:
        t = FestivalEscapeTask()
        t.reset(_world(0x0A))
        self.assertEqual(t.step(_world(0x15)).status, TaskStatus.SUCCESS)

    def test_new_year_and_star_night_maps_covered(self) -> None:
        self.assertIn(0x3A, FESTIVAL_MAPS)
        self.assertIn(0x0A, FESTIVAL_MAPS)

    def test_planner_takes_over_on_a_festival_map_only(self) -> None:
        task = MultiDayPlannerTask()
        task._sync_festival_phase(_world(0x15))
        self.assertNotEqual(task._phase, "festival")
        self.assertIsNone(task._festival_escape)
        task._sync_festival_phase(_world(0x0A))
        self.assertEqual(task._phase, "festival")
        result = task._current_task.step(_world(0x0A))
        self.assertEqual(result.status, TaskStatus.RUNNING)
        task._sync_festival_phase(_world(0x15))
        self.assertNotEqual(task._phase, "festival")
        self.assertIsNone(task._festival_escape)
        self.assertEqual(task._phase, "plan_day")

    def test_failed_escape_falls_through_to_the_interrupted_day(self) -> None:
        task = MultiDayPlannerTask()
        seen = {"n": 0}

        class _Day:
            def step(self, world):
                seen["n"] += 1
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        task._phase = "return_home"
        task._current_task = _Day()
        task._sync_festival_phase(_world(0x0A))
        task._festival_escape.timeout = 0
        result = task._step_inner(_world(0x0A))
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "return_home")
        self.assertEqual(seen["n"], 1)
        self.assertIs(task._current_task.__class__, _Day)

    def test_flower_festival_keeps_map_08_and_the_day_floor(self) -> None:
        class FlowerFestivalTask:
            pass

        class _Plan:
            def __init__(self) -> None:
                self._current_task = FlowerFestivalTask()

            def step(self, world):
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        plan = _Plan()
        task = MultiDayPlannerTask()
        task._phase = "plan_day"
        task._current_task = plan
        task.floor.frame_budget = 1
        task.floor.begin_day(0)
        self.assertTrue(day_plan_owns_festival(plan))
        for _ in range(4):
            result = task._step_inner(_world(0x08))
            self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "plan_day")
        self.assertIs(task._current_task, plan)
        self.assertIsNone(task._festival_escape)
        self.assertFalse(task._floor_abandoned)

        bare = MultiDayPlannerTask()
        bare._sync_festival_phase(_world(0x08))
        self.assertEqual(bare._phase, "festival")
        self.assertIsNotNone(bare._festival_escape)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("source,tilemap", [("egg_festival", 0x0B), ("harvest_festival", 0x09)])
def test_scheduled_berry_festival_keeps_control_in_the_day_plan(source, tilemap):
    from harvest.tasks.power_berry import PowerBerryTask
    plan = SimpleNamespace(_current_task=PowerBerryTask(source=source))
    campaign = MultiDayPlannerTask()
    campaign._current_task = plan
    campaign._sync_festival_phase(_world(tilemap))
    assert campaign._phase == "plan_day"
    assert campaign._current_task is plan
    assert campaign._festival_escape is None
    plan._current_task = None
    campaign._sync_festival_phase(_world(tilemap))
    assert campaign._phase == "festival"


def _happiness_ram(**fields):
    from harvest.core.ram_catalog import field_spec, ram_index

    ram = np.zeros(0x24000, dtype=np.uint8)
    values = dict(year=1, season=2, day=12, hour=6, money=108950, happiness=710,
                  tilemap=0x17, input_lock=1, power_berry_count=10, max_stamina=200)
    for key, value in (values | fields).items():
        spec = field_spec(key)
        address = ram_index(ram, spec.address, live_offset=spec.live_offset)
        value = value // 10 if key == "money" else value
        ram[address] = value & 255
        for byte in range(1, {"u8": 1, "u16": 2, "u24": 3}[spec.kind]):
            ram[address + byte] = (value >> (8 * byte)) & 255
    return ram


def test_happiness_trip_survives_ten_berry_calendar_and_owns_its_festival(monkeypatch):
    from harvest.planner.day_plan_phases import build_day_phases_from_ram
    from harvest.planner.day_phase_registry import build_phase_task, TaskBuildContext
    from harvest.planner.tasks.festival_escape import HarvestHappinessTask
    from retro_harness import WorldState

    monkeypatch.setenv("HARVEST_STRATEGY", "score_herd")
    ram = _happiness_ram(marriage_flags=0x10, eve_hearts=502, egg_available=3, num_chickens=2)
    phases = build_day_phases_from_ram(ram)
    assert not any(p.phase in {"GIVE_WIFE", "FETCH_FEATHER", "BUY_FEATHER"} for p in phases)
    spec = next(p for p in phases if p.phase == "HARVEST_HAPPINESS")
    assert spec.params["ends_day"]
    world = WorldState(frame=0, ram=ram, info={}, obs=None)
    task = build_phase_task(TaskBuildContext(), spec, world)
    assert isinstance(task, HarvestHappinessTask)
    assert day_plan_owns_festival(SimpleNamespace(_current_task=task))
    assert day_plan_owns_festival(SimpleNamespace(_current_task=None, current_phase=spec))


def test_missing_harvest_berry_keeps_its_own_phase_before_happiness(monkeypatch):
    from harvest.planner.day_plan_phases import build_day_phases_from_ram

    monkeypatch.setenv("HARVEST_STRATEGY", "score_herd")
    names = [p.phase for p in build_day_phases_from_ram(_happiness_ram(power_berry_count=6))]
    assert names.index("COLLECT_POWER_BERRY_HARVEST_FESTIVAL") < names.index("HARVEST_HAPPINESS")


def test_ten_berry_happiness_task_enters_without_requesting_an_eleventh():
    from harvest.planner.tasks.festival_escape import HarvestHappinessTask
    from harvest.tasks.power_berry import FestivalEntryTask
    from retro_harness import WorldState

    ram = _happiness_ram(tilemap=0)
    world = WorldState(frame=0, ram=ram, info={}, obs=None)
    task = HarvestHappinessTask()
    task.reset(world)
    task._child = SimpleNamespace(step=lambda _world: TaskResult(TaskStatus.SUCCESS))
    assert task.step(world).status == TaskStatus.RUNNING
    assert isinstance(task._child, FestivalEntryTask)


def test_happiness_wait_does_not_reopen_the_last_girls_dialogue():
    from harvest.planner.tasks.festival_escape import FestivalEscapeTask
    from retro_harness import WorldState

    ram = _happiness_ram(tilemap=9, game_state=0x4041, player_action=9)
    task = FestivalEscapeTask(happiness_target=999)
    task.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
    task._steps = 24
    assert not task._happiness_step(ram).action[8]


def test_happiness_counter_does_not_spend_the_cash_reserve():
    from harvest.planner.tasks.festival_escape import FestivalEscapeTask
    from retro_harness import WorldState

    ram = _happiness_ram(tilemap=9, money=60290, game_state=0x4001)
    task = FestivalEscapeTask(happiness_target=999)
    task.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
    assert task._happiness_step(ram) is None
    assert task.drinks == []


@pytest.mark.parametrize("fields", [dict(year=0), dict(money=60000), dict(happiness=999), dict(hour=15)])
def test_happiness_trip_respects_year_one_cash_target_and_deadline(monkeypatch, fields):
    from harvest.planner.tasks.festival_escape import harvest_happiness_due

    monkeypatch.setenv("HARVEST_STRATEGY", "score_herd")
    assert not harvest_happiness_due(_happiness_ram(**fields))


def test_happiness_uses_existing_dated_cash_floor_and_keeps_sixty_thousand():
    from harvest.planner.tasks.festival_escape import happiness_cash_floor

    assert happiness_cash_floor(_happiness_ram()) == 60000
    assert happiness_cash_floor(_happiness_ram(year=2, season=1, day=29)) >= 99840


@pytest.mark.rom
def test_native_harvest_stall_reaches_999_and_finishes_the_festival():
    import gzip
    import hashlib
    import json
    from harvest.core.ram_catalog import read_ram_value
    from harvest.paths import GAME_DIR, PROJECT_DIR
    from harvest.runtime.retro_setup import make_harvest_env
    from retro_harness import AuditedEnv, AuditCapabilities, WorldState

    # One initial load, then native entry, conversations, purchases and sleep.
    pin = "PR_t5_s51_Y2_Fall_D12"
    source = GAME_DIR / f"{pin}.state"
    if not source.exists():
        pytest.skip("local native festival conversation checkpoint is absent")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.happiness_stall"))
    try:
        env.reset()
        before = {k: read_ram_value(env.get_ram(), k) for k in ("happiness", "money", "day", "hour")}
        from harvest.planner.tasks.inventory import ExitToFarmTask
        from harvest.tasks.power_berry import FestivalEntryTask
        from harvest.planner.tasks.festival_escape import HarvestHappinessTask
        preparation_frames = 0
        for entry in (ExitToFarmTask(), FestivalEntryTask()):
            entry.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
            for tick in range(18000):
                result = entry.step(WorldState(frame=tick, ram=env.get_ram(), info={}, obs=None))
                if result.status != TaskStatus.RUNNING:
                    break
                env.step(result.action.action if result.action else make_action())
            assert result.status == TaskStatus.SUCCESS, result.reason
            preparation_frames += tick
        task = HarvestHappinessTask()
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(660_000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if frame % 20000 == 0:
                print(f"[HAPPINESS] frame={frame} drinks={len(task.escape.drinks)} happiness={read_ram_value(env.get_ram(), 'happiness')} failure={task.escape.happiness_failure}", flush=True)
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        after = {k: read_ram_value(env.get_ram(), k) for k in ("happiness", "money", "day", "hour", "tilemap")}
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert task.escape.happiness_failure is None
        assert after["happiness"] == 999
        assert len(task.escape.drinks) == (999 - before["happiness"] + 2) // 3
        assert before["money"] - after["money"] == len(task.escape.drinks) * 300
        assert after["money"] >= 60000 and after["day"] == 13
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        name = "LN_happiness_999_native_20261112"
        (GAME_DIR / f"{name}.state").write_bytes(gzip.compress(env.unwrapped.em.get_state(), mtime=0))
        out = PROJECT_DIR / "recordings/happiness_20261112"
        out.mkdir(exist_ok=True, parents=True)
        report = {"source": pin, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "before": before, "after": after, "frames": frame, "preparation_frames": preparation_frames, "drinks": task.escape.drinks,
                  "saved_state": name, "ram_writes": env.audit().ram_writes, "mid_run_loads": env.audit().mid_run_loads}
        (out / "stall_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()
