"""Day floor: no single day may end the run (rr-bu62.4)."""
from __future__ import annotations

import sys
import unittest
import pytest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from day_plan_test_helpers import make_date_world, set_player_pos

from harvest.core.recovery import DayFloor, PhaseFaultInjector
from harvest.planner.day_plan import (
    ADDR_DAY,
    ADDR_HOUR,
    ADDR_TILEMAP,
    ActionResult,
    TaskResult,
    TaskStatus,
)
from harvest.planner.multi_day_planner import MultiDayPlannerTask
from harvest.tasks.nav import make_action


@pytest.mark.rom
def test_after_d2_evening_reaches_the_next_morning():
    """Natural D2 completion used to lose Eve's dialogue to the day floor."""
    from retro_harness import AuditCapabilities, AuditedEnv, WorldState
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.day_plan_status import read_world_date

    env = AuditedEnv(make_harvest_env("LN_d2_perfect_poweron", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.d2_evening"))
    try:
        env.reset()
        task = MultiDayPlannerTask(target_days=1)
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(30_000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, (frame, result.reason)
        assert read_world_date(env.get_ram()) == (0, 3)
        assert task.floor.total_bad_days == task.floor.rescues == 0
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


class _Task:
    def __init__(self, status=TaskStatus.RUNNING, reason="", on_step=None):
        self.status, self.reason, self.on_step = status, reason, on_step
        self.steps = 0

    def reset(self, world):
        self.steps = 0

    def can_start(self, world):
        return True

    def step(self, world):
        self.steps += 1
        if self.on_step:
            self.on_step(world)
        return TaskResult(
            status=self.status, reason=self.reason, action=ActionResult(make_action())
        )


def _advance_day(world):
    world.ram[ADDR_DAY + 0x4000] = int(world.ram[ADDR_DAY + 0x4000]) + 1


def _planner(**tasks):
    class P(MultiDayPlannerTask):
        def _build_day_task(self, world):
            return tasks["day"]()

        def _build_return_home_task(self):
            return tasks["home"]()

        def _build_sleep_task(self):
            return tasks["sleep"]()

    p = P(target_days=1, max_days=50, morning_settle_frames=1)
    return p


def _run(planner, world, n):
    planner.reset(world)
    result = None
    for _ in range(n):
        result = planner.step(world)
        if result.status in (TaskStatus.SUCCESS, TaskStatus.FAILURE):
            break
    return result


class DayFloorLogicTests(unittest.TestCase):
    def test_finished_festival_does_not_own_followup_overnight(self):
        from harvest.planner.day_plan import DayPlanTask, PhaseSpec
        planner = MultiDayPlannerTask(target_days=1)
        world = make_date_world(0x15, season=2, day=20)
        planner.reset(world)
        child = DayPlanTask(phase_sequence=[
            PhaseSpec("EGG_FESTIVAL", "power_berry", {"ends_day": True}),
            PhaseSpec("EXIT_TO_FARM", "farm_building_exit"),
        ])
        child.reset(world)
        child._phase_index = 1
        planner._current_task = child
        world.ram[ADDR_DAY + 0x4000] = 21
        result = planner._observe_running_side_effects(world, _Task().step(world))
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner.phase_text, "SETTLE_MORNING")
        self.assertEqual(planner._days_completed, 1)

    def test_morning_wait_has_a_bound_even_with_day_floor_disabled(self):
        world = make_date_world(0x00, season=2, day=21)
        world.ram[ADDR_HOUR + 0x4000] = 18
        planner = MultiDayPlannerTask(target_days=1)
        planner.reset(world)
        planner._finish_sleep(world, "festival date roll")
        for _ in range(4000):
            planner.step(world)
            if planner.phase_text != "SETTLE_MORNING":
                break
        self.assertNotEqual(planner.phase_text, "SETTLE_MORNING")
        self.assertEqual(planner.floor.rescues, 1)

    def test_long_previous_day_cannot_abandon_a_fresh_morning_before_activation(self):
        child = _Task()
        planner = _planner(day=lambda: child, home=lambda: _Task(), sleep=lambda: _Task())
        world = make_date_world(0x15, season=0, day=9)
        planner.reset(world)
        planner._step_count = 1000
        planner.floor.frame_budget = 100
        planner.floor.begin_day(0)
        planner._last_day_phase_results = [{"phase": "HARVEST_ROUTE", "shipped_count": 8}]
        result = planner.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner._phase, "plan_day")
        self.assertEqual(child.steps, 1)
        self.assertEqual(planner._last_day_phase_results, [])
        self.assertEqual(planner.floor._started_at, 1001)
        self.assertFalse(planner.day_failures)

    def test_bounded_evening_work_keeps_the_hard_day_budget(self) -> None:
        floor = DayFloor(frame_budget=100, clock_grace=10)
        floor.begin_day(0)
        self.assertIsNone(floor.check(1, 18))
        self.assertIsNone(floor.check(30, 18, allow_late_work=True))
        self.assertIsNone(floor.check(40, 18))
        self.assertIn("clock", floor.check(41, 18))
        floor.begin_day(0)
        self.assertIn("budget", floor.check(101, 18, allow_late_work=True))

    def test_frame_budget_trips_once(self) -> None:
        floor = DayFloor(frame_budget=100, clock_grace=10**9)
        floor.begin_day(0)
        self.assertIsNone(floor.check(100, 8))
        self.assertIn("budget", floor.check(101, 8))
        self.assertIsNone(floor.check(500, 8))

    def test_clock_deadline_needs_grace(self) -> None:
        floor = DayFloor(frame_budget=10**9, clock_grace=50)
        floor.begin_day(0)
        self.assertIsNone(floor.check(10, 17))
        self.assertIsNone(floor.check(20, 18))
        self.assertIsNone(floor.check(70, 18))
        self.assertIn("clock", floor.check(71, 18))

    def test_bad_day_counter_reports_and_resets(self) -> None:
        floor = DayFloor()
        floor.begin_day(0)
        floor.note_bad()
        floor.end_day(clean=False)
        floor.begin_day(1)
        floor.end_day(clean=False)
        self.assertEqual((floor.bad_days, floor.total_bad_days), (2, 2))
        floor.begin_day(2)
        floor.end_day(clean=True)
        self.assertEqual((floor.bad_days, floor.total_bad_days), (0, 2))

    def test_rescues_run_out_only_when_wedged(self) -> None:
        floor = DayFloor(max_rescues=2)
        floor.begin_day(0)
        self.assertEqual([floor.next_rescue() for _ in range(3)], [True, True, False])
        floor.begin_day(1)
        self.assertTrue(floor.next_rescue())


class BadDayDoesNotDisarmTheFloorTests(unittest.TestCase):
    def test_a_failure_recorded_earlier_in_the_day_still_lets_the_budget_trip(self) -> None:
        floor = DayFloor(frame_budget=1000)
        floor.begin_day(0)
        floor.note_bad()  # e.g. an optional CLEAR_FIELD failure
        self.assertIsNone(floor.check(500, 12))
        self.assertEqual(floor.check(1001, 12), "day frame budget 1000 exceeded")
        self.assertIsNone(floor.check(1002, 12))  # it trips once

    def test_a_bad_day_is_counted_even_when_the_floor_never_tripped(self) -> None:
        floor = DayFloor()
        floor.begin_day(0)
        floor.note_bad()
        floor.end_day(clean=True)
        self.assertEqual((floor.bad_days, floor.total_bad_days), (1, 1))
        floor.begin_day(10)
        floor.end_day(clean=True)
        self.assertEqual(floor.bad_days, 0)


class FaultInjectorTests(unittest.TestCase):
    def test_env_parse_and_end_of_day_immune(self) -> None:
        inj = PhaseFaultInjector.from_env(
            {"HARVEST_FAULT_PHASES": "crop:fail,harvest:stall,*:fail"}
        )
        self.assertEqual(inj.decide("crop", "CROP_WATER"), "fail")
        self.assertEqual(inj.decide("harvest", "H"), "stall")
        self.assertEqual(inj.decide("nav", "N"), "fail")
        self.assertIsNone(inj.decide("return_home", "R"))
        self.assertIsNone(inj.decide("sleep", "S"))

    def test_off_without_env(self) -> None:
        self.assertIsNone(PhaseFaultInjector.from_env({}))

    def test_phase_jitter_is_seeded_bounded_and_spares_end_of_day(self) -> None:
        env = {"HARVEST_PHASE_JITTER": "90", "HARVEST_FAULT_SEED": "5"}

        def waits(inj: PhaseFaultInjector, kind: str, phases: int = 40) -> list[int]:
            out = []
            for key in range(phases):
                n = 0
                while inj.jitter_wait(key, kind):
                    n += 1
                out.append(n)
            return out

        a, b = PhaseFaultInjector.from_env(env), PhaseFaultInjector.from_env(env)
        draws = waits(a, "crop")
        self.assertEqual(draws, waits(b, "crop"))
        self.assertTrue(all(0 <= n <= 90 for n in draws))
        self.assertGreater(len(set(draws)), 5)
        self.assertEqual(set(waits(a, "return_home")), {0})
        self.assertEqual(set(waits(a, "sleep")), {0})
        # Jitter alone never injects a fault, and zero jitter builds no injector.
        self.assertIsNone(a.decide("crop", "CROP_WATER"))
        self.assertIsNone(PhaseFaultInjector.from_env({"HARVEST_PHASE_JITTER": "0"}))

    def test_phase_jitter_can_start_in_a_later_year(self) -> None:
        """A power-on run seeds only Year 3 (``HARVEST_PHASE_JITTER_FROM_YEAR=2``): Years 1-2 play exactly as unjittered."""
        env = {"HARVEST_PHASE_JITTER": "90", "HARVEST_FAULT_SEED": "5", "HARVEST_PHASE_JITTER_FROM_YEAR": "2"}
        inj = PhaseFaultInjector.from_env(env)
        self.assertEqual(inj.jitter_from_year, 2)
        for year in (0, 1):  # no draw, no wait, in Years 1 and 2
            self.assertEqual(sum(inj.jitter_wait(("k", year, i), "crop", year) for i in range(500)), 0)
        waited = 0
        for i in range(40):
            while inj.jitter_wait(("y3", i), "crop", 2):
                waited += 1
        self.assertGreater(waited, 0)
        # Without the variable (or without a year) every phase jitters as before.
        plain = PhaseFaultInjector.from_env({"HARVEST_PHASE_JITTER": "90", "HARVEST_FAULT_SEED": "5"})
        self.assertEqual(plain.jitter_from_year, 0)
        self.assertTrue(any(plain.jitter_wait(("p", i), "crop", 0) for i in range(60)))


class PlannerFloorTests(unittest.TestCase):
    def test_stuck_day_work_is_abandoned_then_slept(self) -> None:
        planner = _planner(
            day=lambda: _Task(),  # never finishes
            home=lambda: _Task(TaskStatus.SUCCESS, "home"),
            sleep=lambda: _Task(TaskStatus.SUCCESS, "slept", _advance_day),
        )
        planner.floor.frame_budget = 30
        world = make_date_world(0x00, season=0, day=5)
        result = _run(planner, world, 200)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(planner.day_journal[0]["floor_reason"].split()[0], "day")
        self.assertTrue(planner.day_journal[0]["bad_day"])
        self.assertEqual(planner.floor.bad_days, 1)

    def test_day_work_exception_fails_the_run(self) -> None:
        def boom(world):
            raise RuntimeError("planner bug")

        planner = _planner(
            day=lambda: _Task(on_step=boom),
            home=lambda: _Task(TaskStatus.SUCCESS),
            sleep=lambda: _Task(TaskStatus.SUCCESS, "slept", _advance_day),
        )
        world = make_date_world(0x00, season=0, day=5)
        with self.assertRaises(RuntimeError):
            _run(planner, world, 100)

    def test_return_home_exhaustion_rescues_instead_of_failing(self) -> None:
        calls = {"home": 0}

        def home():
            calls["home"] += 1
            # Fails until the fifth build: 3 retries + first rescue's builds.
            ok = calls["home"] >= 6
            return _Task(TaskStatus.SUCCESS if ok else TaskStatus.FAILURE, "home")

        planner = _planner(
            day=lambda: _Task(TaskStatus.SUCCESS),
            home=home,
            sleep=lambda: _Task(TaskStatus.SUCCESS, "slept", _advance_day),
        )
        world = make_date_world(0x00, season=0, day=5)
        result = _run(planner, world, 3000)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertGreaterEqual(planner.floor.rescues, 1)

    def test_sleep_failure_rescues(self) -> None:
        builds = {"n": 0}

        def sleep():
            builds["n"] += 1
            if builds["n"] == 1:
                return _Task(TaskStatus.FAILURE, "sleep timeout")
            return _Task(TaskStatus.SUCCESS, "slept", _advance_day)

        planner = _planner(
            day=lambda: _Task(TaskStatus.SUCCESS),
            home=lambda: _Task(TaskStatus.SUCCESS),
            sleep=sleep,
        )
        world = make_date_world(0x15, season=0, day=5)
        result = _run(planner, world, 3000)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(builds["n"], 2)

    def test_provably_wedged_farmer_finally_fails(self) -> None:
        planner = _planner(
            day=lambda: _Task(TaskStatus.SUCCESS),
            home=lambda: _Task(TaskStatus.FAILURE, "stuck"),
            sleep=lambda: _Task(),
        )
        planner.floor.max_rescues = 2
        world = make_date_world(0x00, season=0, day=5)
        result = _run(planner, world, 20000)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("wedged", result.reason)

    def test_watchdog_can_restart_a_stuck_sleep(self) -> None:
        planner = _planner(
            day=lambda: _Task(TaskStatus.SUCCESS),
            home=lambda: _Task(TaskStatus.SUCCESS),
            sleep=lambda: _Task(),
        )
        world = make_date_world(0x15, season=0, day=5)
        planner.reset(world)
        for _ in range(6):
            planner.step(world)
        self.assertEqual(planner.phase_text, "SLEEP")
        self.assertTrue(planner.force_end_day(world, "no progress"))
        self.assertEqual(planner.phase_text, "RETURN_HOME")

    def test_max_days_does_not_end_a_date_target_run(self) -> None:
        planner = _planner(
            day=lambda: _Task(TaskStatus.SUCCESS),
            home=lambda: _Task(TaskStatus.SUCCESS),
            sleep=lambda: _Task(TaskStatus.SUCCESS, "slept", _advance_day),
        )
        planner.target_days = None
        planner.until_season, planner.until_day, planner.max_days = 0, 9, 1
        world = make_date_world(0x15, season=0, day=5)
        result = _run(planner, world, 400)
        self.assertEqual(result.status, TaskStatus.SUCCESS)


if __name__ == "__main__":
    unittest.main()


class SleepWhenDoneTests(unittest.TestCase):
    def test_sleep_early_and_no_shipping_wait_by_default(self) -> None:
        p = MultiDayPlannerTask(target_days=1)
        self.assertEqual(p._build_sleep_task().earliest_sleep_hour, 0)
        world = make_date_world(0x00, season=0, day=5)
        self.assertFalse(p._needs_farm_shipping_wait(world))

    def test_opt_out_restores_evening_gate(self) -> None:
        p = MultiDayPlannerTask(target_days=1, sleep_when_done=False)
        self.assertEqual(p._build_sleep_task().earliest_sleep_hour, 18)


def test_every_ordinary_phase_has_a_stall_backstop_while_the_day_floor_is_off():
    """Y2 Spring D8 (compose6): the day floor is off by design (owner: no per-day cap), and only CROP/HARVEST had a
    watchdog, so a silently looping ENSURE_ANIMAL_TOOLS ran for 2M frames. The longest ordinary phase in a finished
    Year 3 run is 31k frames (COVER_FIELD, exempt below), so 120k leaves ample room for real work."""
    from harvest.planner.day_phase_types import PHASE_BACKSTOP_FRAMES, PhaseKind, PhaseSpec
    from harvest.planner.day_plan_orchestrator import _phase_watchdog_frames

    assert PHASE_BACKSTOP_FRAMES >= 3 * 31_000
    assert _phase_watchdog_frames(PhaseSpec("CROP_WATER", PhaseKind.CROP)) == 24_000                      # unchanged
    for name, kind in (("ENSURE_ANIMAL_TOOLS", PhaseKind.ENSURE_ANIMAL_TOOLS), ("COW_CHORES", PhaseKind.COW_CHORES),
                       ("SOCIAL_TOWN", PhaseKind.TOWN_COURT), ("BUY_SEEDS", PhaseKind.SHOP_BUY)):
        assert _phase_watchdog_frames(PhaseSpec(name, kind)) == PHASE_BACKSTOP_FRAMES, name
    for name in ("COVER_FIELD", "COVER_TILL", "EVE_TALK_LOOP", "HARVEST_HAPPINESS"):                  # long on purpose
        assert _phase_watchdog_frames(PhaseSpec(name, PhaseKind.GRASS)) == 0, name
    assert _phase_watchdog_frames(PhaseSpec("ANY", PhaseKind.COOP_CHORES, {"ends_day": True})) == 0           # the day rolls inside
    assert _phase_watchdog_frames(PhaseSpec("ANY", PhaseKind.COOP_CHORES, {"watchdog_frames": 5_000})) == 5_000
    assert _phase_watchdog_frames(PhaseSpec("ANY", PhaseKind.COOP_CHORES, {"timeout": 360_000})) == 720_000    # never below its own timeout
    assert _phase_watchdog_frames(PhaseSpec("D2", PhaseKind.COOP_CHORES, {"d2_expanded": True})) == 0
