"""Day floor: no single day may end the run (rr-bu62.4)."""
from __future__ import annotations

import sys
import unittest
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

    def test_rate_is_seeded(self) -> None:
        a = PhaseFaultInjector(faults={"crop": "fail"}, rate=0.5, seed=3)
        b = PhaseFaultInjector(faults={"crop": "fail"}, rate=0.5, seed=3)
        self.assertEqual(
            [a.decide("crop", "x") for _ in range(20)],
            [b.decide("crop", "x") for _ in range(20)],
        )


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
