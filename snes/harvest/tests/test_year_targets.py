"""Year-aware date reads and run targets (Y1 Winter D30 -> Y2 Spring D1, Y3 S1 D30)."""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from harvest.core.ram_catalog import field_spec, ram_index
from harvest.planner.day_plan_status import (
    format_world_date,
    read_world_date,
    read_world_year,
    read_world_ymd,
)
from harvest.planner.multi_day_planner import MultiDayPlannerTask
from harvest.planner.tasks.ending import EndingTask
from harvest.scripts.run_to_day2 import (
    _gate_b_full_ok,
    _goal_reached,
    _parse_args,
    _previous_day,
    _build_task,
    _year_report_fields,
    days_between,
    main,
    resolve_until_target,
)


def _ram(year: int, season: int, day: int) -> np.ndarray:
    ram = np.zeros(0x24000, dtype=np.uint8)
    for name, value in (("year", year), ("season", season), ("day", day)):
        spec = field_spec(name)
        ram[ram_index(ram, spec.address, live_offset=spec.live_offset)] = value
    return ram


class YearReadTests(unittest.TestCase):
    def test_read_helpers(self) -> None:
        ram = _ram(2, 1, 30)
        self.assertEqual(read_world_year(ram), 2)
        self.assertEqual(read_world_ymd(ram), (2, 1, 30))
        self.assertEqual(read_world_date(ram), (1, 30))  # signature unchanged
        self.assertEqual(format_world_date(ram), "Y3S1D30")
        self.assertEqual(format_world_date(_ram(1, 0, 1)), "Y2S0D1")

    def test_ymd_orders_across_year_rollover(self) -> None:
        winter_end = read_world_ymd(_ram(0, 3, 30))
        new_year = read_world_ymd(_ram(1, 0, 1))
        self.assertLess(winter_end, new_year)
        # the (season, day) pair alone gets this wrong
        self.assertGreater(read_world_date(_ram(0, 3, 30)), read_world_date(_ram(1, 0, 1)))


class MultiDayTargetTests(unittest.TestCase):
    def test_year1_bound_keeps_meaning(self) -> None:
        task = MultiDayPlannerTask(until_season=0, until_day=30)  # end-of-spring
        self.assertFalse(task._target_reached(_ram(0, 0, 30)))
        self.assertTrue(task._target_reached(_ram(0, 1, 1)))

    def test_rollover_does_not_mis_stop(self) -> None:
        # Y1 Winter D28 -> Y2 Spring D1 with a Y2 Spring D5 target
        task = MultiDayPlannerTask(until_year=1, until_season=0, until_day=4)
        self.assertFalse(task._target_reached(_ram(0, 3, 28)))
        self.assertFalse(task._target_reached(_ram(0, 3, 30)))
        self.assertFalse(task._target_reached(_ram(1, 0, 1)))
        self.assertFalse(task._target_reached(_ram(1, 0, 4)))
        self.assertTrue(task._target_reached(_ram(1, 0, 5)))

    def test_later_year_is_past_every_year1_target(self) -> None:
        task = MultiDayPlannerTask(until_season=0, until_day=3)
        self.assertTrue(task._target_reached(_ram(1, 0, 1)))

    def test_target_days_unaffected(self) -> None:
        task = MultiDayPlannerTask(target_days=2)
        self.assertFalse(task._target_reached(_ram(2, 3, 30)))


class RunnerTargetTests(unittest.TestCase):
    def test_perfect_rancher_is_explicit_and_reaches_the_planner(self) -> None:
        with patch("harvest.scripts.run_to_day2.MultiDayPlannerTask") as planner:
            _build_task(_parse_args(["--days", "1"]), (0, 0, 3))
            self.assertNotIn("perfect_rancher", planner.call_args.kwargs)
            _build_task(_parse_args(["--perfect-rancher", "--days", "1"]), (0, 0, 3))
            self.assertTrue(planner.call_args.kwargs["perfect_rancher"])
        with self.assertRaises(SystemExit):
            _build_task(_parse_args(["--perfect-rancher", "--day-plan", "mountain_berry"]), (0, 0, 3))

    def test_previous_day(self) -> None:
        self.assertEqual(_previous_day((0, 0, 4)), (0, 0, 3))
        self.assertEqual(_previous_day((0, 1, 1)), (0, 0, 30))
        self.assertEqual(_previous_day((1, 0, 1)), (0, 3, 30))
        self.assertEqual(_previous_day((0, 0, 1)), (0, 0, 0))

    def test_days_between(self) -> None:
        self.assertEqual(days_between((0, 3, 30), (1, 0, 1)), 1)
        self.assertEqual(days_between((0, 0, 1), (2, 1, 30)), 2 * 120 + 30 + 29)

    def test_legacy_flags_resolve_against_start(self) -> None:
        start = (0, 1, 5)
        args = _parse_args(["--until-day", "9"])
        self.assertEqual(resolve_until_target(args, start), (0, 1, 9))
        args = _parse_args(["--until-season", "2", "--until-day", "1"])
        self.assertEqual(resolve_until_target(args, start), (0, 2, 1))
        self.assertIsNone(resolve_until_target(_parse_args([]), start))

    def test_until_year_defaults_to_spring_d1(self) -> None:
        start = (0, 3, 20)
        self.assertEqual(resolve_until_target(_parse_args(["--until-year", "2"]), start), (1, 0, 1))
        args = _parse_args(["--until-year", "2", "--until-season", "1", "--until-day", "7"])
        self.assertEqual(resolve_until_target(args, start), (1, 1, 7))

    def test_until_day_from_a_year2_pin_stays_in_year2(self) -> None:
        args = _parse_args(["--until-day", "4"])
        self.assertEqual(resolve_until_target(args, (1, 0, 2)), (1, 0, 4))

    def test_goal_across_rollover(self) -> None:
        args = _parse_args(["--until-year", "2"])
        start = (0, 3, 29)
        self.assertFalse(_goal_reached(start=start, end=(0, 3, 30), days_completed=1, args=args))
        self.assertTrue(_goal_reached(start=start, end=(1, 0, 1), days_completed=2, args=args))

    def test_goal_legacy_year1(self) -> None:
        args = _parse_args(["--until-season", "1", "--until-day", "1"])
        self.assertFalse(_goal_reached(start=(0, 0, 28), end=(0, 0, 30), days_completed=2, args=args))
        self.assertTrue(_goal_reached(start=(0, 0, 28), end=(0, 1, 1), days_completed=3, args=args))

    def test_goal_days_across_season(self) -> None:
        args = _parse_args(["--days", "2"])
        self.assertTrue(_goal_reached(start=(0, 0, 29), end=(0, 1, 1), days_completed=None, args=args))

    def test_end_of_spring(self) -> None:
        args = _parse_args(["--end-of-spring"])
        self.assertFalse(_goal_reached(start=(0, 0, 2), end=(0, 0, 30), days_completed=None, args=args))
        self.assertTrue(_goal_reached(start=(0, 0, 2), end=(0, 1, 1), days_completed=None, args=args))

    def test_build_task_cross_year_bound(self) -> None:
        args = _parse_args(["--until-year", "2"])
        task = _build_task(args, (0, 3, 20))
        self.assertEqual((task.until_year, task.until_season, task.until_day), (0, 3, 30))
        self.assertGreaterEqual(task.max_days, 15)

    def test_build_task_legacy_bound(self) -> None:
        args = _parse_args(["--until-day", "4"])
        task = _build_task(args, (0, 0, 2))
        self.assertEqual((task.until_year, task.until_season, task.until_day), (0, 0, 3))

    def test_until_ending_has_no_date_target(self) -> None:
        args = _parse_args(["--until-ending"])
        task = _build_task(args, (0, 0, 2))
        self.assertIsNone(task.target_days)
        self.assertFalse(task._target_reached(_ram(2, 1, 29)))
        self.assertFalse(task._target_reached(_ram(2, 1, 30)))
        self.assertGreaterEqual(task.max_days, 300)

    def test_build_task_power_on_and_end_of_spring_keep_daily_clear(self) -> None:
        args_poweron = _parse_args(["--power-on", "--until-season", "1", "--until-day", "1"])
        task_poweron = _build_task(args_poweron, (0, 0, 2))
        self.assertTrue(task_poweron.policy.include_field_clear)

        args_eos = _parse_args(["--end-of-spring"])
        task_eos = _build_task(args_eos, (0, 0, 2))
        self.assertTrue(task_eos.policy.include_field_clear)


class EndingHandoffReportTests(unittest.TestCase):
    def test_year_fields_come_from_the_planner_not_ending_task(self) -> None:
        planner = MultiDayPlannerTask()
        planner._days_completed = 12
        planner._day_journal.append(
            {
                "shipped_count": 4,
                "harvested_count": 1,
                "money": 800,
                "rescues": 2,
                "phase_results": [],
            }
        )
        planner._day_failures.append({"phase": "sleep", "reason": "bed"})
        planner.floor.bad_days = 1
        planner.floor.total_bad_days = 3
        planner.scene_guard.log.append({"step": 1, "mode": "dialogue"})

        fields = _year_report_fields(planner)
        self.assertEqual(fields["days_completed"], 12)
        self.assertEqual(fields["day_journal"][0]["shipped_count"], 4)
        self.assertEqual(fields["day_journal"][0]["rescues"], 2)
        self.assertEqual(fields["day_failures"], [{"phase": "sleep", "reason": "bed"}])
        self.assertEqual(fields["journal_summary"]["total_shipped"], 4)
        self.assertEqual(fields["journal_summary"]["total_harvested"], 1)
        self.assertEqual(fields["journal_summary"]["overnights"], 1)
        self.assertEqual(
            fields["floor_report"],
            {
                "consecutive_bad_days": 1,
                "total_bad_days": 3,
                "scene_guard": [{"step": 1, "mode": "dialogue"}],
            },
        )
        self.assertEqual(fields["floor_report"], planner.floor_report)

        # The credits waiter must not be what the report reads.
        blank = _year_report_fields(EndingTask())
        self.assertIsNone(blank["days_completed"])
        self.assertEqual(blank["day_journal"], [])
        self.assertEqual(blank["day_failures"], [])
        self.assertIsNone(blank["floor_report"])
        self.assertEqual(blank["journal_summary"]["total_shipped"], 0)
        self.assertEqual(blank["journal_summary"]["total_harvested"], 0)

    def test_missing_floor_report_is_null(self) -> None:
        class _NoFloor:
            _days_completed = 1
            day_journal = ({"shipped_count": 1, "harvested_count": 0, "phase_results": []},)
            day_failures = ()

        fields = _year_report_fields(_NoFloor())
        self.assertIsNone(fields["floor_report"])
        self.assertEqual(fields["days_completed"], 1)
        self.assertEqual(fields["journal_summary"]["total_shipped"], 1)

    def test_until_ending_does_not_replace_the_planner(self) -> None:
        src = inspect.getsource(main)
        self.assertNotIn("task = _start_ending", src)
        self.assertIn("ending_task.step", src)
        self.assertIn("_year_report_fields(planner)", src)

    def test_until_ending_is_not_gate_b(self) -> None:
        # Summer entry plus harvest income still is not Gate B on a credits run.
        self.assertFalse(
            _gate_b_full_ok(
                power_on=True,
                until_ending=True,
                end_key=(0, 1, 1),
                total_shipped=5,
                total_harvested=5,
                mid_run_state_loads=0,
            )
        )

    def test_gate_b_is_summer_entry_with_harvest_income(self) -> None:
        summer = dict(
            power_on=True,
            until_ending=False,
            end_key=(0, 1, 1),
            mid_run_state_loads=0,
        )
        self.assertTrue(_gate_b_full_ok(**summer, total_shipped=1, total_harvested=0))
        self.assertTrue(_gate_b_full_ok(**summer, total_shipped=0, total_harvested=2))
        # Money and "past Spring D30" are not enough: fall, Y3 ending, no income.
        self.assertFalse(
            _gate_b_full_ok(
                power_on=True,
                until_ending=False,
                end_key=(0, 2, 1),
                total_shipped=10,
                total_harvested=10,
                mid_run_state_loads=0,
            )
        )
        self.assertFalse(
            _gate_b_full_ok(
                power_on=True,
                until_ending=False,
                end_key=(2, 1, 30),
                total_shipped=10,
                total_harvested=1,
                mid_run_state_loads=0,
            )
        )
        self.assertFalse(_gate_b_full_ok(**summer, total_shipped=0, total_harvested=0))


if __name__ == "__main__":
    unittest.main()
