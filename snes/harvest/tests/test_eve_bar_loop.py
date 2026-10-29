"""Eve bar night loop: outlook model and the phase machine on fake RAM."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import unittest

from day_plan_test_helpers import make_world, set_live_u16, set_player_pos
from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.tasks import eve_loop_task as eve
from harvest.tasks.eve_loop_task import EveTalkLoopTask, bar_talk_outlook
from retro_harness import TaskStatus

ADDR_ACTION = field_spec("player_action").address
ADDR_DIALOG_TEXT = field_spec("dialog_text_id").address
ADDR_CHOICE = field_spec("dialog_choice_result").address


def _set_date(world, season: int, day: int) -> None:
    set_live_u16(world.ram, "season", season)
    set_live_u16(world.ram, "day", day)


def _bar_world(hearts: int, *, season: int = 0, day: int = 10):
    world = make_world(eve.BAR_TILEMAP)
    set_live_u16(world.ram, "eve_hearts", hearts)
    _set_date(world, season, day)
    return world


def _eve_at(x: int, y: int):
    obj = SimpleNamespace(is_player=False, sprite_table_idx=0x0246, pixel=(x, y))
    return mock.patch.object(eve, "game_objects", return_value=[obj])


def _task(world, **kw) -> EveTalkLoopTask:
    task = EveTalkLoopTask(target_points=250, **kw)
    task.reset(world)
    return task


class OutlookTest(unittest.TestCase):
    def test_bands_follow_the_script(self) -> None:
        kw = dict(season=0, day=10, weather_flags=0)
        self.assertEqual(bar_talk_outlook(0, **kw), "juice")
        self.assertEqual(bar_talk_outlook(119, **kw), "juice")
        self.assertEqual(bar_talk_outlook(120, **kw), "dead_zone")
        self.assertEqual(bar_talk_outlook(149, **kw), "dead_zone")
        self.assertEqual(bar_talk_outlook(150, **kw), "choice")
        self.assertEqual(bar_talk_outlook(200, **kw), "choice")
        self.assertEqual(bar_talk_outlook(201, **kw), "juice")

    def test_choice_band_pays_in_every_season(self) -> None:
        for season in range(4):
            self.assertEqual(
                bar_talk_outlook(160, season=season, day=5, weather_flags=0), "choice"
            )

    def test_spring_closed_days(self) -> None:
        self.assertEqual(bar_talk_outlook(10, season=0, day=22, weather_flags=0), "none")
        self.assertEqual(bar_talk_outlook(10, season=0, day=10, weather_flags=eve.RAIN), "none")

    def test_summer_and_fall_pay_nothing(self) -> None:
        for season in (1, 2):
            self.assertEqual(bar_talk_outlook(10, season=season, day=5, weather_flags=0), "none")

    def test_winter_pays_two_unless_snow_or_festival_eve(self) -> None:
        self.assertEqual(bar_talk_outlook(10, season=3, day=5, weather_flags=0), "winter")
        self.assertEqual(bar_talk_outlook(10, season=3, day=23, weather_flags=0), "none")
        self.assertEqual(bar_talk_outlook(10, season=3, day=5, weather_flags=eve.SNOW), "none")
        self.assertEqual(bar_talk_outlook(10, season=3, day=30, weather_flags=eve.SNOW), "winter")


class TaskTest(unittest.TestCase):
    def test_target_already_met_is_success_outside_the_bar(self) -> None:
        world = make_world(eve.TOWN_TILEMAP)
        set_live_u16(world.ram, "eve_hearts", 260)
        _set_date(world, 0, 10)
        result = _task(world).step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("eve hearts 260/250", result.reason)

    def test_target_met_inside_the_bar_walks_out_first(self) -> None:
        world = _bar_world(260)
        task = _task(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "leave")
        world.ram[eve.ADDR_TILEMAP] = eve.TOWN_TILEMAP
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_dead_zone_stops_with_a_named_reason(self) -> None:
        world = make_world(eve.TOWN_TILEMAP)
        set_live_u16(world.ram, "eve_hearts", 124)
        _set_date(world, 0, 10)
        result = _task(world).step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("dead_zone at hearts=124/250", result.reason)

    def test_summer_stops_before_entering(self) -> None:
        world = make_world(eve.TOWN_TILEMAP)
        _set_date(world, 1, 10)
        result = _task(world).step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("none at hearts=0", result.reason)

    def test_outlook_check_can_be_disabled(self) -> None:
        world = make_world(eve.TOWN_TILEMAP)
        set_player_pos(world.ram, 152, 872)
        _set_date(world, 1, 10)
        result = _task(world, check_outlook=False).step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)

    def test_rejects_a_map_that_is_not_the_bar_exterior(self) -> None:
        world = make_world(0x22)
        set_player_pos(world.ram, 152, 872)
        _set_date(world, 0, 10)
        task = _task(world, align_timeout=2)
        task.step(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("expected Eve loop origin", result.reason)

    def test_farm_map_walks_to_the_bar(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 136, 424)
        _set_date(world, 0, 10)
        task = _task(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "to_town")
        self.assertIsNotNone(task._nav)
        self.assertEqual(task._nav.waypoints[-1].target_px, eve.DOOR_PX)

    def test_aligns_toward_the_door_on_the_town_map(self) -> None:
        world = make_world(eve.TOWN_TILEMAP)
        set_player_pos(world.ram, 152, 953)
        _set_date(world, 0, 10)
        result = _task(world).step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(int(result.action.action[4]), 1)  # Up

    def test_far_in_town_it_delegates_to_the_town_walk(self) -> None:
        world = make_world(eve.TOWN_TILEMAP)
        set_player_pos(world.ram, 681, 54)
        _set_date(world, 0, 10)
        nav = mock.MagicMock()
        nav.step.return_value = eve.TaskResult(
            status=TaskStatus.RUNNING, action=eve.ActionResult(eve.make_action(down=True)), reason="nav"
        )
        with mock.patch.object(eve, "MultiMapNavTask", return_value=nav) as nav_cls:
            task = _task(world)
            first = task.step(world)
            task.step(world)
        self.assertEqual(first.reason, "walk_to_bar")
        self.assertEqual(int(first.action.action[5]), 1)  # Down, from the nav task
        self.assertEqual(nav_cls.call_count, 1)  # one nav task, stepped each frame
        wp = nav_cls.call_args.kwargs["waypoints"][0]
        self.assertEqual((wp.tilemap, wp.target_px), (eve.TOWN_TILEMAP, eve.DOOR_PX))

    def test_a_town_walk_that_fails_ends_the_task(self) -> None:
        world = make_world(eve.TOWN_TILEMAP)
        set_player_pos(world.ram, 681, 54)
        _set_date(world, 0, 10)
        nav = mock.MagicMock()
        nav.step.return_value = eve.TaskResult(status=TaskStatus.FAILURE, reason="blocked")
        with mock.patch.object(eve, "MultiMapNavTask", return_value=nav):
            result = _task(world).step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("could not reach the bar door: blocked", result.reason)

    def test_at_the_door_it_walks_in(self) -> None:
        world = make_world(eve.TOWN_TILEMAP)
        set_player_pos(world.ram, 152, 872)
        _set_date(world, 0, 10)
        task = _task(world)
        task.step(world)
        result = task.step(world)
        self.assertEqual(result.reason, "enter_bar")
        self.assertEqual(int(result.action.action[4]), 1)  # Up
        self.assertEqual(int(result.action.action[0]), 1)  # B runs

    def test_enter_times_out_when_the_door_stays_shut(self) -> None:
        world = make_world(eve.TOWN_TILEMAP)
        set_player_pos(world.ram, 152, 872)
        _set_date(world, 0, 10)
        task = _task(world)
        task.step(world)
        task._phase_frames = eve.ENTER_TIMEOUT
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("door did not open", result.reason)

    def test_first_meeting_spot_is_skipped_by_leaving(self) -> None:
        world = _bar_world(0)
        task = _task(world)
        with _eve_at(168, 440):
            result = task.step(world)
        self.assertEqual(result.reason, "first_visit_skip")
        self.assertEqual(task._phase, "leave")

    def test_regular_spot_starts_the_approach(self) -> None:
        world = _bar_world(0)
        task = _task(world)
        with _eve_at(56, 448):
            result = task.step(world)
        self.assertEqual(result.reason, "bar_ready")
        self.assertEqual(task._phase, "approach")

    def test_approach_walks_up_then_left_then_talks(self) -> None:
        world = _bar_world(0)
        task = _task(world)
        task._set_phase("approach")
        with _eve_at(56, 448):
            set_player_pos(world.ram, 144, 456)
            up = task.step(world)
            self.assertEqual(up.reason, "approach_eve")
            self.assertEqual(int(up.action.action[4]), 1)
            set_player_pos(world.ram, 144, 446)
            left = task.step(world)
            self.assertEqual(int(left.action.action[6]), 1)
            set_player_pos(world.ram, 78, 446)
            ready = task.step(world)
        self.assertEqual(ready.reason, "talk_ready")
        self.assertEqual(task._phase, "talk")

    def test_juice_choice_takes_the_arrow_then_no_thank_you(self) -> None:
        world = _bar_world(0)
        task = _task(world)
        task._set_phase("talk")
        task._seen_lock = True
        world.ram[ADDR_INPUT_LOCK] = 2
        world.ram[ADDR_DIALOG_TEXT] = eve.TEXT_JUICE & 0xFF
        world.ram[ADDR_DIALOG_TEXT + 1] = eve.TEXT_JUICE >> 8
        wait = task.step(world)
        self.assertEqual(wait.reason, "choose_wait")
        world.ram[ADDR_ACTION] = eve.ACTION_ARROW
        opened = task.step(world)
        self.assertEqual(opened.reason, "choose_open")
        self.assertEqual(int(opened.action.action[8]), 1)  # A
        world.ram[ADDR_ACTION] = 0
        down = None
        for _ in range(4):  # Down is held for half of every eight frames
            down = task.step(world)
        self.assertEqual(down.reason, "choose_down")
        world.ram[ADDR_CHOICE] = 1
        picked = task.step(world)
        self.assertEqual(picked.reason, "choose_no_thank_you")
        self.assertEqual(int(picked.action.action[8]), 1)

    def test_a_parked_answer_ends_the_entry(self) -> None:
        world = _bar_world(0)
        task = _task(world)
        task._set_phase("talk")
        task._texts = [eve.TEXT_JUICE, 0x00E8]
        set_live_u16(world.ram, "eve_hearts", 8)
        result = task._talk_done(world.ram)
        self.assertEqual(result.reason, "heart_gain")
        self.assertEqual(task._phase, "leave")

    def test_winter_gain_keeps_talking(self) -> None:
        world = _bar_world(0, season=3)
        task = _task(world)
        task._set_phase("talk")
        task._texts = [0x00ED]
        set_live_u16(world.ram, "eve_hearts", 2)
        self.assertEqual(task._talk_done(world.ram).reason, "talk_again")
        self.assertEqual(task._phase, "talk")

    def test_entries_without_a_gain_count_toward_the_cap(self) -> None:
        world = _bar_world(0)
        task = _task(world, max_talks_per_entry=2)
        task._set_phase("talk")
        task._texts = [0x00E5]
        self.assertEqual(task._talk_done(world.ram).reason, "talk_next")
        task._texts = [0x00E5]
        self.assertEqual(task._talk_done(world.ram).reason, "entry_exhausted")
        self.assertEqual(task._empty_entries, 1)
        self.assertEqual(task._phase, "leave")

    def test_leave_walks_right_then_down(self) -> None:
        world = _bar_world(0)
        task = _task(world)
        task._set_phase("leave")
        set_player_pos(world.ram, 78, 446)
        right = task.step(world)
        self.assertEqual(int(right.action.action[7]), 1)
        set_player_pos(world.ram, 144, 446)
        down = task.step(world)
        self.assertEqual(int(down.action.action[5]), 1)

    def test_leaving_resumes_with_a_fresh_entry(self) -> None:
        world = _bar_world(0)
        task = _task(world)
        task._set_phase("leave")
        world.ram[eve.ADDR_TILEMAP] = eve.TOWN_TILEMAP
        self.assertEqual(task.step(world).reason, "outside_bar")
        self.assertEqual(task._phase, "outside")
        task._phase_frames = eve.OUTSIDE_FRAMES
        task.step(world)
        self.assertEqual(task._phase, "enter")


if __name__ == "__main__":
    unittest.main()
