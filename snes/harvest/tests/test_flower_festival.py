"""Flower Festival task: lattice route planning and the stop state machine."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import unittest

from day_plan_test_helpers import make_world, set_live_u16, set_player_pos
from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.planner.day_phase_calendar import attend_festival_phase
from harvest.planner.day_phase_types import PhaseKind
from harvest.tasks import flower_festival as ff
from harvest.tasks.flower_festival import HOPS, FlowerFestivalTask, WalkGrid
from retro_harness import TaskStatus

ADDR_ACTION = field_spec("player_action").address
ADDR_TEXT = field_spec("dialog_text_id").address


def _festival_world(*, day: int = 23, flags: int = 0xF7FF):
    world = make_world(ff.FESTIVAL_TILEMAP)
    set_live_u16(world.ram, "day", day)
    set_live_u16(world.ram, "season", 0)
    set_player_pos(world.ram, 264, 312)
    world.ram[0x11F74], world.ram[0x11F75] = flags & 0xFF, flags >> 8
    return world


def _set_flags(world, flags: int) -> None:
    world.ram[0x11F74], world.ram[0x11F75] = flags & 0xFF, flags >> 8


def _task(world, stop_index: int = 0, stop=None) -> FlowerFestivalTask:
    task = FlowerFestivalTask()
    task.reset(world)
    task._start_stop(stop_index, stop)
    return task


class GridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = WalkGrid.load()

    def test_entry_cell_is_the_origin(self) -> None:
        self.assertEqual(self.grid.origin, (264, 312))
        self.assertEqual(self.grid.cell_of((264, 312)), (0, 0))
        self.assertEqual(self.grid.center((2, 3)), (280, 336))

    def test_every_open_hop_lands_on_a_known_cell(self) -> None:
        for (i, j), mask in self.grid.cells.items():
            for name, (di, dj, bit) in HOPS.items():
                if mask & bit:
                    self.assertIn((i + di, j + dj), self.grid.cells, f"{(i, j)} {name}")

    def test_route_to_every_stop_exists_from_the_entry(self) -> None:
        start = self.grid.nearest((264, 312))
        for stop in ff.STOPS + (ff.PARTNER,):
            goal = self.grid.nearest(stop.stand)
            self.assertIsNotNone(goal, stop.name)
            self.assertIsNotNone(self.grid.path(start, goal), stop.name)

    def test_path_is_shortest_and_obeys_closed_edges(self) -> None:
        self.assertEqual(self.grid.path((0, 0), (0, 2)), ["down", "down"])
        detour = self.grid.path((0, 0), (0, 2), frozenset({((0, 0), "down")}))
        self.assertGreater(len(detour), 2)
        self.assertNotEqual(detour[0], "down")

    def test_unreachable_goal_has_no_path(self) -> None:
        self.assertIsNone(self.grid.path((0, 0), (500, 500)))


class StopsTest(unittest.TestCase):
    def test_eleven_bits_are_cleared_once_each(self) -> None:
        bits = sorted(s.bit for s in ff.STOPS if s.bit is not None)
        self.assertEqual(bits, list(range(11)))

    def test_stands_are_close_to_their_npc(self) -> None:
        for stop in ff.STOPS + (ff.HOST, ff.PARTNER):
            dx, dy = stop.stand[0] - stop.npc[0], stop.stand[1] - stop.npc[1]
            self.assertLessEqual(max(abs(dx), abs(dy)), 40, stop.name)

    def test_partner_is_eve(self) -> None:
        self.assertEqual(ff.PARTNER.npc, ff.STOPS[0].npc)
        self.assertEqual(ff.STOPS[0].name, "eve")

    def test_all_talked_mask_is_the_eleven_bits(self) -> None:
        self.assertEqual(ff.ALL_TALKED_MASK, 0x07FF)
        self.assertEqual(ff.HOST_BIT, 0x0800)


class CalendarTest(unittest.TestCase):
    def test_flower_festival_phase_uses_the_task_and_ends_the_day(self) -> None:
        phase = attend_festival_phase("spring", 23)
        self.assertEqual(phase.kind, PhaseKind.FLOWER_FESTIVAL)
        self.assertTrue(phase.params["ends_day"])
        self.assertEqual(phase.failure_policy, "optional")

    def test_other_festivals_keep_the_recording(self) -> None:
        self.assertEqual(attend_festival_phase("fall", 12).kind, PhaseKind.RECORDED)


class EntryTest(unittest.TestCase):
    def test_start_in_the_ground_waits_out_the_scene_first(self) -> None:
        world = _festival_world()
        task = FlowerFestivalTask()
        task.reset(world)
        self.assertEqual(task._phase, "settle")
        for _ in range(ff.SETTLE_FRAMES - 1):
            self.assertEqual(task.step(world).reason, "festival_settle")
        self.assertEqual(task.step(world).reason, "festival_ready")
        self.assertEqual(task._phase, "walk")

    def test_west_edge_in_town_goes_straight_to_the_exit(self) -> None:
        world = make_world(ff.TOWN_TILEMAP)
        set_player_pos(world.ram, 10, 150)
        task = FlowerFestivalTask()
        task.reset(world)
        self.assertEqual(task.step(world).reason, "at_edge")
        result = task.step(world)
        self.assertEqual(result.reason, "enter_festival")
        self.assertEqual(int(result.action.action[6]), 1)  # Left

    def test_enter_gives_up_when_the_ground_does_not_load(self) -> None:
        world = make_world(ff.TOWN_TILEMAP)
        set_player_pos(world.ram, 10, 150)
        task = FlowerFestivalTask()
        task.reset(world)
        task.step(world)
        task._frames = ff.ENTER_TIMEOUT
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("did not load", result.reason)

    def test_route_to_the_edge_depends_on_the_map(self) -> None:
        farm = FlowerFestivalTask._hops_to_edge(0x00)
        path = FlowerFestivalTask._hops_to_edge(ff.PATH_TILEMAP)
        self.assertGreater(len(farm), len(path))
        self.assertTrue(path[-1].is_exit)
        town = FlowerFestivalTask._hops_to_edge(ff.TOWN_TILEMAP)
        self.assertEqual(town[0].tilemap, ff.TOWN_TILEMAP)
        self.assertTrue(town[-1].is_exit)
        self.assertIsNone(FlowerFestivalTask._hops_to_edge(0x1E))

    def test_far_map_fails_with_a_reason(self) -> None:
        world = make_world(0x1E)
        task = FlowerFestivalTask()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("cannot start", result.reason)

    def test_town_walk_is_delegated(self) -> None:
        world = make_world(ff.PATH_TILEMAP)
        set_player_pos(world.ram, 100, 200)
        nav = mock.MagicMock()
        nav.step.return_value = ff.TaskResult(
            status=TaskStatus.RUNNING, action=ff.ActionResult(ff.make_action(up=True)), reason="nav"
        )
        with mock.patch.object(ff, "MultiMapNavTask", return_value=nav) as cls:
            task = FlowerFestivalTask()
            task.reset(world)
            first = task.step(world)
        self.assertEqual(first.reason, "walk_to_edge")
        self.assertEqual(cls.call_count, 1)


class WalkTest(unittest.TestCase):
    def test_first_step_takes_a_lattice_hop_toward_the_stand(self) -> None:
        world = _festival_world()
        task = _task(world)
        result = task.step(world)
        self.assertEqual(result.reason, "hop")
        self.assertIsNotNone(task._hop)
        self.assertEqual(int(result.action.action[5]), 1)  # eve is down and right of the entry

    def test_hop_finishes_when_the_axis_arrives(self) -> None:
        world = _festival_world()
        task = _task(world)
        task.step(world)
        name, target = task._hop
        set_player_pos(world.ram, *target)
        self.assertEqual(task.step(world).reason, "hop_done")
        self.assertIsNone(task._hop)

    def test_a_stalled_hop_closes_the_edge_and_replans(self) -> None:
        world = _festival_world()
        task = _task(world)
        task.step(world)
        reasons = [task.step(world).reason for _ in range(ff.STALL_FRAMES + 2)]
        self.assertIn("hop_blocked", reasons)
        self.assertEqual(len(task._closed), 1)

    def test_too_many_blocked_hops_fail_the_walk(self) -> None:
        world = _festival_world()
        task = _task(world)
        task._replans = 6
        task.step(world)
        results = [task.step(world) for _ in range(ff.STALL_FRAMES + 2)]
        failed = [r for r in results if r.status == TaskStatus.FAILURE]
        self.assertTrue(failed)
        self.assertIn("blocked", failed[0].reason)

    def test_at_the_stand_it_turns_to_talk(self) -> None:
        world = _festival_world()
        task = _task(world)
        set_player_pos(world.ram, *ff.STOPS[0].stand)
        self.assertEqual(task.step(world).reason, "at_stand")
        self.assertEqual(task._phase, "talk")

    def test_the_host_is_reached_along_the_row(self) -> None:
        world = _festival_world(flags=0xF000)
        task = _task(world, 12, ff.HOST)
        set_player_pos(world.ram, 138, 251)
        first = task.step(world)
        self.assertEqual(first.reason, "host_walk")
        self.assertEqual(int(first.action.action[6]), 1)  # Left
        set_player_pos(world.ram, 93, 250)
        self.assertEqual(task.step(world).reason, "host_row_done")


class TalkTest(unittest.TestCase):
    def _talking(self, world, index, texts=(0x0231,), stop=None):
        task = _task(world, index, stop)
        task._begin("talk")
        task._reset_talk()
        task._texts = list(texts)
        return task

    def test_first_presses_hold_the_facing_direction(self) -> None:
        world = _festival_world()
        task = _task(world, 0)
        task._begin("talk")
        task._reset_talk()
        result = task.step(world)
        self.assertEqual(result.reason, "talk")
        self.assertEqual(int(result.action.action[5]), 1)  # Down, facing Eve
        self.assertEqual(int(result.action.action[8]), 1)  # A

    def test_the_snow_seed_seller_is_declined(self) -> None:
        world = _festival_world()
        stop = next(s for s in ff.STOPS if s.name == ff.SNOW_SEED_STOP)
        task = _task(world, 6, stop)
        task._begin("talk")
        task._reset_talk()
        world.ram[ADDR_INPUT_LOCK] = 2
        world.ram[0x019B] = 0x10
        world.ram[0x018F] = 0
        decline = task.step(world)
        self.assertEqual(int(decline.action.action[5]), 1)  # Down, off "Buy it"
        self.assertEqual(int(decline.action.action[8]), 0)
        world.ram[0x018F] = 1
        confirm = task.step(world)
        self.assertEqual(int(confirm.action.action[5]), 0)
        self.assertEqual(int(confirm.action.action[8]), 1)  # A on "Not this time"

    def test_the_dance_host_still_confirms_row_zero(self) -> None:
        world = _festival_world(flags=0xF000)
        task = _task(world, 12, ff.HOST)
        task._begin("talk")
        task._reset_talk()
        world.ram[ADDR_INPUT_LOCK] = 2
        world.ram[0x019B] = 0x10
        world.ram[0x018F] = 0
        result = task.step(world)
        self.assertEqual(int(result.action.action[8]), 1)  # A
        self.assertEqual(int(result.action.action[5]), 0)

    def test_dialog_pulses_a_then_waits_for_quiet(self) -> None:
        world = _festival_world()
        task = _task(world, 0)
        task._begin("talk")
        task._reset_talk()
        world.ram[ADDR_INPUT_LOCK] = 2
        task.step(world)
        self.assertTrue(task._seen_lock)
        world.ram[ADDR_INPUT_LOCK] = 1
        for _ in range(ff.QUIET_FRAMES):
            result = task.step(world)
        self.assertEqual(result.reason, "talk_close")
        result = task.step(world)
        self.assertNotEqual(result.reason, "talk_close")

    def test_cleared_bit_moves_to_the_next_stop(self) -> None:
        world = _festival_world(flags=0xF3FF)  # Eve's bit 10 clear
        task = self._talking(world, 0)
        self.assertEqual(task._after_talk(world.ram).reason, "next_stop")
        self.assertEqual(task._stop.name, "maria")

    def test_a_missed_talk_retries_then_fails(self) -> None:
        world = _festival_world(flags=0xF7FF)
        task = self._talking(world, 0, texts=())
        self.assertEqual(task._after_talk(world.ram).reason, "retry_stop")
        task._texts = []
        self.assertEqual(task._after_talk(world.ram).reason, "retry_stop")
        task._texts = []
        result = task._after_talk(world.ram)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("did not register", result.reason)

    def test_the_blue_woman_does_not_move_while_a_villager_bit_is_set(self) -> None:
        world = _festival_world(flags=0xF010)  # one villager bit still set
        task = self._talking(world, len(ff.STOPS) - 1, texts=(0x021F,))
        for _ in range(2):
            self.assertEqual(task._after_talk(world.ram).reason, "retry_stop")
        result = task._after_talk(world.ram)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("blue_woman", result.reason)

    def test_blue_woman_then_host(self) -> None:
        world = _festival_world(flags=0xF000)
        task = self._talking(world, len(ff.STOPS) - 1, texts=(0x021F,))
        self.assertEqual(task._after_talk(world.ram).reason, "to_host")
        self.assertIs(task._stop, ff.HOST)

    def test_host_needs_the_dance_bit_before_the_partner(self) -> None:
        world = _festival_world(flags=0xF000)
        task = self._talking(world, 12, texts=(0x0234, 0x0235), stop=ff.HOST)
        self.assertEqual(task._after_talk(world.ram).reason, "retry_stop")
        _set_flags(world, 0xF800)
        task._texts = [0x0234, 0x0235]
        self.assertEqual(task._after_talk(world.ram).reason, "to_partner")
        self.assertEqual(task._phase, "to_partner")
        self.assertIs(task._stop, ff.PARTNER)

    def test_partner_talk_starts_the_dance(self) -> None:
        world = _festival_world(flags=0xF800)
        task = self._talking(world, 13, texts=(), stop=ff.PARTNER)
        self.assertEqual(task._after_talk(world.ram).reason, "dance")
        self.assertEqual(task._phase, "dance")

    def test_leaving_the_alcove_walks_right_along_the_row(self) -> None:
        world = _festival_world(flags=0xF800)
        task = _task(world, 13, ff.PARTNER)
        task._begin("to_partner")
        set_player_pos(world.ram, 93, 250)
        result = task.step(world)
        self.assertEqual(result.reason, "leave_alcove")
        self.assertEqual(int(result.action.action[7]), 1)  # Right
        set_player_pos(world.ram, 140, 250)
        self.assertEqual(task.step(world).reason, "to_partner_walk")
        self.assertEqual(task._phase, "walk")


class FinishTest(unittest.TestCase):
    def test_the_day_rolling_is_success(self) -> None:
        world = _festival_world()
        task = _task(world)
        task._begin("dance")
        set_live_u16(world.ram, "day", 24)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_dance_text_is_dismissed_with_a(self) -> None:
        world = _festival_world(flags=0)
        task = _task(world)
        task._begin("dance")
        world.ram[ADDR_INPUT_LOCK] = 2
        result = task.step(world)
        self.assertEqual(result.reason, "dance_text")

    def test_timeout_fails(self) -> None:
        world = _festival_world()
        task = _task(world)
        task._steps = task.timeout
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("timeout", result.reason)


if __name__ == "__main__":
    unittest.main()
