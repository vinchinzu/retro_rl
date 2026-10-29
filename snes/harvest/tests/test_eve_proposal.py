"""Eve proposal task: blockers, route choice and the phase machine on fake RAM."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import unittest

from day_plan_test_helpers import make_world, set_live_u16, set_player_pos
from harvest.core.carry import ADDR_TOOL_BACKPACK, ADDR_TOOL_SELECTED
from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.planner.day_phase_catalog import PHASE_SEQUENCES
from harvest.planner.day_phase_types import PhaseKind, coerce_phase_kind
from harvest.tasks import eve_proposal as ep
from harvest.tasks.eve_proposal import EveProposalTask, proposal_blockers
from retro_harness import TaskStatus


def _set_u8(world, field: str, value: int) -> None:
    world.ram[field_spec(field).address] = value


def _world(tilemap: int = ep.MOUNTAIN_TILEMAP, *, hearts: int = 250, weekday: int = 6, hour: int = 9):
    world = make_world(tilemap)
    set_live_u16(world.ram, "eve_hearts", hearts)
    _set_u8(world, "weekday", weekday)
    _set_u8(world, "hour", hour)
    _set_u8(world, "day", 27)
    return world


def _eve_at(x: int = 504, y: int = 216):
    obj = SimpleNamespace(is_player=False, sprite_table_idx=0x0245, pixel=(x, y))
    return mock.patch.object(ep, "game_objects", return_value=[obj])


def _task(world, **kw) -> EveProposalTask:
    task = EveProposalTask(**kw)
    task.reset(world)
    return task


class BlockerTest(unittest.TestCase):
    def test_a_sunny_saturday_with_250_hearts_is_clear(self) -> None:
        self.assertIsNone(proposal_blockers(_world().ram))

    def test_hearts_below_the_floor(self) -> None:
        self.assertIn("249 < 250", proposal_blockers(_world(hearts=249).ram))

    def test_only_saturday(self) -> None:
        self.assertIn("Saturdays", proposal_blockers(_world(weekday=0).ram))

    def test_not_after_closing(self) -> None:
        self.assertIn("17:00", proposal_blockers(_world(hour=17).ram))

    def test_rain_snow_and_storm(self) -> None:
        for flag in (ep.RAIN, ep.SNOW, ep.STORM):
            world = _world()
            world.ram[ep.ADDR_WEATHER_FLAGS] = flag
            self.assertIn("sunny", proposal_blockers(world.ram), hex(flag))

    def test_already_married(self) -> None:
        world = _world()
        world.ram[ep.ADDR_WIFE_FLAGS] = 0x02
        self.assertEqual(proposal_blockers(world.ram), "already married")


class RouteTest(unittest.TestCase):
    def test_route_depends_on_the_starting_map(self) -> None:
        mountain = EveProposalTask.hops_to_eve(ep.MOUNTAIN_TILEMAP)
        path = EveProposalTask.hops_to_eve(ep.PATH_TILEMAP)
        farm = EveProposalTask.hops_to_eve(0x00)
        self.assertEqual(len(mountain), len(ep.MOUNTAIN_TO_EVE))
        self.assertGreater(len(path), len(mountain))
        self.assertGreater(len(farm), len(path))
        self.assertIsNone(EveProposalTask.hops_to_eve(0x1E))

    def test_route_ends_below_eve(self) -> None:
        self.assertEqual(ep.MOUNTAIN_TO_EVE[-1], (496, 240))

    def test_named_sequence_and_kind_exist(self) -> None:
        phase = PHASE_SEQUENCES["propose_eve"][0]
        self.assertEqual(coerce_phase_kind(phase.kind), PhaseKind.PROPOSE_EVE)
        self.assertTrue(phase.params["ends_day"])


class StartTest(unittest.TestCase):
    def test_blockers_stop_the_walk_before_it_starts(self) -> None:
        world = _world(hearts=100)
        result = _task(world).step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("blocked", result.reason)

    def test_blockers_can_be_disabled(self) -> None:
        world = _world(hearts=100)
        with mock.patch.object(ep, "MultiMapNavTask") as nav:
            nav.return_value.step.return_value = ep.TaskResult(
                status=TaskStatus.RUNNING, action=ep.ActionResult(ep.make_action()), reason="nav"
            )
            result = _task(world, check_blockers=False).step(world)
        self.assertEqual(result.reason, "walk_to_eve")

    def test_unsupported_map_fails(self) -> None:
        world = _world(0x1E)
        result = _task(world).step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("cannot start", result.reason)

    def test_the_walk_is_one_nav_task_with_the_probed_route(self) -> None:
        world = _world(ep.PATH_TILEMAP)
        with mock.patch.object(ep, "MultiMapNavTask") as nav:
            nav.return_value.step.return_value = ep.TaskResult(
                status=TaskStatus.RUNNING, action=ep.ActionResult(ep.make_action(up=True)), reason="nav"
            )
            task = _task(world)
            task.step(world)
            task.step(world)
        self.assertEqual(nav.call_count, 1)
        waypoints = nav.call_args.kwargs["waypoints"]
        self.assertEqual(waypoints[-1].target_px, (496, 240))


class ProposeTest(unittest.TestCase):
    def _at(self, phase: str, world, frames: int = 0) -> EveProposalTask:
        task = _task(world)
        task._begin(phase)
        task._frames = frames
        return task

    def test_arriving_waits_then_finds_eve(self) -> None:
        world = _world()
        task = self._at("settle", world)
        with _eve_at():
            self.assertEqual(task.step(world).reason, "mountain_settle")
            task._frames = 30
            self.assertEqual(task.step(world).reason, "eve_found")
        self.assertEqual(task._phase, "equip")

    def test_no_eve_is_a_named_failure(self) -> None:
        world = _world()
        task = self._at("settle", world, 30)
        with mock.patch.object(ep, "game_objects", return_value=[]):
            result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("not on the mountain", result.reason)

    def test_feather_already_selected(self) -> None:
        world = _world()
        world.ram[ADDR_TOOL_SELECTED] = ep.FEATHER
        task = self._at("equip", world)
        self.assertEqual(task.step(world).reason, "feather_ready")
        self.assertEqual(task._phase, "align")

    def test_feather_in_the_backpack_is_swapped_in_with_x(self) -> None:
        world = _world()
        world.ram[ADDR_TOOL_BACKPACK] = ep.FEATHER
        task = self._at("equip", world)
        result = task.step(world)
        self.assertEqual(result.reason, "swap_tools")
        self.assertEqual(int(result.action.action[9]), 1)  # X

    def test_no_feather_fails(self) -> None:
        world = _world()
        task = self._at("equip", world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("no blue feather", result.reason)

    def test_align_walks_to_the_exact_pixel_below_her(self) -> None:
        world = _world()
        task = self._at("align", world)
        task._eve = (504, 216)
        with _eve_at():
            set_player_pos(world.ram, 496, 240)
            up = task.step(world)
            self.assertEqual(int(up.action.action[4]), 1)  # the longer error first
            set_player_pos(world.ram, 496, 229)
            right = task.step(world)
            self.assertEqual(int(right.action.action[7]), 1)
            set_player_pos(world.ram, 503, 229)
            self.assertEqual(task.step(world).reason, "below_eve")
        self.assertEqual(task._phase, "use")

    def test_two_pixels_low_is_not_close_enough(self) -> None:
        world = _world()
        task = self._at("align", world)
        task._eve = (504, 216)
        with _eve_at():
            set_player_pos(world.ram, 504, 232)
            self.assertEqual(task.step(world).reason, "align")

    def test_use_swings_the_feather_with_y_facing_up(self) -> None:
        world = _world()
        task = self._at("use", world, 3)
        result = task.step(world)  # frame 4
        self.assertEqual(result.reason, "use_feather")
        self.assertEqual(int(result.action.action[1]), 1)  # Y
        self.assertEqual(int(result.action.action[4]), 1)  # Up

    def test_proposal_text_is_dismissed_with_a(self) -> None:
        world = _world()
        task = self._at("use", world)
        world.ram[ADDR_INPUT_LOCK] = 2
        self.assertEqual(task.step(world).reason, "proposal_text")

    def test_the_wife_bit_starts_the_wedding_and_the_day_roll_is_success(self) -> None:
        world = _world()
        task = self._at("use", world)
        world.ram[ep.ADDR_WIFE_FLAGS] = ep.WIFE_BIT
        task.step(world)
        self.assertEqual(task._phase, "wedding")
        _set_u8(world, "day", 28)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        set_live_u16(world.ram, "upgrade_flags", 0x40)
        world.ram[ep.ADDR_TILEMAP] = 0x15
        self.assertEqual(task.step(world).status, TaskStatus.RUNNING)
        world.ram[ep.ADDR_TILEMAP] = 0x16
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("married Eve", result.reason)

    def test_use_gives_up_without_a_proposal(self) -> None:
        world = _world()
        task = self._at("use", world, ep.USE_TIMEOUT)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("did not propose", result.reason)


class FeatherBuyRouteTest(unittest.TestCase):
    def test_east_gate_landing_is_outside_the_first_town_hop(self) -> None:
        from harvest.tasks.proposal import _peddler_hops_from_farm, _peddler_hops_from_path

        for hops in (_peddler_hops_from_farm(), _peddler_hops_from_path()):
            town = [wp for wp in hops if wp.tilemap == 0x04]
            first = town[0]
            self.assertEqual(first.target_px, (700, 424))
            self.assertEqual(first.run_direction, "left")
            self.assertTrue(first.force_run)
            self.assertGreater(abs(744 - first.target_px[0]), first.radius)
            self.assertEqual(town[-1].target_px, (568, 288))
            self.assertGreater(len(town), 2)


if __name__ == "__main__":
    unittest.main()
