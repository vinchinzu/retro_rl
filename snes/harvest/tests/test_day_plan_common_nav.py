"""Nav / directional transition / multi-map day-plan sequences.

Split from test_day_plan_common (under 1k LOC soft max).
"""
from __future__ import annotations

from pathlib import Path
import sys

# Path-stable import of sibling helpers (works under unittest and pytest importlib mode).
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from day_plan_test_helpers import (
    make_navigation_ram,
    make_transition_world,
    make_world,
    set_player_pos,
)

import unittest
from types import SimpleNamespace

from harvest.planner.day_plan import (
    DirectionalTransitionTask,
    MultiMapNavTask,
    NavTask,
    PhaseSpec,
    TaskResult,
    TaskStatus,
)
from harvest.planner.day_task_factory import DayTaskFactory
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    WEED,
)
from harvest.tasks.nav import (
    Navigator,
    Pathfinder,
    Point,
)
from harvest.tasks.farm_clearer import TileScanner
from harvest.tasks.harvest_task import ADDR_SHIPPING_MONEY
from harvest.maps.map_config import ROUTES, Waypoint
from harvest.core.ram_catalog import field_spec


class DayPlanSequenceCommonNavTests(unittest.TestCase):
    def test_navigator_treats_transition_tile_0x76_as_stale_and_repaths(self) -> None:
        ram = make_navigation_ram()
        navigator = Navigator(Pathfinder(TileScanner()))
        navigator.update(ram)
        navigator.path = [(14, 8)]

        action = navigator.follow_path(ram)

        self.assertIsNone(action)
        self.assertEqual(navigator.path, [])

    def test_navigator_clears_path_for_known_nonwalkable_tile(self) -> None:
        ram = make_navigation_ram(blocked_tile=(14, 8), blocked_id=0xA6)
        navigator = Navigator(Pathfinder(TileScanner()))
        navigator.update(ram)
        navigator.path = [(14, 8)]

        action = navigator.follow_path(ram)

        self.assertIsNone(action)
        self.assertEqual(navigator.path, [])
        self.assertIn((14, 8), navigator.pathfinder.temp_blocked)

    def test_directional_transition_uses_stand_tile_before_holding_direction(self) -> None:
        world = make_transition_world(0x28, current_tile=(8, 11))
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x28,
            target_tilemap=0x00,
            stand_tile=(8, 12),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)

    def test_directional_transition_routes_to_nonwalkable_exit_tile(self) -> None:
        ram = make_navigation_ram(current_tile=(2, 1), blocked_tile=(1, 1), blocked_id=0xA6)
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_MAP + 0 * 64 + 0] = 0xC0
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = DirectionalTransitionTask(
            direction="left",
            origin_tilemap=0x00,
            target_tilemap=0x0C,
            stand_tile=(0, 0),
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[6]), 0)

    def test_directional_transition_can_settle_after_tilemap_change(self) -> None:
        origin = make_transition_world(0x28, current_tile=(8, 12))
        target = make_transition_world(0x00, current_tile=(8, 13))
        set_player_pos(target.ram, 456, 360)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x28,
            target_tilemap=0x00,
            min_frames_before_success=1,
            settle_frames=3,
        )

        task.reset(origin)
        self.assertEqual(task.step(origin).status, TaskStatus.RUNNING)
        self.assertEqual(task.step(target).status, TaskStatus.RUNNING)
        self.assertEqual(task.step(target).status, TaskStatus.RUNNING)
        result = task.step(target)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tilemap=0x00")

    def test_directional_transition_waits_for_target_stand_tile(self) -> None:
        origin = make_transition_world(0x00)
        transitional = make_transition_world(0x27)
        landed = make_transition_world(0x27)
        set_player_pos(origin.ram, 20 * 16 + 8, 22 * 16 + 8)
        set_player_pos(transitional.ram, 20 * 16 + 8, 21 * 16 + 8)
        set_player_pos(landed.ram, 8 * 16 + 8, 22 * 16 + 8)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x27,
            min_frames_before_success=1,
            settle_frames=2,
            target_stand_tile=(8, 22),
            target_stand_tolerance=1,
        )

        task.reset(origin)
        self.assertEqual(task.step(origin).status, TaskStatus.RUNNING)
        waiting = task.step(transitional)
        self.assertEqual(waiting.status, TaskStatus.RUNNING)
        self.assertEqual(waiting.reason, "transition target settle")
        self.assertEqual(int(waiting.action.action[4]), 1)
        self.assertEqual(task.step(landed).status, TaskStatus.RUNNING)
        result = task.step(landed)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tilemap=0x27")

    def test_directional_transition_factory_passes_target_stand_tile(self) -> None:
        spec = PhaseSpec(
            "ENTER_BARN",
            "directional_transition",
            {
                "direction": "up",
                "origin_tilemap": 0x00,
                "target_tilemap": 0x27,
                "target_stand_tile": (8, 22),
                "target_stand_tolerance": 1,
            },
        )

        task = DayTaskFactory().make_task(spec, make_transition_world(0x00))

        self.assertIsInstance(task, DirectionalTransitionTask)
        self.assertEqual(task.target_stand_tile, (8, 22))
        self.assertEqual(task.target_stand_tolerance, 1)

    def test_directional_transition_factory_passes_door_entry_params(self) -> None:
        spec = PhaseSpec(
            "ENTER_COOP",
            "directional_transition",
            {
                "direction": "up",
                "origin_tilemap": 0x00,
                "target_tilemap": 0x28,
                "stand_tile": (28, 22),
                "door_align_px": 456,
                "overshoot_limit_px": 330,
                "require_empty_hands": True,
            },
        )

        task = DayTaskFactory().make_task(spec, make_transition_world(0x00))

        self.assertIsInstance(task, DirectionalTransitionTask)
        self.assertEqual(task.door_align_px, 456)
        self.assertEqual(task.overshoot_limit_px, 330)
        self.assertTrue(task.require_empty_hands)

    def test_directional_transition_clears_hands_before_door(self) -> None:
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_transition_world(0x00, current_tile=(8, 26))
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0D
        world.ram[field_spec("player_state").address + base] = 0x03
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x15,
            stand_tile=(8, 26),
            require_empty_hands=True,
            door_align_px=136,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "clear hands for door")
        self.assertEqual(task._hands_attempts, 1)
        self.assertTrue(task._action_queue)

    def test_directional_transition_overshoot_restands(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 20))
        set_player_pos(world.ram, 136, 310)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x15,
            stand_tile=(8, 26),
            overshoot_limit_px=328,
            door_align_px=136,
        )

        task.reset(world)
        task._stand_reached = True
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertFalse(task._stand_reached)
        # Re-stand now strafes off the embedded door column before moving down.
        self.assertEqual(int(result.action.action[6]), 1)

    def test_directional_transition_aligns_door_column_before_push(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 26))
        set_player_pos(world.ram, 120, 424)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x15,
            stand_tile=(8, 26),
            door_align_px=136,
        )

        task.reset(world)
        task._stand_reached = True
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[4]), 0)

    def test_nav_task_soft_arrives_when_stuck_near_target(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 121, 519)
        task = NavTask(
            target_px=Point(136, 520),
            radius=12,
            soft_radius=48,
            soft_stasis=5,
            timeout=200,
        )
        task.reset(world)
        task._navigator.stasis = 10

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("soft arrived", result.reason or "")

    def test_directional_transition_accepts_seasonal_farm_origin(self) -> None:
        world = make_world(0x01)
        set_player_pos(world.ram, 20 * 16 + 8, 22 * 16 + 8)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x27,
            min_frames_before_success=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn("expected origin", result.reason or "")

    def test_multi_nav_treats_seasonal_farm_as_farm_waypoint(self) -> None:
        world = make_world(0x01)
        set_player_pos(world.ram, 137, 344)
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(137, 344), radius=12)],
            initial_settle_frames=0,
        )

        task.reset(world)
        first = task.step(world)
        result = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_directional_transition_does_not_recenter_after_stand_reached(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 12))
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x28,
            stand_tile=(8, 12),
        )

        task.reset(world)
        first = task.step(world)
        self.assertEqual(int(first.action.action[4]), 1)

        set_player_pos(world.ram, 8 * 16 + 8, 12 * 16 + 6)
        second = task.step(world)

        self.assertEqual(int(second.action.action[4]), 1)
        self.assertEqual(int(second.action.action[5]), 0)

    def test_multi_nav_blocks_wrong_tilemap_before_farm_route(self) -> None:
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(137, 375), run_direction="down")],
            initial_settle_frames=0,
        )
        world = make_transition_world(0x15, current_tile=(8, 7))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("expected tilemap 0x00", result.reason or "")

    def test_multi_nav_run_direction_corrects_off_lane_before_running(self) -> None:
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(200, 200), radius=12, run_direction="right")],
            initial_settle_frames=0,
        )
        world = make_transition_world(0x00, current_tile=(10, 15))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[7]), 0)

    def test_multi_nav_bfs_routes_around_farm_weeds(self) -> None:
        world = make_transition_world(0x00, current_tile=(10, 10))
        world.ram[ADDR_MAP + 10 * 64 + 11] = WEED
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(16 * 16 + 8, 10 * 16 + 8), radius=8)],
            initial_settle_frames=0,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn((11, 10), task._farm_weed_blocks)
        self.assertNotIn((11, 10), task._navigator.path)
        # First move must take the clear vertical detour, not step right onto weed.
        self.assertEqual(int(result.action.action[7]), 0)

    def test_multi_nav_lift_throw_skips_when_gate_already_clear(self) -> None:
        world = make_transition_world(0x00, current_tile=(36, 59))
        # Clean dirt north of stand — no thrash A presses.
        world.ram[ADDR_MAP + 58 * 64 + 36] = 0x01
        task = MultiMapNavTask(
            waypoints=[
                Waypoint(
                    tilemap=0x00,
                    target_px=(36 * 16 + 8, 59 * 16 + 8),
                    radius=8,
                    action_on_arrive="lift_throw",
                    action_face="up",
                ),
                Waypoint(tilemap=0x00, target_px=(37 * 16 + 8, 57 * 16 + 8), radius=8),
            ],
            initial_settle_frames=0,
        )
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._wp_index, 1)
        self.assertIn("already clear", result.reason or "")

    def test_multi_nav_lift_throw_queues_lift_when_weed_present(self) -> None:
        world = make_transition_world(0x00, current_tile=(36, 59))
        world.ram[ADDR_MAP + 58 * 64 + 36] = WEED
        task = MultiMapNavTask(
            waypoints=[
                Waypoint(
                    tilemap=0x00,
                    target_px=(36 * 16 + 8, 59 * 16 + 8),
                    radius=8,
                    action_on_arrive="lift_throw",
                    action_face="up",
                    action_frames=4,
                    action_cooldown=2,
                )
            ],
            initial_settle_frames=0,
        )
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "lift_throw_drain")
        self.assertGreater(len(task._action_queue), 0)

    def test_farm_exit_uses_south_return_route_after_berry_ship(self) -> None:
        from harvest.planner.tasks.inventory_exit import FarmExitTask

        world = make_transition_world(0x00, current_tile=(62, 60))
        task = FarmExitTask(timeout=10_000)

        task.reset(world)

        self.assertIsInstance(task._nav, MultiMapNavTask)
        self.assertEqual(task._nav.waypoints, ROUTES["farm_south_to_west_gate"])

    def test_berry_ship_fails_closed_without_shipping_money_delta(self) -> None:
        from harvest.tasks.berry_ship import BerryShipTask

        class DoneNav:
            _wp_index = 1
            _action_queue = []

            def step(self, world):
                return TaskResult(status=TaskStatus.SUCCESS, reason="route done")

        world = make_transition_world(0x00, current_tile=(62, 60))
        task = BerryShipTask(waypoints=[Waypoint(0x00, (1001, 969))])
        task._shipping_before = 0
        task._nav = DoneNav()

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("unverified", result.reason or "")

        world.ram[ADDR_SHIPPING_MONEY] = 15
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("0->150", result.reason or "")



if __name__ == "__main__":
    unittest.main()
