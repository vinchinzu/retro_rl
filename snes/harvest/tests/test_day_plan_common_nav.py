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
from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_TILEMAP,
    WEED,
)
from harvest.tasks.nav import (
    Navigator,
    Pathfinder,
    Point,
)
from harvest.tasks.farm_clearer import TileScanner
from harvest.maps.map_config import ROUTES, Waypoint
from harvest.core.ram_catalog import field_spec


class DayPlanSequenceCommonNavTests(unittest.TestCase):
    def test_navigator_treats_unloaded_tile_as_stale_and_repaths(self) -> None:
        ram = make_navigation_ram(blocked_id=0xFF)
        navigator = Navigator(Pathfinder(TileScanner()))
        navigator.update(ram)
        navigator.path = [(14, 8)]

        action = navigator.follow_path(ram)

        self.assertIsNone(action)
        self.assertEqual(navigator.path, [])

    def test_navigator_stops_leftright_center_after_flips(self) -> None:
        """In-place L/R centering against a wall must yield to the path dir."""
        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        navigator = Navigator(Pathfinder(TileScanner()))
        set_player_pos(ram, 13 * 16 + 3, 8 * 16 + 8)  # 5px left of tile center
        navigator.update(ram)
        navigator.path = [(13, 9)]  # go down — wants X center first

        first = navigator.follow_path(ram)
        self.assertIsNotNone(first)
        self.assertEqual(int(first[7]), 1)  # right toward center

        set_player_pos(ram, 13 * 16 + 13, 8 * 16 + 8)  # overshoot right
        navigator.update(ram)
        second = navigator.follow_path(ram)
        self.assertIsNotNone(second)
        self.assertEqual(int(second[6]), 1)  # left
        self.assertGreaterEqual(navigator._center_flips, 1)

        # Keep flipping until the limiter trips, then push the path (down).
        yielded = False
        for _ in range(8):
            x = 13 * 16 + (3 if navigator._center_dir == "left" else 13)
            set_player_pos(ram, x, 8 * 16 + 8)
            navigator.update(ram)
            action = navigator.follow_path(ram)
            self.assertIsNotNone(action)
            if int(action[5]) == 1 and int(action[6]) == 0 and int(action[7]) == 0:
                yielded = True
                self.assertGreaterEqual(navigator._center_flips, 4)
                break
        self.assertTrue(yielded, "centering never yielded after left/right flips")

    def test_stalled_centering_against_a_solid_yields_to_the_path(self) -> None:
        """Y1 Fall D12 Harvest Festival (score_policy_final): the farmer stood 7 px above the tile centre
        with a solid below, so the vertical centring before a sideways step could never finish.

        Every pixel-stuck replan reset ``stasis`` to 0, which re-armed centring (it yields at stasis 30,
        but the 48-frame replan always came first), and four replans failed the walk. Centring that does
        not move the farmer for ``CENTER_STALL_FRAMES`` frames must give way; centring that works must not.
        """
        from harvest.tasks.nav import CENTER_STALL_FRAMES

        from harvest.core.ram_catalog import live_wram_base

        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        ram[ADDR_INPUT_LOCK] = 1
        ram[0x0025 + live_wram_base(ram)] = 15  # faded in: the pad is live
        navigator = Navigator(Pathfinder(TileScanner()))
        navigator.center_stall_limit = CENTER_STALL_FRAMES
        set_player_pos(ram, 13 * 16 + 8, 8 * 16 + 1)  # 7 px above centre, cannot move down
        navigator.update(ram)
        navigator.path = [(14, 8)]  # a sideways step wants Y centred first

        for _ in range(CENTER_STALL_FRAMES + 1):  # the first frame only records the pixel
            action = navigator.follow_path(ram)
            self.assertEqual(int(action[5]), 1)  # down toward the centre
            self.assertEqual(int(action[7]), 0)
            navigator.update(ram)  # same pixel: the solid held the farmer
            navigator.stasis = 0  # replans reset it, which is what kept the old 30-frame valve from ever firing
        walking = navigator.follow_path(ram)
        self.assertEqual(int(walking[7]), 1)  # right, no centring
        self.assertEqual(int(walking[5]), 0)

        navigator.update(ram)
        set_player_pos(ram, 14 * 16 + 8, 8 * 16 + 1)  # reached the next tile: centring is armed again
        navigator.update(ram)
        navigator.path = [(15, 8)]
        again = navigator.follow_path(ram)
        self.assertEqual(int(again[5]), 1)

    def test_a_dead_pad_does_not_count_as_stalled_centering(self) -> None:
        """The farm's fade-in (WRAM $0025 below 15) and any non-free player state ignore the pad for ~70 frames.

        compose3 (first stall detector) counted those frames, skipped centring before the farmer reached the
        one-tile house walkway, and died on Y1 Spring D2 with ``ensure_hoe ... pixel_stuck (131,415)``.
        """
        from harvest.core.ram_catalog import live_wram_base
        from harvest.tasks.nav import CENTER_STALL_FRAMES

        for label, poke in (("fade-in", lambda ram: ram.__setitem__(0x0025 + live_wram_base(ram), 7)),
                            ("tool animation", lambda ram: ram.__setitem__(0x00D2 + live_wram_base(ram), 4))):
            ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
            ram[ADDR_INPUT_LOCK] = 1
            poke(ram)
            navigator = Navigator(Pathfinder(TileScanner()))
            navigator.center_stall_limit = CENTER_STALL_FRAMES
            set_player_pos(ram, 13 * 16 + 8, 8 * 16 + 1)
            navigator.update(ram)
            navigator.path = [(14, 8)]
            for step in range(CENTER_STALL_FRAMES * 3):
                action = navigator.follow_path(ram)
                self.assertEqual(int(action[7]), 0, f"{label}: gave up centring at step {step} while the pad was dead")
                navigator.update(ram)
                navigator.stasis = 0

    def test_stall_detector_is_off_unless_a_walk_asks_for_it(self) -> None:
        """compose4 died on Y1 Spring D2 with the detector on for every walk. Default navigators keep the
        original behaviour: only ``stasis`` yields centring, and replans reset it."""
        from harvest.core.ram_catalog import live_wram_base
        from harvest.tasks.nav import CENTER_STALL_FRAMES

        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        ram[ADDR_INPUT_LOCK] = 1
        ram[0x0025 + live_wram_base(ram)] = 15
        navigator = Navigator(Pathfinder(TileScanner()))
        self.assertIsNone(navigator.center_stall_limit)
        set_player_pos(ram, 13 * 16 + 8, 8 * 16 + 1)
        navigator.update(ram)
        navigator.path = [(14, 8)]
        for step in range(CENTER_STALL_FRAMES * 3):
            action = navigator.follow_path(ram)
            self.assertEqual(int(action[5]), 1, f"default navigator stopped centring at step {step}")
            navigator.update(ram)
            navigator.stasis = 0

    def test_multinav_hands_the_stall_limit_to_every_navigator_it_builds(self) -> None:
        from harvest.tasks.nav import CENTER_STALL_FRAMES

        plain = MultiMapNavTask(waypoints=[Waypoint(tilemap=0x00, target_px=(136, 120), radius=8)])
        self.assertIsNone(plain._navigator.center_stall_limit)
        task = MultiMapNavTask(waypoints=[Waypoint(tilemap=0x00, target_px=(136, 120), radius=8)],
                               center_stall_frames=CENTER_STALL_FRAMES)
        self.assertEqual(task._navigator.center_stall_limit, CENTER_STALL_FRAMES)
        task._rebuild_pathfinder(0x10)  # every map change builds a new navigator
        self.assertEqual(task._navigator.center_stall_limit, CENTER_STALL_FRAMES)

    def test_cover_walk_that_runs_out_of_time_ends_where_the_farmer_stands(self) -> None:
        """Y3 Summer: the route to the first cover stand hit a bank every evening for 12 days and forfeited the sowing."""
        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        world = type("W", (), {"ram": ram})()
        wp = [Waypoint(tilemap=0x00, target_px=(136, 120), radius=8)]
        hard = MultiMapNavTask(waypoints=list(wp), timeout=0)
        soft = MultiMapNavTask(waypoints=list(wp), timeout=0, soft_timeout=True)
        for task in (hard, soft):
            task.reset(world)
        self.assertEqual(hard.step(world).status, TaskStatus.FAILURE)
        result = soft.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("soft", result.reason)

    def test_centering_that_moves_the_farmer_never_counts_as_a_stall(self) -> None:
        from harvest.tasks.nav import CENTER_STALL_FRAMES

        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        ram[ADDR_INPUT_LOCK] = 1
        navigator = Navigator(Pathfinder(TileScanner()))
        navigator.center_stall_limit = CENTER_STALL_FRAMES
        navigator.path = [(14, 8)]
        for step in range(CENTER_STALL_FRAMES * 2):  # twice the stall budget, pixel always changing
            set_player_pos(ram, 13 * 16 + 8, 8 * 16 + (1 + step % 3))  # creeping, never the same pixel twice in a row
            navigator.update(ram)
            navigator.stasis = 0  # replans reset it
            action = navigator.follow_path(ram)
            self.assertIsNotNone(action)
            self.assertEqual(int(action[7]), 0, f"yielded to the path at step {step} while centring was moving")

    def test_multinav_fails_closed_on_pixel_stuck_leftright(self) -> None:
        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        world = SimpleNamespace(ram=ram, info={}, obs=None, frame=0)
        task = MultiMapNavTask(
            waypoints=[
                Waypoint(
                    tilemap=0x00,
                    target_px=(13 * 16 + 8, 20 * 16 + 8),
                    radius=12,
                )
            ],
            timeout=800,
            initial_settle_frames=0,
        )
        task.reset(world)
        task._navigator.path = [(13, 9)]
        task._pixel_stuck = 48
        task._pixel_replans = 3

        result = task._pixel_stuck_replan()

        self.assertIsNotNone(result)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("pixel_stuck", result.reason or "")

    def test_plain_waypoint_on_a_fence_tile_is_reached_from_its_standable_neighbour(self) -> None:
        """field_to_shed aims at (21, 31); Summer regrows that tile as fence (0x05)."""
        ram = make_navigation_ram(current_tile=(21, 30), blocked_tile=(21, 31), blocked_id=0x05)
        world = SimpleNamespace(ram=ram, info={}, obs=None, frame=0)
        fence_wp = Waypoint(tilemap=0x00, target_px=(344, 504), radius=12)
        task = MultiMapNavTask(waypoints=[fence_wp], timeout=800, initial_settle_frames=0)
        task.reset(world)
        task._navigator.update(ram)

        self.assertFalse(task._at_wp_target(fence_wp))  # the pixel test alone is 16px short
        self.assertTrue(task._at_wp_target(fence_wp, ram))

        exact = [
            Waypoint(tilemap=0x00, target_px=(344, 504), radius=12, is_exit=True, exit_direction="down"),
            Waypoint(tilemap=0x00, target_px=(344, 504), radius=12, action_on_arrive="press_a"),
        ]
        for wp in exact:
            self.assertFalse(task._at_wp_target(wp, ram))

        set_player_pos(ram, 21 * 16 + 8, 32 * 16 + 8)  # south of the fence: not there yet
        task._navigator.update(ram)
        self.assertFalse(task._at_wp_target(fence_wp, ram))

    def test_waypoint_on_walkable_ground_keeps_the_pixel_arrival_test(self) -> None:
        ram = make_navigation_ram(current_tile=(21, 30), blocked_tile=(63, 63), blocked_id=0xA1)
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(344, 504), radius=12)],
            timeout=800,
            initial_settle_frames=0,
        )
        task.reset(SimpleNamespace(ram=ram, info={}, obs=None, frame=0))
        task._navigator.update(ram)

        self.assertFalse(task._at_wp_target(task.waypoints[0], ram))

    def _path_map_ram(self):
        """Path 0x0C at the mountain-gate descent: 0xFF seals (7,7)/(8,7)."""
        ram = make_navigation_ram(current_tile=(8, 6), blocked_tile=(63, 63))
        ram[ADDR_TILEMAP] = 0x0C
        for ty in range(64):
            for tx in range(64):
                ram[ADDR_MAP + ty * 64 + tx] = 0xA0
        ram[ADDR_MAP + 7 * 64 + 7] = 0xFF
        ram[ADDR_MAP + 7 * 64 + 8] = 0xFF
        ram[ADDR_INPUT_LOCK] = 1
        set_player_pos(ram, 133, 101)
        return ram

    def _crossroads_nav(self, ram):
        world = SimpleNamespace(ram=ram, info={}, obs=None, frame=0)
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x0C, target_px=(132, 128), radius=10)],
            timeout=800,
            initial_settle_frames=0,
        )
        task.reset(world)
        return task, world

    def _pinned_nav(self, **state):
        """Localize first: reset leaves the navigator at (0,0), so the first
        step always reads as tile movement and zeroes stasis."""
        task, world = self._crossroads_nav(self._path_map_ram())
        task.step(world)
        task._navigator.path = [(8, 7)]
        task._navigator.stasis = 45  # past the close-range cutoff, as at the pin
        task._soft_solid_pin_frames = 10_000
        for key, value in state.items():
            setattr(task, key, value)
        return task, world

    def test_close_range_walk_routes_around_a_concave_seal(self) -> None:
        """The run13 path pin: (8,7) sealed, target two tiles straight down.

        The close-range walk used to answer a blocked primary with a
        sideways secondary that was already on target — stepping off and
        back across a tile boundary, which resets ``Navigator.stasis`` and
        so hides the pin from every stall guard until the leg fails
        (``soft_solid pin held=0x03 pos=(133,101) stasis=40``, 6x).
        """
        task, world = self._crossroads_nav(self._path_map_ram())

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertTrue(task._navigator.path, "expected a BFS detour, not a bounce")
        self.assertNotIn((8, 7), task._navigator.path)
        self.assertIn(task._navigator.path[0], {(7, 6), (9, 6)})

    def test_close_range_walk_hands_a_stalled_waypoint_to_bfs(self) -> None:
        task, world = self._crossroads_nav(self._path_map_ram())
        task._close_latch.best = 0
        task._close_latch.stall = 29

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._close_bail_wp, 0)
        self.assertIn("close-range stalled", result.reason or "")

    def test_soft_solid_pin_recovers_before_failing_the_leg(self) -> None:
        task, world = self._pinned_nav()

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("pin recovery 1", result.reason or "")
        self.assertIn((8, 7), task._pathfinder.temp_blocked)
        self.assertEqual(task._soft_solid_pin_frames, 0)

    def test_soft_solid_pin_still_fails_closed_after_the_recovery_budget(self) -> None:
        task, world = self._pinned_nav(_pin_recoveries=4)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("soft_solid pin", result.reason or "")
        self.assertIn("recoveries=4", result.reason or "")

    def test_nav_yields_to_a_sprite_standing_in_the_next_tile(self) -> None:
        """Mountain/path NPCs wander into proven corridors; wait, don't shove."""
        task, world = self._pinned_nav(_entity_tiles={(8, 7)})

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("yield to sprite", result.reason or "")
        self.assertEqual(int(result.action.action.sum()), 0)
        self.assertEqual(task._soft_solid_pin_frames, 0)

    def test_nav_stops_yielding_to_a_parked_sprite(self) -> None:
        task, world = self._pinned_nav(_entity_tiles={(8, 7)}, _yield_frames=90)

        result = task.step(world)

        self.assertNotIn("yield to sprite", result.reason or "")

    def test_a_parked_town_sprite_gets_a_padded_ring(self) -> None:
        """Y2 Spring D16 (compose7): a villager parked in front of the animal shop sealed the walk home.

        Ring-padding only existed on the mountain map, so on the town map BFS routed tight against the villager
        and its hit box (+-12 px) spilled onto the neighbouring tile: four pixel-stuck replans failed the leg and the
        rescue chain ended inside the shop. Once a sprite is judged parked (the yield timed out), pad its ring.
        """
        from unittest import mock

        sprite = "harvest.planner.tasks.nav_corridor.entity_tiles"
        task, world = self._pinned_nav(_entity_tiles={(8, 7)})
        world.ram[ADDR_TILEMAP] = 0x04
        with mock.patch(sprite, return_value={(8, 7)}):
            task._sync_entity_blocks(world.ram)
        self.assertFalse(getattr(task, "_pad_parked", False))
        self.assertEqual(task._pathfinder.no_go_tiles & {(8, 7), (9, 7), (7, 7)}, {(8, 7)},
                         "a passing sprite is only waited for, never padded")

        task, world = self._pinned_nav(_entity_tiles={(8, 7)}, _yield_frames=90)
        world.ram[ADDR_TILEMAP] = 0x04
        task.waypoints = [wp._replace(tilemap=0x04) for wp in task.waypoints]
        task.step(world)
        self.assertTrue(task._pad_parked)
        with mock.patch(sprite, return_value={(8, 7)}):
            task._sync_entity_blocks(world.ram)
        ring = {(8 + dx, 7 + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)} - {task._navigator.current_tile}
        self.assertTrue(ring <= task._pathfinder.no_go_tiles, ring - task._pathfinder.no_go_tiles)
        task._advance_waypoint()
        self.assertFalse(task._pad_parked)

    def test_parked_farm_dog_does_not_seal_the_approach_target(self):
        from unittest import mock
        ram = self._path_map_ram()
        ram[ADDR_TILEMAP] = 0x00
        task, world = self._crossroads_nav(ram)
        task._navigator.update(ram)
        task._pad_parked = True
        # As on D1: the target is beside the dog's tile. A padded ring
        # makes that ground unreachable even though a path skirts the dog.
        with mock.patch("harvest.planner.tasks.nav_corridor.entity_tiles", return_value={(8, 7)}):
            task._sync_entity_blocks(ram)
        path = task._pathfinder.find_path(ram, (8, 6), (8, 8))
        self.assertTrue(path)
        self.assertNotIn((8, 7), path)

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
        target.ram[0x0025] = 15  # farm faded in
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

        task = build_phase_task(TaskBuildContext(), spec, make_transition_world(0x00))

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

        task = build_phase_task(TaskBuildContext(), spec, make_transition_world(0x00))

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

    def test_walk_into_door_aligns_and_pushes_without_b(self) -> None:
        world = make_transition_world(0x00, current_tile=(26, 30))
        set_player_pos(world.ram, 433, 496)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x26,
            stand_tile=(26, 30),
            door_align_px=424,
            walk_into_door=True,
        )
        task.reset(world)
        task._stand_reached = True
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(int(result.action.action[0]), 0)
        self.assertEqual(int(result.action.action[6]), 1)

        set_player_pos(world.ram, 424, 489)
        task._step_count = 0
        push = task.step(world)
        self.assertEqual(int(push.action.action[0]), 0)
        self.assertEqual(int(push.action.action[4]), 1)

        task._step_count = 8
        idle = task.step(world)
        self.assertEqual(int(idle.action.action[0]), 0)
        self.assertEqual(int(idle.action.action[4]), 0)

    def test_shed_enter_transition_walks_into_door(self) -> None:
        from harvest.planner.tasks.inventory_shed import shed_enter_transition

        task = shed_enter_transition(name="enter_shed_test", timeout=1500)
        self.assertTrue(task.walk_into_door)
        self.assertEqual(task.direction, "up")
        self.assertEqual(task.door_align_px, 424)

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

    def test_multi_nav_relocalization_restarts_destination_settle(self) -> None:
        origin = make_transition_world(0x0C, current_tile=(15, 8))
        landed = make_transition_world(0x00, current_tile=(15, 6))
        task = MultiMapNavTask(
            waypoints=[
                Waypoint(tilemap=0x0C, target_px=(244, 128)),
                Waypoint(tilemap=0x00, target_px=(80, 424)),
            ],
            initial_settle_frames=5,
        )
        task.reset(origin)
        task._initial_settle = task.initial_settle_frames

        result = task.step(landed)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "relocalized after map transition")
        self.assertEqual(task._wp_index, 1)
        self.assertEqual(task._initial_settle, 0)

    def test_multi_nav_exit_can_push_into_destination_before_settle(self) -> None:
        origin = make_transition_world(0x0C, current_tile=(15, 8))
        landed = make_transition_world(0x00, current_tile=(15, 6))
        task = MultiMapNavTask(
            waypoints=[
                Waypoint(
                    tilemap=0x0C,
                    target_px=(244, 128),
                    is_exit=True,
                    exit_direction="right",
                    exit_push_frames=2,
                )
            ],
            initial_settle_frames=0,
        )
        task.reset(origin)
        task._phase = "exit_settle"

        first = task.step(landed)
        second = task.step(landed)
        third = task.step(landed)

        self.assertEqual(int(first.action.action[7]), 1)
        self.assertEqual(int(second.action.action[7]), 1)
        self.assertEqual(int(third.action.action.sum()), 0)

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

    def test_multi_nav_settles_unregistered_tilemap_instead_of_failing(self) -> None:
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x0C, target_px=(132, 128))],
            initial_settle_frames=0,
        )
        world = make_transition_world(0x57, current_tile=(0, 26))
        set_player_pos(world.ram, 10, 422)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("tilemap settle", result.reason or "")
        world.ram[ADDR_TILEMAP] = 0x0C
        set_player_pos(world.ram, 232, 128)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn("expected tilemap", result.reason or "")

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

    def test_farm_exit_uses_south_route_from_shipping_bin(self) -> None:
        """Live grape return is tile (8,28), north of the y=31 fence.

        farm_south_to_west_gate first-hops berry (55,60) and 10k-hugs.
        Join the y=26 pinch, then west to the gate.
        """
        from harvest.planner.tasks.inventory_exit import FarmExitTask

        world = make_transition_world(0x00, current_tile=(8, 28))
        task = FarmExitTask(timeout=10_000)
        task.reset(world)
        hops = [wp.target_px for wp in task._nav.waypoints]
        self.assertNotEqual(task._nav.waypoints, ROUTES["farm_south_to_west_gate"])
        self.assertNotIn((888, 968), hops)
        self.assertIn((136, 424), hops)
        self.assertIn((72, 424), hops)
        self.assertEqual(hops[-1], (40, 424))
        self.assertTrue(task._nav.waypoints[-1].is_exit)
        self.assertTrue(all(wp.tilemap == 0x00 for wp in task._nav.waypoints))

    def test_farm_exit_from_bin_west_joins_pinch_row_before_sidestep(self) -> None:
        """Power-on grape-return (7,27)/(115,432): do not sidestep on y=27.

        run_left to (72,424) with dy==radius holds LEFT into (1,27)=FF.
        Up onto y=26 at current x, then east to the pinch column.
        """
        from harvest.planner.tasks.inventory_exit import FarmExitTask

        world = make_transition_world(0x00, current_tile=(7, 27))
        set_player_pos(world.ram, 115, 432)
        task = FarmExitTask(timeout=10_000)
        task.reset(world)
        hops = [wp.target_px for wp in task._nav.waypoints]
        row_join = next(wp for wp in task._nav.waypoints if wp.target_px == (115, 424))
        self.assertIn((115, 424), hops)
        self.assertNotIn((136, 432), hops)
        self.assertIn((136, 424), hops)
        self.assertIn((72, 424), hops)
        self.assertEqual(hops[-1], (40, 424))
        self.assertLess(hops.index((115, 424)), hops.index((136, 424)))
        self.assertLessEqual(row_join.radius, 6)
        self.assertGreater(abs(432 - 424), row_join.radius)

    def test_multi_nav_run_left_aligns_when_dy_equals_radius(self) -> None:
        """(7,27) y=432 vs pinch-west y=424 r=8: 8>=8 must UP, not LEFT."""
        task = MultiMapNavTask(
            waypoints=[
                Waypoint(
                    tilemap=0x00,
                    target_px=(72, 424),
                    radius=8,
                    run_direction="left",
                )
            ],
            initial_settle_frames=0,
        )
        world = make_transition_world(0x00, current_tile=(7, 27))
        set_player_pos(world.ram, 115, 432)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[4]), 1)  # up
        self.assertEqual(int(result.action.action[6]), 0)  # not left

    def test_farm_exit_waits_have_lunch(self) -> None:
        from harvest.planner.tasks.inventory_exit import FarmExitTask
        from harvest.core.ram_catalog import field_spec

        world = make_transition_world(0x00, current_tile=(7, 27))
        world.ram[field_spec("hour").address] = 12
        task = FarmExitTask(timeout=10_000)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
    def test_animal_shop_staging_route_has_safe_corridors(self) -> None:
        """rr-4zvl: farm_to_animal_shop_staging must route through gate pinch, crossroads, and town y=890 corridor."""
        route = ROUTES["farm_to_animal_shop_staging"]
        tilemaps = [wp.tilemap for wp in route]
        self.assertIn(0x00, tilemaps)
        self.assertIn(0x0C, tilemaps)
        self.assertIn(0x04, tilemaps)
        self.assertEqual(route[-1].tilemap, 0x24)

        farm_hops = [wp.target_px for wp in route if wp.tilemap == 0x00]
        self.assertIn((136, 424), farm_hops)
        self.assertIn((72, 424), farm_hops)
        self.assertIn((40, 424), farm_hops)

        path_hops = [wp.target_px for wp in route if wp.tilemap == 0x0C]
        self.assertIn((132, 128), path_hops)
        self.assertIn((10, 128), path_hops)

        town_hops = [wp.target_px for wp in route if wp.tilemap == 0x04]
        self.assertIn((696, 890), town_hops)
        self.assertIn((601, 890), town_hops)


if __name__ == "__main__":
    unittest.main()
