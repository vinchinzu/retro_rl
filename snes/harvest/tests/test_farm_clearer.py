"""Unit tests for farm debris clearing — no ROM needed."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.core.carry import ADDR_TOOL_BACKPACK
from harvest.core.animal_status import ADDR_HELD_ITEM
from harvest.core.tile_catalog import (
    ADDR_MAP,
    ADDR_STAMINA,
    LARGE_ROCK_TL,
    MAP_WIDTH,
    ROCK,
    STONE,
    STUMP_TL,
    TILE_SIZE,
    TILE_TO_DEBRIS,
    WEED,
    DebrisType,
)
from harvest.tasks.farm_toss import HELD_STONE
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseKind
from harvest.planner.day_plan_phases import PHASE_SEQUENCES, build_day_phases
from harvest.tasks.farm_clear_task import FarmClearTask, choose_clear_target
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_TILEMAP,
    ADDR_TOOL,
    ADDR_X,
    ADDR_Y,
    Tool,
)
from harvest.tasks.nav import (
    Point,
    VIEWPORT_HOP_TILES,
)
from harvest.tasks.farm_clearer import (
    DEFAULT_PRIORITY,
    FarmClearer,
    Target,
    TileScanner,
    use_tool,
)
from harvest.tasks.farm_ops import sort_targets_cluster

from retro_harness import TaskStatus, WorldState


def _set_player(ram: np.ndarray, tile: tuple[int, int]) -> None:
    px = tile[0] * TILE_SIZE + 8
    py = tile[1] * TILE_SIZE + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF


def _set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


def _place_stump(ram: np.ndarray, tx: int, ty: int) -> None:
    _set_tile(ram, tx, ty, 0x09)
    _set_tile(ram, tx + 1, ty, 0x0A)
    _set_tile(ram, tx, ty + 1, 0x0B)
    _set_tile(ram, tx + 1, ty + 1, 0x0C)


def _place_large_rock(ram: np.ndarray, tx: int, ty: int) -> None:
    _set_tile(ram, tx, ty, 0x0D)
    _set_tile(ram, tx + 1, ty, 0x0E)
    _set_tile(ram, tx, ty + 1, 0x0F)
    _set_tile(ram, tx + 1, ty + 1, 0x10)


def _make_farm_ram(
    *,
    player_tile: tuple[int, int] = (10, 10),
    stamina: int = 100,
    tool: int = int(Tool.HAMMER),
) -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = 0x00
    ram[ADDR_INPUT_LOCK] = 1
    ram[ADDR_STAMINA] = stamina
    ram[ADDR_TOOL] = tool
    for i in range(MAP_WIDTH * MAP_WIDTH):
        ram[ADDR_MAP + i] = 0xA1
    _set_player(ram, player_tile)
    return ram


class TestDebrisClassification(unittest.TestCase):
    def test_stump_and_rock_families_map_correctly(self) -> None:
        self.assertEqual(TILE_TO_DEBRIS[STUMP_TL], DebrisType.STUMP)
        self.assertEqual(TILE_TO_DEBRIS[0x0A], DebrisType.STUMP)
        self.assertEqual(TILE_TO_DEBRIS[LARGE_ROCK_TL], DebrisType.ROCK)
        self.assertEqual(TILE_TO_DEBRIS[0x0E], DebrisType.ROCK)
        self.assertEqual(TILE_TO_DEBRIS[WEED], DebrisType.WEED)
        self.assertEqual(TILE_TO_DEBRIS[STONE], DebrisType.STONE)

    def test_default_priority_opens_pathing_first(self) -> None:
        self.assertEqual(
            DEFAULT_PRIORITY,
            [
                DebrisType.ROCK,
                DebrisType.STUMP,
                DebrisType.STONE,
                DebrisType.WEED,
            ],
        )

    def test_required_hits_small_vs_large_rock(self) -> None:
        small = Target((5, 5), Point(88, 88), DebrisType.ROCK, ROCK)
        large = Target((5, 5), Point(88, 88), DebrisType.ROCK, LARGE_ROCK_TL)
        stump = Target((5, 5), Point(88, 88), DebrisType.STUMP, STUMP_TL)
        damaged = Target((5, 5), Point(88, 88), DebrisType.ROCK, 0x11)
        self.assertEqual(small.required_hits, 1)
        self.assertEqual(large.required_hits, 6)
        self.assertEqual(stump.required_hits, 6)
        self.assertEqual(damaged.required_hits, 6)


class TestTileScanner(unittest.TestCase):
    def test_collapses_2x2_stump_and_rock_to_top_left(self) -> None:
        ram = _make_farm_ram()
        _place_stump(ram, 20, 20)
        _place_large_rock(ram, 30, 30)
        _set_tile(ram, 12, 12, WEED)
        _set_tile(ram, 13, 12, STONE)

        targets = TileScanner().scan(ram)
        by_type = {t.debris_type: t for t in targets}
        self.assertEqual(len(targets), 4)
        self.assertEqual(by_type[DebrisType.STUMP].tile, (20, 20))
        self.assertEqual(by_type[DebrisType.ROCK].tile, (30, 30))
        self.assertEqual(by_type[DebrisType.WEED].tile, (12, 12))
        self.assertEqual(by_type[DebrisType.STONE].tile, (13, 12))

    def test_has_clearable_debris(self) -> None:
        ram = _make_farm_ram()
        self.assertFalse(TileScanner().has_clearable_debris(ram))
        _set_tile(ram, 8, 8, WEED)
        self.assertTrue(TileScanner().has_clearable_debris(ram))

    def test_scan_accepts_bytes_ram(self) -> None:
        """Save-state loaders return bytes; scanner must not crash on them."""
        ram = _make_farm_ram()
        _place_stump(ram, 16, 16)
        as_bytes = ram.tobytes()
        targets = TileScanner().scan(as_bytes)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].debris_type, DebrisType.STUMP)

    def test_scan_respects_pocket_bounds(self) -> None:
        ram = _make_farm_ram()
        _set_tile(ram, 16, 20, WEED)
        _set_tile(ram, 16, 40, STONE)
        pocket = (3, 14, 28, 30)
        tiles = {t.tile for t in TileScanner().scan(ram, pocket)}
        self.assertIn((16, 20), tiles)
        self.assertNotIn((16, 40), tiles)

    def test_collapses_damage_tiles_to_one_target(self) -> None:
        ram = _make_farm_ram()
        _set_tile(ram, 20, 20, 0x11)
        _set_tile(ram, 21, 20, 0x12)
        _set_tile(ram, 20, 21, 0x13)
        _set_tile(ram, 21, 21, 0x14)
        targets = TileScanner().scan(ram)
        rocks = [t for t in targets if t.debris_type == DebrisType.ROCK]
        self.assertEqual(len(rocks), 1)
        self.assertEqual(rocks[0].tile, (20, 20))
        self.assertEqual(rocks[0].required_hits, 6)

    def test_use_tool_frames_have_no_dpad(self) -> None:
        for action in use_tool(frames=20, cooldown=10):
            if int(action[1]) == 1:
                self.assertFalse(any(int(action[i]) for i in range(4, 8)))


class TestPocketClearTask(unittest.TestCase):
    def test_locked_bounds_do_not_expand_to_south_farm(self) -> None:
        ram = _make_farm_ram(player_tile=(16, 20))
        _set_tile(ram, 16, 20, WEED)
        _set_tile(ram, 16, 40, STONE)
        clearer = FarmClearer()
        clearer.configure(farm_bounds=(3, 14, 28, 30))
        clearer.navigator.update(ram)
        self.assertEqual(clearer._locked_bounds, (3, 14, 28, 30))
        nxt = clearer._handle_scanning(ram)
        self.assertEqual(clearer.farm_bounds, (3, 14, 28, 30))
        self.assertEqual(clearer._locked_bounds, (3, 14, 28, 30))
        self.assertIn(nxt, ("navigating", "clearing", "scanning"))

    def test_factory_passes_pocket_bounds(self) -> None:
        from harvest.planner.day_plan_phases import pocket_clear_phase
        from harvest.planner.day_task_factory import DayTaskFactory

        world = WorldState(frame=0, ram=_make_farm_ram(), info={}, obs=None)
        task = DayTaskFactory().make_task(pocket_clear_phase(), world)
        self.assertIsInstance(task, FarmClearTask)
        self.assertEqual(task.farm_bounds, (3, 14, 28, 30))
        self.assertFalse(task.fetch_tools)
        self.assertTrue(task.prefer_lift_for_stones)
        self.assertEqual(task.clearer.priority, [DebrisType.WEED, DebrisType.STONE])

    def test_pocket_clear_does_not_succeed_off_farm(self) -> None:
        ram = _make_farm_ram(player_tile=(2, 26), tool=0)
        ram[ADDR_TILEMAP] = 0x0C
        _set_tile(ram, 13, 28, WEED)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            farm_bounds=(3, 14, 28, 30),
            prefer_lift_for_weeds=True,
            prefer_lift_for_stones=True,
            timeout=7000,
        )
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("approach", (result.reason or "").lower())
        self.assertFalse(task._pocket_arrived)

    def test_pocket_clear_does_not_succeed_from_west_gate(self) -> None:
        ram = _make_farm_ram(player_tile=(2, 26), tool=0)
        _set_tile(ram, 13, 28, WEED)
        _set_tile(ram, 14, 27, WEED)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            farm_bounds=(3, 14, 28, 30),
            timeout=7000,
        )
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertFalse(task._pocket_arrived)
        self.assertNotIn("field_clear", result.reason or "")

    def test_pocket_clear_can_start_when_gate_scan_is_empty(self) -> None:
        ram = _make_farm_ram(player_tile=(2, 26), tool=0)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(fetch_tools=False, farm_bounds=(3, 14, 28, 30))
        self.assertTrue(task.can_start(world))

    def test_pocket_clear_does_not_ready_at_west_fence(self) -> None:
        ram = _make_farm_ram(player_tile=(3, 28), tool=0)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            farm_bounds=(3, 14, 28, 30),
            timeout=7000,
        )
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertFalse(task._pocket_arrived)
        self.assertIn("approach", (result.reason or "").lower())

    def test_south_exit_staging_navs_instead_of_leftright_thrash(self) -> None:
        ram = _make_farm_ram(player_tile=(4, 33), tool=0)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(fetch_tools=False, farm_bounds=(3, 14, 28, 30))
        task.reset(world)
        task._pocket_arrived = True
        result = task._maybe_stage_then_success(world, "partial_clear")
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task._exit_nav)
        self.assertEqual(len(task._staging_queue), 0)
        lefts = sum(1 for act in task._staging_queue if int(act[6]) == 1)
        rights = sum(1 for act in task._staging_queue if int(act[7]) == 1)
        self.assertEqual(lefts, 0)
        self.assertEqual(rights, 0)

    def test_pocket_clear_succeeds_when_already_in_clean_pocket(self) -> None:
        ram = _make_farm_ram(player_tile=(12, 28), tool=0)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            farm_bounds=(3, 14, 28, 30),
            timeout=7000,
        )
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("field_clear", result.reason or "")
        self.assertTrue(task._pocket_arrived)

    def test_pocket_clear_hands_off_with_unrelated_debris_remaining(self) -> None:
        ram = _make_farm_ram(player_tile=(12, 28), tool=0)
        _set_tile(ram, 20, 15, WEED)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            farm_bounds=(3, 14, 28, 30),
            timeout=7000,
        )
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("plot_ring_clear", result.reason or "")
        self.assertEqual(ram[ADDR_MAP + 15 * MAP_WIDTH + 20], WEED)

    def test_plot_scan_bounds_cover_hoe_stands(self) -> None:
        task = FarmClearTask(farm_bounds=(3, 14, 28, 30))
        self.assertEqual(task._plot_scan_bounds(), (11, 26, 15, 30))

    def test_pocket_complete_fails_while_ring_is_dirty(self) -> None:
        ram = _make_farm_ram(player_tile=(12, 28), tool=0)
        _set_tile(ram, 14, 27, WEED)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            farm_bounds=(3, 14, 28, 30),
            timeout=7000,
        )
        task.reset(world)
        task._pocket_arrived = True
        task._lock_clearer_to_plot()
        self.assertEqual(
            task._complete_status(world, remaining=["weed"]),
            TaskStatus.FAILURE,
        )

    def test_pocket_clear_does_not_hand_off_with_ring_weed(self) -> None:
        ram = _make_farm_ram(player_tile=(12, 28), tool=0)
        _set_tile(ram, 14, 27, WEED)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            farm_bounds=(3, 14, 28, 30),
            timeout=7000,
        )
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(ram[ADDR_MAP + 27 * MAP_WIDTH + 14], WEED)


class TestFarmClearerSelection(unittest.TestCase):
    def test_lower_2x2_footprint_stand_enters_clearing(self) -> None:
        """A stand beside BL is valid even when two steps from the TL anchor."""
        ram = _make_farm_ram(player_tile=(9, 11))
        _place_large_rock(ram, 10, 10)
        clearer = FarmClearer(priority=[DebrisType.ROCK])
        clearer.navigator.update(ram)
        clearer.current_target = Target(
            (10, 10),
            Point(10 * 16 + 8, 10 * 16 + 8),
            DebrisType.ROCK,
            LARGE_ROCK_TL,
        )
        clearer.approach_tile = (9, 11)
        clearer.frame_count = 1

        self.assertIsNone(clearer._handle_clearing(ram))

    def test_cluster_sort_prefers_nearby_then_row_order(self) -> None:
        clearer = FarmClearer()
        targets = [
            Target((20, 10), Point(20 * 16 + 8, 10 * 16 + 8), DebrisType.WEED, WEED),
            Target((5, 10), Point(5 * 16 + 8, 10 * 16 + 8), DebrisType.WEED, WEED),
            Target((6, 10), Point(6 * 16 + 8, 10 * 16 + 8), DebrisType.WEED, WEED),
        ]
        ordered = sort_targets_cluster(targets, Point(5 * 16 + 8, 10 * 16 + 8))
        self.assertEqual(ordered[0].tile, (5, 10))
        self.assertEqual(ordered[1].tile, (6, 10))

    def test_navigation_watchdog_rejects_cross_tile_oscillation_approach(self) -> None:
        """Pixel oscillation across a tile edge must not reset target progress."""
        ram = _make_farm_ram(player_tile=(49, 55))
        _set_tile(ram, 51, 56, WEED)
        clearer = FarmClearer(priority=[DebrisType.WEED])
        clearer.current_target = Target(
            (51, 56), Point(51 * 16 + 8, 56 * 16 + 8), DebrisType.WEED, WEED
        )
        clearer.approach_tile = (50, 56)
        clearer.state = "navigating"
        clearer.max_nav_no_progress = 6

        verdict = None
        for frame, pos in enumerate(((785, 894), (790, 902)) * 5, start=1):
            ram[ADDR_X] = pos[0] & 0xFF
            ram[ADDR_X + 1] = pos[0] >> 8
            ram[ADDR_Y] = pos[1] & 0xFF
            ram[ADDR_Y + 1] = pos[1] >> 8
            clearer.frame_count = frame
            clearer.navigator.update(ram)
            verdict = clearer._handle_navigating(ram)
            if verdict == "scanning":
                break

        self.assertEqual(verdict, "scanning")
        self.assertIn(((51, 56), (50, 56)), clearer.failed_approaches)
        self.assertNotIn((51, 56), clearer.failed_tiles)

    def test_pathable_stand_beats_nearer_blocked_neighbor(self) -> None:
        """An isolated closer stand must lose to a farther reachable one."""
        ram = _make_farm_ram(player_tile=(8, 10))
        _set_tile(ram, 4, 10, WEED)
        _set_tile(ram, 5, 10, 0xA1)  # nearer stand, boxed
        _set_tile(ram, 6, 10, STONE)
        _set_tile(ram, 5, 9, STONE)
        _set_tile(ram, 5, 11, STONE)
        clearer = FarmClearer(priority=[DebrisType.WEED])
        clearer.navigator.update(ram)
        weed = Target((4, 10), Point(4 * 16 + 8, 10 * 16 + 8), DebrisType.WEED, WEED)

        chosen = choose_clear_target(clearer, ram, [weed])
        self.assertIsNotNone(chosen)
        target, stand, path = chosen
        self.assertEqual(target.tile, (4, 10))
        self.assertNotEqual(stand, (5, 10))
        self.assertEqual(path[-1], stand)

    def test_boxed_weed_opens_adjacent_stone(self) -> None:
        ram = _make_farm_ram(player_tile=(16, 15))
        _set_tile(ram, 18, 15, WEED)
        _set_tile(ram, 19, 15, WEED)
        _set_tile(ram, 17, 15, STONE)
        _set_tile(ram, 18, 14, 0x05)
        _set_tile(ram, 18, 16, 0xA6)
        clearer = FarmClearer(priority=[DebrisType.WEED])
        clearer.navigator.update(ram)
        weed = Target((18, 15), Point(18 * 16 + 8, 15 * 16 + 8), DebrisType.WEED, WEED)

        chosen = choose_clear_target(clearer, ram, [weed])
        self.assertIsNotNone(chosen)
        target, stand, _path = chosen
        self.assertEqual(target.debris_type, DebrisType.STONE)
        self.assertEqual(target.tile, (17, 15))
        self.assertEqual(stand, (16, 15))

    def test_replan_miss_rejects_approach_not_whole_tile(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10))
        _set_tile(ram, 20, 20, WEED)
        clearer = FarmClearer(priority=[DebrisType.WEED])
        clearer.navigator.update(ram)
        clearer.current_target = Target(
            (20, 20), Point(20 * 16 + 8, 20 * 16 + 8), DebrisType.WEED, WEED
        )
        clearer.approach_tile = (21, 20)
        clearer.pathfinder.temp_blocked.update(
            {(11, 10), (10, 11), (9, 10), (10, 9)}
        )

        verdict = clearer._replan_nav_hop(ram)
        self.assertEqual(verdict, "scanning")
        self.assertIn(((20, 20), (21, 20)), clearer.failed_approaches)
        self.assertNotIn((20, 20), clearer.failed_tiles)

    def test_lift_verify_does_not_claim_while_stone_is_held(self) -> None:
        ram = _make_farm_ram(player_tile=(11, 29), tool=0)
        _set_tile(ram, 11, 28, 0xA1)
        ram[ADDR_HELD_ITEM] = HELD_STONE
        clearer = FarmClearer()
        clearer.prefer_lift_for_stones = True
        clearer.navigator.update(ram)
        clearer.current_target = Target(
            (11, 28), Point(11 * 16 + 8, 28 * 16 + 8), DebrisType.STONE, STONE
        )
        clearer._pending_lift_verify = (11, 28)
        clearer.clearing_start_frame = 1
        nxt = clearer._handle_clearing(ram)
        self.assertEqual(nxt, "scanning")
        self.assertNotIn((11, 28), clearer.tiles_cleared)
        self.assertNotIn((11, 28), clearer.failed_tiles)
        self.assertEqual(clearer.cleared_count, 0)
        self.assertEqual(clearer._pending_toss_origin, (11, 28))
        self.assertIsNotNone(clearer._toss_skill)

    def test_viewport_hop_limits_path_length(self) -> None:
        ram = _make_farm_ram(player_tile=(5, 5))
        clearer = FarmClearer()
        clearer.navigator.update(ram)
        path = clearer.pathfinder.find_path(
            ram,
            (5, 5),
            (5, 25),
            max_steps=VIEWPORT_HOP_TILES,
        )
        self.assertIsNotNone(path)
        assert path is not None
        self.assertLessEqual(len(path), VIEWPORT_HOP_TILES)
        self.assertEqual(path[-1], (5, 5 + VIEWPORT_HOP_TILES))

    def test_find_path_hops_when_goal_is_outside_loaded_walkable(self) -> None:
        """Stale far tiles must not prevent a viewport hop toward debris."""
        ram = _make_farm_ram(player_tile=(10, 10))
        # Make distant corridor unwalkable (simulates SNES stale 0x72).
        for y in range(18, 30):
            for x in range(0, MAP_WIDTH):
                _set_tile(ram, x, y, 0x72)
        # Leave a walkable pocket near the player and a rock far away.
        for y in range(8, 16):
            for x in range(8, 16):
                _set_tile(ram, x, y, 0xA1)
        _place_large_rock(ram, 10, 24)

        clearer = FarmClearer()
        path = clearer.pathfinder.find_path(
            ram,
            (10, 10),
            (10, 23),
            max_steps=VIEWPORT_HOP_TILES,
        )
        self.assertIsNotNone(path)
        assert path is not None
        self.assertLessEqual(len(path), VIEWPORT_HOP_TILES)
        self.assertGreater(path[-1][1], 10)

    def test_scan_miss_streak_completes_instead_of_hanging(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10))
        _place_large_rock(ram, 40, 40)
        # Surround the rock so no approach tile is walkable.
        for dx, dy in ((-1, 0), (2, 0), (0, -1), (0, 2), (1, -1), (1, 2), (-1, 1), (2, 1)):
            _set_tile(ram, 40 + dx, 40 + dy, 0x72)

        clearer = FarmClearer(priority=[DebrisType.ROCK])
        clearer.startup_done = True
        clearer.max_scan_misses = 3
        clearer.navigator.update(ram)

        for _ in range(3):
            nxt = clearer._handle_scanning(ram)
        self.assertEqual(nxt, "complete")

    def test_missing_startup_tools_continue_lift_only(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        _set_tile(ram, 11, 10, WEED)
        clearer = FarmClearer(priority=[DebrisType.ROCK, DebrisType.WEED])
        clearer.startup_done = True
        clearer._enable_lift_only_mode([int(Tool.HAMMER), int(Tool.AXE)])
        clearer.navigator.update(ram)

        self.assertEqual(clearer.priority, [DebrisType.WEED])
        self.assertTrue(clearer.prefer_lift_for_stones)
        nxt = clearer._handle_scanning(ram)
        self.assertIn(nxt, {"navigating", "clearing"})
        assert clearer.current_target is not None
        self.assertEqual(clearer.current_target.debris_type, DebrisType.WEED)

    def test_missing_startup_tools_do_not_fail_farm_clear_task(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(fetch_tools=False)
        task.reset(world)
        task.clearer.tools_missing = True
        task.clearer.startup_done = True
        task.clearer.state = "complete"

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("lift_only", result.reason or "")

    def test_adjacent_opportunity_picks_priority_debris(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10))
        _set_tile(ram, 11, 10, WEED)
        _place_large_rock(ram, 10, 11)
        clearer = FarmClearer()
        clearer.navigator.update(ram)
        nxt = clearer._try_adjacent_opportunity(ram, (10, 10))
        self.assertEqual(nxt, "clearing")
        assert clearer.current_target is not None
        self.assertEqual(clearer.current_target.debris_type, DebrisType.ROCK)

    def test_failed_stone_lift_stops_adjacent_thrash(self) -> None:
        """Unclearable adjacent stone must not lift forever (Spring D3 thrash)."""
        ram = _make_farm_ram(player_tile=(10, 10), tool=int(Tool.WATERING_CAN))
        _set_tile(ram, 11, 10, STONE)
        clearer = FarmClearer(priority=[DebrisType.STONE, DebrisType.WEED])
        clearer.startup_done = True
        clearer.prefer_lift_for_stones = True
        clearer.tools_missing = True
        clearer.state = "scanning"
        clearer.max_scan_misses = 5

        for _ in range(250):
            _set_player(ram, (10, 10))
            ram[ADDR_INPUT_LOCK] = 1
            _set_tile(ram, 11, 10, STONE)  # never actually clears
            clearer.tick(ram)
            if clearer.state == "complete":
                break

        self.assertIn((11, 10), clearer.failed_tiles)
        self.assertEqual(clearer.state, "complete")

    def test_hit_count_does_not_mark_cleared_until_tile_gone(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=int(Tool.AXE))
        _place_stump(ram, 11, 10)
        clearer = FarmClearer(priority=[DebrisType.STUMP])
        clearer.navigator.update(ram)
        clearer.tool_manager.update(ram)
        clearer.current_target = Target(
            (11, 10),
            Point(11 * 16 + 8, 10 * 16 + 8),
            DebrisType.STUMP,
            STUMP_TL,
        )
        clearer.approach_tile = (10, 10)
        clearer.state = "clearing"
        clearer.target_hits = 6
        clearer.clearing_start_frame = clearer.frame_count or 1
        clearer.navigator.stasis = 10

        before = clearer.cleared_count
        nxt = clearer._handle_clearing(ram)
        self.assertIsNone(nxt)
        self.assertEqual(clearer.cleared_count, before)

        # Remove stump and verify count advances.
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            _set_tile(ram, 11 + dx, 10 + dy, 0xA1)
        clearer.target_hits = 6
        nxt = clearer._handle_clearing(ram)
        self.assertEqual(nxt, "scanning")
        self.assertEqual(clearer.cleared_count, before + 1)

    def test_low_stamina_still_lifts_weeds(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), stamina=2)
        _set_tile(ram, 11, 10, WEED)
        clearer = FarmClearer()
        clearer.startup_done = True
        clearer.navigator.update(ram)
        action = clearer.tick(ram)
        self.assertIsNotNone(action)
        self.assertFalse(clearer.stamina_exhausted)
        self.assertNotEqual(clearer.state, "complete")

    def test_hammer_in_backpack_still_targets_rock(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=int(Tool.WATERING_CAN))
        ram[ADDR_TOOL_BACKPACK] = int(Tool.HAMMER)
        _place_large_rock(ram, 12, 10)
        clearer = FarmClearer()
        clearer.startup_done = True
        clearer.tool_manager.update(ram)
        clearer._finalize_startup_tools()
        self.assertIn(DebrisType.ROCK, clearer.priority)
        self.assertNotIn(DebrisType.STUMP, clearer.priority)
        clearer.navigator.update(ram)
        nxt = clearer._handle_scanning(ram)
        self.assertIn(nxt, {"navigating", "clearing"})
        assert clearer.current_target is not None
        self.assertEqual(clearer.current_target.debris_type, DebrisType.ROCK)

    def test_missing_hammer_does_not_target_rock(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        _place_large_rock(ram, 12, 10)
        clearer = FarmClearer()
        clearer.startup_done = True
        clearer.tool_manager.update(ram)
        clearer._finalize_startup_tools()
        self.assertNotIn(DebrisType.ROCK, clearer.priority)
        clearer.navigator.update(ram)
        nxt = clearer._handle_scanning(ram)
        self.assertEqual(nxt, "complete")
        self.assertIsNone(clearer.current_target)

    def test_adjacent_snap_collapses_damage_family(self) -> None:
        ram = _make_farm_ram(player_tile=(12, 10))
        _set_tile(ram, 10, 10, 0x11)
        _set_tile(ram, 11, 10, 0x12)
        _set_tile(ram, 10, 11, 0x13)
        _set_tile(ram, 11, 11, 0x14)
        clearer = FarmClearer()
        clearer.navigator.update(ram)
        nxt = clearer._try_adjacent_opportunity(ram, (12, 10))
        self.assertEqual(nxt, "clearing")
        assert clearer.current_target is not None
        self.assertEqual(clearer.current_target.tile, (10, 10))
        self.assertEqual(clearer.current_target.debris_type, DebrisType.ROCK)


class TestFarmClearTask(unittest.TestCase):
    def test_success_when_field_clean(self) -> None:
        ram = _make_farm_ram()
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(fetch_tools=False)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("field_clear", result.reason or "")

    def test_progress_snapshot_includes_cleared(self) -> None:
        task = FarmClearTask(fetch_tools=False)
        snap = task.progress_snapshot()
        self.assertEqual(snap.task_name, "farm_clear")
        self.assertIn(("cleared", 0), snap.details)

    def test_unbounded_clear_off_farm_does_not_succeed(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        ram[ADDR_TILEMAP] = 0x04
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(fetch_tools=False)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertNotIn("field_clear", result.reason or "")

    def test_unbounded_leftover_debris_is_not_success(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        _place_large_rock(ram, 20, 20)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(fetch_tools=False)
        task.reset(world)
        task.clearer.startup_done = True
        task.clearer.state = "complete"
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("partial_clear", result.reason or "")

    def test_type_clear_does_not_succeed_on_plot_ring(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        _set_tile(ram, 40, 40, 0x03)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            priority=[DebrisType.WEED],
            handoff="type_clear",
            timeout=0,
        )
        task.reset(world)
        task._pocket_arrived = True
        task.farm_bounds = (3, 14, 28, 30)
        result = task.step(world)
        self.assertNotEqual(result.status, TaskStatus.SUCCESS)
        self.assertNotIn("plot_ring_clear", result.reason or "")

    def test_type_clear_succeeds_when_selected_debris_is_exhausted(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        _place_large_rock(ram, 20, 20)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            priority=[DebrisType.WEED],
            handoff="type_clear",
        )
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("field_clear", result.reason or "")

    def test_type_clear_fails_while_selected_debris_remains(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        _set_tile(ram, 20, 20, 0x03)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            priority=[DebrisType.WEED],
            handoff="type_clear",
            timeout=1,
        )
        task.reset(world)
        task.step(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("remaining=1", result.reason or "")

    def test_type_clear_retries_failed_stands_after_field_changes(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        _set_tile(ram, 20, 20, 0x03)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            priority=[DebrisType.WEED],
            handoff="type_clear",
        )
        task.reset(world)
        task.clearer.failed_tiles.add((20, 20))
        task.clearer.failed_approaches.add(((20, 20), (19, 20)))
        task.clearer.state = "complete"
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("retry type clear pass 1", result.reason or "")
        self.assertEqual(task.clearer.failed_tiles, set())
        self.assertEqual(task.clearer.failed_approaches, set())

    def test_type_clear_retries_when_timeout_is_unbounded(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        _set_tile(ram, 20, 20, 0x03)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            fetch_tools=False,
            priority=[DebrisType.WEED],
            handoff="type_clear",
            timeout=0,
        )
        task.reset(world)
        task.clearer.failed_tiles.add((20, 20))
        task.clearer.failed_approaches.add(((20, 20), (21, 20)))
        task.clearer.state = "complete"
        task._step_count = 80_000
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("retry type clear pass 1", result.reason or "")
        self.assertEqual(task.clearer.failed_approaches, set())

    def test_low_stamina_does_not_start_six_hit_or_succeed(self) -> None:
        ram = _make_farm_ram(
            player_tile=(10, 10), stamina=5, tool=int(Tool.HAMMER)
        )
        _place_large_rock(ram, 11, 10)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = FarmClearTask(fetch_tools=True)
        task.reset(world)
        task.clearer.startup_done = True
        task.clearer._tool_scan_done = True
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertNotIn("field_clear", result.reason or "")
        self.assertEqual(ram[ADDR_MAP + 10 * MAP_WIDTH + 11], LARGE_ROCK_TL)
        self.assertEqual(ram[ADDR_MAP + 10 * MAP_WIDTH + 12], 0x0E)


class TestDayPlanClearField(unittest.TestCase):
    def test_named_clear_sequence_exists(self) -> None:
        phases = PHASE_SEQUENCES["clear"]
        kinds = [p.kind for p in phases]
        self.assertIn(PhaseKind.CLEAR_FIELD, kinds)

    def test_build_day_phases_includes_clear_when_debris(self) -> None:
        phases = build_day_phases(
            has_debris=True,
            has_chickens=False,
            has_cows=False,
            has_harvest=False,
            has_waterable=False,
            has_seeds=False,
            policy=DayPlannerPolicy(
                include_chickens=False,
                include_cows=False,
                include_shop_run=False,
                include_berry_run=False,
                include_end_day=False,
            ),
        )
        kinds = [p.kind for p in phases]
        self.assertIn(PhaseKind.CLEAR_FIELD, kinds)
        self.assertEqual(kinds[0], PhaseKind.FARM_BUILDING_EXIT)
        self.assertEqual(kinds[1], PhaseKind.CLEAR_FIELD)

    def test_build_day_phases_skips_clear_when_clean(self) -> None:
        phases = build_day_phases(
            has_debris=False,
            has_chickens=False,
            has_cows=False,
            has_harvest=False,
            has_waterable=False,
            has_seeds=False,
            policy=DayPlannerPolicy(
                include_chickens=False,
                include_cows=False,
                include_shop_run=False,
                include_berry_run=False,
                include_end_day=False,
            ),
        )
        kinds = [p.kind for p in phases]
        self.assertNotIn(PhaseKind.CLEAR_FIELD, kinds)


if __name__ == "__main__":
    unittest.main()
