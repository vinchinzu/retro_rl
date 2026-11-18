"""D2 leftover clear quotas — RAM-count stop, not whole-farm wipe."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.core.tile_catalog import (
    ADDR_MAP,
    ADDR_TILEMAP,
    FENCE,
    LARGE_ROCK_TL,
    MAP_WIDTH,
    ROCK,
    STONE,
    STUMP_TL,
    TILE_SIZE,
    WEED,
    DebrisType,
    Tool,
)
from harvest.maps.map_config import WEST_PLANT_POCKET_BOUNDS
from harvest.tasks.farm_clear_quota import (
    ClearQuota,
    DebrisCounts,
    capped_quota,
    classify_target,
    count_debris,
    farm_map_loaded,
    needs_shed_door_step_off,
    yard_load_action,
    quota_counts_met,
    quota_satisfied,
    unmet_debris_types,
)
from harvest.tasks.farm_clear_task import FarmClearTask
from harvest.core.carry import ADDR_TOOL_BACKPACK
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_STAMINA, ADDR_TOOL, ADDR_X, ADDR_Y
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


def _world(ram: np.ndarray) -> WorldState:
    return WorldState(frame=0, ram=ram, info={}, obs=None)


def _quota_task(**kwargs) -> FarmClearTask:
    params = dict(
        fetch_tools=False,
        handoff="quota",
        timeout=7000,
    )
    params.update(kwargs)
    return FarmClearTask(**params)


class QuotaCarryCompletionTests(unittest.TestCase):
    def test_met_quota_finishes_with_empty_hands(self) -> None:
        from harvest.core.animal_status import ADDR_HELD_ITEM

        ram = _make_farm_ram(player_tile=(37, 58))
        ram[ADDR_HELD_ITEM] = 0x09
        ram[0xD2] = 2
        task = _quota_task(quota={"weeds": 10000}, priority=[DebrisType.WEED])
        task.reset(_world(ram))
        result = task.step(_world(ram))
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("quota_met", task._pending_finish_reason)
        ram[ADDR_HELD_ITEM] = 0
        ram[0xD2] = 0
        task._drop_queue.clear()
        self.assertEqual(task.step(_world(ram)).status, TaskStatus.SUCCESS)


def _place_weeds(ram: np.ndarray, n: int, *, origin: tuple[int, int] = (8, 8)):
    tiles = []
    ox, oy = origin
    for i in range(n):
        tx, ty = ox + (i % 8), oy + (i // 8)
        _set_tile(ram, tx, ty, WEED)
        tiles.append((tx, ty))
    return tiles


class TestDebrisCountContract(unittest.TestCase):
    def test_four_large_rocks_count_as_four_not_sixteen(self) -> None:
        ram = _make_farm_ram()
        for tx, ty in ((12, 12), (16, 12), (20, 12), (24, 12)):
            _place_large_rock(ram, tx, ty)
        counts = count_debris(ram)
        self.assertEqual(counts.large_rocks, 4)
        self.assertEqual(counts.small_rocks, 0)
        cells = sum(
            1
            for y in range(MAP_WIDTH)
            for x in range(MAP_WIDTH)
            if int(ram[ADDR_MAP + y * MAP_WIDTH + x]) in (0x0D, 0x0E, 0x0F, 0x10)
        )
        self.assertEqual(cells, 16)

    def test_small_rocks_distinct_from_large_2x2(self) -> None:
        ram = _make_farm_ram()
        for i in range(10):
            _set_tile(ram, 8 + i, 8, ROCK)
        _place_large_rock(ram, 20, 20)
        _place_large_rock(ram, 24, 20)
        counts = count_debris(ram)
        self.assertEqual(counts.small_rocks, 10)
        self.assertEqual(counts.large_rocks, 2)
        self.assertEqual(
            classify_target(ROCK, DebrisType.ROCK),
            "small_rocks",
        )
        self.assertEqual(
            classify_target(LARGE_ROCK_TL, DebrisType.ROCK),
            "large_rocks",
        )
        self.assertNotEqual(
            classify_target(ROCK, DebrisType.ROCK),
            classify_target(LARGE_ROCK_TL, DebrisType.ROCK),
        )

    def test_fence_posts_are_counted(self) -> None:
        ram = _make_farm_ram()
        for x in range(11, 16):
            _set_tile(ram, x, 31, FENCE)
        counts = count_debris(ram)
        self.assertEqual(counts.fences, 5)
        self.assertEqual(classify_target(FENCE, DebrisType.FENCE), "fences")

    def test_stumps_collapse_to_one_per_2x2(self) -> None:
        ram = _make_farm_ram()
        _place_stump(ram, 10, 10)
        _place_stump(ram, 14, 10)
        counts = count_debris(ram)
        self.assertEqual(counts.stumps, 2)
        self.assertEqual(classify_target(STUMP_TL, DebrisType.STUMP), "stumps")


class TestQuotaTaskHandoff(unittest.TestCase):
    def test_quota_ram_drop_without_swings_is_not_success(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=int(Tool.HAMMER))
        tiles = _place_weeds(ram, 12)
        world = _world(ram)
        task = _quota_task(quota={"weeds": 10}, priority=[DebrisType.WEED])
        task.reset(world)
        for tx, ty in tiles[:10]:
            _set_tile(ram, tx, ty, 0xA1)
        result = task.step(world)
        self.assertNotEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(task.clearer.cleared_count, 0)

    def test_ten_weed_quota_success_with_leftover_weeds(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        tiles = _place_weeds(ram, 15)
        world = _world(ram)
        task = _quota_task(quota={"weeds": 10}, priority=[DebrisType.WEED])
        task.reset(world)

        for tx, ty in tiles[:10]:
            _set_tile(ram, tx, ty, 0xA1)
        task.clearer.cleared_count = 10
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("quota_met", result.reason or "")
        leftover = [
            (tx, ty) for tx, ty in tiles[10:] if ram[ADDR_MAP + ty * MAP_WIDTH + tx] == WEED
        ]
        self.assertEqual(len(leftover), 5)
        self.assertEqual(count_debris(ram).weeds, 5)

    def test_two_stump_quota_success_with_leftover(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=int(Tool.AXE))
        _place_stump(ram, 12, 12)
        _place_stump(ram, 16, 12)
        _place_stump(ram, 20, 12)
        world = _world(ram)
        task = _quota_task(quota={"stumps": 2}, priority=[DebrisType.STUMP])
        task.reset(world)
        self.assertEqual(task.clearer.quota_start_counts.stumps, 3)

        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            _set_tile(ram, 12 + dx, 12 + dy, 0xA1)
            _set_tile(ram, 16 + dx, 12 + dy, 0xA1)
        task.clearer.cleared_count = 2
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("quota_met", result.reason or "")
        self.assertEqual(count_debris(ram).stumps, 1)
        self.assertEqual(ram[ADDR_MAP + 12 * MAP_WIDTH + 20], STUMP_TL)

    def test_grown_pasture_is_not_an_unloaded_map(self) -> None:
        """Y2 Summer D6: 57 pasture tiles at 0x76 sent the stump clearer into the barn."""
        from harvest.tasks.grass_planter import PASTURE_BOUNDS

        ram = _make_farm_ram(player_tile=(40, 20), tool=int(Tool.AXE))
        x0, y0, x1, _y1 = PASTURE_BOUNDS
        for i in range(70):
            ram[ADDR_MAP + (y0 + i // (x1 - x0 + 1)) * MAP_WIDTH + x0 + i % (x1 - x0 + 1)] = 0x76
        self.assertTrue(farm_map_loaded(ram))
        for i in range(70):
            ram[ADDR_MAP + (2 + i // 30) * MAP_WIDTH + 30 + i % 30] = 0xFF
        self.assertFalse(farm_map_loaded(ram))

    def test_whole_field_cover_grass_is_not_an_unloaded_map(self) -> None:
        """Y2 Spring D8 (compose6): the cover's young grass (0x75/0x76) outside the pasture read as stale ids.

        389 such tiles made ``farm_map_loaded`` False for the whole day, so ``EnsureCarryToolTask`` kept starting
        its leave-the-shed-door walk (it succeeds at once) and ``ENSURE_ANIMAL_TOOLS`` ran silently for 2M frames.
        """
        import os
        from unittest import mock

        from harvest.tasks.farm_clear_quota import stale_under_player
        from harvest.tasks.nav import get_pos_from_ram

        def sow(ram, tile_id, count):
            for i in range(count):
                ram[ADDR_MAP + (4 + i // 40) * MAP_WIDTH + 20 + i % 40] = tile_id   # rows 4-11, x 20-59: no pasture

        for tile_id in (0x75, 0x76):
            ram = _make_farm_ram(player_tile=(40, 20), tool=int(Tool.AXE))
            sow(ram, tile_id, 300)
            with mock.patch.dict(os.environ, {"HARVEST_FIELD_COVER": "0"}):
                self.assertFalse(farm_map_loaded(ram), "without the cover, 300 stale ids outside the pasture are unloaded")
            with mock.patch.dict(os.environ, {"HARVEST_FIELD_COVER": "1"}):
                self.assertTrue(farm_map_loaded(ram))
                ram[ADDR_MAP + 20 * MAP_WIDTH + 40] = tile_id          # the farmer stands on sown grass
                self.assertFalse(stale_under_player(ram))
                for i in range(70):                                    # a truly unloaded viewport is still caught
                    ram[ADDR_MAP + (30 + i // 30) * MAP_WIDTH + 30 + i % 30] = 0xFF
                self.assertFalse(farm_map_loaded(ram))
            ram2 = _make_farm_ram(player_tile=(40, 6), tool=int(Tool.AXE))      # outside every layout's pasture
            ram2[ADDR_MAP + 6 * MAP_WIDTH + 40] = tile_id
            with mock.patch.dict(os.environ, {"HARVEST_FIELD_COVER": "0"}):
                self.assertTrue(stale_under_player(ram2), "outside the cover a stale id under the farmer is a stale map")

    def test_capped_empty_quota_is_satisfied_when_farm_loaded(self) -> None:
        ram = _make_farm_ram(player_tile=(40, 40), tool=int(Tool.HAMMER))
        _place_large_rock(ram, 12, 12)
        world = _world(ram)
        se = (32, 32, 63, 63)
        task = _quota_task(
            quota={"large_rocks": 10000},
            farm_bounds=se,
            priority=[DebrisType.ROCK],
        )
        task.reset(world)
        self.assertEqual(task.clearer.quota_start_counts.large_rocks, 0)
        self.assertEqual(task.clearer.cleared_count, 0)
        self.assertTrue(farm_map_loaded(ram))
        self.assertTrue(
            quota_satisfied(ram, task.quota, clearer=task.clearer, bounds=se)
        )
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        reason = result.reason or ""
        self.assertIn("quota_met", reason)
        self.assertIn("no-op", reason)

    def test_capped_empty_stump_chunk_is_loaded_map_noop(self) -> None:
        ram = _make_farm_ram(player_tile=(40, 40), tool=int(Tool.AXE))
        _place_stump(ram, 4, 20)
        world = _world(ram)
        se = (32, 32, 63, 63)
        task = _quota_task(
            quota={"stumps": 10000},
            farm_bounds=se,
            priority=[DebrisType.STUMP],
        )
        task.reset(world)
        self.assertEqual(task.clearer.quota_start_counts.stumps, 0)
        self.assertEqual(task.clearer.cleared_count, 0)
        self.assertTrue(farm_map_loaded(ram))
        self.assertTrue(
            quota_satisfied(ram, task.quota, clearer=task.clearer, bounds=se)
        )
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        reason = result.reason or ""
        self.assertIn("quota_met", reason)
        self.assertIn("no-op", reason)
        self.assertEqual(count_debris(ram).stumps, 1)

    def test_capped_empty_quota_rejected_when_farm_unloaded(self) -> None:
        ram = _make_farm_ram(player_tile=(40, 40), tool=int(Tool.HAMMER))
        _place_large_rock(ram, 12, 12)
        world = _world(ram)
        se = (32, 32, 63, 63)
        task = _quota_task(
            quota={"large_rocks": 10000},
            farm_bounds=se,
            priority=[DebrisType.ROCK],
        )
        task.reset(world)
        _set_player(ram, (25, 30))
        _set_tile(ram, 25, 30, 0xA8)
        for y in range(MAP_WIDTH):
            for x in range(MAP_WIDTH):
                if (x, y) != (25, 30):
                    _set_tile(ram, x, y, 0xFF)
        self.assertFalse(farm_map_loaded(ram))
        self.assertFalse(
            quota_satisfied(ram, task.quota, clearer=task.clearer, bounds=se)
        )
        result = task.step(world)
        self.assertNotEqual(result.status, TaskStatus.SUCCESS)
        self.assertNotIn("quota_met", result.reason or "")

    def test_shed_interior_is_not_a_loaded_farm_map(self) -> None:
        ram = _make_farm_ram(player_tile=(26, 29), tool=int(Tool.HAMMER))
        ram[ADDR_TILEMAP] = 0x26
        self.assertFalse(farm_map_loaded(ram))

    def test_quota_not_met_on_shed_door_unload(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=int(Tool.HAMMER))
        for tx, ty in ((12, 10), (16, 10), (20, 10), (24, 10)):
            _place_large_rock(ram, tx, ty)
        world = _world(ram)
        task = _quota_task(
            quota={"large_rocks": 4},
            priority=[DebrisType.ROCK],
        )
        task.reset(world)
        self.assertEqual(task.clearer.quota_start_counts.large_rocks, 4)
        _set_player(ram, (25, 30))
        _set_tile(ram, 25, 30, 0xA8)
        for y in range(MAP_WIDTH):
            for x in range(MAP_WIDTH):
                if (x, y) != (25, 30):
                    _set_tile(ram, x, y, 0xFF)
        result = task.step(world)
        self.assertNotEqual(result.status, TaskStatus.SUCCESS)
        self.assertNotIn("quota_met", result.reason or "")

    def test_quota_scan_steps_off_shed_door_stale_tile(self) -> None:
        ram = _make_farm_ram(player_tile=(26, 30), tool=int(Tool.HAMMER))
        _set_tile(ram, 26, 30, 0xFF)
        _place_large_rock(ram, 28, 30)
        world = _world(ram)
        task = _quota_task(
            quota={"large_rocks": 1},
            priority=[DebrisType.ROCK],
        )
        task.reset(world)
        task.clearer.navigator.update(ram)
        nxt = task.clearer._handle_scanning(ram)
        self.assertIsNone(nxt)
        from harvest.tasks.nav import make_action
        np.testing.assert_array_equal(task.clearer.action_queue[0], make_action(down=True, b=True))

    def test_quota_scan_steps_off_a8_when_farm_still_unloaded(self) -> None:
        ram = _make_farm_ram(player_tile=(25, 30), tool=int(Tool.HAMMER))
        _set_tile(ram, 25, 30, 0xA8)
        for y in range(MAP_WIDTH):
            for x in range(MAP_WIDTH):
                if (x, y) != (25, 30):
                    _set_tile(ram, x, y, 0xFF)
        _place_large_rock(ram, 28, 28)
        world = _world(ram)
        task = _quota_task(
            quota={"large_rocks": 1},
            priority=[DebrisType.ROCK],
        )
        task.reset(world)
        task.clearer.navigator.update(ram)
        self.assertTrue(needs_shed_door_step_off(ram))
        nxt = task.clearer._handle_scanning(ram)
        self.assertIsNone(nxt)
        self.assertGreaterEqual(len(task.clearer.action_queue), 1)

    def test_yard_load_action_west_gate_runs_right(self) -> None:
        from harvest.tasks.nav import make_action

        west = _make_farm_ram(player_tile=(1, 28))
        _set_tile(west, 1, 28, 0xFF)
        self.assertTrue((yard_load_action(west) == make_action(right=True, b=True)).all())
        east = _make_farm_ram(player_tile=(40, 28))
        self.assertTrue((yard_load_action(east) == make_action(left=True, b=True)).all())

    def test_quota_scan_stops_rocks_once_boulder_count_met(self) -> None:
        ram = _make_farm_ram()
        for i in range(3):
            _set_tile(ram, 6 + i, 8, STONE)
        for tx, ty in ((8, 12), (12, 12), (16, 12), (20, 12), (24, 12)):
            _place_large_rock(ram, tx, ty)
        world = _world(ram)
        task = _quota_task(
            quota={"stones": 2, "large_rocks": 2},
            priority=[DebrisType.ROCK, DebrisType.STONE],
        )
        task.reset(world)
        for tx, ty in ((8, 12), (12, 12)):
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                _set_tile(ram, tx + dx, ty + dy, 0xA1)
        task.clearer.startup_done = True
        task.clearer._tool_scan_done = True
        task.clearer.navigator.update(ram)
        nxt = task.clearer._handle_scanning(ram)
        self.assertIn(nxt, {"navigating", "clearing"})
        assert task.clearer.current_target is not None
        self.assertEqual(task.clearer.current_target.debris_type, DebrisType.STONE)

    def test_rock_quota_counts_small_and_large_separately(self) -> None:
        ram = _make_farm_ram()
        for i in range(12):
            _set_tile(ram, 8 + i, 8, ROCK)
        for tx, ty in ((8, 12), (12, 12), (16, 12), (20, 12), (24, 12)):
            _place_large_rock(ram, tx, ty)
        world = _world(ram)
        task = _quota_task(
            quota={"small_rocks": 10, "large_rocks": 4},
            priority=[DebrisType.ROCK],
            fetch_tools=True,
        )
        task.reset(world)
        start = task.clearer.quota_start_counts
        self.assertEqual(start.small_rocks, 12)
        self.assertEqual(start.large_rocks, 5)

        for i in range(10):
            _set_tile(ram, 8 + i, 8, 0xA1)
        for tx, ty in ((8, 12), (12, 12), (16, 12), (20, 12)):
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                _set_tile(ram, tx + dx, ty + dy, 0xA1)
        task.clearer.cleared_count = 14
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("quota_met", result.reason or "")
        now = count_debris(ram)
        self.assertEqual(now.small_rocks, 2)
        self.assertEqual(now.large_rocks, 1)

    def test_pocket_plot_ring_hands_off_with_unrelated_debris(self) -> None:
        ram = _make_farm_ram(player_tile=(12, 28), tool=0)
        _set_tile(ram, 20, 15, WEED)
        world = _world(ram)
        task = FarmClearTask(
            fetch_tools=False,
            farm_bounds=WEST_PLANT_POCKET_BOUNDS,
            timeout=7000,
        )
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("plot_ring_clear", result.reason or "")
        self.assertEqual(ram[ADDR_MAP + 15 * MAP_WIDTH + 20], WEED)

    def test_quota_pocket_bounds_do_not_shrink_to_hoe_ring(self) -> None:
        ram = _make_farm_ram(player_tile=(13, 26), tool=0)
        _set_tile(ram, 20, 15, WEED)
        world = _world(ram)
        task = _quota_task(
            quota={"weeds": 10},
            farm_bounds=WEST_PLANT_POCKET_BOUNDS,
            priority=[DebrisType.WEED],
        )
        task.reset(world)
        task._pocket_arrived = True
        task._lock_clearer_to_plot()
        self.assertEqual(task.clearer.farm_bounds, WEST_PLANT_POCKET_BOUNDS)
        self.assertEqual(task._scan_bounds(), WEST_PLANT_POCKET_BOUNDS)

    def test_quota_handoff_does_not_plot_ring_success(self) -> None:
        ram = _make_farm_ram(player_tile=(12, 28), tool=0)
        _set_tile(ram, 20, 15, WEED)
        world = _world(ram)
        task = _quota_task(
            quota={"weeds": 10},
            farm_bounds=WEST_PLANT_POCKET_BOUNDS,
            priority=[DebrisType.WEED],
        )
        task.reset(world)
        result = task.step(world)
        self.assertNotEqual(result.status, TaskStatus.SUCCESS)
        self.assertNotIn("plot_ring_clear", result.reason or "")
        self.assertNotIn("quota_met", result.reason or "")
        self.assertEqual(ram[ADDR_MAP + 15 * MAP_WIDTH + 20], WEED)

    def test_stamina_low_unmet_quota_is_failure(self) -> None:
        ram = _make_farm_ram(
            player_tile=(10, 10), stamina=5, tool=int(Tool.HAMMER)
        )
        for tx, ty in ((12, 10), (16, 10), (20, 10), (24, 10)):
            _place_large_rock(ram, tx, ty)
        world = _world(ram)
        task = _quota_task(
            fetch_tools=True,
            quota={"large_rocks": 4},
            priority=[DebrisType.ROCK],
        )
        task.reset(world)
        task.clearer.startup_done = True
        task.clearer._tool_scan_done = True
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("stamina_low", result.reason or "")
        self.assertNotIn("quota_met", result.reason or "")
        self.assertEqual(count_debris(ram).large_rocks, 4)
        self.assertFalse(
            quota_satisfied(ram, task.quota, clearer=task.clearer)
        )

    def test_fetch_tools_false_keeps_rock_when_hammer_in_carry(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=int(Tool.WATERING_CAN))
        ram[ADDR_TOOL_BACKPACK] = int(Tool.HAMMER)
        _place_large_rock(ram, 12, 10)
        world = _world(ram)
        task = _quota_task(
            quota={"large_rocks": 4},
            priority=[DebrisType.ROCK],
        )
        task.reset(world)
        self.assertIn(DebrisType.ROCK, task.clearer.priority)
        self.assertTrue(task.clearer.tool_manager.has(int(Tool.HAMMER)))
        task.clearer.navigator.update(ram)
        nxt = task.clearer._handle_scanning(ram)
        self.assertIn(nxt, {"navigating", "clearing"})
        assert task.clearer.current_target is not None
        self.assertEqual(task.clearer.current_target.debris_type, DebrisType.ROCK)

    def test_fetch_tools_false_drops_rock_when_hammer_missing(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        _place_large_rock(ram, 12, 10)
        _set_tile(ram, 11, 10, WEED)
        world = _world(ram)
        task = FarmClearTask(
            fetch_tools=False,
            priority=[DebrisType.ROCK, DebrisType.WEED],
        )
        task.reset(world)
        self.assertNotIn(DebrisType.ROCK, task.clearer.priority)
        self.assertIn(DebrisType.WEED, task.clearer.priority)


class TestQuotaSatisfiedSnapshot(unittest.TestCase):
    def test_empty_quota_is_not_satisfied(self) -> None:
        ram = _make_farm_ram()
        task = _quota_task(quota={})
        task.reset(_world(ram))
        self.assertTrue(ClearQuota.from_mapping({}).is_empty())
        self.assertFalse(quota_satisfied(ram, {}, clearer=task.clearer))
        self.assertFalse(quota_satisfied(ram, None, clearer=task.clearer))

    def test_start_snapshot_required_for_honest_count(self) -> None:
        ram = _make_farm_ram()
        _set_tile(ram, 8, 8, WEED)
        class _NoSnap:
            farm_bounds = None

        self.assertFalse(
            quota_satisfied(ram, {"weeds": 1}, clearer=_NoSnap())
        )

    def test_small_rock_quota_caps_to_spawned_count(self) -> None:
        start = DebrisCounts(small_rocks=0, large_rocks=5, stones=20)
        want = ClearQuota(small_rocks=10, large_rocks=4, stones=10)
        effective = capped_quota(want, start)
        self.assertEqual(effective.small_rocks, 0)
        self.assertEqual(effective.large_rocks, 4)
        self.assertEqual(effective.stones, 10)
        now = DebrisCounts(small_rocks=0, large_rocks=1, stones=10)
        self.assertTrue(quota_counts_met(start, now, want))
        still_short = DebrisCounts(small_rocks=0, large_rocks=2, stones=10)
        self.assertFalse(quota_counts_met(start, still_short, want))

    def test_unmet_types_drop_rock_after_boulder_quota(self) -> None:
        start = DebrisCounts(small_rocks=0, large_rocks=5, stones=20)
        want = ClearQuota(small_rocks=10, large_rocks=4, stones=10)
        after_boulders = DebrisCounts(small_rocks=0, large_rocks=1, stones=20)
        unmet = unmet_debris_types(start, after_boulders, want)
        self.assertEqual(unmet, {DebrisType.STONE})
        after_both = DebrisCounts(small_rocks=0, large_rocks=1, stones=10)
        self.assertEqual(unmet_debris_types(start, after_both, want), set())


if __name__ == "__main__":
    unittest.main()
