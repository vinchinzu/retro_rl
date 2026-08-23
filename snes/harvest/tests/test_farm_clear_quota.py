"""D2 leftover clear quotas — RAM-count stop, not whole-farm wipe."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.core.tile_catalog import (
    ADDR_MAP,
    ADDR_TILEMAP,
    LARGE_ROCK_TL,
    MAP_WIDTH,
    ROCK,
    STUMP_TL,
    TILE_SIZE,
    WEED,
    DebrisType,
    Tool,
)
from harvest.maps.map_config import WEST_PLANT_POCKET_BOUNDS
from harvest.tasks.farm_clear_quota import (
    ClearQuota,
    classify_target,
    count_debris,
    quota_satisfied,
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

    def test_stumps_collapse_to_one_per_2x2(self) -> None:
        ram = _make_farm_ram()
        _place_stump(ram, 10, 10)
        _place_stump(ram, 14, 10)
        counts = count_debris(ram)
        self.assertEqual(counts.stumps, 2)
        self.assertEqual(classify_target(STUMP_TL, DebrisType.STUMP), "stumps")


class TestQuotaTaskHandoff(unittest.TestCase):
    def test_ten_weed_quota_success_with_leftover_weeds(self) -> None:
        ram = _make_farm_ram(player_tile=(10, 10), tool=0)
        tiles = _place_weeds(ram, 15)
        world = _world(ram)
        task = _quota_task(quota={"weeds": 10}, priority=[DebrisType.WEED])
        task.reset(world)

        for tx, ty in tiles[:10]:
            _set_tile(ram, tx, ty, 0xA1)
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
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("quota_met", result.reason or "")
        self.assertEqual(count_debris(ram).stumps, 1)
        self.assertEqual(ram[ADDR_MAP + 12 * MAP_WIDTH + 20], STUMP_TL)

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


if __name__ == "__main__":
    unittest.main()
