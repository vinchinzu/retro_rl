"""Crop planter refill/pond ranking and fence-access corridor tests.

Split from test_crop_planter_logic monofile.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

# Path-stable import of sibling helpers (works under unittest and pytest importlib mode).
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from crop_planter_test_helpers import blank_ram, set_player_tile, set_tile

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, WRAM_SNAPSHOT_SIZE
from harvest.tasks.crop_planter import (
    ADDR_WATER_LEVEL,
    BAD_REFILL_STAND_BOUNDS,
    CropWaterTask,
    REFILL_BAND_MID,
    REFILL_BAND_NORTH,
    REFILL_BAND_POND,
    REFILL_BAND_SOUTH,
    REFILL_PREFERRED_WATER_TILES,
    REFILL_WATER_TILES,
    edge_water_tile_id,
    find_pond_edges,
    is_bad_refill_stand,
    is_main_pond_stand,
    pond_access_blocking_fences,
    refill_edge_sort_key,
    refill_stand_band,
)
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_TOOL,
)


class CropPlanterRefillTests(unittest.TestCase):
    def test_refill_preferred_water_tiles_documented(self) -> None:
        """CheckToolSuccess fill set must stay F0/F9–FD (decomp bank_82)."""
        self.assertEqual(
            REFILL_PREFERRED_WATER_TILES,
            frozenset({0xF0, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD}),
        )
        # Still searchable, but not preferred fill properties:
        for tid in (0xF1, 0xF2, 0xF7, 0xF8):
            self.assertIn(tid, REFILL_WATER_TILES)
            self.assertNotIn(tid, REFILL_PREFERRED_WATER_TILES)

    def test_refill_stand_bands_prefer_main_pond_then_south(self) -> None:
        north = (15, 16)   # north spur stand
        stream = (18, 23)  # F8 north stream stand
        shipping = (9, 28)  # F2 shipping pocket (bad)
        south = (14, 48)   # FC south stream stand
        pond = (32, 34)    # main F0 pond south lip
        mid_other = (49, 35)  # east FB spur

        self.assertEqual(refill_stand_band(pond), REFILL_BAND_POND)
        self.assertTrue(is_main_pond_stand(pond))
        self.assertEqual(refill_stand_band(north), REFILL_BAND_NORTH)
        self.assertEqual(refill_stand_band(stream), REFILL_BAND_NORTH)
        self.assertEqual(refill_stand_band(south), REFILL_BAND_SOUTH)
        self.assertEqual(refill_stand_band(mid_other), REFILL_BAND_MID)
        self.assertTrue(is_bad_refill_stand(shipping))
        self.assertFalse(is_bad_refill_stand(north))

        player = (5, 35)  # west field — shipping is closest by Manhattan
        # Same preferred rank (none): band order pond → south → north → bad.
        edges = [
            (shipping, "right"),
            (south, "down"),
            (north, "left"),
            (pond, "up"),
        ]
        edges.sort(key=lambda e: refill_edge_sort_key(e, player, water_tid=-1))
        self.assertEqual(edges[0][0], pond)
        self.assertEqual(edges[1][0], south)
        self.assertEqual(edges[2][0], north)
        self.assertEqual(edges[3][0], shipping)

    def test_refill_edge_sort_prefers_fc_over_north_f8(self) -> None:
        """South FC (preferred property) beats closer north F8."""
        north_f8 = (18, 23)
        south_fc = (14, 48)
        player = (18, 24)  # next to north F8 stand — F8 is much closer
        edges = [
            (north_f8, "up"),
            (south_fc, "down"),
        ]
        edges.sort(
            key=lambda e: refill_edge_sort_key(
                e,
                player,
                water_tid=0xF8 if e[0] == north_f8 else 0xFC,
            )
        )
        self.assertEqual(edges[0][0], south_fc)
        self.assertEqual(edges[1][0], north_f8)
        # Preferred rank alone decides before band/distance.
        self.assertLess(
            refill_edge_sort_key((south_fc, "down"), player, 0xFC),
            refill_edge_sort_key((north_f8, "up"), player, 0xF8),
        )

    def test_refill_edge_sort_prefers_main_pond_f0_over_south_fc(self) -> None:
        pond = (32, 34)
        south_fc = (14, 48)
        player = (13, 27)  # west plant pocket — FC is closer by Manhattan
        self.assertLess(
            refill_edge_sort_key((pond, "up"), player, 0xF0),
            refill_edge_sort_key((south_fc, "down"), player, 0xFC),
        )

    def test_find_pond_edges_excludes_bad_shipping_stands(self) -> None:
        ram = blank_ram()
        for ty in range(64):
            for tx in range(64):
                set_tile(ram, tx, ty, 0xA1)
        # North stream water + stand
        set_tile(ram, 18, 22, 0xF8)
        set_tile(ram, 18, 23, 0xA1)
        # Shipping F2 pocket water + stand
        set_tile(ram, 9, 29, 0xF2)
        set_tile(ram, 9, 28, 0xA1)
        # South stream
        set_tile(ram, 14, 49, 0xFC)
        set_tile(ram, 14, 48, 0xA1)

        all_edges = find_pond_edges(ram, (3, 10, 62, 60), water_tiles=REFILL_WATER_TILES)
        stands = {t for t, _f in all_edges}
        self.assertIn((18, 23), stands)
        self.assertIn((9, 28), stands)
        self.assertIn((14, 48), stands)

        good_edges = find_pond_edges(
            ram,
            (3, 10, 62, 60),
            water_tiles=REFILL_WATER_TILES,
            exclude_bad_stands=True,
        )
        good_stands = {t for t, _f in good_edges}
        self.assertIn((18, 23), good_stands)
        self.assertIn((14, 48), good_stands)
        self.assertNotIn((9, 28), good_stands)
        for t in good_stands:
            self.assertFalse(is_bad_refill_stand(t), msg=f"bad stand leaked: {t}")

        # South stand faces FC (preferred); north faces F8 (fallback).
        by_stand = {t: f for t, f in good_edges}
        self.assertEqual(edge_water_tile_id(ram, (14, 48), by_stand[(14, 48)]), 0xFC)
        self.assertEqual(edge_water_tile_id(ram, (18, 23), by_stand[(18, 23)]), 0xF8)

    def test_start_refill_prefers_preferred_water_over_f8_and_skips_bad(self) -> None:
        ram = blank_ram()
        for ty in range(64):
            for tx in range(64):
                set_tile(ram, tx, ty, 0xA1)
        # North F8 (closer, not preferred CheckToolSuccess property)
        set_tile(ram, 18, 22, 0xF8)
        # Shipping F2 (closer still — must not be chosen)
        set_tile(ram, 9, 29, 0xF2)
        # South FC (preferred fill property)
        set_tile(ram, 14, 49, 0xFC)

        player = (7, 30)  # next to shipping pocket
        set_player_tile(ram, player)
        ram[ADDR_TOOL] = 0x10
        ram[ADDR_WATER_LEVEL] = 0

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(refill_bounds=(3, 10, 62, 60))
        task.reset(world)
        task._navigator.update(ram)
        task._plots = [(5, 35)]
        task._plot_index = 0
        task._plot_phase = "water"
        task._water_steps = [((4, 34), (5, 35), "left")]
        task._water_index = 0

        task._start_refill(ram)

        self.assertEqual(task._plot_phase, "refill")
        self.assertIsNotNone(task._refill_pond_tile)
        assert task._refill_pond_tile is not None
        self.assertFalse(is_bad_refill_stand(task._refill_pond_tile))
        # Preferred FC south beats closer F8 north when both pathable.
        self.assertEqual(task._refill_pond_tile, (14, 48))
        water = edge_water_tile_id(
            ram, task._refill_pond_tile, task._refill_pond_face or "down"
        )
        self.assertIn(water, REFILL_PREFERRED_WATER_TILES)
        x0, y0, x1, y1 = BAD_REFILL_STAND_BOUNDS
        rx, ry = task._refill_pond_tile
        self.assertFalse(x0 <= rx <= x1 and y0 <= ry <= y1)

    def test_start_refill_prefers_main_pond_f0_when_pathable(self) -> None:
        """Main F0 pond beats closer south FC when both preferred and pathable."""
        ram = blank_ram()
        for ty in range(64):
            for tx in range(64):
                set_tile(ram, tx, ty, 0xA1)
        set_tile(ram, 33, 31, 0xF0)  # main pond water
        set_tile(ram, 14, 49, 0xFC)  # south stream

        player = (14, 40)  # closer to FC
        set_player_tile(ram, player)
        ram[ADDR_TOOL] = 0x10
        ram[ADDR_WATER_LEVEL] = 0

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(refill_bounds=(3, 10, 62, 60))
        task.reset(world)
        task._navigator.update(ram)
        task._plots = [(13, 35)]
        task._plot_index = 0
        task._plot_phase = "water"
        task._water_steps = [((12, 34), (13, 35), "left")]
        task._water_index = 0

        task._start_refill(ram)

        self.assertEqual(task._plot_phase, "refill")
        assert task._refill_pond_tile is not None
        water = edge_water_tile_id(
            ram, task._refill_pond_tile, task._refill_pond_face or "down"
        )
        self.assertEqual(water, 0xF0)
        self.assertTrue(is_main_pond_stand(task._refill_pond_tile))

    def test_start_refill_ignores_only_nonfill_water(self) -> None:
        """F1/F8-only maps must not pretend to refill — no refill stand chosen."""
        ram = blank_ram()
        for ty in range(64):
            for tx in range(64):
                set_tile(ram, tx, ty, 0xA1)
        set_tile(ram, 18, 22, 0xF8)
        set_tile(ram, 9, 29, 0xF2)  # bad stand — excluded

        player = (7, 30)
        set_player_tile(ram, player)
        ram[ADDR_TOOL] = 0x10
        ram[ADDR_WATER_LEVEL] = 0

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(refill_bounds=(3, 10, 62, 60))
        task.reset(world)
        task._navigator.update(ram)
        task._plots = [(5, 35)]
        task._plot_index = 0
        task._plot_phase = "water"
        task._water_steps = [((4, 34), (5, 35), "left")]
        task._water_index = 0

        task._start_refill(ram)

        # No preferred water on map → never enter refill at F8.
        self.assertIsNone(task._refill_pond_tile)
        self.assertNotEqual(task._plot_phase, "refill")
        self.assertNotEqual(task._state, "fence_open")

    def test_pond_access_blocking_fences_detects_y31_wall(self) -> None:
        ram = blank_ram()
        for ty in range(64):
            for tx in range(64):
                set_tile(ram, tx, ty, 0xA1)
        for tx in range(11, 30):
            set_tile(ram, tx, 31, 0x05)
        fences = pond_access_blocking_fences(ram)
        self.assertEqual(len(fences), 19)
        self.assertIn((20, 31), fences)
        self.assertEqual(pond_access_blocking_fences(ram), fences)

    def test_try_open_pond_access_stages_west_pocket_first(self) -> None:
        """West plant pocket must stage before FenceClearLoopTask.

        ROM trap: pure-south from (13,27) soft-blocks even when tile IDs look
        walkable; staging via (12,29)/(15,29) is the proven detour.
        """
        ram = blank_ram()
        for ty in range(64):
            for tx in range(64):
                set_tile(ram, tx, ty, 0xA1)
        for tx in range(11, 30):
            set_tile(ram, tx, 31, 0x05)
        # Preferred pond water so refill path exists after open.
        set_tile(ram, 33, 31, 0xF0)

        player = (13, 27)
        set_player_tile(ram, player)
        ram[ADDR_TOOL] = 0x10
        ram[ADDR_WATER_LEVEL] = 0

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(refill_bounds=(3, 10, 62, 60), work_mode="water")
        task.reset(world)
        task._navigator.update(ram)
        fences = pond_access_blocking_fences(ram)
        self.assertTrue(fences)
        started = task._try_open_pond_access(ram, fences)
        self.assertTrue(started)
        self.assertEqual(task._plot_phase, "stage_pond")
        self.assertIn(task._approach_tile, task._pond_access_staging_tiles())
        self.assertTrue(task._pond_staged)

        # After staging is marked done, skip_stage starts the fence subtask.
        started2 = task._try_open_pond_access(ram, fences, skip_stage=True)
        self.assertTrue(started2)
        self.assertEqual(task._plot_phase, "open_pond")
        self.assertEqual(task._state, "fence_open")
        self.assertIsNotNone(task._fence_subtask)

    def test_empty_can_triggers_refill_before_water(self) -> None:
        ram = blank_ram()
        for ty in range(64):
            for tx in range(64):
                set_tile(ram, tx, ty, 0xA1)
        # Preferred fill water (not non-fill F8)
        set_tile(ram, 33, 31, 0xF0)
        # Dry crop tile target so water step is considered waterable
        set_tile(ram, 12, 24, 0x5A)

        player = (13, 25)
        set_player_tile(ram, player)
        ram[ADDR_TOOL] = 0x10  # watering can selected
        ram[ADDR_WATER_LEVEL] = 0  # empty
        ram[ADDR_INPUT_LOCK] = 1

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(refill_bounds=(3, 10, 62, 60), work_mode="water")
        task.reset(world)
        task._navigator.update(ram)
        task._plots = [(13, 25)]
        task._plot_index = 0
        task._plot_phase = "water"
        task._state = "act"
        task._water_steps = [((12, 24), (13, 25), "left")]
        task._water_index = 0
        task._target_tile = (12, 24)
        task._approach_tile = (13, 25)
        task._face_direction = "left"
        task._tool_mgr.update(ram)
        task._refill_exhausted = False

        result = task._act_water(ram)

        self.assertIsNone(result)
        self.assertEqual(task._plot_phase, "refill")
        self.assertIsNotNone(task._refill_pond_tile)
        assert task._refill_pond_tile is not None
        self.assertFalse(is_bad_refill_stand(task._refill_pond_tile))
        water = edge_water_tile_id(
            ram, task._refill_pond_tile, task._refill_pond_face or "down"
        )
        self.assertIn(water, REFILL_PREFERRED_WATER_TILES)

    def test_water_level_uses_catalog_live_offset(self) -> None:
        # Live-sized RAM: watering_can at address + LIVE_RAM_WRAM_OFFSET
        live = np.zeros(WRAM_SNAPSHOT_SIZE + 0x8000, dtype=np.uint8)
        live[ADDR_WATER_LEVEL] = 3  # raw/save-style offset must be ignored
        live[ADDR_WATER_LEVEL + LIVE_RAM_WRAM_OFFSET] = 17
        self.assertEqual(CropWaterTask._water_level(live), 17)

        # Tiny test RAM: catalog uses no live offset, reads 0x0926 directly
        tiny = blank_ram()
        tiny[ADDR_WATER_LEVEL] = 11
        self.assertEqual(CropWaterTask._water_level(tiny), 11)

    def test_start_refill_skips_sealed_f9_multihop(self) -> None:
        """Sealed F9 (manhattan hop only) must NOT multihop-commit.

        ROM dry fixture: F9 island is disconnected from west plant pocket by
        the y=13 fence bar. False multihop thrash blocked fence-open forever.
        Fall through to fence corridor instead.
        """
        ram = blank_ram()
        for ty in range(64):
            for tx in range(64):
                set_tile(ram, tx, ty, 0x05)
        # Local pocket around player (not connected to F9).
        for ty in range(24, 30):
            for tx in range(10, 18):
                set_tile(ram, tx, ty, 0xA1)
        # Staging tiles near fence wall.
        for tx in range(10, 16):
            set_tile(ram, tx, 29, 0xA1)
            set_tile(ram, tx, 30, 0xA1)
        # F9 island far north (not connected) + y=31 fence wall.
        for ty in range(10, 14):
            for tx in range(24, 28):
                set_tile(ram, tx, ty, 0xA1)
        set_tile(ram, 26, 12, 0xF9)
        for tx in range(11, 30):
            set_tile(ram, tx, 31, 0x05)

        player = (13, 27)
        set_player_tile(ram, player)
        ram[ADDR_TOOL] = 0x10
        ram[ADDR_WATER_LEVEL] = 0
        ram[ADDR_INPUT_LOCK] = 1

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(refill_bounds=(3, 10, 62, 60), work_mode="water")
        task.reset(world)
        task._navigator.update(ram)
        task._plots = [(13, 25)]
        task._plot_index = 0
        task._plot_phase = "water"
        task._water_steps = [((12, 24), (13, 25), "left")]
        task._water_index = 0

        self.assertIsNone(task._pathfinder.find_path(ram, player, (25, 12)))

        task._start_refill(ram)

        # Must not lock onto sealed F9 multihop.
        if task._refill_pond_tile is not None:
            water = edge_water_tile_id(
                ram, task._refill_pond_tile, task._refill_pond_face or "up"
            )
            self.assertNotEqual(
                water,
                0xF9,
                msg=f"sealed F9 must not multihop-commit, got stand={task._refill_pond_tile}",
            )
        # Prefer fence/stage corridor for main pond.
        self.assertIn(
            task._plot_phase,
            ("open_pond", "stage_pond", "refill"),
            msg=f"phase={task._plot_phase} state={task._state}",
        )
        if task._plot_phase == "refill" and task._refill_pond_tile is not None:
            # Main-pond multihop after gap is OK; F9 is not.
            self.assertTrue(
                is_main_pond_stand(task._refill_pond_tile)
                or task._refill_pond_tile[1] >= 30,
                msg=f"unexpected sealed-edge stand {task._refill_pond_tile}",
            )

    def test_start_refill_prefers_f9_before_fence_open(self) -> None:
        """West pocket + sealed south: pathable north F9 must win over fence-open.

        Empty-can natural refill used to burn the day on FenceClearLoopTask
        before ever ranking preferred edges north of the wall. Seal the whole
        y=31 row so main-pond BFS cannot sneak east around x=11–29.
        """
        ram = blank_ram()
        for ty in range(64):
            for tx in range(64):
                set_tile(ram, tx, ty, 0xA1)
        # Full east-west barrier: main pond + south FC unreachable from pocket.
        for tx in range(0, 64):
            set_tile(ram, tx, 31, 0x05)
        # Track wall fences for corridor_needs_fence_open (x=11–29 subset).
        for tx in range(11, 30):
            set_tile(ram, tx, 31, 0x05)
        # Main pond F0 south of barrier (unreachable).
        for ty in range(32, 35):
            for tx in range(31, 35):
                set_tile(ram, tx, ty, 0xF0)
        # North spur F9 (preferred fill, north of wall, pathable).
        set_tile(ram, 26, 12, 0xF9)

        player = (13, 27)
        set_player_tile(ram, player)
        ram[ADDR_TOOL] = 0x10
        ram[ADDR_WATER_LEVEL] = 0
        ram[ADDR_INPUT_LOCK] = 1

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(refill_bounds=(3, 10, 62, 60), work_mode="water")
        task.reset(world)
        task._navigator.update(ram)
        task._plots = [(13, 25)]
        task._plot_index = 0
        task._plot_phase = "water"
        task._water_steps = [((12, 24), (13, 25), "left")]
        task._water_index = 0

        # Precondition: main pond stands not full-pathable from player.
        self.assertIsNone(
            task._pathfinder.find_path(ram, player, (32, 34)),
            msg="test map must block main pond full BFS",
        )

        task._start_refill(ram)

        self.assertEqual(
            task._plot_phase,
            "refill",
            msg=f"phase={task._plot_phase} state={task._state} stand={task._refill_pond_tile}",
        )
        self.assertNotEqual(task._state, "fence_open")
        self.assertIsNone(task._fence_subtask)
        self.assertIsNotNone(task._refill_pond_tile)
        assert task._refill_pond_tile is not None
        water = edge_water_tile_id(
            ram, task._refill_pond_tile, task._refill_pond_face or "up"
        )
        self.assertEqual(water, 0xF9)
        self.assertFalse(is_bad_refill_stand(task._refill_pond_tile))


if __name__ == "__main__":
    unittest.main()
