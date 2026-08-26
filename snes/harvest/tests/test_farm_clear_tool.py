"""Stationary hammer/axe policy — d-pad freeze and delayed RAM hit credit."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_STAMINA,
    ADDR_TILEMAP,
    ADDR_TOOL,
    ADDR_X,
    ADDR_Y,
    LARGE_ROCK_TL,
    MAP_WIDTH,
    TILE_SIZE,
    DebrisType,
    Tool,
)
from harvest.tasks.farm_clear_tool import (
    MAX_OBSERVE_EXTRA,
    MAX_STAND_MISSES,
    POST_SWING_OBSERVE_FRAMES,
    handle_tool_clear,
    tool_clear_is_planted,
)
from harvest.tasks.farm_clearer import FarmClearer, Target
from harvest.tasks.nav import Point, make_action

ADDR_TOOL_HITS = field_spec("tool_hit_counter").address


def _set_player(ram: np.ndarray, tile: tuple[int, int], *, dx: int = 0, dy: int = 0) -> None:
    px = tile[0] * TILE_SIZE + 8 + dx
    py = tile[1] * TILE_SIZE + 8 + dy
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF


def _set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


def _place_large_rock(ram: np.ndarray, tx: int, ty: int) -> None:
    _set_tile(ram, tx, ty, 0x0D)
    _set_tile(ram, tx + 1, ty, 0x0E)
    _set_tile(ram, tx, ty + 1, 0x0F)
    _set_tile(ram, tx + 1, ty + 1, 0x10)


def _make_ram(*, stamina: int = 65, hits: int = 0) -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = 0x00
    ram[ADDR_INPUT_LOCK] = 1
    ram[ADDR_STAMINA] = stamina
    ram[field_spec("max_stamina").address] = 100
    ram[ADDR_TOOL] = int(Tool.HAMMER)
    ram[ADDR_TOOL_HITS] = hits
    for i in range(MAP_WIDTH * MAP_WIDTH):
        ram[ADDR_MAP + i] = 0xA1
    _set_player(ram, (9, 11))
    _place_large_rock(ram, 10, 10)
    return ram


def _ready_clearer(ram: np.ndarray) -> FarmClearer:
    clearer = FarmClearer(priority=[DebrisType.ROCK])
    clearer.navigator.update(ram)
    clearer.tool_manager.update(ram)
    clearer.current_target = Target(
        (10, 10),
        Point(10 * 16 + 8, 10 * 16 + 8),
        DebrisType.ROCK,
        LARGE_ROCK_TL,
    )
    clearer.approach_tile = (9, 11)
    clearer.frame_count = 10
    clearer.navigator.stasis = 10
    return clearer


def _has_dpad(action: np.ndarray) -> bool:
    return any(int(action[i]) for i in range(4, 8))


class PlantedSwingTests(unittest.TestCase):
    def test_first_swing_faces_then_y_only(self) -> None:
        ram = _make_ram()
        clearer = _ready_clearer(ram)
        nxt = handle_tool_clear(clearer, ram, player=(9, 11), target=(10, 10))
        self.assertIsNone(nxt)
        self.assertTrue(clearer._tool_faced)
        self.assertTrue(tool_clear_is_planted(clearer))
        self.assertTrue(_has_dpad(clearer.action_queue[0]))
        y_frames = [a for a in clearer.action_queue if int(a[1]) == 1]
        self.assertGreaterEqual(len(y_frames), 20)
        for action in y_frames:
            self.assertFalse(_has_dpad(action))

    def test_later_swings_never_touch_dpad(self) -> None:
        ram = _make_ram(hits=2)
        clearer = _ready_clearer(ram)
        handle_tool_clear(clearer, ram, player=(9, 11), target=(10, 10))
        clearer.action_queue.clear()
        ram[ADDR_TOOL_HITS] = 2
        nxt = handle_tool_clear(clearer, ram, player=(9, 11), target=(10, 10))
        self.assertIsNone(nxt)
        self.assertTrue(clearer.action_queue)
        self.assertFalse(any(_has_dpad(a) for a in clearer.action_queue))
        self.assertGreaterEqual(
            sum(1 for a in clearer.action_queue if int(a.sum()) == 0),
            POST_SWING_OBSERVE_FRAMES,
        )

    def test_late_ram_hit_is_credited_not_a_miss(self) -> None:
        ram = _make_ram(stamina=63, hits=0)
        clearer = _ready_clearer(ram)
        handle_tool_clear(clearer, ram, player=(9, 11), target=(10, 10))
        clearer.action_queue.clear()
        ram[ADDR_TOOL_HITS] = 1
        ram[ADDR_STAMINA] = 61
        nxt = handle_tool_clear(clearer, ram, player=(9, 11), target=(10, 10))
        self.assertIsNone(nxt)
        self.assertEqual(clearer.target_hits, 1)
        self.assertEqual(clearer._tool_misses, 0)
        self.assertEqual(clearer.approach_tile, (9, 11))

    def test_stamina_drop_counts_when_counter_is_one_frame_late(self) -> None:
        ram = _make_ram(stamina=63, hits=0)
        clearer = _ready_clearer(ram)
        handle_tool_clear(clearer, ram, player=(9, 11), target=(10, 10))
        clearer.action_queue.clear()
        ram[ADDR_TOOL_HITS] = 0
        ram[ADDR_STAMINA] = 61
        nxt = handle_tool_clear(clearer, ram, player=(9, 11), target=(10, 10))
        self.assertIsNone(nxt)
        self.assertGreaterEqual(clearer.target_hits, 1)
        self.assertEqual(clearer._tool_misses, 0)

    def test_genuine_miss_stays_on_the_same_stand(self) -> None:
        ram = _make_ram(stamina=63, hits=0)
        clearer = _ready_clearer(ram)
        handle_tool_clear(clearer, ram, player=(9, 11), target=(10, 10))
        clearer.action_queue.clear()
        clearer._tool_observe_extra = MAX_OBSERVE_EXTRA
        nxt = handle_tool_clear(clearer, ram, player=(9, 11), target=(10, 10))
        self.assertIsNone(nxt)
        self.assertEqual(clearer._tool_misses, 1)
        self.assertEqual(clearer.approach_tile, (9, 11))
        self.assertNotIn(((10, 10), (9, 11)), clearer.failed_approaches)

    def test_three_misses_try_another_footprint_side(self) -> None:
        ram = _make_ram()
        clearer = _ready_clearer(ram)
        clearer._tool_seq_key = (10, 10, 9, 11)
        clearer._tool_faced = True
        clearer._tool_swing_pending = True
        clearer._tool_last_hits = 0
        clearer._tool_last_stam = 65
        clearer._tool_observe_extra = MAX_OBSERVE_EXTRA
        clearer._tool_misses = MAX_STAND_MISSES - 1
        nxt = handle_tool_clear(clearer, ram, player=(9, 11), target=(10, 10))
        self.assertEqual(nxt, "navigating")
        self.assertIn(((10, 10), (9, 11)), clearer.failed_approaches)
        self.assertNotEqual(clearer.approach_tile, (9, 11))
        self.assertFalse(clearer._tool_faced)

    def test_mid_hit_does_not_recenter(self) -> None:
        ram = _make_ram(hits=3)
        _set_player(ram, (9, 11), dx=4)
        clearer = _ready_clearer(ram)
        clearer._tool_faced = True
        clearer._tool_seq_key = (10, 10, 9, 11)
        clearer.target_hits = 3
        clearer.clearing_start_frame = 1
        nxt = clearer._handle_clearing(ram)
        self.assertIsNone(nxt)
        self.assertTrue(clearer.action_queue)
        queued = list(clearer.action_queue)
        self.assertFalse(any(_has_dpad(a) and int(a[1]) == 0 for a in queued))


class IdleFrameTests(unittest.TestCase):
    def test_observe_frames_are_blank(self) -> None:
        idle = make_action()
        self.assertEqual(int(idle.sum()), 0)
