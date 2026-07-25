from __future__ import annotations

import unittest

import numpy as np

from harvest.tasks.farm_clearer import ADDR_INPUT_LOCK, ADDR_MAP, ADDR_TILEMAP, ADDR_X, ADDR_Y, MAP_WIDTH
from harvest.runtime.probe_utils import (
    frame_in_ranges,
    parse_field_list,
    parse_frame_ranges,
    snapshot_from_ram,
    task_debug_snapshot,
    watch_changes,
    watch_values,
)


def _make_ram() -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = 0x28
    ram[ADDR_INPUT_LOCK] = 1
    px = 13 * 16 + 8
    py = 11 * 16 + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = px >> 8
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = py >> 8
    ram[ADDR_MAP + 11 * MAP_WIDTH + 13] = 0xA1
    return ram


class ProbeUtilsTests(unittest.TestCase):
    def test_parse_field_list_accepts_repeated_commas(self) -> None:
        self.assertEqual(parse_field_list(["held_item,egg_available", "incubator_flags"]), [
            "held_item",
            "egg_available",
            "incubator_flags",
        ])

    def test_parse_frame_ranges_accepts_ranges_and_singletons(self) -> None:
        ranges = parse_frame_ranges(["10:20, 30", "40:42"])

        self.assertEqual(ranges, [(10, 20), (30, 30), (40, 42)])
        self.assertTrue(frame_in_ranges(15, ranges))
        self.assertTrue(frame_in_ranges(30, ranges))
        self.assertFalse(frame_in_ranges(31, ranges))

    def test_snapshot_from_ram_reports_position_tile_and_buttons(self) -> None:
        ram = _make_ram()
        action = [0] * 12
        action[7] = 1
        action[8] = 1

        snapshot = snapshot_from_ram(ram, frame=44, action=action)

        self.assertEqual(snapshot.tilemap, 0x28)
        self.assertEqual((snapshot.tx, snapshot.ty), (13, 11))
        self.assertEqual(snapshot.tile_id, 0xA1)
        self.assertEqual(snapshot.buttons, ("Right", "A"))

    def test_watch_values_and_changes_use_named_ram_fields(self) -> None:
        before = _make_ram()
        after = before.copy()
        after[ADDR_INPUT_LOCK] = 0

        before_values = watch_values(before, ["tilemap", "input_lock"])
        after_values = watch_values(after, ["tilemap", "input_lock"])

        self.assertEqual(before_values["tilemap"], 0x28)
        self.assertEqual(watch_changes(before_values, after_values), {
            "input_lock": {"from": 1, "to": 0},
        })

    def test_task_debug_snapshot_reports_nav_state(self) -> None:
        class DummyNavigator:
            current_tile = (2, 3)
            path = [(3, 3), (4, 3)]
            stasis = 12

            class Pos:
                x = 40
                y = 56

            current_pos = Pos()

        class DummyTask:
            _phase = "nav"
            _navigator = DummyNavigator()
            _action_queue = [1, 2]

        row = task_debug_snapshot(DummyTask())

        self.assertEqual(row["class"], "DummyTask")
        self.assertEqual(row["phase"], "nav")
        self.assertEqual(row["nav_tile"], [2, 3])
        self.assertEqual(row["path_len"], 2)
        self.assertEqual(row["action_queue_len"], 2)


if __name__ == "__main__":
    unittest.main()
