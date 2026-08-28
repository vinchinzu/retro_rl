from __future__ import annotations

import unittest

import numpy as np

from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_TILEMAP,
    ADDR_X,
    ADDR_Y,
)
from harvest.core.npc_catalog import GOBJ_INITIALIZED, GOBJ_STRUCT_BASE, GOBJ_STRUCT_STRIDE
from harvest.core.ram_catalog import CHICKEN_SLOT_BASE, CHICKEN_SLOT_SIZE
from harvest.runtime.recording_trace import recording_trace_entry, summarize_recording


def _make_ram(*, tilemap: int = 0x28, tile: tuple[int, int] = (2, 7)) -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    ram[ADDR_INPUT_LOCK] = 1
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF
    ram[ADDR_MAP + tile[1] * 64 + tile[0]] = 0xA1
    return ram


def _add_chicken_object(ram: np.ndarray, tile: tuple[int, int], *, slot: int = 1) -> None:
    offset = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
    ram[offset] = GOBJ_INITIALIZED & 0xFF
    ram[offset + 1] = (GOBJ_INITIALIZED >> 8) & 0xFF
    ram[offset + 2] = 0xE1
    ram[offset + 3] = 0x01
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[offset + 0x08] = px & 0xFF
    ram[offset + 0x09] = (px >> 8) & 0xFF
    ram[offset + 0x0A] = py & 0xFF
    ram[offset + 0x0B] = (py >> 8) & 0xFF


def _add_chicken_slot(ram: np.ndarray, slot: int, tile: tuple[int, int], status: int) -> None:
    offset = CHICKEN_SLOT_BASE + slot * CHICKEN_SLOT_SIZE
    ram[offset] = status
    ram[offset + 1] = 0x28
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[offset + 4] = px & 0xFF
    ram[offset + 5] = (px >> 8) & 0xFF
    ram[offset + 6] = py & 0xFF
    ram[offset + 7] = (py >> 8) & 0xFF


class RecordingTraceTests(unittest.TestCase):
    def test_recording_trace_entry_includes_coop_chicken_entities(self) -> None:
        ram = _make_ram(tile=(2, 7))
        _add_chicken_object(ram, (3, 7))
        action = [0] * 12
        action[7] = 1

        row = recording_trace_entry(ram, frame=12, action=action)

        self.assertEqual(row["frame"], 12)
        self.assertEqual(row["tm"], 0x28)
        self.assertEqual(row["tx"], 2)
        self.assertEqual(row["buttons"], ["Right"])
        chickens = [entity for entity in row["entities"] if entity["label"] == "chicken"]
        self.assertEqual(chickens[0]["tile"], [3, 7])

    def test_recording_trace_entry_includes_chicken_slot_stage(self) -> None:
        ram = _make_ram(tile=(2, 7))
        _add_chicken_slot(ram, 0, (13, 9), 0x09)
        _add_chicken_slot(ram, 1, (13, 11), 0x05)

        row = recording_trace_entry(ram, frame=12, action=[0] * 12)

        slot_entities = [entity for entity in row["entities"] if entity["source"] == "animal_slot"]
        self.assertEqual(
            [(entity["stage"], entity["tile"]) for entity in slot_entities],
            [("adult", [13, 9]), ("chick", [13, 11])],
        )

    def test_summarize_recording_reports_stasis_and_coop_tiles(self) -> None:
        trace = []
        frames = []
        for frame in range(50):
            action = [0] * 12
            action[7] = 1
            frames.append(action)
            trace.append(
                {
                    "frame": frame,
                    "x": 40,
                    "y": 120,
                    "tx": 2,
                    "ty": 7,
                    "tm": 0x28,
                    "map": "coop",
                    "tile_id": 0xA1,
                    "tile_label": "structure",
                    "buttons": ["Right"],
                    "entities": [{"label": "chicken", "tile": [3, 7]}],
                    "egg_available": 1,
                    "held_item": 0,
                    "stored_grass": 10,
                    "fed_chickens_n": 0,
                    "shipping_money": 0,
                }
            )

        summary = summarize_recording(frames=frames, trace=trace)

        self.assertEqual(summary["coop"]["frame_count"], 50)
        self.assertEqual(summary["coop"]["chicken_tiles"], [{"tile": [3, 7]}])
        self.assertEqual(summary["coop"]["unknown_chicken_tiles"], [{"tile": [3, 7]}])
        self.assertEqual(summary["stasis_windows"][0]["start"], 0)
        self.assertEqual(summary["stasis_windows"][0]["length"], 50)
        self.assertEqual(summary["stasis_windows"][0]["nearest_chicken_distance_min"], 1)
        self.assertEqual(
            summary["push_faces"],
            [
                {
                    "tile": [2, 7],
                    "tilemap": 0x28,
                    "length": 50,
                    "buttons": ["Right"],
                }
            ],
        )

    def test_farm_trace_includes_facing_and_neighbor_cells(self) -> None:
        ram = _make_ram(tilemap=0x00, tile=(17, 20))
        ram[ADDR_MAP + 20 * 64 + 18] = 0xA1
        ram[ADDR_MAP + 19 * 64 + 17] = 0x00
        ram[0x00DA] = 2  # face right
        action = [0] * 12
        action[7] = 1

        row = recording_trace_entry(ram, frame=0, action=action)

        self.assertEqual(row["tm"], 0)
        self.assertEqual(row["tx"], 17)
        self.assertEqual(row["ty"], 20)
        self.assertEqual(row["facing"], "right")
        self.assertEqual(row["facing_tile"]["tile"], [18, 20])
        self.assertEqual(row["facing_tile"]["hex"], "0xA1")
        self.assertEqual(row["neighbors"]["up"]["tile"], [17, 19])
        self.assertEqual(row["neighbors"]["up"]["hex"], "0x00")
        self.assertEqual(row["buttons"], ["Right"])


if __name__ == "__main__":
    unittest.main()
