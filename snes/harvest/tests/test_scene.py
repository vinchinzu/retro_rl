from __future__ import annotations

import unittest

import numpy as np

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec, read_ram_value
from harvest.core.scene import (
    SceneLocation,
    SceneMode,
    classify_scene,
    classify_scene_from_ram,
    morning_scene_ready,
    scene_indicates_ending,
)
from harvest.core.world_snapshot import WorldSnapshot
from harvest.runtime.rom_tools import parse_save_state, resolve_state_path


def _ram(size: int = 0x24000) -> np.ndarray:
    ram = np.zeros(size, dtype=np.uint8)
    ram[field_spec("input_lock").address] = 1
    _write_u16(ram, field_spec("player_x").address + _live_base(ram), 136)
    _write_u16(ram, field_spec("player_y").address + _live_base(ram), 424)
    return ram


def _live_base(ram: np.ndarray) -> int:
    return LIVE_RAM_WRAM_OFFSET if len(ram) > 0x20000 else 0


def _write_u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


class SceneClassifierTests(unittest.TestCase):
    def test_classifies_seasonal_farm_as_normal_farm(self) -> None:
        for tilemap in (0x00, 0x01, 0x02, 0x03):
            with self.subTest(tilemap=tilemap):
                ram = _ram()
                ram[field_spec("tilemap").address] = tilemap

                scene = classify_scene_from_ram(ram)

                self.assertEqual(scene.mode, SceneMode.NORMAL)
                self.assertEqual(scene.location, SceneLocation.FARM)
                self.assertIn("farm", scene.variant)

    def test_classifies_house_variants(self) -> None:
        expected = {0x15: "base", 0x16: "level1", 0x17: "level2"}
        for tilemap, variant in expected.items():
            with self.subTest(tilemap=tilemap):
                ram = _ram()
                ram[field_spec("tilemap").address] = tilemap
                _write_u16(ram, field_spec("player_y").address + _live_base(ram), 120)

                scene = classify_scene_from_ram(ram)

                self.assertEqual(scene.mode, SceneMode.NORMAL)
                self.assertEqual(scene.location, SceneLocation.HOUSE)
                self.assertIn(variant, scene.variant)

    def test_classifies_known_domain_locations(self) -> None:
        expected = {
            0x04: SceneLocation.TOWN,
            0x05: SceneLocation.TOWN,
            0x10: SceneLocation.MOUNTAIN,
            0x1C: SceneLocation.SHOP,
            0x24: SceneLocation.SHOP,
            0x26: SceneLocation.SHED,
            0x27: SceneLocation.BARN,
            0x28: SceneLocation.COOP,
        }
        for tilemap, location in expected.items():
            with self.subTest(tilemap=tilemap):
                ram = _ram()
                ram[field_spec("tilemap").address] = tilemap

                scene = classify_scene_from_ram(ram)

                self.assertEqual(scene.mode, SceneMode.NORMAL)
                self.assertEqual(scene.location, location)

    def test_classifies_town_event_stale_dialogue_as_normal(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x05
        ram[field_spec("input_lock").address] = 1
        _write_u16(ram, field_spec("dialog_text_id").address, 0x0316)
        ram[field_spec("dialog_text_mode").address] = 0x02
        _write_u16(ram, field_spec("player_x").address + _live_base(ram), 440)
        _write_u16(ram, field_spec("player_y").address + _live_base(ram), 160)

        scene = classify_scene_from_ram(ram)

        self.assertEqual(scene.mode, SceneMode.NORMAL)
        self.assertEqual(scene.location, SceneLocation.TOWN)

    def test_classifies_dialogue_menu_and_generic_input_lock(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x00
        ram[field_spec("input_lock").address] = 0
        _write_u16(ram, field_spec("dialog_text_id").address, 0x03A6)

        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.DIALOGUE)

        ram[field_spec("dialog_menu_cursor").address] = 2
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.MENU)

        _write_u16(ram, field_spec("dialog_text_id").address, 0)
        ram[field_spec("dialog_text_mode").address] = 0
        ram[field_spec("dialog_menu_cursor").address] = 0
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.INPUT_LOCKED)

        ram[field_spec("tilemap").address] = 0xFE
        _write_u16(ram, field_spec("dialog_text_id").address, 0x0315)
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.DIALOGUE)

        ram[field_spec("input_lock").address] = 1
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.DIALOGUE)

    def test_classifies_transition_sleep_ending_unknown_and_invalid(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x00
        ram[field_spec("player_state").address + _live_base(ram)] = 0x80
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.MAP_TRANSITION)

        ram = _ram()
        ram[field_spec("tilemap").address] = 0x0F
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.SLEEP_WAKE_TRANSITION)

        ram = _ram()
        ram[field_spec("tilemap").address] = 0x00
        ram[field_spec("ending_scene_index").address + _live_base(ram)] = 0x25
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.ENDING_CREDITS)

        ram = _ram()
        ram[field_spec("tilemap").address] = 0xFE
        scene = classify_scene_from_ram(ram)
        self.assertEqual(scene.mode, SceneMode.CUTSCENE_EVENT)
        self.assertTrue(scene.is_recoverable)
        self.assertTrue(scene.needs_input_dismiss)

        ram = _ram()
        ram[field_spec("tilemap").address] = 0x00
        _write_u16(ram, field_spec("player_x").address + _live_base(ram), 0)
        _write_u16(ram, field_spec("player_y").address + _live_base(ram), 0)
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.INVALID_COORDINATES)

    def test_classifies_festival_event_location(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x04
        ram[field_spec("weather_tomorrow").address + _live_base(ram)] = 6

        scene = classify_scene_from_ram(ram)

        self.assertEqual(scene.mode, SceneMode.NORMAL)
        self.assertEqual(scene.location, SceneLocation.FESTIVAL)

    def test_classifies_festival_dialogue_as_cutscene_event(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x04
        ram[field_spec("weather_tomorrow").address + _live_base(ram)] = 6
        _write_u16(ram, field_spec("dialog_text_id").address, 0x0316)
        ram[field_spec("dialog_text_mode").address] = 0x02

        scene = classify_scene_from_ram(ram)

        self.assertEqual(scene.mode, SceneMode.CUTSCENE_EVENT)
        self.assertEqual(scene.location, SceneLocation.FESTIVAL)
        self.assertTrue(scene.needs_input_dismiss)

    def test_morning_scene_ready_and_ending_helpers(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x15
        _write_u16(ram, field_spec("player_y").address + _live_base(ram), 120)
        ram[field_spec("hour").address + _live_base(ram)] = 6
        scene = classify_scene_from_ram(ram)
        self.assertTrue(morning_scene_ready(scene, hour=6))
        self.assertFalse(morning_scene_ready(scene, hour=14))

        ram[field_spec("ending_scene_index").address + _live_base(ram)] = 0x04
        ending = classify_scene_from_ram(ram)
        self.assertTrue(scene_indicates_ending(ending))
        self.assertFalse(morning_scene_ready(ending, hour=6))

    def test_classifies_existing_world_snapshot(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x27
        snapshot = WorldSnapshot.from_ram(ram, bounds=(0, 0, 1, 1))

        scene = classify_scene(snapshot)

        self.assertEqual(scene.mode, SceneMode.NORMAL)
        self.assertEqual(scene.location, SceneLocation.BARN)

    def test_pinned_morning_after_sleep_fixture_is_normal_house_scene(self) -> None:
        state = parse_save_state(resolve_state_path("Y1_After_Sleep"))

        scene = classify_scene_from_ram(state.ram)

        self.assertEqual(scene.mode, SceneMode.NORMAL)
        self.assertEqual(scene.location, SceneLocation.HOUSE)
        self.assertEqual(read_ram_value(state.ram, "season", raw=True), 0)
        self.assertEqual(read_ram_value(state.ram, "day", raw=True), 3)
        self.assertEqual(read_ram_value(state.ram, "hour", raw=True), 6)
        self.assertEqual(read_ram_value(state.ram, "minute", raw=True), 0)


if __name__ == "__main__":
    unittest.main()
