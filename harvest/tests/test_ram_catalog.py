from __future__ import annotations

import unittest

import numpy as np

from harvest.core.ram_catalog import (
    LIVE_RAM_WRAM_OFFSET,
    LiveRamEditor,
    RamExpectation,
    check_expectations,
    field_spec,
    parse_ram_patch,
    read_ram_value,
)


class _DummyMemory:
    def __init__(self, ram: np.ndarray) -> None:
        self.ram = ram
        self.assignments: list[tuple[int, str, int]] = []

    def assign(self, addr: int, kind: str, value: int) -> None:
        self.assignments.append((addr, kind, value))
        self.ram[addr] = value & 0xFF


class _DummyData:
    def __init__(self, ram: np.ndarray) -> None:
        self.memory = _DummyMemory(ram)
        self.set_values: list[tuple[str, int]] = []

    def set_value(self, key: str, value: int) -> None:
        self.set_values.append((key, value))


class _DummyEnv:
    def __init__(self) -> None:
        self.ram = np.zeros(0x24000, dtype=np.uint8)
        self.data = _DummyData(self.ram)

    def get_ram(self) -> np.ndarray:
        return self.ram


class RamCatalogTests(unittest.TestCase):
    def test_aliases_resolve_to_canonical_specs(self) -> None:
        self.assertEqual(field_spec("weather").key, "weather_tomorrow")
        self.assertEqual(field_spec("gold").key, "money")
        self.assertEqual(field_spec("hay").key, "stored_grass")
        self.assertEqual(field_spec("item_in_hand").key, "tool_selected")
        self.assertEqual(field_spec("item_on_hand").key, "held_item")
        self.assertEqual(field_spec("child_flags").key, "incubator_flags")
        self.assertEqual(field_spec("family_status_flags").key, "incubator_flags")
        self.assertEqual(field_spec("global_happiness").key, "happiness")
        self.assertEqual(field_spec("ending_index").key, "ending_scene_index")
        self.assertEqual(field_spec("farm_development").key, "ranch_development")
        self.assertEqual(field_spec("ending_power_berries").key, "power_berry_count")
        self.assertEqual(field_spec("dog_pickups").key, "dog_hugs")
        self.assertEqual(field_spec("player_state_flags").key, "game_state")

    def test_read_ram_value_uses_live_wram_offset_and_money_scaling(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        addr = field_spec("money").address + LIVE_RAM_WRAM_OFFSET
        ram[addr] = 0xBC
        ram[addr + 1] = 0x02

        self.assertEqual(read_ram_value(ram, "money", raw=True), 700)
        self.assertEqual(read_ram_value(ram, "money"), 7000)

    def test_dialog_text_id_reads_low_wram_without_live_offset(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[field_spec("dialog_text_id").address] = 0xA6
        ram[field_spec("dialog_text_id").address + 1] = 0x03

        self.assertEqual(read_ram_value(ram, "dialog_text_id", raw=True), 0x03A6)

    def test_ending_evaluation_fields_read_from_live_wram(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[field_spec("happiness").address + LIVE_RAM_WRAM_OFFSET] = 0x84
        ram[field_spec("happiness").address + LIVE_RAM_WRAM_OFFSET + 1] = 0x03
        ram[field_spec("ending_scene_index").address + LIVE_RAM_WRAM_OFFSET] = 0x25
        ram[field_spec("power_berry_count").address + LIVE_RAM_WRAM_OFFSET] = 10
        ram[field_spec("shipped_corn").address + LIVE_RAM_WRAM_OFFSET] = 0xC8
        ram[field_spec("ranch_mastery").address + LIVE_RAM_WRAM_OFFSET] = 0xE7
        ram[field_spec("ranch_mastery").address + LIVE_RAM_WRAM_OFFSET + 1] = 0x03

        self.assertEqual(read_ram_value(ram, "happiness", raw=True), 900)
        self.assertEqual(read_ram_value(ram, "ending_scene_index", raw=True), 0x25)
        self.assertEqual(read_ram_value(ram, "power_berry_count", raw=True), 10)
        self.assertEqual(read_ram_value(ram, "shipped_corn", raw=True), 200)
        self.assertEqual(read_ram_value(ram, "ranch_mastery", raw=True), 999)

    def test_parse_ram_patch_accepts_weather_names_and_raw_suffix(self) -> None:
        self.assertEqual(parse_ram_patch("weather=rain").value, 1)
        patch = parse_ram_patch("money:raw=700")
        self.assertEqual(patch.field, "money")
        self.assertTrue(patch.raw)
        self.assertEqual(patch.value, 700)

    def test_live_ram_editor_writes_multibyte_display_values_to_live_offset(self) -> None:
        env = _DummyEnv()

        storage = LiveRamEditor(env).set_field("money", 7000)

        self.assertEqual(storage, 700)
        base = field_spec("money").address + LIVE_RAM_WRAM_OFFSET
        self.assertEqual(env.ram[base], 0xBC)
        self.assertEqual(env.ram[base + 1], 0x02)
        self.assertEqual(env.ram[base + 2], 0x00)

    def test_live_ram_editor_prefers_integration_key_for_simple_fields(self) -> None:
        env = _DummyEnv()

        storage = LiveRamEditor(env).set_field("day", 12)

        self.assertEqual(storage, 12)
        self.assertEqual(env.data.set_values, [("day", 12)])
        self.assertEqual(env.data.memory.assignments, [])

    def test_expectations_report_field_mismatches(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[field_spec("day").address + LIVE_RAM_WRAM_OFFSET] = 9

        failures = check_expectations(
            ram,
            [
                RamExpectation("day", 9, raw=True),
                RamExpectation("hour", 6, raw=True),
            ],
        )

        self.assertEqual(failures, ["hour expected 6 raw, got 0"])


if __name__ == "__main__":
    unittest.main()
