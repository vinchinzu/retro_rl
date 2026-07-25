from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import stable_retro as retro

from harvest.core.harvest_state import (
    CURRENT_MAP_ARRAY_ADDR,
    DAY_ORDINAL_ADDR,
    DOG_ENDING_PICKUPS_REQUIRED,
    FARM_MAP_ARRAY_ADDR,
    FULLY_GROWN_GRASS_TILE,
    HORSE_ADULT_AGE,
    HORSE_LONG_NAME_ADDR,
    HORSE_OWNED_FLAG,
    HORSE_PICKUP_SCENE_COMPLETE_FLAG,
    HORSE_SADDLEBAG_SCENE_COMPLETE_FLAG,
    HORSE_STREET_PREREQUISITE_FLAGS,
    HOUSE_UPGRADE_1_FLAG,
    HOUSE_UPGRADE_2_FLAG,
    KID_BIRTH_EVENT_FLAG,
    KID_EXISTS_FLAGS,
    KID_LONG_NAME_ADDRS,
    KID_GROWTH_EVENT_FLAGS,
    KID_STAGE_AGES,
    MAP_TILE_COUNT,
    SEASON_NAME_ADDR,
    WEEKDAY_NAME_ADDR,
    HarvestStateDocument,
    projected_farm_development_percent,
    weekday_for_date,
)
from harvest.runtime.rom_tools import MutableSaveState, SaveStateArchive, wram_offset


SCRIPT_DIR = Path(__file__).resolve().parents[1]
STATES_DIR = SCRIPT_DIR / "custom_integrations" / "HarvestMoon-Snes"


def _has_state(state_name: str) -> bool:
    return (STATES_DIR / f"{state_name}.state").exists()


class SaveStateArchiveTests(unittest.TestCase):
    def test_wram_offset_accepts_relative_and_absolute_addresses(self) -> None:
        self.assertEqual(wram_offset(0x0010), 0x0010)
        self.assertEqual(wram_offset(0x7E0010), 0x0010)
        self.assertEqual(wram_offset(0x7F1F1C), 0x11F1C)

    def test_mutable_save_state_round_trips_unknown_blocks(self) -> None:
        raw = b"#!s9xsnp\nRAM:4:\x00\x01\x02\x03VRA:2:\xAA\xBBFOO:3:xyz"
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "sample.state"
            dst = Path(tmpdir) / "patched.state"
            src.write_bytes(raw)

            state = MutableSaveState.load(src)
            state.write_u8(0x7E0001, 0x99)
            state.save(dst)

            archive = SaveStateArchive.load(dst)

        self.assertEqual(archive.require_block("RAM"), b"\x00\x99\x02\x03")
        self.assertEqual(archive.require_block("VRA"), b"\xAA\xBB")
        self.assertEqual(archive.require_block("FOO"), b"xyz")

    def test_archive_round_trip_preserves_zero_padded_size_fields(self) -> None:
        raw = b"#!s9xsnp:0009\nNAM:000010:MemoryROM\x00RAM:000004:\x00\x01\x02\x03"
        archive = SaveStateArchive.from_bytes(raw)
        self.assertEqual(archive.to_bytes(), raw)


class CalendarHelperTests(unittest.TestCase):
    def test_weekday_for_date_matches_game_rollover_epoch(self) -> None:
        self.assertEqual(weekday_for_date(game_year=1, season="spring", day=1), 1)
        self.assertEqual(weekday_for_date(game_year=1, season="summer", day=1), 3)
        self.assertEqual(weekday_for_date(game_year=2, season="summer", day=28), 3)


@unittest.skipUnless(
    _has_state("Y1_Watered_Planted_Test"),
    "Harvest Moon save states not available locally",
)
class HarvestStateDocumentIntegrationTests(unittest.TestCase):
    def test_document_reads_known_scalar_values(self) -> None:
        document = HarvestStateDocument.load("Y1_After_Buy_Potato")
        self.assertEqual(document.scalar_value("hour"), 9)
        self.assertEqual(document.scalar_value("minute"), 4)
        self.assertEqual(document.scalar_value("potato_seeds"), 1)
        self.assertEqual(document.scalar_value("money"), 10)

    def test_document_can_patch_farm_tile_and_save(self) -> None:
        document = HarvestStateDocument.load("Y1_Watered_Planted_Test")
        tile = document.farm_tile(26, 33)
        self.assertEqual(tile.persistent_value, 0x54)
        self.assertEqual(tile.visible_value, 0x55)

        document.set_farm_tile_value(26, 33, 0x02)
        patched = document.farm_tile(26, 33)
        self.assertEqual(patched.persistent_value, 0x02)
        self.assertEqual(patched.visible_value, 0x02)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "patched.state"
            written = document.save_as(out_path)
            reloaded = MutableSaveState.load(written)

        self.assertEqual(reloaded.read_u8(0x7EA4E6 + 33 * 64 + 26), 0x02)
        self.assertEqual(reloaded.read_u8(0x09B6 + 33 * 64 + 26), 0x02)

    def test_document_sets_calendar_clock_and_cached_date_labels(self) -> None:
        document = HarvestStateDocument.load("Y1_After_Buy_Potato")

        document.set_calendar_date(game_year=2, season="summer", day=28)
        document.set_clock(hour=13, minute=45, second=12, time_running=1)

        self.assertEqual(document.scalar_value("year"), 1)
        self.assertEqual(document.scalar_value("season"), 1)
        self.assertEqual(document.scalar_value("day"), 28)
        self.assertEqual(document.scalar_value("weekday"), weekday_for_date(game_year=2, season="summer", day=28))
        self.assertEqual(document.scalar_value("hour"), 13)
        self.assertEqual(document.scalar_value("minute"), 45)
        self.assertEqual(document.scalar_value("second"), 12)
        self.assertEqual(document.scalar_value("time_running"), 1)

        self.assertEqual(document.mutable_state.read_u16(SEASON_NAME_ADDR), 0x002C)
        self.assertEqual(document.mutable_state.read_u16(WEEKDAY_NAME_ADDR), 0x0030)
        self.assertEqual(document.mutable_state.read_u16(DAY_ORDINAL_ADDR), 0x0013)

    def test_document_sets_house_level_and_kid_growth_stages(self) -> None:
        document = HarvestStateDocument.load("Y1_After_Buy_Potato")

        document.set_house_level(2)
        document.set_kid_stage(1, "child", name_bytes=(0x01, 0x02, 0x03, 0x04))
        document.set_kid_stage(2, "grown", name_bytes=(0x05, 0x06, 0x07, 0x08))

        self.assertEqual(document.scalar_value("house_size"), 2)
        self.assertTrue(document.scalar_value("upgrade_flags") & HOUSE_UPGRADE_1_FLAG)
        self.assertTrue(document.scalar_value("upgrade_flags") & HOUSE_UPGRADE_2_FLAG)
        self.assertEqual(document.scalar_value("kid1_age"), KID_STAGE_AGES["child"])
        self.assertEqual(document.scalar_value("kid2_age"), KID_STAGE_AGES["grown"])

        status_flags = document.scalar_value("incubator_flags")
        self.assertTrue(status_flags & KID_EXISTS_FLAGS[1])
        self.assertTrue(status_flags & KID_EXISTS_FLAGS[2])
        self.assertFalse(status_flags & KID_BIRTH_EVENT_FLAG)
        self.assertFalse(document.scalar_value("family_event_flags") & KID_GROWTH_EVENT_FLAGS[1])
        self.assertFalse(document.scalar_value("family_event_flags") & KID_GROWTH_EVENT_FLAGS[2])

        kids = document.kids()
        self.assertEqual(kids[0].stage, "child")
        self.assertEqual(kids[0].name_bytes, (0x01, 0x02, 0x03, 0x04))
        self.assertEqual(kids[1].stage, "grown")
        self.assertEqual(kids[1].name_bytes, (0x05, 0x06, 0x07, 0x08))
        self.assertEqual(document.mutable_state.read_u16(KID_LONG_NAME_ADDRS[1]), 0x0001)

        document.clear_kids()
        self.assertEqual(document.scalar_value("kid1_age"), 0)
        self.assertEqual(document.scalar_value("kid2_age"), 0)
        self.assertFalse(document.scalar_value("incubator_flags") & KID_EXISTS_FLAGS[1])
        self.assertFalse(document.scalar_value("incubator_flags") & KID_EXISTS_FLAGS[2])
        self.assertEqual(document.kids()[0].stage, "absent")

    def test_document_sets_owned_adult_horse(self) -> None:
        document = HarvestStateDocument.load("Y1_After_Buy_Potato")

        document.set_horse_owned(name_bytes=(0x09, 0x0A, 0x0B, 0xB1))

        self.assertTrue(document.scalar_value("event_flags_1f68") & HORSE_OWNED_FLAG)
        self.assertEqual(
            document.scalar_value("event_flags_1f68") & HORSE_STREET_PREREQUISITE_FLAGS,
            HORSE_STREET_PREREQUISITE_FLAGS,
        )
        self.assertTrue(document.scalar_value("event_flags_1f68") & HORSE_PICKUP_SCENE_COMPLETE_FLAG)
        self.assertTrue(document.scalar_value("upgrade_flags") & HORSE_SADDLEBAG_SCENE_COMPLETE_FLAG)
        self.assertEqual(document.scalar_value("horse_map"), 0)
        self.assertEqual(document.scalar_value("horse_age"), HORSE_ADULT_AGE)
        self.assertEqual(document.mutable_state.read_u16(HORSE_LONG_NAME_ADDR), 0x0009)

    def test_document_fills_farm_ground_with_mature_grass(self) -> None:
        document = HarvestStateDocument.load("Y1_After_Buy_Potato")
        before = [document.mutable_state.read_u8(FARM_MAP_ARRAY_ADDR + index) for index in range(MAP_TILE_COUNT)]
        visible_before = [
            document.mutable_state.read_u8(CURRENT_MAP_ARRAY_ADDR + index) for index in range(MAP_TILE_COUNT)
        ]
        ground_count = sum(1 for value in before if value < 0xA0)
        expected_converted = sum(1 for value in before if value < 0xA0 and value != FULLY_GROWN_GRASS_TILE)

        result = document.fill_farm_ground_with_grass(update_visible_map=False)

        after = [document.mutable_state.read_u8(FARM_MAP_ARRAY_ADDR + index) for index in range(MAP_TILE_COUNT)]
        visible_after = [
            document.mutable_state.read_u8(CURRENT_MAP_ARRAY_ADDR + index) for index in range(MAP_TILE_COUNT)
        ]
        self.assertEqual(result.converted_tiles, expected_converted)
        self.assertEqual(result.grass_tiles, ground_count)
        self.assertEqual(result.development_tiles, ground_count)
        self.assertEqual(result.projected_development_percent, projected_farm_development_percent(ground_count))
        self.assertEqual(document.scalar_value("planted_grass"), ground_count)
        self.assertEqual(document.scalar_value("development_rate"), result.projected_development_percent)
        self.assertEqual(document.scalar_value("ranch_development"), result.projected_development_percent)
        self.assertEqual(visible_after, visible_before)
        for before_value, after_value in zip(before, after):
            if before_value < 0xA0:
                self.assertEqual(after_value, FULLY_GROWN_GRASS_TILE)
            else:
                self.assertEqual(after_value, before_value)

    def test_document_sets_dog_pickup_counter(self) -> None:
        document = HarvestStateDocument.load("Y1_After_Buy_Potato")

        document.set_dog_pickups(DOG_ENDING_PICKUPS_REQUIRED)

        self.assertEqual(document.scalar_value("dog_hugs"), DOG_ENDING_PICKUPS_REQUIRED)

    def test_saved_state_loads_patched_money_into_live_emulator_wram(self) -> None:
        retro.data.Integrations.add_custom_path(str((SCRIPT_DIR / "custom_integrations").resolve()))
        document = HarvestStateDocument.load("Y1_After_Buy_Potato")
        document.set_scalar_value("money", 43210)

        out_path = STATES_DIR / "_tmp_money_live_check.state"
        self.addCleanup(out_path.unlink, missing_ok=True)
        document.save_as(out_path)

        env = retro.make(
            game="HarvestMoon-Snes",
            state=out_path.stem,
            inttype=retro.data.Integrations.ALL,
            use_restricted_actions=retro.Actions.ALL,
            render_mode="rgb_array",
        )
        self.addCleanup(env.close)
        env.reset()
        ram = env.data.memory.blocks[0x7E0000]
        live_money = ram[0x11F04] | (ram[0x11F05] << 8) | (ram[0x11F06] << 16)

        self.assertEqual(live_money, 43210)


if __name__ == "__main__":
    unittest.main()
