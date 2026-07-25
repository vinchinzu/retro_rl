from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

from harvest.core.harvest_state import HarvestStateDocument
from harvest.tools.livestock_builder import (
    FOUR_COW_MILK_FIXTURE,
    SHED_ITEMS_ROW_2_ADDR,
    SHED_ROW_2_BRUSH_BIT,
    SHED_ROW_2_MILKER_BIT,
    SHED_ROW_2_WATERING_CAN_BIT,
    build_compact_livestock_states,
    build_four_cow_milk_fixture_state,
    verify_state,
)
from harvest.core.tile_catalog import Tool


SCRIPT_DIR = Path(__file__).resolve().parents[1]
STATES_DIR = SCRIPT_DIR / "custom_integrations" / "HarvestMoon-Snes"


def _has_state(state_name: str) -> bool:
    return (STATES_DIR / f"{state_name}.state").exists()


@unittest.skipUnless(
    _has_state("Y1_After_Buy_Potato"),
    "Harvest Moon save states not available locally",
)
class LivestockBuilderTests(unittest.TestCase):
    def _build_states(self) -> dict[str, Path]:
        prefix = f"UT_Livestock_{uuid4().hex[:8]}"
        outputs = build_compact_livestock_states(
            base_state="Y1_After_Buy_Potato",
            prefix=prefix,
            money=5000,
            hay=99,
            chicken_feed=20,
            cow_feed=20,
        )
        for path in outputs.values():
            self.addCleanup(path.unlink, missing_ok=True)
        return outputs

    def test_build_compact_livestock_states_sets_expected_values(self) -> None:
        outputs = self._build_states()

        resources = HarvestStateDocument.load(outputs["resources"].stem)
        chicken = HarvestStateDocument.load(outputs["chicken"].stem)
        chicken_cow = HarvestStateDocument.load(outputs["chicken_cow"].stem)

        self.assertEqual(resources.scalar_value("money"), 5000)
        self.assertEqual(resources.scalar_value("stored_grass"), 99)
        self.assertEqual(resources.scalar_value("num_chickens"), 0)
        self.assertEqual(resources.scalar_value("num_cows"), 0)

        self.assertEqual(chicken.scalar_value("num_chickens"), 1)
        self.assertEqual(chicken.scalar_value("num_cows"), 0)
        chicken0 = chicken.chickens()[0]
        self.assertEqual(chicken0.status_raw, 0x09)
        self.assertEqual(chicken0.raw_1, 0x28)
        self.assertEqual(chicken0.position_x, 0x0018)
        self.assertEqual(chicken0.position_y, 0x0048)

        self.assertEqual(chicken_cow.scalar_value("num_chickens"), 1)
        self.assertEqual(chicken_cow.scalar_value("num_cows"), 1)
        cow0 = chicken_cow.cows()[0]
        self.assertEqual(cow0.status_raw, 0x05)
        self.assertEqual(cow0.home_map_raw, 0x27)
        self.assertEqual(cow0.position_x, 0x00A8)
        self.assertEqual(cow0.position_y, 0x0116)

    def test_verify_state_matches_emulator_loaded_snapshot(self) -> None:
        outputs = self._build_states()
        saved, loaded = verify_state(outputs["chicken_cow"].stem)
        self.assertEqual(saved.money, loaded.money)
        self.assertEqual(saved.hay, loaded.hay)
        self.assertEqual(saved.chickens, loaded.chickens)
        self.assertEqual(saved.cows, loaded.cows)
        self.assertEqual(saved.chicken_slot0, loaded.chicken_slot0)
        self.assertEqual(saved.cow_slot0, loaded.cow_slot0)

    def test_build_four_cow_milk_fixture_state_sets_known_cow_slots(self) -> None:
        state_name = f"UT_FourCow_{uuid4().hex[:8]}"
        path = build_four_cow_milk_fixture_state(
            base_state="Y1_After_Buy_Potato",
            output_name=state_name,
            money=7000,
            hay=88,
            cow_feed=12,
        )
        self.addCleanup(path.unlink, missing_ok=True)

        document = HarvestStateDocument.load(state_name)
        cows = document.cows()

        self.assertEqual(document.scalar_value("money"), 7000)
        self.assertEqual(document.scalar_value("stored_grass"), 88)
        self.assertEqual(document.scalar_value("cow_feed"), 12)
        self.assertEqual(document.scalar_value("fed_cows_n"), 0)
        self.assertEqual(document.scalar_value("fed_cows_flags"), 0)
        self.assertEqual(document.scalar_value("num_cows"), 4)
        self.assertEqual(document.scalar_value("tool_selected"), int(Tool.BRUSH))
        self.assertEqual(document.scalar_value("tool_backpack"), int(Tool.MILKER))
        shed_row_2 = document.mutable_state.read_u8(SHED_ITEMS_ROW_2_ADDR)
        self.assertTrue(shed_row_2 & SHED_ROW_2_WATERING_CAN_BIT)
        self.assertFalse(shed_row_2 & SHED_ROW_2_BRUSH_BIT)
        self.assertFalse(shed_row_2 & SHED_ROW_2_MILKER_BIT)

        for spec in FOUR_COW_MILK_FIXTURE:
            cow = cows[spec.slot]
            self.assertEqual(cow.status_raw, spec.status_raw)
            self.assertEqual(cow.raw_1, spec.raw_1)
            self.assertEqual(cow.home_map_raw, 0x27)
            self.assertEqual(cow.happiness, spec.happiness)
            self.assertEqual(cow.name_bytes, spec.name_bytes)
            self.assertEqual(cow.position_x, spec.position_tile[0] * 16 + 8)
            self.assertEqual(cow.position_y, spec.position_tile[1] * 16 + 8)

        self.assertEqual(
            len({cows[spec.slot].name_bytes for spec in FOUR_COW_MILK_FIXTURE}),
            len(FOUR_COW_MILK_FIXTURE),
        )
        self.assertTrue(all(cow.status_raw == 0 for cow in cows[4:]))

    def test_four_cow_fixture_restores_shed_animal_tools_when_not_carried(self) -> None:
        state_name = f"UT_FourCow_NoCarry_{uuid4().hex[:8]}"
        path = build_four_cow_milk_fixture_state(
            base_state="Y1_After_Buy_Potato",
            output_name=state_name,
            carry_animal_tools=False,
        )
        self.addCleanup(path.unlink, missing_ok=True)

        document = HarvestStateDocument.load(state_name)
        shed_row_2 = document.mutable_state.read_u8(SHED_ITEMS_ROW_2_ADDR)

        self.assertTrue(shed_row_2 & SHED_ROW_2_BRUSH_BIT)
        self.assertTrue(shed_row_2 & SHED_ROW_2_MILKER_BIT)


if __name__ == "__main__":
    unittest.main()
