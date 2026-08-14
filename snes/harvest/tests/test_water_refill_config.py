"""Unit tests for named pond corridor config + skills."""

from __future__ import annotations

import unittest

from harvest.maps.map_config import (
    FARM_MAIN_POND_STANDS,
    FARM_POND_REFILL_CORRIDOR,
    farm_pond_refill_primary_stand,
    player_in_west_plant_pocket,
    WEST_PLANT_POCKET_BOUNDS,
    WEST_POCKET_PLANT_CENTER,
)
from harvest.tasks.skills import farm_nav_to_pond_refill_skill, farm_pond_refill_face


class PondCorridorConfigTests(unittest.TestCase):
    def test_primary_stand_is_south_lip(self) -> None:
        stand, face = farm_pond_refill_primary_stand()
        self.assertEqual(stand, (32, 34))
        self.assertEqual(face, "up")
        self.assertEqual(FARM_MAIN_POND_STANDS[0], (stand, face))

    def test_corridor_steps_named(self) -> None:
        self.assertIn("stage_west_of_fence", FARM_POND_REFILL_CORRIDOR)
        self.assertIn("open_fence_row_y31", FARM_POND_REFILL_CORRIDOR)
        self.assertIn("fill_at_main_pond", FARM_POND_REFILL_CORRIDOR)

    def test_west_pocket_predicate(self) -> None:
        self.assertTrue(player_in_west_plant_pocket((13, 27)))
        self.assertTrue(player_in_west_plant_pocket((12, 29)))
        self.assertFalse(player_in_west_plant_pocket((32, 34)))
        self.assertFalse(player_in_west_plant_pocket((10, 40)))
        x0, y0, x1, y1 = WEST_PLANT_POCKET_BOUNDS
        self.assertEqual((x0, y0, x1, y1), (3, 14, 28, 30))
        self.assertTrue(x0 <= 15 <= x1 and y0 <= 29 <= y1)
        self.assertEqual(WEST_POCKET_PLANT_CENTER, (13, 28))
        self.assertTrue(x0 <= 13 <= x1 and y0 <= 28 <= y1)

    def test_pond_nav_skill_targets_primary_stand(self) -> None:
        skill = farm_nav_to_pond_refill_skill()
        self.assertEqual(skill.target_px, (32 * 16 + 8, 34 * 16 + 8))
        self.assertEqual(farm_pond_refill_face(), "up")


if __name__ == "__main__":
    unittest.main()
