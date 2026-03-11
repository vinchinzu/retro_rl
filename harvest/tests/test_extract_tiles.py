from __future__ import annotations

import unittest

from extract_tiles import FARM_COVERAGE_RECIPES, REQUIRED_FARM_TILE_IDS, camera_offset, run_extraction


class CameraOffsetTests(unittest.TestCase):
    def test_camera_offset_clamps_left_and_top_edges(self) -> None:
        self.assertEqual(camera_offset(93, 448), (0, 336))
        self.assertEqual(camera_offset(983, 52), (768, 0))

    def test_camera_offset_tracks_centered_positions(self) -> None:
        self.assertEqual(camera_offset(419, 536), (291, 424))
        self.assertEqual(camera_offset(121, 565), (0, 453))


class FarmCoverageRecipeTests(unittest.TestCase):
    def test_farm_coverage_recipes_capture_missing_tiles(self) -> None:
        atlas = run_extraction(recipes=FARM_COVERAGE_RECIPES, load_existing=False)
        self.assertTrue(REQUIRED_FARM_TILE_IDS.issubset(set(atlas)))


if __name__ == "__main__":
    unittest.main()
