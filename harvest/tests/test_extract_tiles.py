from __future__ import annotations

import unittest

import numpy as np

from extract_tiles import (
    FARM_COVERAGE_RECIPES,
    REQUIRED_FARM_TILE_IDS,
    camera_offset,
    run_extraction,
    structural_tile_error,
)


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


class StructuralValidationTests(unittest.TestCase):
    def test_structural_error_ignores_brightness_shift(self) -> None:
        base = np.zeros((16, 16, 3), dtype=np.uint8)
        base[:, :8] = 40
        shifted = np.clip(base.astype(np.int16) + 80, 0, 255).astype(np.uint8)
        self.assertLess(structural_tile_error(base, shifted), 0.05)

    def test_structural_error_detects_layout_change(self) -> None:
        left_right = np.zeros((16, 16, 3), dtype=np.uint8)
        left_right[:, :8] = 255
        top_bottom = np.zeros((16, 16, 3), dtype=np.uint8)
        top_bottom[:8, :] = 255
        self.assertGreater(structural_tile_error(left_right, top_bottom), 0.5)


if __name__ == "__main__":
    unittest.main()
