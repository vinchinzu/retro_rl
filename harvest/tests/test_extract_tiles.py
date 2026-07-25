from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage

from harvest.maps.extract_tiles import (
    camera_offset,
    mean_rgb_error,
    save_rgb_image,
    structural_tile_error,
)


class CameraOffsetTests(unittest.TestCase):
    def test_camera_offset_clamps_left_and_top_edges(self) -> None:
        self.assertEqual(camera_offset(93, 448), (0, 336))
        self.assertEqual(camera_offset(983, 52), (768, 0))

    def test_camera_offset_tracks_centered_positions(self) -> None:
        self.assertEqual(camera_offset(419, 536), (291, 424))
        self.assertEqual(camera_offset(121, 565), (0, 453))


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

    def test_save_rgb_image_writes_png(self) -> None:
        img = np.zeros((6, 8, 3), dtype=np.uint8)
        img[:, :, 1] = 200
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.png"
            save_rgb_image(path, img)
            loaded = QImage(str(path))

        self.assertTrue(path.name.endswith(".png"))
        self.assertFalse(loaded.isNull())
        self.assertEqual((loaded.width(), loaded.height()), (8, 6))

    def test_mean_rgb_error_zero_for_identical(self) -> None:
        a = np.full((16, 16, 3), 128, dtype=np.uint8)
        self.assertAlmostEqual(mean_rgb_error(a, a), 0.0)

    def test_mean_rgb_error_nonzero_for_different(self) -> None:
        a = np.zeros((16, 16, 3), dtype=np.uint8)
        b = np.full((16, 16, 3), 100, dtype=np.uint8)
        self.assertAlmostEqual(mean_rgb_error(a, b), 100.0)


if __name__ == "__main__":
    unittest.main()
