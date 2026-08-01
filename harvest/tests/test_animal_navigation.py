"""Unit coverage for shared animal-interior navigation helpers."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.tasks.animal_navigation import (
    align_to_pixel,
    fallback_action,
    find_path_around_blockers,
)
from harvest.tasks.farm_clearer import make_action


class _WalkablePathfinder:
    def __init__(self, blocked: set[tuple[int, int]] | None = None) -> None:
        self.blocked = blocked or set()
        self.current_positions: list[tuple[int, int]] = []

    def is_walkable(
        self, _ram: np.ndarray, x: int, y: int, *, current_pos: tuple[int, int]
    ) -> bool:
        self.current_positions.append(current_pos)
        return (x, y) not in self.blocked


class AnimalNavigationTests(unittest.TestCase):
    def test_path_detours_dynamic_blockers_without_mutating_snapshot(self) -> None:
        ram = np.zeros(1, dtype=np.uint8)
        blockers = {(1, 0)}
        pathfinder = _WalkablePathfinder()

        path = find_path_around_blockers(ram, pathfinder, (0, 0), (2, 0), blockers)

        self.assertEqual(path, [(0, 1), (1, 1), (2, 1), (2, 0)])
        self.assertEqual(blockers, {(1, 0)})
        self.assertEqual(set(pathfinder.current_positions), {(0, 0)})

    def test_path_rejects_goal_occupied_by_an_animal(self) -> None:
        path = find_path_around_blockers(
            np.zeros(1, dtype=np.uint8),
            _WalkablePathfinder(),
            (0, 0),
            (1, 0),
            {(1, 0)},
        )

        self.assertIsNone(path)

    def test_pixel_alignment_and_fallback_keep_existing_axis_priority(self) -> None:
        np.testing.assert_array_equal(
            align_to_pixel((10, 10), (14, 12)),
            make_action(right=True),
        )
        self.assertIsNone(align_to_pixel((10, 10), (11, 9), tolerance=1))
        np.testing.assert_array_equal(
            fallback_action((2, 2), (4, 3)),
            make_action(right=True, b=True),
        )


if __name__ == "__main__":
    unittest.main()
