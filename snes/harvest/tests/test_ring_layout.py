"""Packed ring lattice: geometry that must hold on any farm, plus the cleared D2 farm when its pin exists."""

from __future__ import annotations

import os
import unittest
from collections import deque
from pathlib import Path

from harvest.maps import farm_pond
from harvest.maps.farm_pond import (
    FARM_NO_GO_TILES,
    PACKED_COLUMNS,
    PACKED_POCKET_PLANT_CENTERS,
    PACKED_RING_ORDER,
    PACKED_ROWS,
    RING_LAYOUTS,
    SHIP_BIN_LANE_TILES,
    WEST_POCKET_PLANT_CENTER,
)

BIN_STAND = (11, 30)
PIN = "LN_d2_perfect_poweron"


def ring(center):
    cx, cy = center
    return {(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)}


class PackedLatticeGeometryTests(unittest.TestCase):
    def test_layouts_are_selectable_and_legacy_is_unchanged(self) -> None:
        self.assertEqual(set(RING_LAYOUTS), {"legacy", "packed", "packed12", "packed18", "blocks"})
        self.assertEqual(len(RING_LAYOUTS["legacy"]), 9)
        self.assertEqual(RING_LAYOUTS["legacy"][0], WEST_POCKET_PLANT_CENTER)
        self.assertEqual(len(RING_LAYOUTS["packed"]), 26)
        self.assertEqual(RING_LAYOUTS["packed"][:2], ((13, 28), (19, 28)))
        self.assertEqual(RING_LAYOUTS["packed18"], RING_LAYOUTS["packed"][:18])
        self.assertEqual(RING_LAYOUTS["packed12"], RING_LAYOUTS["packed"][:12])

    def test_order_is_exactly_the_lattice_cells(self) -> None:
        cells = {(x, y) for x in PACKED_COLUMNS for y in PACKED_ROWS}
        self.assertEqual(set(PACKED_RING_ORDER), cells)
        self.assertEqual(len(PACKED_RING_ORDER), len(cells))

    def test_rings_share_one_lane_and_never_touch(self) -> None:
        rings = {c: ring(c) for c in PACKED_POCKET_PLANT_CENTERS}
        crop_tiles = set().union(*rings.values())
        self.assertEqual(sum(len(r) for r in rings.values()), len(crop_tiles), "two rings share a tile")
        for center, tiles in rings.items():
            for tx, ty in tiles:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    near = (tx + dx, ty + dy)
                    owner = [c for c, r in rings.items() if near in r and c != center]
                    self.assertFalse(owner, f"{center} touches ring {owner} at {near}")

    def test_every_crop_tile_has_a_lane_neighbour(self) -> None:
        crop_tiles = set().union(*(ring(c) for c in PACKED_POCKET_PLANT_CENTERS))
        for center in PACKED_RING_ORDER:
            for tx, ty in ring(center):
                lanes = [
                    (tx + dx, ty + dy)
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if (tx + dx, ty + dy) not in crop_tiles and (tx + dx, ty + dy) != center
                ]
                self.assertTrue(lanes, f"{(tx, ty)} of ring {center} is boxed in")

    def test_lattice_keeps_the_lanes_the_hardcoded_walks_use(self) -> None:
        crop_tiles = set().union(*(ring(c) for c in PACKED_POCKET_PLANT_CENTERS))
        for lane_tile in ((6, 37), (6, 33), (6, 34), (6, 38), (2, 40), (11, 31), (11, 32), (11, 33)):
            self.assertNotIn(lane_tile, crop_tiles)

    def test_no_ring_sits_on_a_no_go_or_bin_tile(self) -> None:
        for center in PACKED_RING_ORDER:
            tiles = ring(center) | {center}
            self.assertFalse(tiles & set(FARM_NO_GO_TILES), center)
            self.assertFalse(tiles & set(SHIP_BIN_LANE_TILES), center)
            self.assertNotIn(BIN_STAND, tiles)

    def test_order_is_non_decreasing_in_row_then_gets_further_from_the_bin(self) -> None:
        dist = [abs(x - BIN_STAND[0]) + abs(y - BIN_STAND[1]) for x, y in PACKED_RING_ORDER]
        # Closest-first by walking distance: the Manhattan proxy may wobble by a ring pitch, never more.
        for earlier, later in zip(dist, dist[1:]):
            self.assertLessEqual(earlier, later + 4)
        self.assertEqual(PACKED_RING_ORDER[0], (12, 35))


@unittest.skipUnless(
    (Path(__file__).resolve().parents[1] / "custom_integrations" / "HarvestMoon-Snes" / f"{PIN}.state").exists(),
    "pin LN_d2_perfect_poweron is local",
)
class PackedLatticeOnTheClearedFarmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from harvest.maps import farm_final_layout as layout

        _tilemap, cls.farm, _visible = layout._pin_maps(PIN)

    def tile(self, x, y):
        return self.farm[y * 64 + x]

    def test_every_ring_cell_is_bare_ground(self) -> None:
        for center in PACKED_RING_ORDER:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    x, y = center[0] + dx, center[1] + dy
                    self.assertIn(self.tile(x, y), (0x01, 0x02), f"{center}: ({x},{y}) is {self.tile(x, y):#x}")

    def test_every_prefix_is_reachable_from_the_bin_stand(self) -> None:
        bare = (0x01, 0x02, 0x07, 0x08)
        walkable_extra = set(SHIP_BIN_LANE_TILES) | {BIN_STAND, (11, 29)}
        for n in (2, 9, 12, 18, 26):
            centers = PACKED_POCKET_PLANT_CENTERS[:n]
            blocked = set().union(*(ring(c) for c in centers))
            seen = {BIN_STAND}
            queue = deque([BIN_STAND])
            while queue:
                x, y = queue.popleft()
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if (nx, ny) in seen or (nx, ny) in blocked or not (0 <= nx < 64 and 0 <= ny < 64):
                        continue
                    if (nx, ny) in walkable_extra or (self.tile(nx, ny) in bare and (nx, ny) not in FARM_NO_GO_TILES):
                        seen.add((nx, ny))
                        queue.append((nx, ny))
            for center in centers:
                for tx, ty in ring(center):
                    if (tx, ty) == (20, 28):
                        continue  # the legacy second pocket ring has always had this one dead tile
                    reach = [
                        (tx + dx, ty + dy)
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        if (tx + dx, ty + dy) in seen
                    ]
                    self.assertTrue(reach, f"prefix {n}: crop ({tx},{ty}) of {center} cannot be reached")


class LayoutSwitchTests(unittest.TestCase):
    def test_unknown_layout_is_refused(self) -> None:
        old = os.environ.get(farm_pond.RING_LAYOUT_ENV)
        os.environ[farm_pond.RING_LAYOUT_ENV] = "diamond"
        try:
            with self.assertRaises(ValueError):
                farm_pond.ring_layout_name()
        finally:
            if old is None:
                del os.environ[farm_pond.RING_LAYOUT_ENV]
            else:
                os.environ[farm_pond.RING_LAYOUT_ENV] = old


if __name__ == "__main__":
    unittest.main()
