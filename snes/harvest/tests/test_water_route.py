"""plan_water_route: one tour, shared lane stands, every dry crop tile served."""

from __future__ import annotations

import unittest

from harvest.maps.farm_pond import PACKED_POCKET_PLANT_CENTERS
from harvest.tasks.crop_geometry import build_water_steps, plan_water_route, plot_tiles
from tests.crop_planter_test_helpers import blank_ram, set_tile

DRY_YOUNG = 0x54  # dry growing potato, walkable-looking id
WET_YOUNG = 0x55
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def field(centers, tid=DRY_YOUNG):
    ram = blank_ram()
    for y in range(3, 60):
        for x in range(2, 62):
            set_tile(ram, x, y, 0x01)
    for center in centers:
        for tx, ty in plot_tiles(center):
            set_tile(ram, tx, ty, tid)
    return ram


class WaterRouteTests(unittest.TestCase):
    centers = list(PACKED_POCKET_PLANT_CENTERS[2:8])

    def test_every_dry_tile_is_served_once_from_a_neighbouring_open_tile(self) -> None:
        ram = field(self.centers)
        route = plan_water_route(ram, self.centers, start_tile=(11, 30))
        targets = [t for t, _s, _f in route]
        wanted = {tile for c in self.centers for tile in plot_tiles(c)}
        self.assertEqual(sorted(targets), sorted(wanted))
        self.assertEqual(len(targets), len(set(targets)))
        for target, stand, face in route:
            dx, dy = DELTA[face]
            self.assertEqual((stand[0] + dx, stand[1] + dy), target, f"{stand} {face} does not face {target}")
            self.assertNotIn(stand, wanted, "a stand must be open ground, not a crop")

    def test_shared_lane_stand_waters_both_sides(self) -> None:
        ram = field(self.centers)
        route = plan_water_route(ram, self.centers, start_tile=(11, 30))
        by_stand = {}
        for target, stand, _face in route:
            by_stand.setdefault(stand, []).append(target)
        self.assertTrue(any(len(v) == 2 for v in by_stand.values()), "no stand serves two tiles")
        legacy = [s for c in self.centers for s in build_water_steps(ram, c, start_tile=(11, 30))]
        self.assertLess(len(by_stand), len({s for _t, s, _f in legacy}))

    def test_stands_of_one_tile_group_are_emitted_together(self) -> None:
        ram = field(self.centers)
        route = plan_water_route(ram, self.centers, start_tile=(11, 30))
        stands = [s for _t, s, _f in route]
        runs = [s for i, s in enumerate(stands) if i == 0 or stands[i - 1] != s]
        self.assertEqual(len(runs), len(set(runs)), "a stand was visited twice")

    def test_wet_and_skipped_tiles_are_left_alone(self) -> None:
        ram = field(self.centers)
        center = self.centers[0]
        tiles = plot_tiles(center)
        set_tile(ram, *tiles[0], WET_YOUNG)
        route = plan_water_route(ram, self.centers, start_tile=(11, 30), skip_tiles={tiles[1]})
        targets = {t for t, _s, _f in route}
        self.assertNotIn(tiles[0], targets)
        self.assertNotIn(tiles[1], targets)
        self.assertEqual(len(targets), 6 * 8 - 2)

    def test_tour_walks_far_less_than_the_per_ring_order(self) -> None:
        ram = field(PACKED_POCKET_PLANT_CENTERS[2:14])
        centers = list(PACKED_POCKET_PLANT_CENTERS[2:14])
        route = plan_water_route(ram, centers, start_tile=(11, 30))
        legacy = []
        for c in sorted(centers, key=lambda c: abs(c[0] - 11) + abs(c[1] - 30)):
            legacy += build_water_steps(ram, c, start_tile=(11, 30))

        def walk(steps):
            pos, total = (11, 30), 0
            for _t, stand, _f in steps:
                total += abs(stand[0] - pos[0]) + abs(stand[1] - pos[1])
                pos = stand
            return total

        self.assertEqual(len(route), len(legacy))
        self.assertLess(walk(route) * 2, walk(legacy))

    def test_nothing_dry_means_no_route(self) -> None:
        ram = field(self.centers, tid=WET_YOUNG)
        self.assertEqual(plan_water_route(ram, self.centers, start_tile=(11, 30)), [])


if __name__ == "__main__":
    unittest.main()
