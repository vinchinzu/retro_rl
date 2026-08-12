from __future__ import annotations

import unittest

import numpy as np

from harvest.core.tile_catalog import (
    ADDR_MAP,
    ADDR_TILEMAP,
)
from harvest.tasks.nav import (
    MAP_WIDTH,
    Pathfinder,
    WALKABLE_TILES,
)
from harvest.tasks.farm_clearer import TileScanner
from harvest.maps.map_config import (
    FARM_WALKABLE,
    ROUTES,
    find_landmark,
    get_landmarks,
    get_map_name,
    get_no_go_tiles,
    get_walkable_tiles,
)
from harvest.core.tile_catalog import CHURCH_WALKABLE, FARM_WALKABLE as CATALOG_FARM_WALKABLE


class FarmWalkableTests(unittest.TestCase):
    def test_farm_walkables_include_a1_path_tile(self) -> None:
        self.assertIn(0xA1, FARM_WALKABLE)
        self.assertIn(0xA1, WALKABLE_TILES)

    def test_map_and_runtime_farm_walkables_match(self) -> None:
        self.assertEqual(set(FARM_WALKABLE), set(WALKABLE_TILES))

    def test_farm_walkables_are_shared_from_tile_catalog(self) -> None:
        self.assertIs(FARM_WALKABLE, CATALOG_FARM_WALKABLE)
        self.assertIs(WALKABLE_TILES, CATALOG_FARM_WALKABLE)

    def test_seasonal_farm_tilemaps_share_farm_metadata(self) -> None:
        for tilemap in (0x00, 0x01, 0x02, 0x03):
            with self.subTest(tilemap=tilemap):
                self.assertIs(get_walkable_tiles(tilemap), FARM_WALKABLE)
                self.assertIn((17, 27), get_no_go_tiles(tilemap))
                self.assertIn("shipping_bin", {landmark.name for landmark in get_landmarks(tilemap)})

        self.assertEqual(get_map_name(0x01), "farm_summer")

    def test_town_explore_route_spans_farm_path_town(self) -> None:
        route = ROUTES["town_explore"]
        tilemaps = {wp.tilemap for wp in route}
        self.assertIn(0x00, tilemaps)
        self.assertIn(0x0C, tilemaps)
        self.assertIn(0x04, tilemaps)
        self.assertTrue(any(wp.is_exit for wp in route))
        # Last hop returns toward farm via path east exit.
        self.assertEqual(route[-1].tilemap, 0x0C)
        self.assertTrue(route[-1].is_exit)

    def test_berry_routes_start_south_and_repeat_from_bin(self) -> None:
        first = ROUTES["berry_ship"]
        repeat = ROUTES["berry_ship_repeat"]

        self.assertGreaterEqual(first[0].target_px[1] // 16, 35)
        self.assertEqual(first[0].target_px[0] // 16, 27)
        self.assertEqual(repeat[0].target_px, (55 * 16 + 8, 60 * 16 + 8))
        self.assertEqual(first[-1].target_px, (61 * 16 + 8, 60 * 16 + 8))
        self.assertEqual(repeat[-1].target_px, (61 * 16 + 8, 60 * 16 + 8))
        weed_gate_actions = [
            wp
            for wp in first
            if wp.action_on_arrive == "press_a"
            and wp.action_face == "up"
            and wp.target_px[0] // 16 == 37
        ]
        self.assertEqual(
            [wp.target_px[1] // 16 for wp in weed_gate_actions],
            [60, 59],
        )

    def test_south_farm_return_crosses_fence_at_west_end(self) -> None:
        route = ROUTES["farm_south_to_west_gate"]
        first_north = next(wp for wp in route if wp.target_px[1] < 31 * 16)

        self.assertLess(first_north.target_px[0] // 16, 11)
        self.assertEqual(route[-1].target_px, (40, 424))

    def test_map_registry_has_named_landmarks(self) -> None:
        self.assertEqual(get_map_name(0x15), "house")
        self.assertEqual(get_map_name(0x16), "house_level1")
        self.assertEqual(get_map_name(0x17), "house_level2")
        self.assertEqual(get_map_name(0x26), "shed")
        self.assertEqual(get_map_name(0x27), "barn")
        self.assertEqual(get_map_name(0x10), "mountain_spring")
        self.assertEqual(get_map_name(0x1B), "church")
        farm_landmarks = {landmark.name for landmark in get_landmarks(0x00)}
        self.assertIn("shipping_bin", farm_landmarks)
        self.assertIn("pond_edge", farm_landmarks)
        self.assertIn("pond_edge_north", farm_landmarks)
        self.assertIn("shipping_ditch", farm_landmarks)
        self.assertIn("north_stream", farm_landmarks)
        self.assertIn("house_door", farm_landmarks)
        self.assertIn("shed_door", farm_landmarks)
        house_door = next(lm for lm in get_landmarks(0x00) if lm.name == "house_door")
        shed_door = next(lm for lm in get_landmarks(0x00) if lm.name == "shed_door")
        pond_edge = next(lm for lm in get_landmarks(0x00) if lm.name == "pond_edge")
        self.assertEqual(house_door.tile, (8, 26))
        self.assertEqual(shed_door.tile, (26, 30))
        # Main F0 pond south lip (not the non-fill shipping F2 ditch).
        self.assertEqual(pond_edge.tile, (32, 34))
        self.assertEqual(pond_edge.face, "up")
        self.assertEqual(pond_edge.kind, "water_source")
        house_landmarks = {landmark.name for landmark in get_landmarks(0x15)}
        self.assertIn("bed_stand", house_landmarks)

        found_l2_bed = find_landmark("bed_stand", tilemap_id=0x17)
        self.assertIsNotNone(found_l2_bed)
        _tilemap, l2_bed = found_l2_bed
        self.assertEqual(l2_bed.tile, (18, 6))

        found_coop_bin = find_landmark("egg_shipping_bin", tilemap_id=0x28)
        self.assertIsNotNone(found_coop_bin)
        _tilemap, coop_bin = found_coop_bin
        self.assertEqual(coop_bin.tile, (1, 10))
        self.assertEqual(coop_bin.face, "down")

        found = find_landmark("west_stump", tilemap_id=0x10)
        self.assertIsNotNone(found)
        tilemap, landmark = found
        self.assertEqual(tilemap, 0x10)
        self.assertEqual(landmark.tile, (20, 25))

        found_cast_spot = find_landmark("fish_power_berry_cast_spot", tilemap_id=0x10)
        self.assertIsNotNone(found_cast_spot)
        _tilemap, cast_spot = found_cast_spot
        self.assertEqual(cast_spot.tile, (39, 23))
        self.assertEqual(cast_spot.face, "right")

        found_throw_spot = find_landmark("fish_power_berry_throw_spot", tilemap_id=0x10)
        self.assertIsNotNone(found_throw_spot)
        _tilemap, throw_spot = found_throw_spot
        self.assertEqual(throw_spot.tile, (42, 25))
        self.assertEqual(throw_spot.face, "up")

        found_church_door = find_landmark("church_door", tilemap_id=0x04)
        self.assertIsNotNone(found_church_door)
        _tilemap, church_door = found_church_door
        self.assertEqual(church_door.tile, (23, 8))
        self.assertEqual(church_door.face, "up")

        found_ann = find_landmark("church_ann_question_stand", tilemap_id=0x1B)
        self.assertIsNotNone(found_ann)
        _tilemap, ann_stand = found_ann
        self.assertEqual(ann_stand.tile, (12, 25))
        self.assertEqual(ann_stand.face, "right")
        self.assertIn("Ann hearts +4", ann_stand.note)

        found_maria = find_landmark("church_maria_question_stand", tilemap_id=0x1B)
        self.assertIsNotNone(found_maria)
        _tilemap, maria_stand = found_maria
        self.assertEqual(maria_stand.tile, (12, 8))
        self.assertEqual(maria_stand.face, "left")
        self.assertIn("Maria hearts +4", maria_stand.note)

        found_barn_feed = find_landmark("cow_feed_trough", tilemap_id=0x27)
        self.assertIsNotNone(found_barn_feed)
        _tilemap, barn_feed = found_barn_feed
        self.assertEqual(barn_feed.tile, (7, 17))
        self.assertEqual(barn_feed.face, "right")

    def test_buy_cow_routes_are_chunked(self) -> None:
        self.assertIn("farm_to_animal_shop_counter", ROUTES)
        self.assertIn("farm_to_animal_shop_staging", ROUTES)
        self.assertIn("animal_shop_to_town", ROUTES)
        self.assertIn("farm_to_barn", ROUTES)
        self.assertIn("farm_to_house", ROUTES)
        self.assertIn("farm_to_house_level1", ROUTES)
        self.assertIn("farm_to_house_level2", ROUTES)
        self.assertEqual(ROUTES["farm_to_house"][-1].target_px, (136, 424))
        self.assertEqual(ROUTES["farm_to_house_level1"][-1].target_px, (136, 344))
        self.assertEqual(ROUTES["farm_to_animal_shop_staging"][-1].tilemap, 0x24)
        self.assertEqual(ROUTES["farm_to_animal_shop_staging"][-1].target_px, (128, 200))
        self.assertEqual(ROUTES["farm_to_animal_shop_counter"][-1].tilemap, 0x24)
        self.assertEqual(ROUTES["farm_to_animal_shop_counter"][-1].target_px, (201, 158))
        self.assertEqual(ROUTES["farm_to_barn"][-1].target_px, (329, 360))

    def test_mountain_routes_are_chunked(self) -> None:
        self.assertIn("farm_to_mountain", ROUTES)
        self.assertIn("mountain_entry_to_fish_power_berry_spots", ROUTES)
        self.assertIn("farm_to_spa", ROUTES)
        self.assertIn("mountain_entry_to_outdoor_spa", ROUTES)
        self.assertEqual(ROUTES["farm_to_mountain"][0].tilemap, 0x00)
        self.assertTrue(ROUTES["farm_to_mountain"][-1].is_exit)
        self.assertEqual(ROUTES["farm_to_mountain"][-1].target_px, (132, 30))
        self.assertEqual(ROUTES["mountain_entry_to_fish_power_berry_spots"][-1].target_px, (686, 411))
        self.assertEqual(ROUTES["farm_to_spa"][-1].target_px, (619, 201))
        self.assertEqual(ROUTES["mountain_entry_to_outdoor_spa"][-1].target_px, (619, 201))
        # Return path is reverse corridor + exit, not a single south hop.
        self.assertGreaterEqual(len(ROUTES["mountain_to_farm"]), 10)

    def test_church_routes_are_chunked(self) -> None:
        self.assertIn("farm_to_church", ROUTES)
        self.assertIn("church_sunday_talk_loop", ROUTES)
        self.assertIn("church_to_farm", ROUTES)
        self.assertEqual(ROUTES["farm_to_church"][-1].target_px, (375, 139))
        self.assertTrue(ROUTES["farm_to_church"][-1].is_exit)
        self.assertEqual(ROUTES["church_sunday_talk_loop"][1].target_px, (203, 409))
        self.assertEqual(ROUTES["church_sunday_talk_loop"][-1].target_px, (141, 139))
        self.assertEqual(ROUTES["church_to_farm"][0].tilemap, 0x1B)
        self.assertTrue(ROUTES["church_to_farm"][0].is_exit)

    def test_church_walkables_include_recorded_aisle_tiles(self) -> None:
        self.assertIn(0xDB, CHURCH_WALKABLE)
        self.assertIn(0xD5, CHURCH_WALKABLE)

    def test_farm_no_go_tiles_include_well_body(self) -> None:
        no_go = get_no_go_tiles(0x00)
        self.assertIn((17, 27), no_go)
        self.assertIn((15, 26), no_go)

    def test_pathfinder_blocks_coordinate_specific_well_tiles(self) -> None:
        ram = np.zeros(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_MAP + 27 * MAP_WIDTH + 17] = 0xA1

        pathfinder = Pathfinder(TileScanner())

        self.assertFalse(pathfinder.is_walkable(ram, 17, 27))

    def test_default_pathfinder_uses_current_tilemap_walkability(self) -> None:
        ram = np.zeros(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x10
        ram[ADDR_MAP + 5 * MAP_WIDTH + 5] = 0xC3  # mountain walkable, not farm walkable

        pathfinder = Pathfinder(TileScanner())

        self.assertTrue(pathfinder.is_walkable(ram, 5, 5))


if __name__ == "__main__":
    unittest.main()
