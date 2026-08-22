"""Unit tests for hot-spring stamina refill task (no ROM)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    ADDR_X,
    ADDR_Y,
    ADDR_INPUT_LOCK,
)
from harvest.core.tile_catalog import MOUNTAIN_WALKABLE
from harvest.maps.map_config import (
    ROUTES,
    farm_to_spa_waypoints,
    farm_to_west_gate_waypoints,
    slice_route_from_position,
)
from harvest.tasks.hot_spring import (
    HotSpringStaminaTask,
    SPA_TILEMAP,
    MOUNTAIN_TILEMAP,
    CAVE_TILEMAP,
    SPA_OUTDOOR_STAND_PX,
    near_outdoor_spa,
    read_stamina,
    read_max_stamina,
)
from retro_harness import TaskStatus


ADDR_STAMINA = field_spec("stamina").address
ADDR_MAX_STAMINA = field_spec("max_stamina").address


def _blank_ram() -> np.ndarray:
    return np.zeros(0x24000, dtype=np.uint8)


def _world(ram: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(ram=ram, info={}, obs=None)


class HotSpringUnitTests(unittest.TestCase):
    def test_spa_tilemap_is_outdoor_mountain(self) -> None:
        """Hot spring stays on mountain 0x10 — not cave 0x29."""
        self.assertEqual(SPA_TILEMAP, MOUNTAIN_TILEMAP)
        self.assertEqual(MOUNTAIN_TILEMAP, 0x10)
        self.assertEqual(CAVE_TILEMAP, 0x29)
        self.assertNotEqual(SPA_TILEMAP, CAVE_TILEMAP)

    def test_read_stamina_helpers(self) -> None:
        ram = _blank_ram()
        ram[ADDR_STAMINA] = 42
        ram[ADDR_MAX_STAMINA] = 100
        self.assertEqual(read_stamina(ram), 42)
        self.assertEqual(read_max_stamina(ram), 100)

    def test_already_full_on_farm_succeeds(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_STAMINA] = 90
        ram[ADDR_MAX_STAMINA] = 100
        ram[ADDR_INPUT_LOCK] = 1
        task = HotSpringStaminaTask(min_stamina=40)
        task.reset(_world(ram))

        result = task.step(_world(ram))

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("already sufficient", result.reason or "")

    def test_low_stamina_on_farm_starts_route(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_STAMINA] = 5
        ram[ADDR_MAX_STAMINA] = 100
        ram[ADDR_INPUT_LOCK] = 1
        # player_x/y are u16 little-endian
        ram[ADDR_X] = 136 & 0xFF
        ram[ADDR_X + 1] = (136 >> 8) & 0xFF
        ram[ADDR_Y] = 420 & 0xFF
        ram[ADDR_Y + 1] = (420 >> 8) & 0xFF
        task = HotSpringStaminaTask(min_stamina=40)
        task.reset(_world(ram))

        result = task.step(_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.phase_text, "route_mountain")

    def test_mountain_begins_route_or_soak_when_low(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = MOUNTAIN_TILEMAP
        ram[ADDR_STAMINA] = 5
        ram[ADDR_MAX_STAMINA] = 100
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_X] = 120
        ram[ADDR_Y] = 128
        task = HotSpringStaminaTask(min_stamina=40)
        task.reset(_world(ram))

        result = task.step(_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        # Either multi-nav to pond or immediate soak if route missing.
        self.assertIn(task.phase_text, {"route_spa", "soak"})

    def test_soak_queues_b_a_plus_direction(self) -> None:
        """Human bath uses B+A+Right/Left into 0xF7 (not A-alone / B-alone)."""
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = MOUNTAIN_TILEMAP
        ram[ADDR_STAMINA] = 5
        ram[ADDR_MAX_STAMINA] = 100
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_X] = 619 & 0xFF
        ram[ADDR_X + 1] = (619 >> 8) & 0xFF
        ram[ADDR_Y] = 201 & 0xFF
        ram[ADDR_Y + 1] = (201 >> 8) & 0xFF
        task = HotSpringStaminaTask(min_stamina=40, max_jump_cycles=4)
        task.reset(_world(ram))
        task._begin_soak(_world(ram))
        self.assertEqual(task.phase_text, "soak")

        saw_combo = False
        for _ in range(500):
            result = task.step(_world(ram))
            self.assertEqual(result.status, TaskStatus.RUNNING)
            action = getattr(result.action, "action", None)
            if action is not None:
                # SNES: B=0, A=8, Right=7, Left=6
                b = int(action[0]) == 1
                a = int(action[8]) == 1
                right = int(action[7]) == 1
                left = int(action[6]) == 1
                if a and b and (right or left):
                    saw_combo = True
                    break
        self.assertTrue(
            saw_combo, "expected B+A held with left/right into water (human bath)"
        )
        self.assertGreaterEqual(task._jump_cycles, 1)

    def test_soak_finishes_when_target_reached(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = MOUNTAIN_TILEMAP
        ram[ADDR_STAMINA] = 5
        ram[ADDR_MAX_STAMINA] = 100
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_X] = 120
        ram[ADDR_Y] = 128
        task = HotSpringStaminaTask(
            min_stamina=40, soak_plateau_frames=5, return_to_farm=False
        )
        task.reset(_world(ram))
        task._begin_soak(_world(ram))
        self.assertEqual(task.phase_text, "soak")

        # Simulate restore during soak.
        ram[ADDR_STAMINA] = 50
        result = None
        for _ in range(40):
            result = task.step(_world(ram))
            if result.status == TaskStatus.SUCCESS:
                break

        self.assertIsNotNone(result)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("soaked", result.reason or "")

    def test_mountain_with_full_stamina_returns_or_finishes(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = MOUNTAIN_TILEMAP
        ram[ADDR_STAMINA] = 80
        ram[ADDR_MAX_STAMINA] = 100
        ram[ADDR_INPUT_LOCK] = 1
        task = HotSpringStaminaTask(min_stamina=40, return_to_farm=False)
        task.reset(_world(ram))

        result = task.step(_world(ram))

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("sufficient", result.reason or "")

    def test_cave_map_exits_not_soaks(self) -> None:
        """If we land in 0x29 cave, exit to mountain — do not soak there."""
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = CAVE_TILEMAP
        ram[ADDR_STAMINA] = 5
        ram[ADDR_MAX_STAMINA] = 100
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_X] = 120
        ram[ADDR_Y] = 128
        task = HotSpringStaminaTask(min_stamina=40, return_to_farm=False)
        task.reset(_world(ram))

        result = task.step(_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.phase_text, "exit_cave")

    def test_near_outdoor_spa_begins_soak(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = MOUNTAIN_TILEMAP
        ram[ADDR_STAMINA] = 5
        ram[ADDR_MAX_STAMINA] = 100
        ram[ADDR_INPUT_LOCK] = 1
        sx, sy = SPA_OUTDOOR_STAND_PX
        ram[ADDR_X] = sx & 0xFF
        ram[ADDR_X + 1] = (sx >> 8) & 0xFF
        ram[ADDR_Y] = sy & 0xFF
        ram[ADDR_Y + 1] = (sy >> 8) & 0xFF
        self.assertTrue(near_outdoor_spa(ram))
        task = HotSpringStaminaTask(min_stamina=40, return_to_farm=False)
        task.reset(_world(ram))
        result = task.step(_world(ram))
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.phase_text, "soak")

    def test_slice_route_from_fish_skips_entry(self) -> None:
        full = ROUTES["fish_spot_to_outdoor_spa"]
        # Fish / camp stand ~(686, 411)
        sliced = slice_route_from_position(full, 686, 411, tilemap=0x10)
        self.assertLessEqual(len(sliced), len(full))
        self.assertEqual(sliced[-1].target_px, (619, 201))
        # First hop should be near fish corridor, not south entry y~718
        self.assertLess(sliced[0].target_px[1], 600)

    def test_spa_corridor_stays_off_fish_pond(self) -> None:
        """Farm→spa uses grape dirt; y>280 hops stay west of camp/Gotz."""
        spa = ROUTES["mountain_entry_to_outdoor_spa"]
        ret = ROUTES["mountain_to_farm"]
        self.assertEqual(spa[0].target_px, (328, 728))
        self.assertIn((72, 368), [wp.target_px for wp in spa])
        self.assertNotIn((686, 430), [wp.target_px for wp in spa])
        self.assertNotIn((620, 488), [wp.target_px for wp in ret])
        for wp in spa:
            if wp.tilemap == 0x10 and wp.target_px[1] > 280:
                self.assertLess(
                    wp.target_px[0],
                    560,
                    f"spa hop {wp.target_px} is in the fish/camp pocket",
                )
        for wp in ret:
            if wp.tilemap == 0x10 and wp.target_px[1] > 280:
                self.assertLess(
                    wp.target_px[0],
                    560,
                    f"return hop {wp.target_px} is in the fish/camp pocket",
                )

    def test_mountain_walkable_includes_path_edge(self) -> None:
        """0xA7 is a common mountain path-edge stand tile (sunday + chop)."""
        self.assertIn(0xA0, MOUNTAIN_WALKABLE)
        self.assertIn(0xA7, MOUNTAIN_WALKABLE)
        self.assertIn(0xA8, MOUNTAIN_WALKABLE)
        self.assertNotIn(0xFF, MOUNTAIN_WALKABLE)
        self.assertNotIn(0xF7, MOUNTAIN_WALKABLE)

    def test_spa_routes_share_lip_and_return_is_long(self) -> None:
        spa = ROUTES["mountain_entry_to_outdoor_spa"]
        farm = ROUTES["farm_to_spa"]
        ret = ROUTES["mountain_to_farm"]
        self.assertEqual(spa[-1].target_px, (619, 201))
        self.assertEqual(farm[-1].target_px, (619, 201))
        self.assertGreaterEqual(len(ret), 10)
        self.assertTrue(ret[-1].tilemap in (0x00, 0x0C) or ret[-2].is_exit)
        # Return walks path plaza then farm gate, not a single mountain-south hop.
        path_hops = [wp for wp in ret if wp.tilemap == 0x0C]
        self.assertGreaterEqual(len(path_hops), 2)

    def test_south_field_farm_to_spa_uses_dirt_row_not_house(self) -> None:
        """Sunday pin ~(78,598) must not first-hop house (137,375) through crops."""
        route = farm_to_spa_waypoints(78, 598, tilemap=0x00)
        sliced = slice_route_from_position(route, 78, 598, tilemap=0x00)
        self.assertEqual(sliced[-1].target_px, (619, 201))
        self.assertEqual(sliced[0].target_px, (136, 600))
        self.assertNotEqual(sliced[0].target_px, (137, 375))
        farm_hops = [wp for wp in sliced if wp.tilemap == 0x00]
        self.assertTrue(any(wp.is_exit and wp.exit_direction == "left" for wp in farm_hops))
        # Dirt-row run stays on y=37; x=13 column hops are tight so arrival
        # cannot sit on the neighboring 0x5E crop tile.
        self.assertEqual(farm_hops[0].run_direction, "right")
        self.assertEqual(farm_hops[1].target_px, (216, 600))
        self.assertLessEqual(farm_hops[1].radius, 6)
        self.assertIsNone(farm_hops[2].run_direction)

    def test_house_farm_to_spa_keeps_house_south_then_gate(self) -> None:
        house = farm_to_west_gate_waypoints(136, 375, tilemap=0x00)
        self.assertEqual(house[0].target_px, (137, 375))
        self.assertEqual(house[1].target_px, (136, 424))
        spa = farm_to_spa_waypoints(136, 375, tilemap=0x00)
        self.assertEqual(spa[-1].target_px, (619, 201))
        self.assertEqual(spa[0].target_px, (137, 375))

    def test_d2_night_farm_to_spa_uses_house_path(self) -> None:
        """Y1_D2_Night_Farm ~(199,486) is north of south-field y=520."""
        route = farm_to_spa_waypoints(199, 486, tilemap=0x00)
        self.assertEqual(route[0].target_px, (137, 375))
        self.assertNotEqual(route[0].target_px, (136, 600))
        self.assertEqual(route[-1].target_px, (619, 201))


if __name__ == "__main__":
    unittest.main()
