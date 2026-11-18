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
    PLAYER_ACTION_JUMP,
    SPA_TILEMAP,
    MOUNTAIN_TILEMAP,
    CAVE_TILEMAP,
    SPA_OUTDOOR_STAND_PX,
    SPA_WATER_MIN_X_PX,
    near_outdoor_spa,
    read_stamina,
    read_max_stamina,
)
from retro_harness import TaskResult, TaskStatus, WorldState


ADDR_STAMINA = field_spec("stamina").address
ADDR_MAX_STAMINA = field_spec("max_stamina").address
ADDR_PLAYER_ACTION = field_spec("player_action").address


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

    def test_a_blocked_walk_to_the_spa_waits_for_the_dog_then_walks_again(self) -> None:
        """ROM probe: the farm dog parked in the west corridor and evening soaks died on pixel_stuck / exit_walk timeout."""
        from harvest.tasks.hot_spring import GATE_BLOCKS, GATE_RETRIES, GATE_WAIT_FRAMES, GATE_WAIT_NIGHT_FRAMES

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_STAMINA] = 0
        ram[ADDR_MAX_STAMINA] = 100
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_X], ram[ADDR_X + 1] = 136 & 0xFF, 0
        ram[ADDR_Y], ram[ADDR_Y + 1] = 420 & 0xFF, 420 >> 8
        task = HotSpringStaminaTask(min_stamina=None)
        task.reset(_world(ram))
        task.step(_world(ram))
        self.assertEqual(task.phase_text, "route_mountain")

        class Fake:
            def __init__(self, status, reason=""):
                self.status, self.reason = status, reason

            def step(self, world):
                return TaskResult(status=self.status, reason=self.reason)

            def reset(self, world):
                pass

        for attempt in range(1, GATE_RETRIES + 1):
            reason = f"route_mountain failed: {GATE_BLOCKS[attempt % len(GATE_BLOCKS)]} pos=(42,440)"
            task._task = Fake(TaskStatus.FAILURE, reason)
            task._phase = "route_mountain"
            result = task.step(_world(ram))
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertEqual(task.phase_text, "gate_retreat")              # the farmer leaves the corridor first
            task._task = Fake(TaskStatus.SUCCESS)
            task.step(_world(ram))
            self.assertEqual(task.phase_text, "gate_wait")
            self.assertEqual(task._gate_wait, GATE_WAIT_FRAMES)             # by day the clock runs through the wait
            for _ in range(GATE_WAIT_FRAMES - 1):
                self.assertEqual(task.step(_world(ram)).status, TaskStatus.RUNNING)
                self.assertEqual(task.phase_text, "gate_wait")
            task.step(_world(ram))                       # the wait is over: the route is built again
            self.assertEqual(task.phase_text, "route_mountain", attempt)
        task._task = Fake(TaskStatus.FAILURE, "route_mountain failed: exit_walk timeout on tilemap 0x00 towards 0x00")
        task._phase = "route_mountain"
        task._route_retries = 0
        from harvest.core.ram_catalog import live_wram_base

        ram[field_spec("hour").address + live_wram_base(ram)] = 18
        task.step(_world(ram))
        self.assertEqual(task._gate_wait, GATE_WAIT_NIGHT_FRAMES)         # after 18:00 the clock is frozen: wait long
        task._route_retries = GATE_RETRIES
        task._task = Fake(TaskStatus.FAILURE, "route_mountain failed: exit_walk timeout on tilemap 0x00 towards 0x00")
        task._phase = "route_mountain"
        gave_up = task.step(_world(ram))
        self.assertEqual(gave_up.status, TaskStatus.FAILURE)       # a wedged gate still ends the trip, after the retries

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
        self.assertEqual(spa[1].target_px, (328, 688))
        self.assertEqual(spa[1].run_direction, "up")
        self.assertTrue(spa[1].force_run)
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
        farm_mtn = [wp for wp in farm if wp.tilemap == 0x10]
        self.assertEqual(farm_mtn[0].target_px, (328, 728))
        self.assertEqual(farm_mtn[1].target_px, (328, 688))
        self.assertEqual(farm_mtn[1].run_direction, "up")
        self.assertTrue(farm_mtn[1].force_run)
        self.assertGreaterEqual(len(ret), 10)
        self.assertTrue(ret[-1].tilemap in (0x00, 0x0C) or ret[-2].is_exit)
        # Return walks the mountain-gate descent then farm gate, not the
        # plaza (132,128) that seals at (8,7).
        path_hops = [wp for wp in ret if wp.tilemap == 0x0C]
        self.assertGreaterEqual(len(path_hops), 2)
        self.assertEqual(path_hops[0].target_px, (137, 40))
        self.assertNotIn((132, 128), [wp.target_px for wp in path_hops])

    def test_south_field_farm_to_spa_uses_dirt_row_not_house(self) -> None:
        """Sunday pin ~(78,598) must not first-hop house (137,375) through crops.

        It takes the west steps (3-4,30-31): the x=13 climb runs through the
        (13,28) ring and a grown ring stopped 18 of 44 spa trips (v2_d3).
        """
        route = farm_to_spa_waypoints(78, 598, tilemap=0x00)
        sliced = slice_route_from_position(route, 78, 598, tilemap=0x00)
        self.assertEqual(sliced[-1].target_px, (619, 201))
        self.assertNotEqual(sliced[0].target_px, (137, 375))
        farm_hops = [wp for wp in sliced if wp.tilemap == 0x00]
        farm_px = [wp.target_px for wp in farm_hops]
        self.assertTrue(any(wp.is_exit and wp.exit_direction == "left" for wp in farm_hops))
        self.assertNotIn((216, 440), farm_px)
        self.assertLess(farm_px.index((104, 536)), farm_px.index((72, 488)))
        self.assertLess(farm_px.index((72, 488)), farm_px.index((72, 424)))
        # The exit row is reached at its centre: from y=430 the left walk caught (1,27).
        self.assertLessEqual(farm_hops[farm_px.index((72, 424))].radius, 3)

    def test_west_strip_shop_exit_takes_the_steps_and_stops_at_the_gate(self) -> None:
        """NAV_FARM_EXIT from the west-strip rings: no (8,35) ring hop, no path map."""
        from harvest.maps.farm_gate import farm_exit_waypoints

        hops = farm_exit_waypoints(100, 597, tilemap=0x00)
        self.assertNotIn((136, 568), [wp.target_px for wp in hops])
        self.assertIn((72, 488), [wp.target_px for wp in hops])
        self.assertTrue(hops[-1].is_exit and hops[-1].exit_direction == "left")
        self.assertTrue(all(wp.tilemap == 0x00 for wp in hops))

    def test_house_farm_to_spa_keeps_house_south_then_gate(self) -> None:
        house = farm_to_west_gate_waypoints(136, 375, tilemap=0x00)
        self.assertEqual(house[0].target_px, (137, 375))
        self.assertEqual(house[1].target_px, (136, 392))
        self.assertNotEqual(house[1].target_px, (136, 424))
        self.assertNotIn((120, 408), [wp.target_px for wp in house])
        self.assertNotIn((136, 440), [wp.target_px for wp in house])
        self.assertNotIn((72, 440), [wp.target_px for wp in house])
        self.assertNotIn((72, 392), [wp.target_px for wp in house])
        self.assertNotIn((8, 392), [wp.target_px for wp in house])
        self.assertIn((136, 424), [wp.target_px for wp in house])
        self.assertIn((72, 424), [wp.target_px for wp in house])
        down = next(wp for wp in house if wp.target_px == (136, 424))
        self.assertEqual(down.run_direction, "down")
        self.assertTrue(down.force_run)
        west = next(wp for wp in house if wp.target_px == (72, 424))
        self.assertEqual(west.run_direction, "left")
        exit_wp = next(wp for wp in house if wp.is_exit)
        self.assertEqual(exit_wp.target_px, (40, 424))
        self.assertEqual(exit_wp.exit_direction, "left")
        self.assertIsNone(exit_wp.run_direction)
        self.assertFalse(exit_wp.force_run)
        self.assertLess(exit_wp.radius, abs(432 - exit_wp.target_px[1]))
        spa = farm_to_spa_waypoints(136, 375, tilemap=0x00)
        self.assertEqual(spa[-1].target_px, (619, 201))
        self.assertEqual(spa[0].target_px, (137, 375))

    def test_north_west_rock_stand_force_runs_outside_house_paddock(self) -> None:
        """Live D2 rock spa start (8,16) cannot walk across the y=17 wall."""
        route = farm_to_spa_waypoints(131, 265, tilemap=0x00)
        farm = [wp for wp in route if wp.tilemap == 0x00]

        self.assertEqual(farm[0].target_px, (131, 265))
        self.assertEqual(farm[1].target_px, (72, 265))
        self.assertEqual(farm[1].run_direction, "left")
        self.assertTrue(farm[1].force_run)
        self.assertEqual(farm[2].target_px, (72, 424))
        self.assertEqual(farm[2].run_direction, "down")
        self.assertTrue(farm[2].force_run)
        self.assertTrue(any(wp.is_exit for wp in farm))

    def test_d2_night_farm_to_spa_uses_house_south_pinch(self) -> None:
        """Y1_D2_Night_Farm ~(199,486) is north of south-field y=520.

        Tile (12,30) is the bin/shed row. Rewinding to house (137,375) then
        down the door column hugs the wall; join the y=26 pinch instead.
        """
        route = farm_to_spa_waypoints(199, 486, tilemap=0x00)
        self.assertNotEqual(route[0].target_px, (137, 375))
        self.assertNotEqual(route[0].target_px, (136, 600))
        self.assertNotIn((888, 968), [wp.target_px for wp in route])
        farm_px = [wp.target_px for wp in route if wp.tilemap == 0x00]
        self.assertIn((136, 424), farm_px)
        self.assertIn((72, 424), farm_px)
        self.assertEqual(route[-1].target_px, (619, 201))

    def test_leftover_rocks_farm_to_spa_sidesteps_join_column(self) -> None:
        """Live spa pin (41,11)/(663,185): boulder at (40-41,12-13).

        Due-south hop (663,272) is the pixel_stuck. West onto x=39 at y=11,
        then south on the join column.
        """
        px, py = 663, 185
        route = farm_to_spa_waypoints(px, py, tilemap=0x00)
        sliced = slice_route_from_position(route, px, py, tilemap=0x00)
        self.assertEqual(sliced[-1].target_px, (619, 201))
        farm_px = [wp.target_px for wp in sliced if wp.tilemap == 0x00]
        self.assertNotIn((px, 272), farm_px)
        self.assertIn((624, py), farm_px)
        self.assertIn((624, 272), farm_px)
        self.assertIn((624, 384), farm_px)
        dx = abs(sliced[0].target_px[0] - px)
        dy = abs(sliced[0].target_px[1] - py)
        self.assertLessEqual(max(dx, dy), 7 * 16)
        prev = (px, py)
        for wp in sliced:
            if wp.tilemap != 0x00:
                break
            gap = max(abs(wp.target_px[0] - prev[0]), abs(wp.target_px[1] - prev[1]))
            if wp.run_direction is None and not wp.is_exit:
                self.assertLessEqual(gap, 7 * 16, f"{prev} -> {wp.target_px}")
            prev = wp.target_px

    def test_upper_north_east_spa_route_avoids_remaining_stump(self) -> None:
        """Live rock spa failure (38,8) is directly above stump (38,9)."""
        px, py = 621, 132
        route = farm_to_spa_waypoints(px, py, tilemap=0x00)
        sliced = slice_route_from_position(route, px, py, tilemap=0x00)
        farm_px = [wp.target_px for wp in sliced if wp.tilemap == 0x00]

        self.assertIn((37 * 16, py), farm_px)
        self.assertIn((37 * 16, 272), farm_px)
        self.assertNotIn((624, 272), farm_px)

    def test_bin_row_spa_route_avoids_remaining_stump(self) -> None:
        """Live rock spa failure (29,28) is boxed by stumps (18,28)/(38,28)."""
        px, py = 470, 455
        route = farm_to_spa_waypoints(px, py, tilemap=0x00)
        sliced = slice_route_from_position(route, px, py, tilemap=0x00)
        farm_px = [wp.target_px for wp in sliced if wp.tilemap == 0x00]

        self.assertIn((px, 24 * 16 + 8), farm_px)
        self.assertIn((136, 392), farm_px)
        self.assertIn((136, 424), farm_px)
        self.assertNotIn((136, py), farm_px)
        for wp in sliced:
            if wp.tilemap != 0x00:
                break
            if wp.target_px == (px, py):
                continue
            tx, ty = wp.target_px[0] // 16, wp.target_px[1] // 16
            self.assertFalse(tx in (18, 19) and ty in (28, 29), wp.target_px)

    def test_ditch_north_spa_route_does_not_rewind_house(self) -> None:
        """Live stump spa failure (13,23) must not first-hop house (8,23)."""
        px, py = 208, 378
        route = farm_to_spa_waypoints(px, py, tilemap=0x00)
        sliced = slice_route_from_position(route, px, py, tilemap=0x00)
        farm_px = [wp.target_px for wp in sliced if wp.tilemap == 0x00]

        self.assertNotEqual(sliced[0].target_px, (137, 375))
        self.assertIn((px, 22 * 16 + 8), farm_px)
        self.assertIn((136, 22 * 16 + 8), farm_px)
        self.assertIn((136, 392), farm_px)
        self.assertIn((136, 424), farm_px)
        self.assertIn((72, 424), farm_px)
        self.assertNotIn((px, 24 * 16 + 8), farm_px)

    def test_after_rocks_farm_to_spa_prefixes_north_east(self) -> None:
        """Y1_D2_After_Rocks ~(633,223) must not first-hop house or barn A1."""
        route = farm_to_spa_waypoints(633, 223, tilemap=0x00)
        sliced = slice_route_from_position(route, 633, 223, tilemap=0x00)
        self.assertEqual(sliced[-1].target_px, (619, 201))
        self.assertNotEqual(sliced[0].target_px, (137, 375))
        dx = abs(sliced[0].target_px[0] - 633)
        dy = abs(sliced[0].target_px[1] - 223)
        self.assertLessEqual(max(dx, dy), 7 * 16)
        farm_px = [wp.target_px for wp in sliced if wp.tilemap == 0x00]
        self.assertIn((624, 384), farm_px)
        self.assertNotIn((400, 352), farm_px)
        self.assertNotIn((512, 320), farm_px)
        self.assertIn((136, 424), farm_px)
        self.assertIn((72, 424), farm_px)
        self.assertNotIn((72, 392), farm_px)
        self.assertNotIn((8, 392), farm_px)
        self.assertNotIn((136, 440), farm_px)
        self.assertNotIn((72, 440), farm_px)
        self.assertNotIn((120, 408), farm_px)
        self.assertNotIn((88, 432), farm_px)
        mtn = [wp for wp in sliced if wp.tilemap == 0x10]
        self.assertGreaterEqual(len(mtn), 3)
        self.assertEqual(mtn[0].target_px, (328, 728))
        self.assertEqual(mtn[1].target_px, (328, 688))
        self.assertEqual(mtn[1].run_direction, "up")
        self.assertTrue(mtn[1].force_run)

    def test_se_leftover_farm_to_spa_uses_y39_row(self) -> None:
        """Y1_D2_Leftover_Partial ~(879,686) must not first-hop (136,600) RIGHT."""
        px, py = 879, 686
        route = farm_to_spa_waypoints(px, py, tilemap=0x00)
        sliced = slice_route_from_position(route, px, py, tilemap=0x00)
        self.assertEqual(sliced[-1].target_px, (619, 201))
        dx = abs(sliced[0].target_px[0] - px)
        dy = abs(sliced[0].target_px[1] - py)
        self.assertLessEqual(max(dx, dy), 7 * 16)
        self.assertNotEqual(sliced[0].target_px, (136, 600))
        self.assertNotEqual(sliced[0].target_px, (137, 375))
        farm = [wp for wp in sliced if wp.tilemap == 0x00]
        farm_px = [wp.target_px for wp in farm]
        self.assertNotIn((136, 600), farm_px)
        from harvest.maps.farm_pond import is_packed_layout

        # The packed lattice keeps y=37 as its lane; the legacy field walks the y=39 row.
        self.assertIn((216, 600 if is_packed_layout() else 632), farm_px)
        # West along the y=38 lane to the steps, never up x=13 through (13,28).
        self.assertNotIn((216, 440), farm_px)
        self.assertIn((104, 536), farm_px)
        self.assertIn((72, 488), farm_px)
        self.assertTrue(any(wp.is_exit and wp.exit_direction == "left" for wp in farm))
        self.assertNotIn((72, 392), farm_px)
        self.assertNotIn((136, 440), farm_px)
        self.assertNotIn((72, 440), farm_px)
        prev = (px, py)
        for wp in farm:
            gap = max(abs(wp.target_px[0] - prev[0]), abs(wp.target_px[1] - prev[1]))
            if wp.run_direction is None and not wp.is_exit:
                self.assertLessEqual(gap, 7 * 16, f"{prev} -> {wp.target_px}")
            prev = wp.target_px
        stones = farm_to_spa_waypoints(478, 566, tilemap=0x00)
        stones_sliced = slice_route_from_position(stones, 478, 566, tilemap=0x00)
        self.assertNotEqual(stones_sliced[0].target_px, (136, 600))
        from harvest.maps.farm_pond import is_packed_layout

        self.assertIn((216, 600 if is_packed_layout() else 632), [wp.target_px for wp in stones_sliced])

    def test_wood_checkpoint_farm_to_spa_densifies_east_of_join(self) -> None:
        """Y1_D2_Wood_Checkpoint ~(774,216) tile (48,13) is 8 tiles east of (39,17)."""
        px, py = 774, 216
        route = farm_to_spa_waypoints(px, py, tilemap=0x00)
        sliced = slice_route_from_position(route, px, py, tilemap=0x00)
        self.assertEqual(sliced[-1].target_px, (619, 201))
        dx = abs(sliced[0].target_px[0] - px)
        dy = abs(sliced[0].target_px[1] - py)
        self.assertLessEqual(max(dx, dy), 7 * 16)
        farm_px = [wp.target_px for wp in sliced if wp.tilemap == 0x00]
        self.assertIn((624, 272), farm_px)
        self.assertIn((624, 384), farm_px)
        self.assertIn((824, 216), farm_px)
        self.assertNotEqual(sliced[0].target_px, (624, 272))
        self.assertNotEqual(sliced[1].target_px, (px, 272))
        prev = (px, py)
        for wp in sliced:
            if wp.tilemap != 0x00:
                break
            gap = max(abs(wp.target_px[0] - prev[0]), abs(wp.target_px[1] - prev[1]))
            if wp.run_direction is None and not wp.is_exit:
                self.assertLessEqual(gap, 7 * 16, f"{prev} -> {wp.target_px}")
            prev = wp.target_px

    def test_ditch_lip_farm_to_spa_joins_pinch_not_house(self) -> None:
        """Wood_Progress ~(190,400) tile (11,25): y=24 then house-column to y=26.

        Live dump: y=25 x=0-6 is A6 pond; (0,24)=A1 gate wall (LEFT+B pins at
        x=22); (1,26)=A8 open face into (0,26)=C0; (1,27)=FF.
        """
        px, py = 190, 400
        route = farm_to_spa_waypoints(px, py, tilemap=0x00)
        sliced = slice_route_from_position(route, px, py, tilemap=0x00)
        self.assertEqual(sliced[-1].target_px, (619, 201))
        self.assertNotEqual(sliced[0].target_px, (137, 375))
        farm = [wp for wp in sliced if wp.tilemap == 0x00]
        farm_px = [wp.target_px for wp in farm]
        self.assertNotIn((200, 408), farm_px)
        self.assertNotIn((137, 375), farm_px)
        self.assertIn((px, 392), farm_px)
        self.assertIn((136, 392), farm_px)
        self.assertIn((136, 424), farm_px)
        self.assertIn((72, 424), farm_px)
        self.assertNotIn((72, 392), farm_px)
        self.assertNotIn((8, 392), farm_px)
        self.assertNotIn((72, 440), farm_px)
        join = next(wp for wp in farm if wp.target_px == (136, 392))
        self.assertIsNone(join.run_direction)
        down = next(wp for wp in farm if wp.target_px == (136, 424))
        self.assertEqual(down.run_direction, "down")
        self.assertTrue(down.force_run)
        dx = abs(sliced[0].target_px[0] - px)
        dy = abs(sliced[0].target_px[1] - py)
        self.assertLessEqual(max(dx, dy), 7 * 16)
        exit_wp = next(wp for wp in farm if wp.is_exit)
        self.assertEqual(exit_wp.target_px, (40, 424))
        self.assertEqual(exit_wp.exit_direction, "left")
        self.assertIsNone(exit_wp.run_direction)
        self.assertFalse(exit_wp.force_run)
        # Live hug (1,27)/(30,432) is the FF wall, not the C0 trigger.
        self.assertGreater(abs(432 - exit_wp.target_px[1]), exit_wp.radius)
        night = farm_to_spa_waypoints(199, 486, tilemap=0x00)
        self.assertNotEqual(night[0].target_px, (137, 375))
        self.assertIn((136, 424), [wp.target_px for wp in night])


def _set_u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


def _spa_lip_ram(*, stamina: int, maximum: int = 100) -> np.ndarray:
    ram = _blank_ram()
    ram[ADDR_TILEMAP] = MOUNTAIN_TILEMAP
    ram[ADDR_STAMINA] = stamina
    ram[ADDR_MAX_STAMINA] = maximum
    ram[ADDR_INPUT_LOCK] = 1
    sx, sy = SPA_OUTDOOR_STAND_PX
    _set_u16(ram, ADDR_X, sx)
    _set_u16(ram, ADDR_Y, sy)
    return ram


def _pulse_jump_exit(
    task: HotSpringStaminaTask, ram: np.ndarray, new_stam: int
) -> TaskResult:
    """One in-water jump then lip exit, with stamina applied on the exit frame."""
    ram[ADDR_PLAYER_ACTION] = PLAYER_ACTION_JUMP
    task.step(_world(ram))
    ram[ADDR_STAMINA] = new_stam
    ram[ADDR_PLAYER_ACTION] = 0
    return task.step(_world(ram))


class _ArriveTask:
    def reset(self, world) -> None:
        return None

    def step(self, world) -> TaskResult:
        return TaskResult(status=TaskStatus.SUCCESS, reason="arrived")


class SpaFillToMaxTests(unittest.TestCase):
    """Full restore soaks ~5–6 jump-exits; do not walk home before current==max."""

    def test_five_jump_exits_keep_soaking_until_max(self) -> None:
        ram = _spa_lip_ram(stamina=17, maximum=100)
        task = HotSpringStaminaTask(min_stamina=None, return_to_farm=True)
        task.reset(_world(ram))
        task._begin_soak(_world(ram))

        # Human fill is ~+14–16 per water exit; 5 exits leave a remainder.
        after_jumps = (33, 49, 65, 81, 97)
        result = None
        for stam in after_jumps:
            result = _pulse_jump_exit(task, ram, stam)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertEqual(task.phase_text, "soak")
            self.assertLess(read_stamina(ram), read_max_stamina(ram))

        self.assertEqual(task._jumps_seen, 5)
        self.assertFalse(task._stamina_ok(ram))

        result = _pulse_jump_exit(task, ram, 100)
        self.assertEqual(task._jumps_seen, 6)
        self.assertEqual(read_stamina(ram), 100)
        self.assertNotEqual(task.phase_text, "soak")
        self.assertIn(task.phase_text, {"post_soak_settle", "return_farm"})
        self.assertNotEqual(result.status, TaskStatus.FAILURE)

    def test_cycle_budget_does_not_end_a_partial_full_restore(self) -> None:
        ram = _spa_lip_ram(stamina=17, maximum=100)
        task = HotSpringStaminaTask(
            min_stamina=None,
            return_to_farm=False,
            max_jump_cycles=4,
            soak_plateau_frames=10_000,
        )
        task.reset(_world(ram))
        task._begin_soak(_world(ram))

        result = None
        for stam in (40, 60, 80, 90):
            result = _pulse_jump_exit(task, ram, stam)
        self.assertEqual(task._jumps_seen, 4)
        self.assertEqual(read_stamina(ram), 90)

        # Drain the queued 4th bath so the cycle cap is live.
        for _ in range(400):
            result = task.step(_world(ram))
            if result.status != TaskStatus.RUNNING:
                break
            if task._jump_cycles >= 4 and not task._action_queue:
                result = task.step(_world(ram))
                break

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.phase_text, "soak")
        self.assertEqual(read_stamina(ram), 90)

        ram[ADDR_STAMINA] = 100
        result = None
        for _ in range(20):
            result = task.step(_world(ram))
            if result.status == TaskStatus.SUCCESS:
                break
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(read_stamina(ram), read_max_stamina(ram))
        self.assertIn("soaked", result.reason or "")

    def test_return_farm_success_requires_max_stamina(self) -> None:
        ram = _spa_lip_ram(stamina=80, maximum=100)
        ram[ADDR_TILEMAP] = 0x00
        task = HotSpringStaminaTask(min_stamina=None, return_to_farm=True)
        task.reset(_world(ram))
        task._phase = "return_farm"
        task._task = _ArriveTask()

        result = task.step(_world(ram))

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("unrestored", result.reason or "")
        self.assertEqual(read_stamina(ram), 80)

    def test_return_farm_success_when_maxed(self) -> None:
        ram = _spa_lip_ram(stamina=100, maximum=100)
        ram[ADDR_TILEMAP] = 0x00
        task = HotSpringStaminaTask(min_stamina=None, return_to_farm=True)
        task.reset(_world(ram))
        task._phase = "return_farm"
        task._task = _ArriveTask()

        result = task.step(_world(ram))

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(read_stamina(ram), read_max_stamina(ram))
        self.assertIn("returned to farm", result.reason or "")

    def test_full_restore_phase_builds_fill_to_max_task(self) -> None:
        from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
        from harvest.planner.day_phase_stamina import full_restore_spa_phase

        ram = _blank_ram()
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = build_phase_task(TaskBuildContext(), full_restore_spa_phase(), world)
        self.assertIsInstance(task, HotSpringStaminaTask)
        self.assertIsNone(task.min_stamina)
        self.assertTrue(task.return_to_farm)
        ram[ADDR_MAX_STAMINA] = 130
        ram[ADDR_STAMINA] = 17
        self.assertEqual(task._stamina_target(ram), 130)
        self.assertFalse(task._stamina_ok(ram))
        ram[ADDR_STAMINA] = 90
        self.assertFalse(task._stamina_ok(ram))
        ram[ADDR_STAMINA] = 130
        self.assertTrue(task._stamina_ok(ram))


class _PondSim:
    """The spa lip as the ROM probe showed it, one frame per ``step``.

    Water blocks the walk at x=617 (west) and x=646 (east). Holding A against the water starts a jump (+12 stamina) that crosses
    it; a jump from the east lands 12 px past the west lip (605), as the failing sweep did. A fresh A press anywhere else is a
    tool swing that eats the press, which is what killed the open-loop script.
    """

    WEST, EAST, WEST_LANDING, EAST_LANDING = 617, 646, 605, 646

    def __init__(self, ram: np.ndarray, *, x: int = 604, can_jump: bool = True) -> None:
        self.ram, self.x, self.can_jump = ram, x, can_jump
        self.jump_left = self.swing_left = 0
        self.jump_dir = 0
        self.held_a = False
        self.swings = 0
        self.jumps = 0
        ram[ADDR_TILEMAP] = MOUNTAIN_TILEMAP
        ram[ADDR_INPUT_LOCK] = 1
        self._write(0)

    def _write(self, action: int) -> None:
        r = self.ram
        r[ADDR_X], r[ADDR_X + 1] = self.x & 0xFF, self.x >> 8
        r[ADDR_Y], r[ADDR_Y + 1] = 201, 0
        r[ADDR_PLAYER_ACTION] = action

    def flush(self, direction: int) -> bool:
        return (direction > 0 and self.x == self.WEST) or (direction < 0 and self.x == self.EAST)

    def step(self, action) -> None:
        right, left, a = int(action[7]) == 1, int(action[6]) == 1, int(action[8]) == 1
        direction = 1 if right else -1 if left else 0
        a_edge = a and not self.held_a
        self.held_a = a
        if self.jump_left:
            self.jump_left -= 1
            if not self.jump_left:
                self.x = self.EAST_LANDING if self.jump_dir > 0 else self.WEST_LANDING
            self._write(PLAYER_ACTION_JUMP if self.jump_left else 0)
            return
        if self.swing_left:
            self.swing_left -= 1
            self._write(4)
            return
        if a and direction and self.flush(direction) and self.can_jump:
            self.jump_left, self.jump_dir, self.jumps = 40, direction, self.jumps + 1
            self.ram[ADDR_STAMINA] = min(int(self.ram[ADDR_MAX_STAMINA]), int(self.ram[ADDR_STAMINA]) + 12)
            self._write(PLAYER_ACTION_JUMP)
            return
        if a_edge:
            self.swing_left, self.swings = 20, self.swings + 1
            self._write(4)
            return
        if direction > 0:
            if self.x < self.WEST:
                self.x = min(self.x + 2, self.WEST)
            elif self.x >= self.EAST:
                self.x += 2
        elif direction < 0:
            if self.x > self.EAST:
                self.x = max(self.x - 2, self.EAST)
            elif self.x <= self.WEST:
                self.x -= 2
        self._write(0)


class HotSpringBathLoopTests(unittest.TestCase):
    def _soak_task(self, ram: np.ndarray, **kwargs) -> HotSpringStaminaTask:
        ram[ADDR_STAMINA], ram[ADDR_MAX_STAMINA] = 0, 160
        task = HotSpringStaminaTask(**kwargs)
        task.reset(_world(ram))
        task._begin_soak(_world(ram))
        task._action_queue.clear()   # the walk to the lip is not under test
        return task

    def test_bath_fills_to_max_despite_a_lip_overshoot_and_never_swings(self) -> None:
        ram = _blank_ram()
        sim = _PondSim(ram, x=_PondSim.WEST)
        task = self._soak_task(ram, return_to_farm=False)
        result = None
        for frame in range(2000):
            result = task.step(_world(ram))
            if result.status != TaskStatus.RUNNING:
                break
            sim.step(result.action.action)
        self.assertEqual(result.status, TaskStatus.SUCCESS, result.reason)
        self.assertEqual(read_stamina(ram), 160)
        self.assertEqual(sim.jumps, 14)
        self.assertEqual(sim.swings, 0, "A pressed away from the water is a tool swing, not a jump")
        self.assertLess(frame, 1500)

    def test_bath_recovers_when_a_jump_lands_past_the_lip(self) -> None:
        ram = _blank_ram()
        sim = _PondSim(ram, x=_PondSim.WEST_LANDING)   # the stall: back on the west shelf, short of the water
        task = self._soak_task(ram, return_to_farm=False)
        for _ in range(2000):
            result = task.step(_world(ram))
            if result.status != TaskStatus.RUNNING:
                break
            sim.step(result.action.action)
        self.assertEqual(result.status, TaskStatus.SUCCESS, result.reason)
        self.assertEqual(sim.swings, 0)

    def test_bath_that_never_jumps_gives_up_instead_of_burning_the_timeout(self) -> None:
        ram = _blank_ram()
        sim = _PondSim(ram, x=_PondSim.WEST, can_jump=False)
        task = self._soak_task(ram, return_to_farm=False)
        for frame in range(3000):
            result = task.step(_world(ram))
            if result.status != TaskStatus.RUNNING:
                break
            sim.step(result.action.action)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("bath stalled", result.reason or "")
        self.assertLess(frame, 1200, "a stalled bath must not wait out the 3,600-frame soak timeout")

    def test_failed_soak_still_leaves_the_pond_before_it_reports_failure(self) -> None:
        """The farmer left at (661,198) failed every later phase's map lock (Y3 Spring D12)."""
        ram = _blank_ram()
        sim = _PondSim(ram, x=_PondSim.EAST_LANDING)
        ram[ADDR_STAMINA], ram[ADDR_MAX_STAMINA] = 48, 160
        task = HotSpringStaminaTask(return_to_farm=True)
        task.reset(_world(ram))
        task._finish_soak(_world(ram), "soak timeout")
        self.assertEqual(task.phase_text, "post_soak_settle")
        self.assertIn("soak ended unrestored (soak timeout)", task._soak_failure)
        for _ in range(600):
            result = task.step(_world(ram))
            if task.phase_text != "post_soak_settle":
                break
            sim.step(result.action.action)
        self.assertLess(sim.x, SPA_WATER_MIN_X_PX, "the farmer must be west of the water before the walk home starts")
        self.assertEqual(sim.swings, 0)


if __name__ == "__main__":
    unittest.main()
