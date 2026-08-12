"""Unit tests for CoopChoresTask — no ROM needed."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from harvest.tasks.coop_task import (
    ADDR_EGG_AVAILABLE,
    ADDR_HAY_COUNT,
    ADDR_INCUBATOR_FLAGS,
    ADDR_ITEM_ON_HAND,
    ADDR_FED_CHICKENS_FLAGS,
    ADDR_FED_CHICKENS_N,
    CHICKEN_SLOT_BASE,
    CHICKEN_SLOT_SIZE,
    CHICKEN_FEED_SPOTS,
    COOP_FALSE_OPEN_COLUMN_X,
    COOP_MAIN_AISLE_TOP,
    CoopChoresTask,
    EGG_PICKUP_STAND,
    EXIT_PREP_ESCAPE_ROUTE,
    EXIT_PREP_STAND,
    FEED_BIN_STAND,
    FEED_CLEAR_STAND,
    INCUBATOR_BIT,
    ITEM_CHICKEN_FEED,
    INCUBATOR_STAND,
    ITEM_EGG,
    MAX_EGG_DEFERRALS,
    MAX_EGG_NAV_FRAMES,
    MAX_EXIT_PREP_FRAMES,
    MAX_FLOCK_SIZE,
    SHIP_BIN_INTERACT_STAND,
    SHIP_BIN_STAND,
    VISIBLE_EGG_SPRITE,
)
from harvest.planner.day_plan import (
    ADDR_CHICKEN_COUNT,
    CHICKEN_PHASES,
)
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_TILEMAP,
    ADDR_X,
    ADDR_Y,
)
from harvest.tasks.nav import MAP_WIDTH

from harvest.tasks.harvest_task import ADDR_SHIPPING_MONEY
from harvest.core.animal_status import ram_needs_chicken_chores
from harvest.core.npc_catalog import GOBJ_INITIALIZED, GOBJ_STRUCT_BASE, GOBJ_STRUCT_STRIDE
from retro_harness import TaskStatus


def _block_tiles(ram: np.ndarray, tiles: list[tuple[int, int]]) -> None:
    """Mark tiles unwalkable in the fake coop map."""
    for tx, ty in tiles:
        ram[ADDR_MAP + ty * MAP_WIDTH + tx] = 0x00


def _make_coop_ram(
    *,
    adults: int = 1,
    chicks: int = 0,
    slot_eggs: int = 0,
    hay: int = 50,
    egg_available: bool | int = True,
    incubating: bool = False,
    holding_egg: bool = False,
    holding_feed: bool = False,
    fed_chickens: int = 0,
    fed_chicken_flags: int = 0,
    shipping_money: int = 0,
    player_tile: tuple = FEED_BIN_STAND,
    live_offset: bool = False,
) -> np.ndarray:
    """Build a fake RAM snapshot inside the chicken coop."""
    ram = np.zeros(0x24000 if live_offset else 0x20000, dtype=np.uint8)
    base = 0x4000 if live_offset else 0
    ram[ADDR_TILEMAP] = 0x28
    ram[ADDR_INPUT_LOCK] = 1

    # Player position (pixel = tile * 16 + 8)
    px = player_tile[0] * 16 + 8
    py = player_tile[1] * 16 + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF

    # Fill coop tilemap with walkable tiles
    for i in range(64 * 64):
        ram[ADDR_MAP + i] = 0xA1

    # Chicken count
    ram[ADDR_CHICKEN_COUNT + base] = adults + chicks + slot_eggs

    # Chicken slots: adults first, then chicks, then eggs
    slot = 0
    for _ in range(adults):
        ram[CHICKEN_SLOT_BASE + base + slot * CHICKEN_SLOT_SIZE] = 0x09  # exists + adult age
        slot += 1
    for _ in range(chicks):
        ram[CHICKEN_SLOT_BASE + base + slot * CHICKEN_SLOT_SIZE] = 0x05  # exists + chick age
        slot += 1
    for _ in range(slot_eggs):
        ram[CHICKEN_SLOT_BASE + base + slot * CHICKEN_SLOT_SIZE] = 0x03  # exists + egg age
        slot += 1

    # Hay
    ram[ADDR_HAY_COUNT + base] = hay & 0xFF
    ram[ADDR_HAY_COUNT + base + 1] = (hay >> 8) & 0xFF
    ram[ADDR_FED_CHICKENS_N + base] = fed_chickens & 0xFF
    ram[ADDR_FED_CHICKENS_FLAGS + base] = fed_chicken_flags & 0xFF
    ram[ADDR_FED_CHICKENS_FLAGS + base + 1] = (fed_chicken_flags >> 8) & 0xFF

    # Egg available bitfield
    egg_flags = int(egg_available) if not isinstance(egg_available, bool) else (1 if egg_available else 0)
    ram[ADDR_EGG_AVAILABLE + base] = egg_flags & 0xFF
    ram[ADDR_EGG_AVAILABLE + base + 1] = (egg_flags >> 8) & 0xFF

    # Incubator
    if incubating:
        flags = INCUBATOR_BIT | 0x0400
    else:
        flags = 0x0400
    ram[ADDR_INCUBATOR_FLAGS + base] = flags & 0xFF
    ram[ADDR_INCUBATOR_FLAGS + base + 1] = (flags >> 8) & 0xFF

    # Held item
    if holding_egg:
        ram[ADDR_ITEM_ON_HAND + base] = ITEM_EGG
    elif holding_feed:
        ram[ADDR_ITEM_ON_HAND + base] = ITEM_CHICKEN_FEED

    # Shipping money
    ram[ADDR_SHIPPING_MONEY + base] = shipping_money & 0xFF
    ram[ADDR_SHIPPING_MONEY + base + 1] = (shipping_money >> 8) & 0xFF

    return ram


def _make_world(ram: np.ndarray):
    return SimpleNamespace(ram=ram, info={}, obs=None)


def _add_chicken_object(ram: np.ndarray, tile: tuple[int, int], *, slot: int = 1) -> None:
    offset = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
    ram[offset] = GOBJ_INITIALIZED & 0xFF
    ram[offset + 1] = (GOBJ_INITIALIZED >> 8) & 0xFF
    ram[offset + 2] = 0xE1
    ram[offset + 3] = 0x01
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[offset + 0x08] = px & 0xFF
    ram[offset + 0x09] = (px >> 8) & 0xFF
    ram[offset + 0x0A] = py & 0xFF
    ram[offset + 0x0B] = (py >> 8) & 0xFF


def _add_egg_object(ram: np.ndarray, tile: tuple[int, int], *, slot: int = 2) -> None:
    offset = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
    ram[offset] = GOBJ_INITIALIZED & 0xFF
    ram[offset + 1] = (GOBJ_INITIALIZED >> 8) & 0xFF
    ram[offset + 2] = VISIBLE_EGG_SPRITE & 0xFF
    ram[offset + 3] = (VISIBLE_EGG_SPRITE >> 8) & 0xFF
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[offset + 0x08] = px & 0xFF
    ram[offset + 0x09] = (px >> 8) & 0xFF
    ram[offset + 0x0A] = py & 0xFF
    ram[offset + 0x0B] = (py >> 8) & 0xFF


def _set_chicken_slot_position(ram: np.ndarray, slot: int, tile: tuple[int, int], *, live_offset: bool = False) -> None:
    base = 0x4000 if live_offset else 0
    offset = CHICKEN_SLOT_BASE + base + slot * CHICKEN_SLOT_SIZE
    ram[offset + 1] = 0x28
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[offset + 4] = px & 0xFF
    ram[offset + 5] = (px >> 8) & 0xFF
    ram[offset + 6] = py & 0xFF
    ram[offset + 7] = (py >> 8) & 0xFF


class CoopChoresTaskTests(unittest.TestCase):
    """Core state machine tests."""

    def test_reset_counts_adults_and_starts_feed(self):
        ram = _make_coop_ram(adults=3, hay=50)
        task = CoopChoresTask()
        task.reset(_make_world(ram))

        self.assertEqual(task._adult_count, 3)
        self.assertEqual(task._feed_remaining, 3)
        self.assertEqual(task._phase, "feed_nav")

    def test_reset_caps_feed_goal_when_configured(self):
        ram = _make_coop_ram(adults=6, hay=50)
        task = CoopChoresTask(max_feed_adults=2)
        task.reset(_make_world(ram))

        self.assertEqual(task._adult_count, 2)
        self.assertEqual(task._feed_remaining, 2)
        self.assertEqual(task._phase, "feed_nav")

    def test_reset_skips_feed_when_no_hay(self):
        ram = _make_coop_ram(adults=3, hay=0, egg_available=True)
        task = CoopChoresTask()
        task.reset(_make_world(ram))

        # No hay but has egg → should go to egg collection
        self.assertEqual(task._phase, "egg_nav")

    def test_reset_reads_live_wram_offset_values(self):
        ram = _make_coop_ram(adults=1, hay=97, egg_available=True, live_offset=True)
        task = CoopChoresTask()
        task.reset(_make_world(ram))

        self.assertEqual(task._adult_count, 1)
        self.assertEqual(task._feed_remaining, 1)
        self.assertEqual(task._hay_before, 97)
        self.assertEqual(task._phase, "feed_nav")

    def test_reset_skips_to_done_when_nothing_to_do(self):
        ram = _make_coop_ram(adults=0, hay=0, egg_available=False)
        task = CoopChoresTask()
        task.reset(_make_world(ram))

        self.assertEqual(task._phase, "exit_prep_nav")

    def test_feed_queues_correct_number_of_presses(self):
        """After navigating to feed bin, feed_act should queue a pickup press."""
        ram = _make_coop_ram(adults=4, hay=50, player_tile=FEED_BIN_STAND)
        task = CoopChoresTask()
        task.reset(_make_world(ram))

        # Manually advance to feed_act
        task._phase = "feed_act"
        task._feed_remaining = 4

        # Step to queue first feed
        world = _make_world(ram)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "feed_verify")
        self.assertEqual(task.fed_count, 0)
        self.assertEqual(task._feed_remaining, 4)
        self.assertGreater(len(task._action_queue), 0)

    def test_feed_stops_when_hay_runs_out(self):
        ram = _make_coop_ram(adults=5, hay=2, player_tile=FEED_BIN_STAND)
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "feed_act"
        task._feed_remaining = 5

        # Simulate hay running out after 2 feeds
        ram2 = ram.copy()
        ram2[ADDR_HAY_COUNT] = 0
        world = _make_world(ram2)

        result = task.step(world)
        # Should transition to egg or done
        self.assertIn(task._phase, ("egg_nav", "exit_prep_nav"))

    def test_feed_verify_moves_to_place_when_feed_is_held(self):
        ram = _make_coop_ram(
            adults=1,
            hay=96,
            egg_available=True,
            holding_feed=True,
            player_tile=FEED_BIN_STAND,
        )
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "feed_verify"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "feed_place_nav")

    def test_feed_place_verify_resumes_egg_nav_after_flag_sets(self):
        ram = _make_coop_ram(
            adults=1,
            hay=96,
            egg_available=True,
            holding_feed=False,
            fed_chickens=1,
            fed_chicken_flags=CHICKEN_FEED_SPOTS[0].flag,
            player_tile=FEED_CLEAR_STAND,
        )
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "feed_place_verify"
        task._adult_count = 1
        task._feed_remaining = 1
        task._fed_before = 0
        task._fed_flags_before = 0

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "egg_nav")

    def test_feed_place_nav_queues_place_press_when_at_open_slot(self):
        ram = _make_coop_ram(adults=1, hay=96, egg_available=True, holding_feed=True, player_tile=FEED_CLEAR_STAND)
        px, py = CHICKEN_FEED_SPOTS[0].interact_px
        ram[ADDR_X] = px & 0xFF
        ram[ADDR_X + 1] = (px >> 8) & 0xFF
        ram[ADDR_Y] = py & 0xFF
        ram[ADDR_Y + 1] = (py >> 8) & 0xFF
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "feed_place_nav"
        task._navigator.update(ram)

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "feed_place_verify")
        self.assertGreater(len(task._action_queue), 0)

    def test_feed_act_does_not_queue_another_feed_while_item_is_still_on_hand(self):
        ram = _make_coop_ram(adults=1, hay=96, egg_available=True, holding_feed=True, player_tile=FEED_BIN_STAND)
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "feed_act"
        task._feed_remaining = 1

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "feed_place_nav")
        self.assertEqual(len(task._action_queue), 0)

    def test_egg_collection_sets_flag(self):
        ram = _make_coop_ram(adults=0, hay=0, egg_available=True, player_tile=EGG_PICKUP_STAND)
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "egg_verify"

        # Simulate egg picked up
        ram2 = ram.copy()
        ram2[ADDR_EGG_AVAILABLE] = 0
        world = _make_world(ram2)
        task.step(world)

        self.assertTrue(task.egg_collected)
        self.assertEqual(task._phase, "decide")

    def test_egg_nav_still_attempts_pickup_while_feed_item_is_clearing(self):
        ram = _make_coop_ram(adults=0, hay=0, egg_available=True, player_tile=EGG_PICKUP_STAND)
        ram[ADDR_ITEM_ON_HAND] = 0x1A
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "egg_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "egg_verify")
        self.assertGreater(len(task._action_queue), 0)

    def test_egg_skipped_when_not_available(self):
        ram = _make_coop_ram(adults=0, hay=0, egg_available=False)
        task = CoopChoresTask()
        task.reset(_make_world(ram))

        self.assertEqual(task._phase, "exit_prep_nav")

    def test_slot_egg_counts_as_available_when_flag_is_clear(self):
        ram = _make_coop_ram(
            adults=3,
            slot_eggs=1,
            hay=50,
            egg_available=False,
            fed_chickens=3,
            fed_chicken_flags=0x0015,
            player_tile=(13, 11),
        )
        _set_chicken_slot_position(ram, 3, (14, 11))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        self.assertTrue(ram_needs_chicken_chores(ram))
        self.assertEqual(task._phase, "egg_nav")
        self.assertEqual(task._egg_pickup_spot(ram), ((13, 11), "right"))

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "egg_verify")
        self.assertGreater(len(task._action_queue), 0)

    def test_incubator_egg_slot_does_not_count_as_available_floor_egg(self):
        ram = _make_coop_ram(
            adults=0,
            slot_eggs=1,
            hay=50,
            egg_available=False,
            incubating=True,
            player_tile=(13, 11),
        )
        _set_chicken_slot_position(ram, 0, (14, 11))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        self.assertFalse(ram_needs_chicken_chores(ram))
        self.assertEqual(task._phase, "exit_prep_nav")
        self.assertEqual(task._egg_tiles(ram), [])

    def test_incubator_visible_egg_object_does_not_count_as_floor_egg(self):
        ram = _make_coop_ram(
            adults=0,
            hay=50,
            egg_available=False,
            incubating=True,
            player_tile=(13, 11),
        )
        _add_egg_object(ram, (14, 11))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        self.assertFalse(task._egg_present(ram))
        self.assertEqual(task._phase, "exit_prep_nav")
        self.assertEqual(task._egg_tiles(ram), [])

    def test_extended_egg_flags_use_spawn_table_pickup_spots(self):
        ram = _make_coop_ram(
            adults=0,
            hay=50,
            egg_available=0x001C,
            incubating=True,
            player_tile=(13, 9),
        )
        _add_egg_object(ram, (14, 11))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        self.assertEqual(task._phase, "egg_nav")
        self.assertEqual(task._egg_tiles(ram), [])
        # Prefer below-egg stand: (5,9) sits on the false-open column.
        self.assertEqual(task._egg_pickup_spot(ram), ((4, 10), "up"))

    def test_lower_egg_route_does_not_force_top_aisle(self):
        ram = _make_coop_ram(
            adults=0,
            hay=50,
            egg_available=0x001C,
            player_tile=(13, 9),
        )
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        route = task._left_top_route((4, 10))

        self.assertEqual(route, ((4, 10),))
        self.assertNotIn(COOP_MAIN_AISLE_TOP, route)

    def test_lower_egg_route_from_ship_bin_uses_left_lane(self):
        ram = _make_coop_ram(
            adults=0,
            hay=50,
            egg_available=0x001C,
            player_tile=(1, 10),
        )
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        route = task._left_top_route((4, 10))

        self.assertEqual(route, ((2, 10), (2, 9), (3, 9), (4, 10)))
        self.assertNotIn(COOP_MAIN_AISLE_TOP, route)

    def test_flag01_egg_avoids_unreachable_default_stand(self):
        """Regression: stand (2,4) is an island when (2,5)/(2,3)/(3,4) are walls."""
        ram = _make_coop_ram(
            adults=0,
            hay=0,
            egg_available=0x0001,
            player_tile=(1, 10),
        )
        _block_tiles(ram, [(2, 3), (2, 5), (3, 4)])
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        spot = task._egg_pickup_spot(ram)

        self.assertIsNotNone(spot)
        self.assertNotEqual(spot[0], (2, 4))
        self.assertIn(spot[0], {(0, 4), (1, 5)})
        self.assertEqual(task._current_egg_flag, 0x0001)

    def test_upper_egg_route_from_ship_climbs_service_lane(self):
        ram = _make_coop_ram(
            adults=0,
            hay=0,
            egg_available=0x0001,
            player_tile=(1, 10),
        )
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        route = task._left_top_route((0, 4))

        self.assertEqual(route[0], (0, 6))
        self.assertEqual(route[-1], (0, 4))

    def test_egg_nav_timeout_skips_flag_instead_of_stalling(self):
        ram = _make_coop_ram(
            adults=0,
            hay=0,
            egg_available=0x0001,
            player_tile=(1, 10),
        )
        _block_tiles(ram, [(2, 3), (2, 5), (3, 4), (0, 4), (1, 5), (0, 5), (0, 3)])
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "egg_nav"
        task._egg_nav_started_step = 1
        task._current_egg_flag = 0x0001
        task._deferred_egg_counts[0x0001] = MAX_EGG_DEFERRALS
        task._step_count = MAX_EGG_NAV_FRAMES + 2

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn(0x0001, task._skipped_egg_flags)
        self.assertEqual(task._phase, "exit_prep_nav")

    def test_incubating_slot_egg_is_not_collision(self):
        ram = _make_coop_ram(
            adults=0,
            slot_eggs=1,
            egg_available=False,
            incubating=True,
            player_tile=(12, 11),
        )
        _set_chicken_slot_position(ram, 0, (14, 11))
        task = CoopChoresTask()
        task.reset(_make_world(ram))

        self.assertNotIn((14, 11), task._chicken_tiles(ram))

    def test_begin_egg_nav_clears_stale_feed_route(self):
        ram = _make_coop_ram(adults=0, hay=0, egg_available=0x0001, player_tile=(1, 10))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._left_top_route_goal = FEED_BIN_STAND
        task._left_top_route_points = ((8, 6), (4, 5), FEED_BIN_STAND)
        task._left_top_route_index = 2

        result = task._begin_egg_nav()

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "egg_nav")
        self.assertIsNone(task._left_top_route_goal)
        self.assertEqual(task._left_top_route_points, ())
        self.assertEqual(task._left_top_route_index, 0)

    def test_egg_pickup_queues_stationary_a_press(self):
        ram = _make_coop_ram(
            adults=0,
            hay=50,
            egg_available=0x001C,
            player_tile=(4, 10),
        )
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)
        task._phase = "egg_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "egg_verify")
        self.assertGreaterEqual(len(task._action_queue), 2)
        face = task._action_queue[0]
        press = task._action_queue[4]
        self.assertEqual(int(face[4]), 1)  # face up into the egg
        self.assertEqual(int(face[5]), 0)
        self.assertEqual(int(press[8]), 1)  # press A without walking
        self.assertEqual(int(press[6]), 0)

    def test_decide_auto_incubates_when_empty_and_room(self):
        ram = _make_coop_ram(adults=3, incubating=False)
        task = CoopChoresTask(egg_mode="auto")
        task.reset(_make_world(ram))
        task._phase = "decide"
        task.egg_collected = True

        task.step(_make_world(ram))
        self.assertEqual(task._phase, "incubate_nav")

    def test_decide_auto_ships_when_incubator_full(self):
        ram = _make_coop_ram(adults=3, incubating=True)
        task = CoopChoresTask(egg_mode="auto")
        task.reset(_make_world(ram))
        task._phase = "decide"
        task.egg_collected = True

        task.step(_make_world(ram))
        self.assertEqual(task._phase, "ship_nav")

    def test_decide_auto_ships_when_flock_at_max(self):
        ram = _make_coop_ram(adults=MAX_FLOCK_SIZE, incubating=False)
        task = CoopChoresTask(egg_mode="auto")
        task.reset(_make_world(ram))
        task._phase = "decide"
        task.egg_collected = True

        task.step(_make_world(ram))
        self.assertEqual(task._phase, "ship_nav")

    def test_decide_gift_mode_goes_to_done(self):
        ram = _make_coop_ram(adults=1, incubating=False)
        task = CoopChoresTask(egg_mode="gift")
        task.reset(_make_world(ram))
        task._phase = "decide"
        task.egg_collected = True

        result = task.step(_make_world(ram))
        self.assertEqual(task._phase, "exit_prep_nav")

    def test_incubate_verify_succeeds(self):
        ram = _make_coop_ram(adults=1, egg_available=False, incubating=False, player_tile=INCUBATOR_STAND)
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "incubate_verify"

        # Simulate incubator bit set
        ram2 = ram.copy()
        flags = INCUBATOR_BIT | 0x0400
        ram2[ADDR_INCUBATOR_FLAGS] = flags & 0xFF
        ram2[ADDR_INCUBATOR_FLAGS + 1] = (flags >> 8) & 0xFF
        task.step(_make_world(ram2))

        self.assertTrue(task.egg_incubated)
        self.assertEqual(task._phase, "exit_prep_nav")

    def test_incubate_nav_recovers_to_left_approach_when_above_incubator(self):
        ram = _make_coop_ram(adults=1, holding_egg=True, player_tile=(13, 9))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "incubate_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_incubate_nav_queues_a_only_press_from_left_stand(self):
        ram = _make_coop_ram(adults=1, holding_egg=True, player_tile=INCUBATOR_STAND)
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "incubate_nav"
        task._navigator.update(ram)

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "incubate_verify")
        self.assertGreater(len(task._action_queue), 8)
        first = task._action_queue[0]
        self.assertEqual(int(first[7]), 1)  # face right into the incubator
        press = task._action_queue[8]
        self.assertEqual(int(press[8]), 1)
        self.assertEqual(int(press[7]), 0)

    def test_ship_verify_succeeds(self):
        ram = _make_coop_ram(adults=1, egg_available=False, player_tile=SHIP_BIN_STAND, shipping_money=0)
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "ship_verify"
        task._ship_money_before = 0

        # Simulate shipping money increase
        ram2 = ram.copy()
        ram2[ADDR_SHIPPING_MONEY] = 5
        task.step(_make_world(ram2))

        self.assertTrue(task.egg_shipped)
        self.assertEqual(task._phase, "exit_prep_nav")

    def test_ship_verify_fails_when_egg_clears_without_money(self):
        ram = _make_coop_ram(adults=1, holding_egg=False, player_tile=SHIP_BIN_STAND, shipping_money=0)
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "ship_verify"
        task._ship_money_before = 0

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("without shipping money", result.reason)
        self.assertFalse(task.egg_shipped)

    def test_ship_nav_sidesteps_left_from_approach_stand(self):
        ram = _make_coop_ram(adults=1, holding_egg=True, player_tile=SHIP_BIN_STAND, shipping_money=0)
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_nav")
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)  # left

    def test_ship_nav_uses_recorded_left_lane_from_egg_area(self):
        ram = _make_coop_ram(adults=1, holding_egg=True, player_tile=(2, 5), shipping_money=0)
        ram[ADDR_X] = 38
        ram[ADDR_X + 1] = 0
        ram[ADDR_Y] = 85
        ram[ADDR_Y + 1] = 0
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_ship_nav_from_mid_coop_paths_before_pixel_lane(self):
        ram = _make_coop_ram(
            adults=0,
            egg_available=0x0030,
            holding_egg=True,
            player_tile=(6, 7),
            shipping_money=0,
        )
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_nav")
        self.assertIsNotNone(result.action)
        self.assertTrue(task._navigator.path)
        self.assertEqual(task._navigator.path[-1], SHIP_BIN_STAND)
        # Far approach uses coop_nav_to_shipping_bin_skill
        self.assertIsNotNone(task._active_skill)
        self.assertEqual(task._active_skill.name, "coop_nav_ship_bin")

    def test_ship_nav_from_lower_right_corner_avoids_bin_corner_dead_edge(self):
        ram = _make_coop_ram(
            adults=0,
            egg_available=0x0020,
            holding_egg=True,
            player_tile=(3, 11),
            shipping_money=0,
        )
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)
        task._navigator.path = [(2, 11), (2, 10)]

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_nav")
        self.assertIsNotNone(result.action)
        self.assertTrue(task._navigator.path)
        self.assertEqual(task._navigator.path[-1], (3, 10))

    def test_ship_nav_from_row_ten_uses_right_lane_corner(self):
        ram = _make_coop_ram(
            adults=0,
            egg_available=0x0038,
            holding_egg=True,
            player_tile=(5, 10),
            shipping_money=0,
        )
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)
        task._navigator.path = [(5, 11), (4, 11), (3, 11), (2, 11), (2, 10)]

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_nav")
        self.assertIsNotNone(result.action)
        self.assertTrue(task._navigator.path)
        self.assertEqual(task._navigator.path[-1], (3, 10))

    def test_ship_nav_from_right_lane_corner_uses_pixel_slide(self):
        ram = _make_coop_ram(
            adults=0,
            egg_available=0x0038,
            holding_egg=True,
            player_tile=(3, 10),
            shipping_money=0,
        )
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_nav")
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)  # slide left toward bin
        self.assertEqual(int(result.action.action[0]), 1)

    def test_ship_nav_queues_press_from_interact_stand(self):
        ram = _make_coop_ram(adults=1, holding_egg=True, player_tile=SHIP_BIN_INTERACT_STAND, shipping_money=0)
        ram[ADDR_X] = 22
        ram[ADDR_X + 1] = 0
        ram[ADDR_Y] = 169
        ram[ADDR_Y + 1] = 0
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_verify")
        self.assertGreater(len(task._action_queue), 0)
        first = task._action_queue[0]
        self.assertEqual(int(first[5]), 1)  # face down into the egg bin
        self.assertEqual(int(first[4]), 0)
        second = task._action_queue[1]
        self.assertEqual(int(second[8]), 1)  # press A without walking
        self.assertEqual(int(second[5]), 0)

    def test_navigation_routes_around_live_chicken_object(self):
        ram = _make_coop_ram(adults=1, egg_available=False, player_tile=(2, 7))
        _add_chicken_object(ram, (3, 7))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (4, 7))

        self.assertIsNotNone(action)
        self.assertNotIn((3, 7), task._navigator.path)

    def test_navigation_blocks_adult_chicken_slot_without_live_object(self):
        ram = _make_coop_ram(adults=1, egg_available=False, player_tile=(2, 7))
        _set_chicken_slot_position(ram, 0, (3, 7))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (4, 7))

        self.assertIsNotNone(action)
        self.assertNotIn((3, 7), task._navigator.path)

    def test_navigation_blocks_egg_slot_without_live_object(self):
        ram = _make_coop_ram(adults=0, slot_eggs=1, egg_available=False, player_tile=(13, 11))
        _set_chicken_slot_position(ram, 0, (14, 11))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (15, 11))

        self.assertIsNotNone(action)
        self.assertNotIn((14, 11), task._navigator.path)

    def test_navigation_blocks_visible_egg_object(self):
        ram = _make_coop_ram(adults=0, egg_available=False, player_tile=(13, 11))
        _add_egg_object(ram, (14, 11))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (15, 11))

        self.assertIsNotNone(action)
        self.assertNotIn((14, 11), task._navigator.path)

    def test_navigation_blocks_flagged_egg_spawn_tiles(self):
        ram = _make_coop_ram(adults=0, egg_available=0x0002, player_tile=(4, 5))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (2, 4))

        self.assertIsNotNone(action)
        self.assertNotIn((3, 5), task._navigator.path)

    def test_feed_nav_from_entrance_uses_center_aisle(self):
        ram = _make_coop_ram(adults=2, hay=50, egg_available=True, player_tile=(8, 12))
        task = CoopChoresTask()
        task.reset(_make_world(ram))

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[6]), 0)
        self.assertEqual(task._left_top_route_points[:2], ((8, 6), (4, 5)))
        # Production path: feed_nav steps coop_nav_to_feed_bin_skill
        self.assertIsNotNone(task._active_skill)
        self.assertEqual(task._active_skill.name, "coop_nav_feed_bin")
        snap = task.progress_snapshot()
        self.assertEqual(snap.phase_text, "feed_nav")
        self.assertIsNotNone(snap.child)
        self.assertEqual(snap.child.task_name, "coop_nav_feed_bin")

    def test_feed_nav_recovers_from_lower_left_coop_corner(self):
        ram = _make_coop_ram(adults=2, hay=50, egg_available=True, player_tile=(2, 12))
        task = CoopChoresTask()
        task.reset(_make_world(ram))

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[4]), 0)
        self.assertEqual(task._left_top_route_points[:2], ((8, 12), (8, 6)))
        self.assertIsNotNone(task._active_skill)
        self.assertEqual(task._active_skill.name, "coop_nav_feed_bin")

    def test_coop_navigation_strictly_centers_before_vertical_step(self):
        ram = _make_coop_ram(adults=0, egg_available=False, player_tile=(2, 12))
        ram[ADDR_X] = 42
        ram[ADDR_Y] = 198
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)
        task._navigator.path = [(2, 11), (2, 10)]

        action = task._navigate_to_tile(ram, (2, 6))

        self.assertIsNotNone(action)
        self.assertEqual(int(action[6]), 1)
        self.assertEqual(int(action[4]), 0)

    def test_navigation_does_not_block_baby_chick_slot(self):
        ram = _make_coop_ram(adults=0, chicks=1, egg_available=False, player_tile=(2, 7))
        _set_chicken_slot_position(ram, 0, (3, 7))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (4, 7))

        self.assertIsNotNone(action)
        self.assertIn((3, 7), task._navigator.path)

    def test_navigation_waits_when_goal_is_occupied_by_chicken(self):
        ram = _make_coop_ram(adults=1, egg_available=False, player_tile=(2, 7))
        _add_chicken_object(ram, (4, 7))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (4, 7))

        self.assertIsNotNone(action)
        self.assertEqual(int(np.sum(action)), 0)
        self.assertEqual(task._navigator.path, [])

    def test_next_feed_spot_skips_filled_and_occupied_slots(self):
        ram = _make_coop_ram(
            adults=2,
            egg_available=False,
            holding_feed=True,
            fed_chicken_flags=CHICKEN_FEED_SPOTS[0].flag,
            player_tile=(2, 7),
        )
        _set_chicken_slot_position(ram, 0, CHICKEN_FEED_SPOTS[1].stand)
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        spot = task._next_feed_spot(ram)

        self.assertIsNotNone(spot)
        self.assertEqual(spot, CHICKEN_FEED_SPOTS[2])

    def test_next_feed_spot_skips_stuck_slots(self):
        ram = _make_coop_ram(adults=2, egg_available=False, holding_feed=True, player_tile=(3, 6))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._blocked_feed_flags.add(CHICKEN_FEED_SPOTS[0].flag)

        spot = task._next_feed_spot(ram)

        self.assertEqual(spot, CHICKEN_FEED_SPOTS[1])

    def test_feed_bin_stand_uses_reachable_aisle_tile(self):
        self.assertEqual(FEED_BIN_STAND, (2, 6))

    def test_chicken_feed_uses_recorded_held_item_id(self):
        self.assertEqual(ITEM_CHICKEN_FEED, 0x1A)

    def test_chicken_feed_spots_use_recorded_interaction_pixels(self):
        self.assertEqual(CHICKEN_FEED_SPOTS[0].interact_px, (38, 62))
        self.assertEqual(CHICKEN_FEED_SPOTS[1].interact_px, (58, 62))

    def test_exit_prep_stand_uses_coop_door_for_exit_handoff(self):
        self.assertEqual(EXIT_PREP_STAND, (8, 12))

    def test_exit_prep_escapes_false_open_column_instead_of_stalling(self):
        """Regression: long runs stuck at (5,11)/(86,183) in exit_prep_nav."""
        ram = _make_coop_ram(adults=0, hay=0, egg_available=False, player_tile=(5, 11))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        self.assertEqual(task._phase, "exit_prep_nav")
        self.assertEqual(COOP_FALSE_OPEN_COLUMN_X, 5)
        self.assertEqual(EXIT_PREP_ESCAPE_ROUTE[0], (5, 12))

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        # Climb above the false-open band before crossing east (B = run).
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_exit_prep_timeout_hands_off_instead_of_watchdog(self):
        ram = _make_coop_ram(adults=0, hay=0, egg_available=False, player_tile=(5, 11))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        task._exit_prep_started_step = 1
        task._step_count = MAX_EXIT_PREP_FRAMES + 2

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "done")

    def test_false_open_tiles_are_blocked_in_pathfinding(self):
        ram = _make_coop_ram(adults=0, hay=0, egg_available=False, player_tile=(2, 11))
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        blocked = task._coop_false_open_tiles()
        self.assertIn((5, 11), blocked)
        path = task._find_path_around_chickens(ram, (2, 11), (8, 12))
        if path is not None:
            self.assertNotIn((5, 11), path)

    def test_scales_to_12_chickens(self):
        """Reset with 12 adults should plan 12 feeds."""
        ram = _make_coop_ram(adults=12, hay=100)
        task = CoopChoresTask()
        task.reset(_make_world(ram))

        self.assertEqual(task._adult_count, 12)
        self.assertEqual(task._feed_remaining, 12)

    def test_mixed_flock_only_feeds_adults(self):
        """Chicks and eggs don't need feeding."""
        ram = _make_coop_ram(adults=5, chicks=3, slot_eggs=2, hay=100)
        task = CoopChoresTask()
        task.reset(_make_world(ram))

        self.assertEqual(task._adult_count, 5)
        self.assertEqual(task._feed_remaining, 5)

    def test_auto_ships_when_chicks_plus_adults_at_max(self):
        """Total flock (adults + chicks + eggs) >= MAX means no incubation."""
        ram = _make_coop_ram(adults=8, chicks=3, slot_eggs=1, incubating=False)
        task = CoopChoresTask(egg_mode="auto")
        task.reset(_make_world(ram))
        task._phase = "decide"
        task.egg_collected = True

        task.step(_make_world(ram))
        self.assertEqual(task._phase, "ship_nav")

    def test_timeout_returns_failure(self):
        ram = _make_coop_ram(adults=1)
        task = CoopChoresTask(timeout=5)
        task.reset(_make_world(ram))

        for _ in range(10):
            result = task.step(_make_world(ram))
            if result.status == TaskStatus.FAILURE:
                self.assertIn("timeout", result.reason)
                return
        self.fail("Expected timeout failure")

    def test_chicken_phases_use_coop_chores_kind(self):
        """Verify CHICKEN_PHASES wiring."""
        chores = [p for p in CHICKEN_PHASES if p.kind == "coop_chores"]
        self.assertEqual(len(chores), 1)
        self.assertEqual(chores[0].params["egg_mode"], "auto")

    def test_progress_text(self):
        ram = _make_coop_ram(adults=3, hay=50)
        task = CoopChoresTask()
        task.reset(_make_world(ram))
        text = task.progress_text
        self.assertIn("fed=0/3", text)
        self.assertIn("egg=N", text)


if __name__ == "__main__":
    unittest.main()
