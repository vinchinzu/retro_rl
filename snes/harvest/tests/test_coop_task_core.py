"""CoopChoresTask core state-machine tests — feed, egg decide, incubate, ship verify."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from coop_task_test_helpers import (  # noqa: E402
    add_egg_object,
    block_tiles,
    make_coop_ram,
    make_world,
    set_chicken_slot_position,
)
from harvest.tasks.coop_task import (  # noqa: E402
    ADDR_EGG_AVAILABLE,
    ADDR_HAY_COUNT,
    ADDR_INCUBATOR_FLAGS,
    ADDR_ITEM_ON_HAND,
    CHICKEN_FEED_SPOTS,
    CoopChoresTask,
    EGG_PICKUP_STAND,
    EXIT_PREP_STAND,
    FEED_BIN_STAND,
    FEED_CLEAR_STAND,
    INCUBATOR_BIT,
    ITEM_CHICKEN_FEED,
    INCUBATOR_STAND,
    MAX_EGG_DEFERRALS,
    MAX_EGG_NAV_FRAMES,
    MAX_FLOCK_SIZE,
    SHIP_BIN_STAND,
)
from harvest.planner.day_plan import CHICKEN_PHASES  # noqa: E402
from harvest.core.tile_catalog import ADDR_X, ADDR_Y  # noqa: E402
from harvest.tasks.harvest_task import ADDR_SHIPPING_MONEY  # noqa: E402
from harvest.core.animal_status import ram_needs_chicken_chores  # noqa: E402
from retro_harness import TaskStatus  # noqa: E402


class CoopChoresTaskCoreTests(unittest.TestCase):
    """Feed / egg / decide / incubate / ship-verify / flock scaling."""

    def test_reset_counts_adults_and_starts_feed(self):
        ram = make_coop_ram(adults=3, hay=50)
        task = CoopChoresTask()
        task.reset(make_world(ram))

        self.assertEqual(task._adult_count, 3)
        self.assertEqual(task._feed_remaining, 3)
        self.assertEqual(task._phase, "feed_nav")

    def test_reset_caps_feed_goal_when_configured(self):
        ram = make_coop_ram(adults=6, hay=50)
        task = CoopChoresTask(max_feed_adults=2)
        task.reset(make_world(ram))

        self.assertEqual(task._adult_count, 2)
        self.assertEqual(task._feed_remaining, 2)
        self.assertEqual(task._phase, "feed_nav")

    def test_reset_skips_feed_when_no_hay(self):
        ram = make_coop_ram(adults=3, hay=0, egg_available=True)
        task = CoopChoresTask()
        task.reset(make_world(ram))

        # No hay but has egg → should go to egg collection
        self.assertEqual(task._phase, "egg_nav")

    def test_reset_reads_live_wram_offset_values(self):
        ram = make_coop_ram(adults=1, hay=97, egg_available=True, live_offset=True)
        task = CoopChoresTask()
        task.reset(make_world(ram))

        self.assertEqual(task._adult_count, 1)
        self.assertEqual(task._feed_remaining, 1)
        self.assertEqual(task._hay_before, 97)
        self.assertEqual(task._phase, "feed_nav")

    def test_reset_skips_to_done_when_nothing_to_do(self):
        ram = make_coop_ram(adults=0, hay=0, egg_available=False)
        task = CoopChoresTask()
        task.reset(make_world(ram))

        self.assertEqual(task._phase, "exit_prep_nav")

    def test_feed_queues_correct_number_of_presses(self):
        """After navigating to feed bin, feed_act should queue a pickup press."""
        ram = make_coop_ram(adults=4, hay=50, player_tile=FEED_BIN_STAND)
        task = CoopChoresTask()
        task.reset(make_world(ram))

        # Manually advance to feed_act
        task._phase = "feed_act"
        task._feed_remaining = 4

        # Step to queue first feed
        world = make_world(ram)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "feed_verify")
        self.assertEqual(task.fed_count, 0)
        self.assertEqual(task._feed_remaining, 4)
        self.assertGreater(len(task._action_queue), 0)

    def test_feed_stops_when_hay_runs_out(self):
        ram = make_coop_ram(adults=5, hay=2, player_tile=FEED_BIN_STAND)
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "feed_act"
        task._feed_remaining = 5

        # Simulate hay running out after 2 feeds
        ram2 = ram.copy()
        ram2[ADDR_HAY_COUNT] = 0
        world = make_world(ram2)

        result = task.step(world)
        # Should transition to egg or done
        self.assertIn(task._phase, ("egg_nav", "exit_prep_nav"))

    def test_feed_verify_moves_to_place_when_feed_is_held(self):
        ram = make_coop_ram(
            adults=1,
            hay=96,
            egg_available=True,
            holding_feed=True,
            player_tile=FEED_BIN_STAND,
        )
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "feed_verify"

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "feed_place_nav")

    def test_feed_place_verify_resumes_egg_nav_after_flag_sets(self):
        ram = make_coop_ram(
            adults=1,
            hay=96,
            egg_available=True,
            holding_feed=False,
            fed_chickens=1,
            fed_chicken_flags=CHICKEN_FEED_SPOTS[0].flag,
            player_tile=FEED_CLEAR_STAND,
        )
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "feed_place_verify"
        task._adult_count = 1
        task._feed_remaining = 1
        task._fed_before = 0
        task._fed_flags_before = 0

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "egg_nav")

    def test_feed_place_nav_queues_place_press_when_at_open_slot(self):
        ram = make_coop_ram(adults=1, hay=96, egg_available=True, holding_feed=True, player_tile=FEED_CLEAR_STAND)
        px, py = CHICKEN_FEED_SPOTS[0].interact_px
        ram[ADDR_X] = px & 0xFF
        ram[ADDR_X + 1] = (px >> 8) & 0xFF
        ram[ADDR_Y] = py & 0xFF
        ram[ADDR_Y + 1] = (py >> 8) & 0xFF
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "feed_place_nav"
        task._navigator.update(ram)

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "feed_place_verify")
        self.assertGreater(len(task._action_queue), 0)

    def test_feed_act_does_not_queue_another_feed_while_item_is_still_on_hand(self):
        ram = make_coop_ram(adults=1, hay=96, egg_available=True, holding_feed=True, player_tile=FEED_BIN_STAND)
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "feed_act"
        task._feed_remaining = 1

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "feed_place_nav")
        self.assertEqual(len(task._action_queue), 0)

    def test_egg_collection_sets_flag(self):
        ram = make_coop_ram(adults=0, hay=0, egg_available=True, player_tile=EGG_PICKUP_STAND)
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "egg_verify"

        # Simulate egg picked up
        ram2 = ram.copy()
        ram2[ADDR_EGG_AVAILABLE] = 0
        world = make_world(ram2)
        task.step(world)

        self.assertTrue(task.egg_collected)
        self.assertEqual(task._phase, "decide")

    def test_egg_nav_still_attempts_pickup_while_feed_item_is_clearing(self):
        ram = make_coop_ram(adults=0, hay=0, egg_available=True, player_tile=EGG_PICKUP_STAND)
        ram[ADDR_ITEM_ON_HAND] = 0x1A
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "egg_nav"

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "egg_verify")
        self.assertGreater(len(task._action_queue), 0)

    def test_egg_skipped_when_not_available(self):
        ram = make_coop_ram(adults=0, hay=0, egg_available=False)
        task = CoopChoresTask()
        task.reset(make_world(ram))

        self.assertEqual(task._phase, "exit_prep_nav")

    def test_slot_egg_counts_as_available_when_flag_is_clear(self):
        ram = make_coop_ram(
            adults=3,
            slot_eggs=1,
            hay=50,
            egg_available=False,
            fed_chickens=3,
            fed_chicken_flags=0x0015,
            player_tile=(13, 11),
        )
        set_chicken_slot_position(ram, 3, (14, 11))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        self.assertTrue(ram_needs_chicken_chores(ram))
        self.assertEqual(task._phase, "egg_nav")
        self.assertEqual(task._egg_pickup_spot(ram), ((13, 11), "right"))

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "egg_verify")
        self.assertGreater(len(task._action_queue), 0)

    def test_incubator_egg_slot_does_not_count_as_available_floor_egg(self):
        ram = make_coop_ram(
            adults=0,
            slot_eggs=1,
            hay=50,
            egg_available=False,
            incubating=True,
            player_tile=(13, 11),
        )
        set_chicken_slot_position(ram, 0, (14, 11))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        self.assertFalse(ram_needs_chicken_chores(ram))
        self.assertEqual(task._phase, "exit_prep_nav")
        self.assertEqual(task._egg_tiles(ram), [])

    def test_incubator_visible_egg_object_does_not_count_as_floor_egg(self):
        ram = make_coop_ram(
            adults=0,
            hay=50,
            egg_available=False,
            incubating=True,
            player_tile=(13, 11),
        )
        add_egg_object(ram, (14, 11))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        self.assertFalse(task._egg_present(ram))
        self.assertEqual(task._phase, "exit_prep_nav")
        self.assertEqual(task._egg_tiles(ram), [])

    def test_egg_nav_timeout_skips_flag_instead_of_stalling(self):
        ram = make_coop_ram(
            adults=0,
            hay=0,
            egg_available=0x0001,
            player_tile=(1, 10),
        )
        block_tiles(ram, [(2, 3), (2, 5), (3, 4), (0, 4), (1, 5), (0, 5), (0, 3)])
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "egg_nav"
        task._egg_nav_started_step = 1
        task._current_egg_flag = 0x0001
        task._deferred_egg_counts[0x0001] = MAX_EGG_DEFERRALS
        task._step_count = MAX_EGG_NAV_FRAMES + 2

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn(0x0001, task._skipped_egg_flags)
        self.assertEqual(task._phase, "exit_prep_nav")

    def test_incubating_slot_egg_is_not_collision(self):
        ram = make_coop_ram(
            adults=0,
            slot_eggs=1,
            egg_available=False,
            incubating=True,
            player_tile=(12, 11),
        )
        set_chicken_slot_position(ram, 0, (14, 11))
        task = CoopChoresTask()
        task.reset(make_world(ram))

        self.assertNotIn((14, 11), task._chicken_tiles(ram))

    def test_begin_egg_nav_clears_stale_feed_route(self):
        ram = make_coop_ram(adults=0, hay=0, egg_available=0x0001, player_tile=(1, 10))
        task = CoopChoresTask()
        task.reset(make_world(ram))
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
        ram = make_coop_ram(
            adults=0,
            hay=50,
            egg_available=0x001C,
            player_tile=(4, 10),
        )
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)
        task._phase = "egg_nav"

        result = task.step(make_world(ram))

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
        ram = make_coop_ram(adults=3, incubating=False)
        task = CoopChoresTask(egg_mode="auto")
        task.reset(make_world(ram))
        task._phase = "decide"
        task.egg_collected = True

        task.step(make_world(ram))
        self.assertEqual(task._phase, "incubate_nav")

    def test_decide_auto_ships_when_incubator_full(self):
        ram = make_coop_ram(adults=3, incubating=True)
        task = CoopChoresTask(egg_mode="auto")
        task.reset(make_world(ram))
        task._phase = "decide"
        task.egg_collected = True

        task.step(make_world(ram))
        self.assertEqual(task._phase, "ship_nav")

    def test_decide_auto_ships_when_flock_at_max(self):
        ram = make_coop_ram(adults=MAX_FLOCK_SIZE, incubating=False)
        task = CoopChoresTask(egg_mode="auto")
        task.reset(make_world(ram))
        task._phase = "decide"
        task.egg_collected = True

        task.step(make_world(ram))
        self.assertEqual(task._phase, "ship_nav")

    def test_decide_gift_mode_goes_to_done(self):
        ram = make_coop_ram(adults=1, incubating=False)
        task = CoopChoresTask(egg_mode="gift")
        task.reset(make_world(ram))
        task._phase = "decide"
        task.egg_collected = True

        result = task.step(make_world(ram))
        self.assertEqual(task._phase, "exit_prep_nav")

    def test_incubate_verify_succeeds(self):
        ram = make_coop_ram(adults=1, egg_available=False, incubating=False, player_tile=INCUBATOR_STAND)
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "incubate_verify"

        # Simulate incubator bit set
        ram2 = ram.copy()
        flags = INCUBATOR_BIT | 0x0400
        ram2[ADDR_INCUBATOR_FLAGS] = flags & 0xFF
        ram2[ADDR_INCUBATOR_FLAGS + 1] = (flags >> 8) & 0xFF
        task.step(make_world(ram2))

        self.assertTrue(task.egg_incubated)
        self.assertEqual(task._phase, "exit_prep_nav")

    def test_incubate_nav_recovers_to_left_approach_when_above_incubator(self):
        ram = make_coop_ram(adults=1, holding_egg=True, player_tile=(13, 9))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "incubate_nav"

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_incubate_nav_queues_a_only_press_from_left_stand(self):
        ram = make_coop_ram(adults=1, holding_egg=True, player_tile=INCUBATOR_STAND)
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "incubate_nav"
        task._navigator.update(ram)

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "incubate_verify")
        self.assertGreater(len(task._action_queue), 8)
        first = task._action_queue[0]
        self.assertEqual(int(first[7]), 1)  # face right into the incubator
        press = task._action_queue[8]
        self.assertEqual(int(press[8]), 1)
        self.assertEqual(int(press[7]), 0)

    def test_ship_verify_succeeds(self):
        ram = make_coop_ram(adults=1, egg_available=False, player_tile=SHIP_BIN_STAND, shipping_money=0)
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "ship_verify"
        task._ship_money_before = 0

        # Simulate shipping money increase
        ram2 = ram.copy()
        ram2[ADDR_SHIPPING_MONEY] = 5
        task.step(make_world(ram2))

        self.assertTrue(task.egg_shipped)
        self.assertEqual(task._phase, "exit_prep_nav")

    def test_ship_verify_fails_when_egg_clears_without_money(self):
        ram = make_coop_ram(adults=1, holding_egg=False, player_tile=SHIP_BIN_STAND, shipping_money=0)
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "ship_verify"
        task._ship_money_before = 0

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("without shipping money", result.reason)
        self.assertFalse(task.egg_shipped)

    def test_next_feed_spot_skips_filled_and_occupied_slots(self):
        ram = make_coop_ram(
            adults=2,
            egg_available=False,
            holding_feed=True,
            fed_chicken_flags=CHICKEN_FEED_SPOTS[0].flag,
            player_tile=(2, 7),
        )
        set_chicken_slot_position(ram, 0, CHICKEN_FEED_SPOTS[1].stand)
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        spot = task._next_feed_spot(ram)

        self.assertIsNotNone(spot)
        self.assertEqual(spot, CHICKEN_FEED_SPOTS[2])

    def test_next_feed_spot_skips_stuck_slots(self):
        ram = make_coop_ram(adults=2, egg_available=False, holding_feed=True, player_tile=(3, 6))
        task = CoopChoresTask()
        task.reset(make_world(ram))
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

    def test_scales_to_12_chickens(self):
        """Reset with 12 adults should plan 12 feeds."""
        ram = make_coop_ram(adults=12, hay=100)
        task = CoopChoresTask()
        task.reset(make_world(ram))

        self.assertEqual(task._adult_count, 12)
        self.assertEqual(task._feed_remaining, 12)

    def test_mixed_flock_only_feeds_adults(self):
        """Chicks and eggs don't need feeding."""
        ram = make_coop_ram(adults=5, chicks=3, slot_eggs=2, hay=100)
        task = CoopChoresTask()
        task.reset(make_world(ram))

        self.assertEqual(task._adult_count, 5)
        self.assertEqual(task._feed_remaining, 5)

    def test_auto_ships_when_chicks_plus_adults_at_max(self):
        """Total flock (adults + chicks + eggs) >= MAX means no incubation."""
        ram = make_coop_ram(adults=8, chicks=3, slot_eggs=1, incubating=False)
        task = CoopChoresTask(egg_mode="auto")
        task.reset(make_world(ram))
        task._phase = "decide"
        task.egg_collected = True

        task.step(make_world(ram))
        self.assertEqual(task._phase, "ship_nav")

    def test_timeout_returns_failure(self):
        ram = make_coop_ram(adults=1)
        task = CoopChoresTask(timeout=5)
        task.reset(make_world(ram))

        for _ in range(10):
            result = task.step(make_world(ram))
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
        ram = make_coop_ram(adults=3, hay=50)
        task = CoopChoresTask()
        task.reset(make_world(ram))
        text = task.progress_text
        self.assertIn("fed=0/3", text)
        self.assertIn("egg=N", text)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
