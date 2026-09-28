"""Unit tests for Coop multi-adult feeding, dynamic egg tile recovery, and composable chore skills (rr-rbk)."""

from __future__ import annotations

import unittest
import numpy as np

from harvest.core.animal_probe import COOP_TILEMAP
from harvest.core.animal_status import (
    ADDR_CHICKEN_COUNT,
    ADDR_EGG_AVAILABLE,
    ADDR_FED_CHICKENS_FLAGS,
    ADDR_FED_CHICKENS_N,
    ADDR_HAY_COUNT,
    CHICKEN_SLOT_BASE,
    CHICKEN_SLOT_COUNT,
    CHICKEN_SLOT_SIZE,
)
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.tasks.coop_layout import (
    CHICKEN_FEED_FLAGS,
    CHICKEN_FEED_SPOTS,
    MAX_EGG_DEFERRALS,
)
from harvest.tasks.coop_task import CoopChoresTask
from harvest.tasks.coop_feed_ops import _advance_after_feed, _next_feed_spot
from harvest.tasks.coop_egg_ops import (
    _after_egg_nav_budget,
    _collectable_egg_present,
    _defer_or_skip_egg,
    _egg_pickup_spot,
)
from harvest.tasks.skills import (
    CoopCollectEggsSkill,
    CoopEggDispositionSkill,
    CoopExitStagingSkill,
    CoopFeedAdultsSkill,
    coop_chores_composed_task,
)
from retro_harness import TaskStatus, WorldState


import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from coop_task_test_helpers import (
    make_coop_ram,
    make_world,
    set_chicken_slot_position,
)


class TestCoopMultiAdultFeeding(unittest.TestCase):
    def test_next_feed_spot_unblocks_when_all_unfed_are_blocked(self):
        """When wandering chickens cause all open slots to be temporarily blocked,
        _next_feed_spot should clear blocked flags to allow retrying open slots."""
        ram = make_coop_ram(adults=12, hay=50, fed_chicken_flags=0x000F)  # first 4 fed
        task = CoopChoresTask()
        task.reset(make_world(ram))

        # Mark all remaining unfed spots (flags 0x0010 through 0x0800) as blocked
        for spot in CHICKEN_FEED_SPOTS[4:]:
            task._blocked_feed_flags.add(spot.flag)

        # _next_feed_spot should unblock them and return a valid spot for retry
        spot = _next_feed_spot(task, ram)
        self.assertIsNotNone(spot)
        self.assertNotIn(spot.flag, (0x0001, 0x0002, 0x0004, 0x0008))
        self.assertIn(spot.flag, CHICKEN_FEED_FLAGS)

    def test_advance_after_feed_clears_pathfinder_temp_blocked(self):
        """Feeding multiple chickens should clear temporary pathfinder blocks
        so navigation routes don't permanently clog over 12 feeds."""
        ram = make_coop_ram(adults=12, hay=50)
        task = CoopChoresTask()
        task.reset(make_world(ram))

        task._pathfinder.temp_blocked.add((5, 5))
        task._pathfinder.temp_blocked.add((6, 6))

        _advance_after_feed(task, ram)
        self.assertEqual(len(task._pathfinder.temp_blocked), 0)

    def test_advance_after_feed_stops_when_hay_exhausted(self):
        """If hay runs out before all adults are fed, feeding terminates cleanly
        and routes to egg nav or exit prep."""
        ram = make_coop_ram(adults=12, hay=0, fed_chickens=2, egg_available=False)
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task.fed_count = 2
        task._feed_remaining = 10

        res = _advance_after_feed(task, ram)
        self.assertEqual(task._feed_remaining, 0)
        self.assertEqual(task._phase, "exit_prep_nav")


class TestCoopDynamicEggRecovery(unittest.TestCase):
    def test_dynamic_floor_egg_tile_defer_and_skip(self):
        """Dynamic floor egg tiles without bitflags (flag=0) should be deferred
        and eventually skipped, preventing the Spring 22 infinite loop."""
        ram = make_coop_ram(adults=6, hay=50, egg_available=0)
        # Add a dynamic floor egg via chicken slot stage="egg"
        addr = CHICKEN_SLOT_BASE + 6 * CHICKEN_SLOT_SIZE
        ram[addr] = 0x01 | (0 << 1)  # stage egg
        ram[addr + 2] = 0x28
        egg_tile = (5, 8)
        ram[addr + 4] = egg_tile[0] * 16  # x
        ram[addr + 5] = 0
        ram[addr + 6] = egg_tile[1] * 16  # y
        ram[addr + 7] = 0

        task = CoopChoresTask()
        task.reset(make_world(ram))

        # Spot lookup finds the dynamic egg
        spot = _egg_pickup_spot(task, ram, require_path=False)
        self.assertIsNotNone(spot)
        self.assertEqual(task._current_egg_flag, 0)
        self.assertEqual(task._current_egg_tile, egg_tile)

        # Defer MAX_EGG_DEFERRALS times
        for count in range(1, MAX_EGG_DEFERRALS + 1):
            task._current_egg_flag = 0
            task._current_egg_tile = egg_tile
            _defer_or_skip_egg(task, "test_stasis")
            self.assertEqual(task._deferred_egg_tile_counts[egg_tile], count)

        # One more deferral should skip the dynamic egg tile
        task._current_egg_flag = 0
        task._current_egg_tile = egg_tile
        _defer_or_skip_egg(task, "test_stasis")
        self.assertIn(egg_tile, task._skipped_egg_tiles)

        # Now collectable egg present should return False and pickup spot should be None
        self.assertFalse(_collectable_egg_present(task, ram))
        self.assertIsNone(_egg_pickup_spot(task, ram, require_path=False))


class TestComposableCoopSkills(unittest.TestCase):
    def test_feed_adults_skill_finishes_when_fed(self):
        ram = make_coop_ram(adults=2, hay=50, fed_chickens=2, fed_chicken_flags=0x0003)
        skill = CoopFeedAdultsSkill(max_feed_adults=2)
        world = make_world(ram)
        skill.reset(world)

        res = skill.step(world)
        self.assertEqual(res.status, TaskStatus.SUCCESS)

    def test_collect_eggs_skill_finishes_when_egg_collected(self):
        ram = make_coop_ram(adults=2, hay=50)
        skill = CoopCollectEggsSkill()
        world = make_world(ram)
        skill.reset(world)
        skill._inner.egg_collected = True

        res = skill.step(world)
        self.assertEqual(res.status, TaskStatus.SUCCESS)

    def test_egg_disposition_skill_finishes_at_exit_prep(self):
        ram = make_coop_ram(adults=2, hay=50)
        skill = CoopEggDispositionSkill(egg_mode="ship")
        world = make_world(ram)
        skill.reset(world)
        skill._inner._phase = "exit_prep_nav"

        res = skill.step(world)
        self.assertEqual(res.status, TaskStatus.SUCCESS)

    def test_exit_staging_skill_finishes_at_done(self):
        ram = make_coop_ram(adults=2, hay=50)
        skill = CoopExitStagingSkill()
        world = make_world(ram)
        skill.reset(world)
        skill._inner._phase = "done"

        res = skill.step(world)
        self.assertEqual(res.status, TaskStatus.SUCCESS)

    def test_coop_chores_composed_task_sequence(self):
        seq = coop_chores_composed_task(egg_mode="auto", max_feed_adults=12)
        self.assertEqual(len(seq.tasks), 4)
        self.assertIsInstance(seq.tasks[0], CoopFeedAdultsSkill)
        self.assertIsInstance(seq.tasks[1], CoopCollectEggsSkill)
        self.assertIsInstance(seq.tasks[2], CoopEggDispositionSkill)
        self.assertIsInstance(seq.tasks[3], CoopExitStagingSkill)


if __name__ == "__main__":
    unittest.main()
