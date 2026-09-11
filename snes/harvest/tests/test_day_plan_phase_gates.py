"""Phase gates: tool lock and the map-lock exit.

run13 D10 bought no seed bag (BUY_SEEDS_WINDOW blew its cutoff), then
CROP_ESTABLISH hoed the whole ring before ``select_carry_0x07`` reported
"0x07 not in carry pair" — an afternoon spent to reach a settled failure.
"""
from __future__ import annotations

import unittest

from harvest.core.carry import ADDR_TOOL_BACKPACK, ADDR_TOOL_SELECTED
from harvest.core.tile_catalog import Tool
from harvest.planner.day_phase_catalog import (
    CROP_ESTABLISH_PHASE,
    CROP_WATER_PHASE,
    ENSURE_CROP_SEEDS_PHASE,
    ENSURE_WATERING_CAN_PHASE,
    EXIT_TO_FARM_PHASE,
)
from harvest.planner.day_plan_orchestrator import DayPlanTask, PhaseSchedule

from tests.day_plan_test_helpers import make_world


def _carry(world, selected: int, backpack: int) -> None:
    world.ram[ADDR_TOOL_SELECTED] = selected
    world.ram[ADDR_TOOL_BACKPACK] = backpack


class ToolLockTests(unittest.TestCase):
    def _task(self) -> DayPlanTask:
        return DayPlanTask.__new__(DayPlanTask)

    def test_establish_without_a_seed_bag_is_no_work(self) -> None:
        world = make_world(0x00)
        _carry(world, int(Tool.HOE), int(Tool.WATERING_CAN))
        reason = self._task()._phase_tool_lock(CROP_ESTABLISH_PHASE, world)
        self.assertIsNotNone(reason)
        self.assertTrue(reason.startswith("no_work:"), reason)
        self.assertIn("seed", reason)

    def test_establish_with_hoe_and_seed_passes(self) -> None:
        world = make_world(0x00)
        _carry(world, int(Tool.HOE), 0x07)  # potato bag
        self.assertIsNone(self._task()._phase_tool_lock(CROP_ESTABLISH_PHASE, world))

    def test_a_missing_hoe_is_not_gated(self) -> None:
        """The hoe comes off a shed shelf; fetching beats skipping."""
        world = make_world(0x00)
        _carry(world, 0x07, int(Tool.WATERING_CAN))
        self.assertIsNone(self._task()._phase_tool_lock(CROP_ESTABLISH_PHASE, world))

    def test_a_missing_watering_can_is_not_gated(self) -> None:
        """CROP_WATER already routes a missing can to EnsureCarryToolTask —
        gating it here would replace a fetch with a skip."""
        world = make_world(0x00)
        _carry(world, int(Tool.HOE), 0x07)
        self.assertIsNone(self._task()._phase_tool_lock(CROP_WATER_PHASE, world))

    def test_ensure_phases_are_exempt(self) -> None:
        """ENSURE_* declares what it fetches; locking it out breaks the chain
        that puts the seed bag into the carry pair in the first place."""
        world = make_world(0x00)
        _carry(world, 0, 0)
        self.assertIsNone(self._task()._phase_tool_lock(ENSURE_CROP_SEEDS_PHASE, world))
        self.assertIsNone(
            self._task()._phase_tool_lock(ENSURE_WATERING_CAN_PHASE, world)
        )

    def test_phase_without_a_tool_contract_is_never_locked(self) -> None:
        world = make_world(0x15)
        _carry(world, 0, 0)
        self.assertIsNone(self._task()._phase_tool_lock(EXIT_TO_FARM_PHASE, world))


class MapLockExitTests(unittest.TestCase):
    """A farm phase must not forfeit the day because the farmer wandered inside.

    run12 D8/D9: CLEAR_FIELD walked indoors mid-phase, so NAV_CROP,
    HARVEST_ROUTE and both berry phases map-locked on 0x15 and the day earned
    nothing. The farmhouse is one EXIT_TO_FARM away.
    """

    def _task(self) -> DayPlanTask:
        task = DayPlanTask.__new__(DayPlanTask)
        task._schedule = PhaseSchedule.from_phases([CROP_WATER_PHASE])
        task._phase_index = 0
        task._current_task = object()
        task._map_lock_exits = set()
        return task

    def test_indoors_splices_an_exit_and_retries(self) -> None:
        task = self._task()
        world = make_world(0x15)
        self.assertTrue(task._try_map_lock_exit(CROP_WATER_PHASE, world, "map_mismatch"))
        self.assertEqual(
            [p.phase for p in task._schedule.active],
            ["EXIT_TO_FARM", "CROP_WATER"],
        )
        self.assertIsNone(task._current_task)

    def test_it_only_fires_once_per_phase_per_day(self) -> None:
        task = self._task()
        world = make_world(0x15)
        task._try_map_lock_exit(CROP_WATER_PHASE, world, "map_mismatch")
        self.assertFalse(
            task._try_map_lock_exit(CROP_WATER_PHASE, world, "map_mismatch")
        )

    def test_off_farm_but_not_indoors_is_left_alone(self) -> None:
        """Stranded on the mountain is a route problem, not a door problem."""
        task = self._task()
        self.assertFalse(
            task._try_map_lock_exit(CROP_WATER_PHASE, make_world(0x10), "map_mismatch")
        )

    def test_a_phase_with_no_farm_requirement_is_left_alone(self) -> None:
        task = self._task()
        self.assertFalse(
            task._try_map_lock_exit(EXIT_TO_FARM_PHASE, make_world(0x15), "map_mismatch")
        )


if __name__ == "__main__":
    unittest.main()
