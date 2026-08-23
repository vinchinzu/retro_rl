"""D2 work-section composition — quotas, order, shop-splice recombination."""

from __future__ import annotations

import unittest

from harvest.core.stamina import Stamina
from harvest.core.tile_catalog import Tool
from harvest.planner.d2_work import (
    D2_QUOTAS,
    bush_quota_phase,
    d2_leftover_phases,
    d2_post_shop_work_phases,
    ensure_axe_phase,
    ensure_hammer_phase,
    leftover_already_queued,
    pocket_water_phase,
    rock_quota_phase,
    stump_quota_phase,
)
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseKind
from harvest.planner.day_plan_phases import pocket_plant_phases


class D2QuotaContractTests(unittest.TestCase):
    def test_quotas_match_d2_work_day(self) -> None:
        self.assertEqual(D2_QUOTAS["plant"], 8)
        self.assertEqual(D2_QUOTAS["water"], 8)
        self.assertEqual(D2_QUOTAS["bushes"], 10)
        self.assertEqual(D2_QUOTAS["small_rocks"], 10)
        self.assertEqual(D2_QUOTAS["large_boulders"], 4)
        self.assertEqual(D2_QUOTAS["stumps"], 2)

    def test_bush_phase_is_lift_quota_not_plot_ring(self) -> None:
        spec = bush_quota_phase()
        self.assertEqual(spec.phase, "CLEAR_BUSHES")
        self.assertEqual(spec.kind, PhaseKind.CLEAR_FIELD)
        self.assertEqual(spec.params["handoff"], "quota")
        self.assertEqual(spec.params["quota"]["weeds"], 10)
        self.assertFalse(spec.params["fetch_tools"])
        self.assertEqual(spec.params["priority"], ["weed"])
        self.assertNotIn("farm_bounds", spec.params)

    def test_rock_phase_needs_hammer_and_counts_small_plus_large(self) -> None:
        spec = rock_quota_phase()
        self.assertEqual(spec.phase, "CLEAR_ROCKS")
        self.assertEqual(spec.params["quota"]["small_rocks"], 10)
        self.assertEqual(spec.params["quota"]["large_rocks"], 4)
        self.assertEqual(spec.params["priority"], ["rock"])
        self.assertEqual(spec.contract.required_tools, ("hammer",))
        self.assertFalse(spec.params["fetch_tools"])

    def test_stump_phase_needs_axe(self) -> None:
        spec = stump_quota_phase()
        self.assertEqual(spec.phase, "CLEAR_STUMPS")
        self.assertEqual(spec.params["quota"]["stumps"], 2)
        self.assertEqual(spec.params["priority"], ["stump"])
        self.assertEqual(spec.contract.required_tools, ("axe",))

    def test_ensure_hammer_and_axe_are_ram_shelf_not_recorded(self) -> None:
        hammer = ensure_hammer_phase()
        axe = ensure_axe_phase()
        self.assertEqual(hammer.kind, PhaseKind.ENSURE_TOOL)
        self.assertEqual(axe.kind, PhaseKind.ENSURE_TOOL)
        self.assertEqual(hammer.params["tool_id"], int(Tool.HAMMER))
        self.assertEqual(axe.params["tool_id"], int(Tool.AXE))
        self.assertNotEqual(hammer.kind, PhaseKind.RECORDED)


class D2LeftoverOrderTests(unittest.TestCase):
    def test_low_stam_inserts_spa_before_hammer_work(self) -> None:
        phases = d2_leftover_phases(stamina=Stamina(current=8, maximum=100))
        names = [p.phase for p in phases]
        self.assertEqual(names[0], "HOT_SPRING_STAMINA")
        self.assertLess(names.index("CLEAR_BUSHES"), names.index("ENSURE_HAMMER"))
        self.assertLess(names.index("ENSURE_HAMMER"), names.index("CLEAR_ROCKS"))
        self.assertLess(names.index("CLEAR_ROCKS"), names.index("ENSURE_AXE"))
        self.assertLess(names.index("ENSURE_AXE"), names.index("CLEAR_STUMPS"))
        self.assertNotIn("CLEAR_FIELD", names)

    def test_full_stam_skips_spa_but_keeps_smash_order(self) -> None:
        phases = d2_leftover_phases(stamina=Stamina(current=100, maximum=100))
        names = [p.phase for p in phases]
        self.assertNotIn("HOT_SPRING_STAMINA", names)
        self.assertEqual(
            names,
            [
                "CLEAR_BUSHES",
                "ENSURE_HAMMER",
                "CLEAR_ROCKS",
                "ENSURE_AXE",
                "CLEAR_STUMPS",
            ],
        )

    def test_policy_can_drop_leftover(self) -> None:
        phases = d2_leftover_phases(
            stamina=Stamina(current=4, maximum=100),
            policy=DayPlannerPolicy(include_field_clear=False),
        )
        self.assertEqual(phases, [])

    def test_hammer_and_axe_are_sequential_not_same_carry(self) -> None:
        names = [p.phase for p in d2_leftover_phases()]
        self.assertLess(names.index("CLEAR_ROCKS"), names.index("ENSURE_AXE"))
        self.assertEqual(names.count("ENSURE_HAMMER"), 1)
        self.assertEqual(names.count("ENSURE_AXE"), 1)


class D2PostShopComposeTests(unittest.TestCase):
    def test_post_shop_is_plant_water_then_leftover(self) -> None:
        names = [p.phase for p in d2_post_shop_work_phases()]
        self.assertEqual(
            names[:6],
            [
                "ENSURE_CROP_SEEDS",
                "CLEAR_PLOT",
                "NAV_CROP",
                "CROP_ESTABLISH",
                "ENSURE_WATERING_CAN",
                "CROP_WATER",
            ],
        )
        self.assertLess(names.index("CROP_WATER"), names.index("CLEAR_BUSHES"))
        self.assertLess(names.index("CROP_ESTABLISH"), names.index("CROP_WATER"))
        self.assertNotIn("CLEAR_FIELD", names)
        self.assertEqual(pocket_water_phase().params["work_mode"], "pocket")
        self.assertEqual(pocket_water_phase().params["min_wet"], 8)

    def test_pocket_plant_phases_delegate_to_d2_work(self) -> None:
        plant = [p.phase for p in pocket_plant_phases()]
        composed = [p.phase for p in d2_post_shop_work_phases()]
        self.assertEqual(plant, composed)

    def test_leftover_already_queued(self) -> None:
        self.assertTrue(leftover_already_queued(["CROP_WATER", "CLEAR_ROCKS"]))
        self.assertFalse(leftover_already_queued(["CROP_WATER", "RETURN_HOME"]))
        self.assertFalse(leftover_already_queued(["HOT_SPRING_STAMINA"]))


if __name__ == "__main__":
    unittest.main()
