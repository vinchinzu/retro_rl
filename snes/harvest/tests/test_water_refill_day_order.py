"""Unit tests for crop phase order vs keep-alive clear."""

from __future__ import annotations

import unittest


class SameDayEstablishWaterOrderTests(unittest.TestCase):
    """Day-plan crop pass order for same-day plant then water (rr-e1p)."""

    def test_crop_establish_then_water_phases_order(self) -> None:
        from harvest.planner.day_plan_phases import crop_establish_phases, crop_water_phases

        establish = [p.phase for p in crop_establish_phases()]
        water = [p.phase for p in crop_water_phases()]
        self.assertIn("CROP_ESTABLISH", establish)
        self.assertIn("ENSURE_CROP_SEEDS", establish)
        self.assertNotIn("ENSURE_WATERING_CAN", establish)
        self.assertIn("ENSURE_WATERING_CAN", water)
        self.assertIn("CROP_WATER", water)
        # Plant pass before water pass when both are scheduled.
        from harvest.planner.day_plan_phases import _crop_work_phases

        from harvest.planner.day_phase_types import DayPlannerPolicy

        phases = _crop_work_phases(
            has_harvest=False,
            has_waterable=True,
            has_seeds=True,
            is_rainy=False,
            late_day=False,
            policy=DayPlannerPolicy(),
        )
        names = [p.phase for p in phases]
        if "CROP_ESTABLISH" in names and "CROP_WATER" in names:
            self.assertLess(names.index("CROP_ESTABLISH"), names.index("CROP_WATER"))
            self.assertLess(
                names.index("CROP_ESTABLISH"),
                names.index("ENSURE_WATERING_CAN"),
            )


class KeepAliveClearOrderTests(unittest.TestCase):
    """CLEAR_FIELD must not starve crop keep-alive water (rr-3v9)."""

    def test_outdoor_water_before_clear_when_dry_crops(self) -> None:
        from harvest.planner.day_plan_phases import build_outdoor_day_phases

        phases = build_outdoor_day_phases(
            weekday=3,
            hour=6,
            has_harvest=False,
            has_waterable=True,
            has_seeds=False,
            has_debris=True,
            is_rainy=False,
        )
        names = [p.phase for p in phases]
        self.assertIn("CROP_WATER", names)
        self.assertIn("CLEAR_FIELD", names)
        self.assertLess(names.index("CROP_WATER"), names.index("CLEAR_FIELD"))
        self.assertLess(names.index("ENSURE_WATERING_CAN"), names.index("CLEAR_FIELD"))

    def test_outdoor_empty_morning_defers_clear_until_after_berry_window(self) -> None:
        from harvest.planner.day_plan_phases import build_outdoor_day_phases

        phases = build_outdoor_day_phases(
            weekday=3,
            hour=6,
            has_harvest=False,
            has_waterable=False,
            has_seeds=False,
            has_debris=True,
            is_rainy=False,
        )
        names = [p.phase for p in phases]
        self.assertIn("SHIP_BERRY_1", names)
        self.assertNotIn("CLEAR_FIELD", names)
        self.assertNotIn("CROP_WATER", names)

    def test_full_day_water_before_clear_when_dry_crops(self) -> None:
        from harvest.planner.day_plan_phases import build_day_phases
        from harvest.planner.day_phase_types import DayPlannerPolicy

        phases = build_day_phases(
            None,
            weekday=3,
            hour=6,
            has_chickens=False,
            has_cows=False,
            has_harvest=False,
            has_waterable=True,
            has_seeds=False,
            has_debris=True,
            policy=DayPlannerPolicy(
                include_chickens=False,
                include_cows=False,
                include_shop_run=False,
                include_berry_run=False,
                include_end_day=False,
            ),
        )
        names = [p.phase for p in phases]
        self.assertIn("CROP_WATER", names)
        self.assertIn("CLEAR_FIELD", names)
        self.assertLess(names.index("CROP_WATER"), names.index("CLEAR_FIELD"))


if __name__ == "__main__":
    unittest.main()
