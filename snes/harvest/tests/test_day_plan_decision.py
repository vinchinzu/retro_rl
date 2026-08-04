from __future__ import annotations

import unittest
from typing import Optional

import numpy as np

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET
from harvest.planner.day_plan import (
    ADDR_DAY,
    ADDR_HOUR,
    ADDR_MINUTE,
    ADDR_POTATO_SEEDS,
    ADDR_TILEMAP,
    ADDR_WEEKDAY,
    DayPlannerPolicy,
    WorldProbe,
)
from harvest.planner.day_plan_decision import (
    DayPlanDecision,
    PlanningFacts,
    auto_day_plan_decision,
    build_day_plan_decision,
    collect_deferred_plans,
)
from harvest.planner.local_llm import apply_advisor_patch


def _ram(*, tilemap: int = 0x00, weekday: int = 1, hour: int = 6, minute: int = 0) -> np.ndarray:
    ram = np.zeros(0x24000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    base = LIVE_RAM_WRAM_OFFSET
    ram[ADDR_DAY + base] = 12
    ram[ADDR_WEEKDAY + base] = weekday
    ram[ADDR_HOUR + base] = hour
    ram[ADDR_MINUTE + base] = minute
    return ram


class _CountingStateLoader:
    def __init__(self, ram: np.ndarray) -> None:
        self.ram = ram
        self.calls = 0

    def __call__(self, state_name: Optional[str]) -> Optional[np.ndarray]:
        self.calls += 1
        return self.ram if state_name else None


class _Advisor:
    def advise_day_plan(self, decision: DayPlanDecision) -> DayPlanDecision:
        return decision.with_notes(["advisor saw plan"], source="fake_advisor")


class DayPlanDecisionTests(unittest.TestCase):
    def test_build_decision_exposes_jsonable_plan(self) -> None:
        decision = build_day_plan_decision(ram=_ram(hour=6))

        self.assertEqual(decision.facts.source, "ram")
        self.assertIn("BUY_SEEDS", decision.phase_names)
        payload = decision.to_jsonable()
        self.assertEqual(payload["facts"]["hour"], 6)
        self.assertIn("phases", payload)
        self.assertIsInstance(payload["phases"][0]["params"], dict)

    def test_world_probe_caches_state_ram_for_multiple_facts(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_WEEKDAY] = 2
        ram[ADDR_HOUR] = 9
        ram[ADDR_MINUTE] = 15
        ram[ADDR_POTATO_SEEDS] = 1
        loader = _CountingStateLoader(ram)
        probe = WorldProbe(
            state_name="fake",
            load_state_ram=loader,
        )

        self.assertEqual(probe.weekday(), 2)
        self.assertEqual(probe.day_time(), (0, 9, 15))
        self.assertTrue(probe.has_any_crop_seeds())
        self.assertEqual(loader.calls, 1)

    def test_collect_deferred_plans_for_tomorrow(self) -> None:
        facts = PlanningFacts(
            source="test",
            weekday=1,
            hour=18,
            late_day=True,
            needs_cows=True,
            has_harvest=True,
            has_seeds=True,
        )

        deferred = collect_deferred_plans(facts, [], policy=DayPlannerPolicy())

        deferred_by_phase = {item.phase: item.reason for item in deferred}
        self.assertEqual(deferred_by_phase["COW_CHORES"], "late_day")
        self.assertEqual(deferred_by_phase["HARVEST_ROUTE"], "late_day")
        self.assertEqual(deferred_by_phase["CROP_WATER"], "late_day")
        self.assertTrue(all(item.retry == "tomorrow" for item in deferred))

    def test_auto_day_plan_decision_accepts_advisor(self) -> None:
        decision = auto_day_plan_decision(ram=_ram(), advisor=_Advisor())

        self.assertEqual(decision.source, "fake_advisor")
        self.assertIn("advisor saw plan", decision.notes)

    def test_local_llm_patch_adds_notes_and_deferrals_without_phase_changes(self) -> None:
        decision = build_day_plan_decision(ram=_ram())

        patched = apply_advisor_patch(
            decision,
            {
                "notes": ["model recommends checking cows tomorrow"],
                "deferred": [{"phase": "COW_CHORES", "reason": "model"}],
                "phase_names": ["UNSAFE_REWRITE"],
            },
            source="rules+local_llm",
        )

        self.assertEqual(patched.phase_names, decision.phase_names)
        self.assertIn("model recommends checking cows tomorrow", patched.notes)
        self.assertIn("advisor_phase_changes_ignored", patched.notes)
        self.assertIn("COW_CHORES", [item.phase for item in patched.deferred])


if __name__ == "__main__":
    unittest.main()
