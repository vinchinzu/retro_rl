"""D2 power-on spine gate: helper rejects, live evidence contract.

The save gate is phase-name + calendar, not shipping_money RAM. Tests lock
that contract and the recorded power-on JSON when present.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from retro_harness import WorldState

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.scripts.run_to_day2 import _d2_spine_checkpoint_evidence

_HARVEST_DIR = Path(__file__).resolve().parents[1]
_SPINE_EVIDENCE = _HARVEST_DIR / "recordings" / "power_on_d2_spine_clear_final.json"
_FAILED_D1_PIN = _HARVEST_DIR / "recordings" / "d2_spine_post_shipper.json"

_REQUIRED_PHASES = (
    {"phase": "MOUNTAIN_BERRY", "status": "success"},
    {"phase": "BUY_SEEDS", "status": "success"},
    {"phase": "WAIT_FARM_SHIPPING", "status": "success"},
)


def _spine_world(*, hour: int, day: int = 2, potato_seeds: int = 0) -> WorldState:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = 0x00
    for key, value in (
        ("season", 0),
        ("day", day),
        ("hour", hour),
        ("minute", 0),
        ("potato_seeds", potato_seeds),
    ):
        addr = field_spec(key).address
        for base in (0, LIVE_RAM_WRAM_OFFSET):
            if addr + base < len(ram):
                ram[addr + base] = value
    return WorldState(frame=0, ram=ram, info={}, obs=None)


def _spine_task(phases: list[dict], *, phase: str = "return_home"):
    task = type("SpineTask", (), {})()
    task._phase = phase
    task._last_day_phase_results = phases
    return task


class D2SpineHelperTests(unittest.TestCase):
    def test_ready_requires_d2_5pm_return_home_and_three_phases(self) -> None:
        evidence = _d2_spine_checkpoint_evidence(
            _spine_task(list(_REQUIRED_PHASES)),
            _spine_world(hour=17, potato_seeds=1),
        )
        self.assertTrue(evidence["ready"])
        self.assertTrue(evidence["mountain_grape_shipped"])
        self.assertTrue(evidence["potato_purchase_complete"])
        self.assertTrue(evidence["shipping_dialogue_cleared"])

    def test_grape_flag_is_phase_name_not_shipping_money(self) -> None:
        """Bin RAM is not in the gate. MOUNTAIN_BERRY success is the grape proxy."""
        evidence = _d2_spine_checkpoint_evidence(
            _spine_task(list(_REQUIRED_PHASES)),
            _spine_world(hour=17, potato_seeds=0),
        )
        self.assertTrue(evidence["mountain_grape_shipped"])
        self.assertNotIn("shipping_money", evidence)

    def test_rejects_d3_overnight_even_with_phase_names(self) -> None:
        evidence = _d2_spine_checkpoint_evidence(
            _spine_task(list(_REQUIRED_PHASES)),
            _spine_world(hour=6, day=3, potato_seeds=0),
        )
        self.assertFalse(evidence["ready"])

    def test_rejects_hour_before_5pm(self) -> None:
        evidence = _d2_spine_checkpoint_evidence(
            _spine_task(list(_REQUIRED_PHASES)),
            _spine_world(hour=16, potato_seeds=1),
        )
        self.assertFalse(evidence["ready"])

    def test_rejects_missing_wait_farm_shipping(self) -> None:
        phases = [
            {"phase": "MOUNTAIN_BERRY", "status": "success"},
            {"phase": "BUY_SEEDS", "status": "success"},
        ]
        evidence = _d2_spine_checkpoint_evidence(
            _spine_task(phases),
            _spine_world(hour=17, potato_seeds=1),
        )
        self.assertFalse(evidence["ready"])
        self.assertFalse(evidence["shipping_dialogue_cleared"])

    def test_rejects_missing_buy_seeds(self) -> None:
        phases = [
            {"phase": "MOUNTAIN_BERRY", "status": "success"},
            {"phase": "WAIT_FARM_SHIPPING", "status": "success"},
        ]
        evidence = _d2_spine_checkpoint_evidence(
            _spine_task(phases),
            _spine_world(hour=17, potato_seeds=0),
        )
        self.assertFalse(evidence["ready"])
        self.assertFalse(evidence["potato_purchase_complete"])

    def test_rejects_skipped_mountain_berry(self) -> None:
        phases = [
            {"phase": "MOUNTAIN_BERRY", "status": "skipped"},
            {"phase": "BUY_SEEDS", "status": "success"},
            {"phase": "WAIT_FARM_SHIPPING", "status": "success"},
        ]
        evidence = _d2_spine_checkpoint_evidence(
            _spine_task(phases),
            _spine_world(hour=17, potato_seeds=1),
        )
        self.assertFalse(evidence["ready"])
        self.assertFalse(evidence["mountain_grape_shipped"])

    def test_rejects_while_still_in_shipping_wait(self) -> None:
        evidence = _d2_spine_checkpoint_evidence(
            _spine_task(list(_REQUIRED_PHASES), phase="WAIT_FARM_SHIPPING"),
            _spine_world(hour=17, potato_seeds=1),
        )
        self.assertFalse(evidence["ready"])

    def test_empty_bag_is_ready_when_buy_seeds_succeeded(self) -> None:
        phases = list(_REQUIRED_PHASES) + [
            {"phase": "CROP_ESTABLISH", "status": "success"},
        ]
        evidence = _d2_spine_checkpoint_evidence(
            _spine_task(phases),
            _spine_world(hour=17, potato_seeds=0),
        )
        self.assertTrue(evidence["ready"])
        self.assertTrue(evidence["potato_purchase_complete"])
        self.assertEqual(evidence["potato_seeds"], 0)


@unittest.skipUnless(_SPINE_EVIDENCE.is_file(), "power-on D2 spine recording not on disk")
class PowerOnD2SpineEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(_SPINE_EVIDENCE.read_text(encoding="utf-8"))

    def test_clean_power_on_with_zero_loads_and_ram_writes(self) -> None:
        clean = self.report["clean_run"]
        self.assertEqual(clean["intervention_class"], "Clean")
        self.assertEqual(clean["initial_state_loads"], 0)
        self.assertEqual(clean["mid_run_state_loads"], 0)
        self.assertEqual(clean["ram_writes"], 0)
        self.assertIsNone(self.report["state"])
        self.assertTrue(self.report["power_on"]["completed"])
        self.assertEqual(self.report["power_on"]["initial_state_loads"], 0)
        self.assertEqual(self.report["power_on"]["ram_writes"], 0)

    def test_d1_handoff_reached_d2_with_starter_tools(self) -> None:
        handoff = self.report["d1_handoff"]
        self.assertEqual(handoff["status"], "success")
        self.assertEqual(handoff["day"], 2)
        self.assertTrue(handoff["has_watering_can"])
        self.assertTrue(handoff["has_grass_seeds"])
        # Mask is a D1-only byte; it is 0 after overnight. Peak 0x3F is not
        # in this end-of-handoff snapshot.
        self.assertEqual(handoff["mask"], 0)
        self.assertFalse(handoff["mask_complete"])

    def test_checkpoint_is_d2_5pm_farm_after_grape_buy_and_shipper(self) -> None:
        spine = self.report["d2_spine_checkpoint"]
        end = self.report["end"]
        start = self.report["start"]
        self.assertTrue(self.report["success"])
        self.assertTrue(spine["ready"])
        self.assertTrue(spine["mountain_grape_shipped"])
        self.assertTrue(spine["potato_purchase_complete"])
        self.assertTrue(spine["shipping_dialogue_cleared"])
        self.assertEqual(end["season"], 0)
        self.assertEqual(end["day"], 2)
        self.assertEqual(end["hour"], 17)
        self.assertEqual(end["tilemap"], 0)
        self.assertEqual(start["money"], 300)
        self.assertEqual(end["money"], 100)
        self.assertIn("MOUNTAIN_BERRY", spine["successful_phases"])
        self.assertIn("BUY_SEEDS", spine["successful_phases"])
        self.assertIn("CROP_ESTABLISH", spine["successful_phases"])
        self.assertIn("WAIT_FARM_SHIPPING", spine["successful_phases"])

    def test_two_dry_potatoes_are_one_cell_residual_not_3x3(self) -> None:
        """Live JSON still has the 1-cell plant. 8 around (13,28) is rr-m7mk."""
        crops = self.report["crop_survival"]
        self.assertTrue(crops["alive"])
        self.assertEqual(crops["wet"], 0)
        self.assertFalse(self.report["gate_b_full_ok"])
        self.assertFalse(self.report["money_gt_100"])
        self.assertEqual(self.report["journal_summary"]["total_shipped"], 0)
        self.assertEqual(self.report["reason"], "day-2 grape/seed spine reached post-5pm checkpoint")
        planted = int(crops["crop"])
        samples = sorted(tuple(row) for row in crops["samples"])
        if planted >= 8:
            ring = {(x, y) for x, y, _tid in samples}
            self.assertGreaterEqual(len(ring), 8)
            self.assertTrue(all(abs(x - 13) <= 1 and abs(y - 28) <= 1 for x, y, _tid in samples))
            return
        self.assertEqual(planted, 2)
        self.assertEqual(crops["dry"], 2)
        self.assertEqual(samples, [(13, 28, 84), (13, 30, 84)])


@unittest.skipUnless(_FAILED_D1_PIN.is_file(), "failed D1-pin recording not on disk")
class FailedD1HandoffPinEvidenceTests(unittest.TestCase):
    def test_d1_shed_pin_is_not_the_d2_spine_gate(self) -> None:
        report = json.loads(_FAILED_D1_PIN.read_text(encoding="utf-8"))
        spine = report["d2_spine_checkpoint"]
        self.assertEqual(report["state"], "Y1_D2_Morning_After_D1")
        self.assertFalse(report["success"])
        self.assertFalse(spine["ready"])
        self.assertFalse(spine["mountain_grape_shipped"])
        self.assertFalse(spine["potato_purchase_complete"])
        self.assertEqual(report["end"]["day"], 3)
        deferred = {
            row["phase"]: row["reason"]
            for row in report["day_journal"][0]["deferred"]
        }
        self.assertIn("MOUNTAIN_BERRY", deferred)
        self.assertIn("return_to_bin", deferred["MOUNTAIN_BERRY"])


if __name__ == "__main__":
    unittest.main()
