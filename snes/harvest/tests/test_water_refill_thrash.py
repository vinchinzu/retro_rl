"""Unit tests for pond corridor densify-thrash rules (no emulator)."""

from __future__ import annotations

import unittest


class CorridorThrashRulesTests(unittest.TestCase):
    """Pure densify-thrash match/priority/fire (no emulator)."""

    def test_past_fence_north_fires_immediate_east_south(self) -> None:
        from harvest.tasks.pond_hop import (
            ThrashChargeKind,
            ThrashCounters,
            evaluate_corridor_thrash,
            match_thrash_rule,
        )

        c = ThrashCounters()
        rule = match_thrash_rule((31, 29), (32, 34), c)
        self.assertIsNotNone(rule)
        self.assertEqual(rule.name, "past_fence_north")
        r = evaluate_corridor_thrash((31, 29), (32, 34), c)
        self.assertTrue(r.fire_charge)
        self.assertEqual(r.charge, ThrashChargeKind.EAST_SOUTH)
        self.assertIn("Past-fence north", r.log)
        self.assertEqual(r.refill_densify_stalls, 0)

    def test_south_thrash_stalls_then_west_lip(self) -> None:
        from harvest.tasks.pond_hop import (
            ThrashChargeKind,
            ThrashCounters,
            evaluate_corridor_thrash,
            match_thrash_rule,
        )

        start, goal = (24, 34), (32, 34)
        c0 = ThrashCounters()
        self.assertEqual(match_thrash_rule(start, goal, c0).name, "south_thrash")
        r1 = evaluate_corridor_thrash(start, goal, c0)
        self.assertFalse(r1.fire_charge)
        self.assertEqual(r1.refill_densify_stalls, 1)
        self.assertEqual(r1.refill_densify_last, (start, goal))

        c1 = ThrashCounters(
            refill_densify_stalls=r1.refill_densify_stalls,
            refill_densify_last=r1.refill_densify_last,
        )
        r2 = evaluate_corridor_thrash(start, goal, c1)
        self.assertTrue(r2.fire_charge)
        self.assertEqual(r2.charge, ThrashChargeKind.WEST_SOUTH_LIP)
        self.assertIn("west→south-lip", r2.log)
        self.assertEqual(r2.refill_densify_stalls, 0)

    def test_near_f0_beats_south_thrash_and_needs_six_stalls(self) -> None:
        from harvest.tasks.pond_hop import (
            ThrashChargeKind,
            ThrashCounters,
            evaluate_corridor_thrash,
            match_thrash_rule,
        )

        # (28,34)→(32,34): dist=4, in near-F0 band (not south_thrash).
        start, goal = (28, 34), (32, 34)
        c = ThrashCounters()
        self.assertEqual(match_thrash_rule(start, goal, c).name, "near_f0")

        stalls = 0
        last = None
        for n in range(1, 6):
            r = evaluate_corridor_thrash(
                start,
                goal,
                ThrashCounters(
                    refill_densify_stalls=stalls,
                    refill_densify_last=last,
                ),
            )
            self.assertFalse(r.fire_charge, msg=f"must not fire at stall {n}")
            stalls = r.refill_densify_stalls
            last = r.refill_densify_last
            self.assertEqual(stalls, n)

        r6 = evaluate_corridor_thrash(
            start,
            goal,
            ThrashCounters(
                refill_densify_stalls=stalls,
                refill_densify_last=last,
            ),
        )
        self.assertTrue(r6.fire_charge)
        self.assertEqual(r6.charge, ThrashChargeKind.WEST_SOUTH_LIP)
        self.assertIn("Near-F0", r6.log)

    def test_north_thrash_priority_over_unrelated_and_east_south_charge(self) -> None:
        from harvest.tasks.pond_hop import (
            ThrashChargeKind,
            ThrashCounters,
            evaluate_corridor_thrash,
            match_thrash_rule,
        )

        start, goal = (25, 30), (32, 34)
        self.assertEqual(
            match_thrash_rule(start, goal, ThrashCounters()).name,
            "north_thrash",
        )
        r1 = evaluate_corridor_thrash(start, goal, ThrashCounters())
        r2 = evaluate_corridor_thrash(
            start,
            goal,
            ThrashCounters(
                refill_densify_stalls=r1.refill_densify_stalls,
                refill_densify_last=r1.refill_densify_last,
            ),
        )
        self.assertTrue(r2.fire_charge)
        self.assertEqual(r2.charge, ThrashChargeKind.EAST_SOUTH)
        self.assertIn("east→south", r2.log)

    def test_east_thrash_match_and_no_match_resets(self) -> None:
        from harvest.tasks.pond_hop import (
            ThrashChargeKind,
            ThrashCounters,
            evaluate_corridor_thrash,
            match_thrash_rule,
        )

        start, goal = (41, 32), (32, 34)
        self.assertEqual(
            match_thrash_rule(start, goal, ThrashCounters()).name,
            "east_thrash",
        )
        r1 = evaluate_corridor_thrash(start, goal, ThrashCounters())
        r2 = evaluate_corridor_thrash(
            start,
            goal,
            ThrashCounters(
                refill_densify_stalls=r1.refill_densify_stalls,
                refill_densify_last=r1.refill_densify_last,
            ),
        )
        self.assertTrue(r2.fire_charge)
        self.assertEqual(r2.charge, ThrashChargeKind.WEST_SOUTH_LIP)

        # Far from thrash regions: clear stalls.
        clear = evaluate_corridor_thrash(
            (10, 20),
            (12, 20),
            ThrashCounters(refill_densify_stalls=3, refill_densify_last=(start, goal)),
        )
        self.assertFalse(clear.fire_charge)
        self.assertIsNone(clear.rule_name)
        self.assertEqual(clear.refill_densify_stalls, 0)
        self.assertIsNone(clear.refill_densify_last)

    def test_charge_caps_suppress_match(self) -> None:
        from harvest.tasks.pond_hop import (
            ThrashCounters,
            match_thrash_rule,
        )

        # east_south exhausted → past-fence / north no longer match.
        self.assertIsNone(
            match_thrash_rule(
                (31, 29),
                (32, 34),
                ThrashCounters(east_south_charges=6),
            )
        )
        # south_lip exhausted → south thrash off.
        self.assertIsNone(
            match_thrash_rule(
                (24, 34),
                (32, 34),
                ThrashCounters(south_lip_charges=8),
            )
        )


if __name__ == "__main__":
    unittest.main()
