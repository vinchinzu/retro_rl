"""Water-step reorder must never drop a target.

run13 D3: the west plot declared ``steps=8`` and then reported
"WATER DONE: 7/7 watered". ``_reorder_remaining_water_steps`` had rebuilt the
list without (12,28) — the one tile whose stands were unreachable at the
instant the reorder ran — so it was never watered, stayed immature (0x58 while
its siblings were 0x60/0x61), and was correctly not harvested.
"""
from __future__ import annotations

import unittest

import numpy as np

from harvest.tasks.crop_planter import PlotPhase
from harvest.tasks.crop_water_ops import CropWaterOpsMixin


class _Nav:
    current_tile = (13, 29)


class _Ops(CropWaterOpsMixin):
    """Bare mixin with only the state ``_reorder_remaining_water_steps`` reads."""

    def __init__(self, steps, unreachable):
        self._plot_phase = PlotPhase.WATER
        self._water_steps = list(steps)
        self._water_index = 0
        self._water_steps_deferred = 0
        self._navigator = _Nav()
        self._unreachable = set(unreachable)
        self._target_tile = None
        self._approach_tile = None
        self._face_direction = None

    def _best_water_variant(self, ram, target, current_tile):
        if target in self._unreachable:
            return None
        # Score by distance so ordering is still exercised.
        return ((target[0], target[1] - 1), "down", abs(target[0] - current_tile[0]))


def _steps(*tiles):
    return [(t, (t[0], t[1] - 1), "down") for t in tiles]


class WaterReorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ram = np.zeros(0x20000, dtype=np.uint8)

    def test_unreachable_target_is_deferred_not_dropped(self) -> None:
        ops = _Ops(_steps((12, 27), (12, 28), (13, 27)), unreachable={(12, 28)})
        ops._reorder_remaining_water_steps(self.ram)
        targets = [s[0] for s in ops._water_steps]
        self.assertIn((12, 28), targets)
        self.assertEqual(len(targets), 3)
        self.assertEqual(ops._water_steps_deferred, 1)

    def test_deferred_target_goes_to_the_tail(self) -> None:
        ops = _Ops(_steps((12, 28), (12, 27), (13, 27)), unreachable={(12, 28)})
        ops._reorder_remaining_water_steps(self.ram)
        self.assertEqual(ops._water_steps[-1][0], (12, 28))

    def test_a_deferred_target_is_retried_once_it_becomes_reachable(self) -> None:
        ops = _Ops(_steps((12, 27), (12, 28)), unreachable={(12, 28)})
        ops._reorder_remaining_water_steps(self.ram)
        ops._unreachable.clear()
        ops._reorder_remaining_water_steps(self.ram)
        self.assertEqual(sorted(s[0] for s in ops._water_steps), [(12, 27), (12, 28)])
        self.assertEqual(ops._water_steps_deferred, 1)

    def test_all_unreachable_leaves_the_plan_untouched(self) -> None:
        ops = _Ops(_steps((12, 27), (12, 28)), unreachable={(12, 27), (12, 28)})
        self.assertFalse(ops._reorder_remaining_water_steps(self.ram))
        self.assertEqual(len(ops._water_steps), 2)

    def test_reachable_only_plan_still_reorders_by_score(self) -> None:
        ops = _Ops(_steps((18, 27), (13, 27)), unreachable=set())
        ops._reorder_remaining_water_steps(self.ram)
        self.assertEqual([s[0] for s in ops._water_steps], [(13, 27), (18, 27)])
        self.assertEqual(ops._water_steps_deferred, 0)


if __name__ == "__main__":
    unittest.main()
