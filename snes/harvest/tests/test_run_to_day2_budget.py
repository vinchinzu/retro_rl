"""Default run_to_day2 day budget must cover a composed D2 leftover."""

from __future__ import annotations

from pathlib import Path
import unittest


class RunToDay2BudgetTests(unittest.TestCase):
    def test_default_max_frames_is_two_million_per_overnight(self) -> None:
        harvest_dir = Path(__file__).resolve().parents[1] / "harvest"
        run_src = (harvest_dir / "scripts" / "run_to_day2.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("2_000_000 * max(1, overnights_budget)", run_src)
        self.assertNotIn("200_000 * max(1, overnights_budget)", run_src)


if __name__ == "__main__":
    unittest.main()
