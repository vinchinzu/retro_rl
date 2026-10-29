"""PR-058 census in the day journal, PR-027 report, PR-086 berry chances."""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

from retro_harness import WorldState

from harvest.core.harvest_state import HarvestStateDocument
from harvest.planner.multi_day_planner import MultiDayPlannerTask
from harvest.tools.pr999_report import (
    build_report,
    farm_layout,
    format_report,
    happiness_floor,
    main,
)

_REPO = Path(__file__).resolve().parents[3]
_HARVEST = Path(__file__).resolve().parents[1]
_Y1_CHECKPOINT = _HARVEST / "recordings" / "pr026_resume.json"
_CHECKPOINT_PIN = "PR_Y1_Summer_D1"


def _cli(path: Path) -> str:
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["HEADLESS"] = "1"
    env["SDL_VIDEODRIVER"] = "dummy"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO), str(_HARVEST), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        ["uv", "run", "python", "-m", "harvest.tools.pr999_report", str(path)],
        cwd=_REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr or proc.stdout)
    return proc.stdout


class JournalCensusTests(unittest.TestCase):
    def test_checkpoint_pin_journal_matches_linter(self) -> None:
        document = HarvestStateDocument.load(_CHECKPOINT_PIN)
        world = WorldState(frame=0, ram=document.ram_array(), info={})
        planner = MultiDayPlannerTask(target_days=1, max_days=1)
        planner.reset(world)
        planner._journal_day_complete(world, sleep_reason="checkpoint")
        row = planner.day_journal[-1]
        census = row["development_census"]
        layout = farm_layout()
        lint = layout.lint_pin(_CHECKPOINT_PIN)
        self.assertEqual(census["counted_tiles"], lint.live_counted)
        self.assertEqual(census["minimum_tiles"], 2950)
        self.assertEqual(census["gap_to_2950"], 2950 - lint.live_counted)
        self.assertEqual(census["source"], lint.source)
        self.assertEqual(census["tilemap"], lint.tilemap)
        got = [
            (item["x"], item["y"], item["live"], tuple(item["steps"]))
            for item in census["uncovered"]
        ]
        want = [(todo.x, todo.y, todo.live, todo.steps) for todo in lint.todos]
        self.assertEqual(got, want)
        self.assertEqual(census["uncovered_count"], len(lint.todos))
        self.assertIn("cows", row)
        self.assertIn("power_berry_count", row["collectible_flags"])


class ReportTests(unittest.TestCase):
    def test_y1_checkpoint_json_prints_gaps_windows_and_berries(self) -> None:
        text = _cli(_Y1_CHECKPOINT)
        self.assertIn("term gaps", text)
        self.assertIn("window states", text)
        self.assertIn("remaining collectible chances", text)
        self.assertIn("berry chances  designed=7  eaten=0  target=10", text)
        self.assertIn("berry_field_1 designed open", text)
        self.assertIn("berry_dup_cave duplication open", text)
        self.assertIn("berry_dup_indoor duplication open", text)
        self.assertIn("berry_dup_transition duplication open", text)
        self.assertIn("must land on 10", text)
        self.assertIn("lagging cows", text)
        self.assertIn("  none", text)
        self.assertNotIn("FLAG lagging cow", text)
        self.assertIn("power_berries gap=", text)
        self.assertIn("window=below", text)

    def test_whatif_flags_a_lagging_cow(self) -> None:
        # cow_feedable_01 latest is index 238. Eighteen mornings left need 128.
        self.assertEqual(happiness_floor(238, 220), 128)
        run = json.loads(_Y1_CHECKPOINT.read_text())
        run["state"] = "LN_mgr-report_whatif_lag_cow"
        run["end"] = {
            "year": 1,
            "season": 3,
            "day": 11,
            "hour": 6,
            "minute": 0,
        }
        run["cows"] = [
            {"slot": 0, "exists": True, "happiness": 0, "status": 1, "milestone": "cow_slot0"},
            {
                "slot": 1,
                "exists": True,
                "happiness": 10,
                "status": 9,
                "milestone": "cow_feedable_01",
            },
        ]
        report = build_report(run)
        text = format_report(report)
        flagged = [cow for cow in report["lagging_cows"] if cow["id"] == "cow_feedable_01"]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["happiness"], 10)
        self.assertEqual(flagged[0]["needed"], 128)
        self.assertEqual(flagged[0]["reason"], "happiness")
        self.assertIn("FLAG lagging cow cow_feedable_01", text)
        self.assertIn("happiness=10", text)
        self.assertIn("needed=128", text)
        self.assertIn("designed=7", text)
        self.assertIn("target=10", text)
        self.assertEqual(report["berry_chances"]["designed"], 7)
        self.assertEqual(report["berry_chances"]["target"], 10)

    def test_module_main_on_y1_checkpoint(self) -> None:
        self.assertEqual(main([str(_Y1_CHECKPOINT)]), 0)


if __name__ == "__main__":
    unittest.main()
