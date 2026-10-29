"""Empty social, family, and cover seams. No new day, skill, or spend behavior."""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from harvest.planner import day_plan_orchestrator as orch
from harvest.planner.spring_opt import (
    DOMAIN_COST_SEAMS,
    domain_cost_g,
    optimize_spring,
    register_domain_cost,
)
from harvest.scripts.run_to_day2 import _parse_args
from harvest.tasks.skills import SKILL_SEAMS, register_skill_seam, skill_seam

_HARVEST = Path(__file__).resolve().parents[1]
_SOURCES = (
    _HARVEST / "harvest" / "scripts" / "run_to_day2.py",
    _HARVEST / "harvest" / "planner" / "day_plan_orchestrator.py",
    _HARVEST / "harvest" / "tasks" / "skills.py",
    _HARVEST / "harvest" / "planner" / "spring_opt.py",
)
_SOFT_MAX = 1000


def _names(task: orch.DayPlanTask) -> tuple[str, ...]:
    return tuple(phase.phase for phase in task.runtime_phases)


def _row(name: str) -> orch.PhaseSpec:
    return orch.PhaseSpec(name, "nav")


class ComposerSeamTests(unittest.TestCase):
    def tearDown(self) -> None:
        for name in orch.COMPOSER_SEAMS:
            orch.register_composer_seam(name, ())
        for name in SKILL_SEAMS:
            register_skill_seam(name, None)
        for name in DOMAIN_COST_SEAMS:
            register_domain_cost(name, None)

    def test_gut_sources_stay_under_the_soft_max(self) -> None:
        for path in _SOURCES:
            count = len(path.read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(count, _SOFT_MAX, path.name)

    def test_empty_composer_seams_leave_the_day_unchanged(self) -> None:
        self.assertEqual(orch.composer_seam_phases(), ())
        bare = _names(orch.DayPlanTask())
        for name in reversed(orch.COMPOSER_SEAMS):
            orch.register_composer_seam(name, (_row(name.upper() + "_ROW"),))
        spliced = _names(
            orch.DayPlanTask(
                phase_sequence=[
                    _row("WORK"),
                    _row("RETURN_HOME"),
                    _row("GO_TO_SLEEP"),
                ]
            )
        )
        self.assertEqual(
            spliced,
            ("WORK", "SOCIAL_ROW", "FAMILY_ROW", "COVER_ROW", "RETURN_HOME", "GO_TO_SLEEP"),
        )
        for name in orch.COMPOSER_SEAMS:
            orch.register_composer_seam(name, ())
        self.assertEqual(orch.composer_seam_phases(), ())
        self.assertEqual(_names(orch.DayPlanTask()), bare)

    def test_unknown_composer_seam_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            orch.register_composer_seam("livestock", ())

    def test_skill_seam_is_empty_until_a_domain_registers(self) -> None:
        self.assertEqual(skill_seam("social"), None)
        factory = lambda world: None  # noqa: E731
        register_skill_seam("family", factory)
        self.assertIs(skill_seam("family"), factory)
        register_skill_seam("family", None)
        self.assertIsNone(skill_seam("family"))
        with self.assertRaises(KeyError):
            skill_seam("livestock")

    def test_domain_cost_defaults_to_zero_and_stays_out_of_the_search(self) -> None:
        self.assertEqual(domain_cost_g("cover"), 0)
        register_domain_cost("cover", lambda day, cost: 12)
        self.assertEqual(domain_cost_g("cover", day=3), 12)
        register_domain_cost("cover", None)
        self.assertEqual(domain_cost_g("cover"), 0)
        with self.assertRaises(KeyError):
            domain_cost_g("livestock")
        self.assertNotIn("domain_cost_g", inspect.getsource(optimize_spring))

    def test_trials_parses_and_defaults_to_one(self) -> None:
        args = _parse_args(["--trials", "1", "--power-on", "--until-ending"])
        self.assertEqual(args.trials, 1)
        self.assertTrue(args.power_on)
        self.assertTrue(args.until_ending)
        self.assertEqual(_parse_args([]).trials, 1)


if __name__ == "__main__":
    unittest.main()
