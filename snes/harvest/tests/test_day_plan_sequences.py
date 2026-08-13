"""Compatibility shim: day-plan sequence tests split by domain.

Prefer running domain modules directly:

- ``tests.test_day_plan_crop_sequences`` / ``tests.test_day_plan_crop_phases``
  (shim: ``tests.test_day_plan_crop``)
- ``tests.test_day_plan_home_sequences`` / ``tests.test_day_plan_home_return``
  (shim: ``tests.test_day_plan_home``)
- ``tests.test_day_plan_coop``
- ``tests.test_day_plan_power_on``
- ``tests.test_day_plan_common``
- ``tests.test_day_plan_common_nav``

Shared fixtures live in ``day_plan_test_helpers`` (same directory).

Legacy invocation still works via ``load_tests``::

    python -m unittest tests.test_day_plan_sequences

Pytest skips this shim (``__test__ = False``) so domain modules are not
double-collected.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from day_plan_test_helpers import (  # noqa: E402, F401
    DayPlanPhaseHelpers,
    make_date_world,
    make_navigation_ram,
    make_time_world,
    make_transition_world,
    make_world,
    set_live_chicken_slots,
    set_live_cow_slot,
    set_live_u16,
    set_money,
    set_player_pos,
)

_DOMAIN_MODULES = (
    "tests.test_day_plan_common",
    "tests.test_day_plan_common_nav",
    "tests.test_day_plan_crop",
    "tests.test_day_plan_home",
    "tests.test_day_plan_coop",
    "tests.test_day_plan_power_on",
)

# Domain modules own the TestCase classes; do not re-collect here under pytest.
__test__ = False


def load_tests(loader, standard_tests, pattern):
    """Aggregate domain suites for ``unittest tests.test_day_plan_sequences``."""
    suite = unittest.TestSuite()
    for name in _DOMAIN_MODULES:
        suite.addTests(loader.loadTestsFromName(name))
    return suite


def __getattr__(name: str):
    """Lazy class aliases for interactive inspection / docs."""
    aliases = {
        "DayPlanSequenceCommonTests": ("tests.test_day_plan_common", "DayPlanSequenceCommonTests"),
        "BuildDayPhasesCommonTests": ("tests.test_day_plan_common", "BuildDayPhasesCommonTests"),
        "DayPlanSequenceCommonNavTests": (
            "tests.test_day_plan_common_nav",
            "DayPlanSequenceCommonNavTests",
        ),
        "DayPlanSequenceCropTests": (
            "tests.test_day_plan_crop_sequences",
            "DayPlanSequenceCropTests",
        ),
        "BuildDayPhasesCropTests": (
            "tests.test_day_plan_crop_phases",
            "BuildDayPhasesCropTests",
        ),
        "DayPlanSequenceHomeTests": (
            "tests.test_day_plan_home_sequences",
            "DayPlanSequenceHomeTests",
        ),
        "BuildDayPhasesHomeTests": (
            "tests.test_day_plan_home_return",
            "BuildDayPhasesHomeTests",
        ),
        "DayPlanSequenceCoopTests": ("tests.test_day_plan_coop", "DayPlanSequenceCoopTests"),
        "BuildDayPhasesCoopTests": ("tests.test_day_plan_coop", "BuildDayPhasesCoopTests"),
        "DayPlanSequencePowerOnTests": ("tests.test_day_plan_power_on", "DayPlanSequencePowerOnTests"),
        "SleepAndPlannerTests": ("tests.test_day_plan_power_on", "SleepAndPlannerTests"),
    }
    if name in aliases:
        mod_name, attr = aliases[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    unittest.main()
