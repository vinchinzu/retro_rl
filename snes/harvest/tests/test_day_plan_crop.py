"""Compatibility shim: crop day-plan tests split by concern.

Prefer running modules directly:

- ``tests.test_day_plan_crop_sequences`` — DayPlanSequenceCropTests
- ``tests.test_day_plan_crop_phases`` — BuildDayPhasesCropTests

Legacy invocation still works via ``load_tests``::

    python -m unittest tests.test_day_plan_crop

Pytest skips this shim (``__test__ = False``) so split modules are not
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

_DOMAIN_MODULES = (
    "tests.test_day_plan_crop_sequences",
    "tests.test_day_plan_crop_phases",
)

# Domain modules own the TestCase classes; do not re-collect here under pytest.
__test__ = False


def load_tests(loader, standard_tests, pattern):
    """Aggregate crop suites for ``unittest tests.test_day_plan_crop``."""
    suite = unittest.TestSuite()
    for name in _DOMAIN_MODULES:
        suite.addTests(loader.loadTestsFromName(name))
    return suite


def __getattr__(name: str):
    """Lazy class aliases for interactive inspection / docs."""
    aliases = {
        "DayPlanSequenceCropTests": (
            "tests.test_day_plan_crop_sequences",
            "DayPlanSequenceCropTests",
        ),
        "BuildDayPhasesCropTests": (
            "tests.test_day_plan_crop_phases",
            "BuildDayPhasesCropTests",
        ),
    }
    if name in aliases:
        mod_name, attr = aliases[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")


if __name__ == "__main__":
    unittest.main()
