"""Compatibility shim: home day-plan tests split by concern.

Prefer running modules directly:

- ``tests.test_day_plan_home_sequences`` — DayPlanSequenceHomeTests
- ``tests.test_day_plan_home_return`` — BuildDayPhasesHomeTests

Legacy invocation still works via ``load_tests``::

    python -m unittest tests.test_day_plan_home

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
    "tests.test_day_plan_home_sequences",
    "tests.test_day_plan_home_return",
)

# Domain modules own the TestCase classes; do not re-collect here under pytest.
__test__ = False


def load_tests(loader, standard_tests, pattern):
    """Aggregate home suites for ``unittest tests.test_day_plan_home``."""
    suite = unittest.TestSuite()
    for name in _DOMAIN_MODULES:
        suite.addTests(loader.loadTestsFromName(name))
    return suite


def __getattr__(name: str):
    """Lazy class aliases for interactive inspection / docs."""
    aliases = {
        "DayPlanSequenceHomeTests": (
            "tests.test_day_plan_home_sequences",
            "DayPlanSequenceHomeTests",
        ),
        "BuildDayPhasesHomeTests": (
            "tests.test_day_plan_home_return",
            "BuildDayPhasesHomeTests",
        ),
    }
    if name in aliases:
        mod_name, attr = aliases[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    unittest.main()
