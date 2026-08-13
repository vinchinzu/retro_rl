"""Compatibility shim: CoopChoresTask tests split by domain.

Prefer running domain modules directly:

- ``tests.test_coop_task_core`` — feed / egg decide / incubate / ship verify
- ``tests.test_coop_task_nav`` — egg routes / ship nav / pathfinding / exit prep

Shared fixtures live in ``coop_task_test_helpers`` (same directory).

Legacy invocation still works via ``load_tests``::

    python -m unittest tests.test_coop_task

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

from coop_task_test_helpers import (  # noqa: E402, F401
    add_chicken_object,
    add_egg_object,
    block_tiles,
    make_coop_ram,
    make_world,
    set_chicken_slot_position,
)

_DOMAIN_MODULES = (
    "tests.test_coop_task_core",
    "tests.test_coop_task_nav",
)

# Domain modules own the TestCase classes; do not re-collect here under pytest.
__test__ = False


def load_tests(loader, standard_tests, pattern):
    """Aggregate domain suites for ``unittest tests.test_coop_task``."""
    suite = unittest.TestSuite()
    for name in _DOMAIN_MODULES:
        suite.addTests(loader.loadTestsFromName(name))
    return suite


def __getattr__(name: str):
    """Lazy class aliases for interactive inspection / docs."""
    aliases = {
        "CoopChoresTaskTests": ("tests.test_coop_task_core", "CoopChoresTaskCoreTests"),
        "CoopChoresTaskCoreTests": ("tests.test_coop_task_core", "CoopChoresTaskCoreTests"),
        "CoopChoresTaskNavTests": ("tests.test_coop_task_nav", "CoopChoresTaskNavTests"),
    }
    if name in aliases:
        mod_name, attr = aliases[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    unittest.main()
