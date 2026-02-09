#!/usr/bin/env python3
"""
Run all tests for the retro_rl fighting game training framework.

Usage:
    python run_all.py                  # Run all tests
    python run_all.py --fast           # Skip slow tests (ROM-dependent)
    python run_all.py --module state   # Run only state integrity tests
    python run_all.py -v               # Verbose output

Test categories:
    FAST (no ROMs needed):
        - test_win_conditions: Win condition logic + source code audit
        - test_reward_shaping: FightingEnv reward calculations
        - test_training_configs: Training script config validation
        - test_model_registry: Registry CRUD operations

    SLOW (requires ROMs):
        - test_state_integrity: Load every state, check health/timer
        - test_env_load: Load each game environment
        - test_training: PPO smoke tests
        - test_menu_nav: Menu navigation tests
"""

import argparse
import os
import sys
import time
import unittest
from pathlib import Path

# Headless rendering
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TESTS_DIR = Path(__file__).parent

# Test modules categorized by speed
FAST_MODULES = [
    "test_win_conditions",
    "test_reward_shaping",
    "test_training_configs",
    "test_model_registry",
]

SLOW_MODULES = [
    "test_state_integrity",
    "test_env_load",
    "test_training",
    "test_menu_nav",
]

MODULE_ALIASES = {
    "state": "test_state_integrity",
    "states": "test_state_integrity",
    "win": "test_win_conditions",
    "reward": "test_reward_shaping",
    "config": "test_training_configs",
    "training": "test_training_configs",
    "registry": "test_model_registry",
    "env": "test_env_load",
    "ppo": "test_training",
    "menu": "test_menu_nav",
}


def run_tests(modules, verbosity=1):
    """Run specific test modules and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for module_name in modules:
        module_path = TESTS_DIR / f"{module_name}.py"
        if not module_path.exists():
            print(f"  WARNING: {module_name}.py not found, skipping")
            continue
        try:
            module_suite = loader.discover(str(TESTS_DIR), pattern=f"{module_name}.py")
            suite.addTests(module_suite)
        except Exception as e:
            print(f"  ERROR loading {module_name}: {e}")

    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def main():
    parser = argparse.ArgumentParser(description="Run retro_rl test suite")
    parser.add_argument("--fast", action="store_true",
                        help="Only run fast tests (no ROMs needed)")
    parser.add_argument("--slow", action="store_true",
                        help="Only run slow tests (ROM-dependent)")
    parser.add_argument("--module", "-m", type=str, default=None,
                        help="Run specific module (e.g., 'state', 'win', 'config')")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    verbosity = 2 if args.verbose else 1

    print("=" * 70)
    print("RETRO RL TEST SUITE")
    print("=" * 70)

    if args.module:
        # Resolve alias
        module_name = MODULE_ALIASES.get(args.module, args.module)
        if not module_name.startswith("test_"):
            module_name = f"test_{module_name}"
        modules = [module_name]
        print(f"Running: {module_name}")
    elif args.fast:
        modules = FAST_MODULES
        print(f"Running FAST tests ({len(modules)} modules, no ROMs needed)")
    elif args.slow:
        modules = SLOW_MODULES
        print(f"Running SLOW tests ({len(modules)} modules, requires ROMs)")
    else:
        modules = FAST_MODULES + SLOW_MODULES
        print(f"Running ALL tests ({len(modules)} modules)")

    print("=" * 70 + "\n")

    start = time.time()
    result = run_tests(modules, verbosity)
    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print(f"SUMMARY: {result.testsRun} tests in {elapsed:.1f}s")
    print(f"  Passed:  {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"  Failed:  {len(result.failures)}")
    print(f"  Errors:  {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print("=" * 70)

    if result.failures:
        print("\nFAILED TESTS:")
        for test, _ in result.failures:
            print(f"  FAIL: {test}")

    if result.errors:
        print("\nERRORS:")
        for test, _ in result.errors:
            print(f"  ERROR: {test}")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
