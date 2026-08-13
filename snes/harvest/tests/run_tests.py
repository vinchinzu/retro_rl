#!/usr/bin/env python3
"""
Test suite for Harvest Moon bot.

Tests are organized by level (L1-L7+) based on the skill tech tree in PLAN.md.

Helpers live in ``run_tests_helpers``; cases in ``run_tests_cases``.
This module is the thin CLI entry:

    python tests/run_tests.py
    python -m tests.run_tests
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from run_tests_helpers import TestResult  # noqa: E402
from run_tests_cases import ALL_TESTS  # noqa: E402

TESTS: list[Callable[[], TestResult]] = list(ALL_TESTS)


def main():
    results: list[TestResult] = []
    for test in TESTS:
        try:
            results.append(test())
        except Exception as exc:
            results.append(TestResult(test.__name__, "FAIL", str(exc)))
    width = max(len(r.name) for r in results)
    for r in results:
        detail = f" - {r.detail}" if r.detail else ""
        print(f"{r.name:<{width}} : {r.status}{detail}")
    passed = sum(1 for r in results if r.status == "PASS")
    skipped = sum(1 for r in results if r.status == "SKIP")
    failed = sum(1 for r in results if r.status == "FAIL")
    print(f"\nTotal: {len(results)} | Passed: {passed} | Skipped: {skipped} | Failed: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
