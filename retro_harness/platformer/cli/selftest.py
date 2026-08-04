"""Self-test command for platformer harness sanity checks."""

from __future__ import annotations

import argparse

from retro_harness.platformer.cli.helpers import _get_action_table, _resolve_config
from retro_harness.platformer.evaluator import Evaluator


def cmd_selftest(args: argparse.Namespace) -> None:
    """Self-test: verify death detection and level-change guards work correctly."""
    import numpy as np

    config = _resolve_config(args)
    print(f"=== Platformer Optimizer Self-Test: {config.display_name} ===\n")
    failures = 0

    evaluator = Evaluator(config)
    evaluator._ensure_env()

    initial_cam = evaluator._initial_camera_x
    initial_values = evaluator._initial_values
    print(f"State: {config.start_state}")
    print(f"  initial_camera_x = {initial_cam:.0f}")
    print(f"  initial_lives    = {initial_values.get('lives', 'N/A')}")

    # Check level_id is correct
    level_id = initial_values.get("level_id", -1)
    if level_id != config.target_level_id:
        print(f"  FAIL: level_id=0x{level_id:04X}, expected 0x{config.target_level_id:04X}")
        failures += 1
    else:
        print(f"  OK: level_id=0x{level_id:04X}")

    # Test 1: sequence that dies must be flagged as died, NOT completed
    print(f"\n[Test 1] Deterministic death probe")
    death_seq = config.selftest_death_actions or (([2] * 40 + [3] * 15 + [2] * 5 + [5] * 10) * 28)
    if not config.selftest_expect_death:
        print("  SKIP: no published deterministic death probe for this start state")
    else:
        result = evaluator.evaluate(death_seq[:2000], early_terminate=False)
        if not result.died:
            print(f"  FAIL: died={result.died}, expected True")
            failures += 1
        elif result.completed:
            print(f"  FAIL: completed={result.completed}, should be False when died")
            failures += 1
        else:
            print(f"  OK: died=True, completed=False, frame={result.total_frames}, progress={result.max_progress:.0f}")

    # Test 2: fitness for dead < alive at same progress
    print(f"\n[Test 2] Death fitness < alive fitness at same progress")
    dead_fitness_at_100 = 100 * config.progress_weight - config.death_penalty
    alive_fitness_at_100 = 100 * config.progress_weight
    if dead_fitness_at_100 >= alive_fitness_at_100:
        print(f"  FAIL: dead_fitness ({dead_fitness_at_100}) >= alive_fitness ({alive_fitness_at_100})")
        failures += 1
    else:
        print(f"  OK: dead@100={dead_fitness_at_100} < alive@100={alive_fitness_at_100}")

    # Test 3: short alive sequence stays in level
    print(f"\n[Test 3] Short alive sequence stays in level")
    short_result = evaluator.evaluate([0] * 60, early_terminate=False)
    if short_result.completed:
        print(f"  FAIL: 60 frames of nothing showed completed=True!")
        failures += 1
    elif short_result.died:
        print(f"  FAIL: 60 frames of nothing showed died=True!")
        failures += 1
    else:
        print(f"  OK: alive, not completed, level_id=0x{short_result.level_id_at_end:04X}")

    # Test 4: determinism
    print(f"\n[Test 4] Determinism check")
    r1 = evaluator.evaluate(death_seq[:500], early_terminate=False)
    r2 = evaluator.evaluate(death_seq[:500], early_terminate=False)
    if r1.fitness != r2.fitness or r1.total_frames != r2.total_frames:
        print(f"  FAIL: run1 fitness={r1.fitness:.0f}/frames={r1.total_frames} != run2")
        failures += 1
    else:
        print(f"  OK: both runs -> fitness={r1.fitness:.0f}, frames={r1.total_frames}")

    # Test 5: first-frame button stability
    print(f"\n[Test 5] First-frame button stability")
    action_table = _get_action_table(config)
    for idx in range(len(action_table)):
        r = evaluator.evaluate([idx], early_terminate=False)
        if r.completed:
            print(f"  FAIL: action {idx} on first frame triggered completion!")
            failures += 1
            break
    else:
        print(f"  OK: all {len(action_table)} actions stable on first frame")

    evaluator.close()

    print(f"\n{'=' * 40}")
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TEST(S) FAILED")
    return failures


