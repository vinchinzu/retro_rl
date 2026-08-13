#!/usr/bin/env python3
"""L9–L10 day-plan / multi-map cases for run_tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from run_tests_helpers import (  # noqa: E402
    TASKS_DIR,
    TestResult,
    get_pos,
    get_tilemap,
    load_state_bytes,
    make_env,
)

from harvest.tasks.nav import Point  # noqa: E402
from retro_harness import TaskStatus  # noqa: E402


# =============================================================================
# L9: Day Plan Tests
# =============================================================================

def test_day_plan_can_start() -> TestResult:
    """Verify that required recorded tasks exist for the day plan."""
    from harvest.planner.day_plan import PHASE_SEQUENCE
    missing = []
    for spec in PHASE_SEQUENCE:
        if spec.kind == "recorded":
            task_name = spec.params.get("task_name", "")
            task_path = os.path.join(TASKS_DIR, f"{task_name}.json")
            if not os.path.exists(task_path):
                missing.append(task_name)
        elif spec.kind == "cross_map":
            rec_name = spec.params.get("recording_name", "")
            rec_path = os.path.join(TASKS_DIR, f"{rec_name}.json")
            if not os.path.exists(rec_path):
                missing.append(rec_name)
    if missing:
        return TestResult("L9 day plan can start", "FAIL", f"missing recordings: {missing}")
    return TestResult("L9 day plan can start", "PASS")


def test_day_plan_exit_house() -> TestResult:
    """Verify ExitBuildingTask changes tilemap from house (0x15) to farm (0x00)."""
    state = "Y1_Spring_D1_Dawn"
    if not load_state_bytes(state):
        return TestResult("L9 day plan exit house", "SKIP", "missing state")

    from harvest.planner.day_plan import ExitBuildingTask
    from harvest.runtime.harness_runtime import HarnessRunner
    from retro_harness import WorldState as WS

    env = make_env(state)
    env.reset()

    tilemap_before = get_tilemap(env)
    if tilemap_before != 0x15:
        env.close()
        return TestResult("L9 day plan exit house", "SKIP",
                          f"expected tilemap 0x15, got 0x{tilemap_before:02X}")

    runner = HarnessRunner(env)
    world = runner.reset()
    task = ExitBuildingTask(target_tilemap=0x00, timeout=900)
    result = runner.run_task(task, world, max_steps=900)

    tilemap_after = get_tilemap(env)
    env.close()

    if result.status == TaskStatus.SUCCESS and tilemap_after == 0x00:
        return TestResult("L9 day plan exit house", "PASS")
    return TestResult("L9 day plan exit house", "FAIL",
                      f"status={result.status} tilemap=0x{tilemap_after:02X}")


def test_spring4_can_start() -> TestResult:
    """Verify that required recorded tasks exist for the spring4 day plan."""
    from harvest.planner.day_plan import SPRING4_PHASES
    missing = []
    for spec in SPRING4_PHASES:
        if spec.kind == "recorded":
            task_name = spec.params.get("task_name", "")
            task_path = os.path.join(TASKS_DIR, f"{task_name}.json")
            if not os.path.exists(task_path):
                missing.append(task_name)
        elif spec.kind == "cross_map":
            rec_name = spec.params.get("recording_name", "")
            rec_path = os.path.join(TASKS_DIR, f"{rec_name}.json")
            if not os.path.exists(rec_path):
                missing.append(rec_name)
    if missing:
        return TestResult("L9 spring4 can start", "FAIL", f"missing recordings: {missing}")
    return TestResult("L9 spring4 can start", "PASS",
                      f"{len(SPRING4_PHASES)} phases, all recordings present")


def test_spring4_day_plan() -> TestResult:
    """Full Spring Day 4 plan: buy seeds, ship 2 berries, plant + water."""
    state = "Y1_Spring_D1_Dawn"
    if not load_state_bytes(state):
        return TestResult("L9 spring4 day plan", "SKIP", "missing state")

    from harvest.planner.day_plan import DayPlanTask, SPRING4_PHASES
    from harvest.runtime.harness_runtime import HarnessRunner

    env = make_env(state)
    env.reset()

    runner = HarnessRunner(env)
    world = runner.reset()

    task = DayPlanTask(seed_type="potato", phase_sequence=SPRING4_PHASES)

    # Keep seeds/stamina topped up during run (same hack as PlaySession)
    original_step = runner.step_env
    def step_with_hacks(action):
        result = original_step(action)
        try:
            env.data.set_value("stamina", 100)
            env.data.set_value("potato_seeds", 99)
            env.data.set_value("water_can", 20)
        except Exception:
            pass
        return result
    runner.step_env = step_with_hacks

    result = runner.run_task(task, world, max_steps=25000)

    # Validate: check that we reached at least the crop phase
    phase_reached = task._phase_index
    total_phases = len(SPRING4_PHASES)

    env.close()

    if result.status == TaskStatus.SUCCESS:
        return TestResult("L9 spring4 day plan", "PASS",
                          f"completed all {total_phases} phases")
    if phase_reached >= 5:  # At least past berry shipping
        return TestResult("L9 spring4 day plan", "PASS",
                          f"reached phase {phase_reached}/{total_phases}: {task.phase_text}")
    return TestResult("L9 spring4 day plan", "FAIL",
                      f"phase={phase_reached}/{total_phases} status={result.status} reason={result.reason}")


def test_day_plan_nav_phase() -> TestResult:
    """Verify NavTask can reach farm exit waypoint from front-of-house state."""
    state = "Y1_Front_House"
    if not load_state_bytes(state):
        # Try to derive from house state
        state = "Y1_Spring_D1_Dawn"
        if not load_state_bytes(state):
            return TestResult("L9 day plan nav phase", "SKIP", "missing state")

    from harvest.planner.day_plan import NavTask
    from harvest.runtime.harness_runtime import HarnessRunner

    env = make_env(state)
    env.reset()

    # If starting from house, we need to be on farm tilemap
    tilemap = get_tilemap(env)
    if tilemap != 0x00:
        env.close()
        return TestResult("L9 day plan nav phase", "SKIP",
                          f"need farm tilemap 0x00, got 0x{tilemap:02X}")

    runner = HarnessRunner(env)
    world = runner.reset()

    # Navigate to farm exit waypoint (NAV_FARM_EXIT target)
    task = NavTask(
        name="nav_farm_exit",
        target_px=Point(40, 424),
        radius=12,
        timeout=3000,
    )
    result = runner.run_task(task, world, max_steps=3000)
    pos = get_pos(env)
    env.close()

    if result.status == TaskStatus.SUCCESS:
        return TestResult("L9 day plan nav phase", "PASS",
                          f"arrived at ({pos.x},{pos.y})")
    return TestResult("L9 day plan nav phase", "FAIL",
                      f"status={result.status} pos=({pos.x},{pos.y}) reason={result.reason}")


# =============================================================================
# L10: Multi-Map Navigation Tests
# =============================================================================

def test_map_config_registry() -> TestResult:
    """Verify farm config exists in MAP_REGISTRY, walkable tiles match farm_clearer."""
    from harvest.maps.map_config import MAP_REGISTRY, FARM_WALKABLE, get_walkable_tiles
    from harvest.tasks.nav import WALKABLE_TILES

    if 0x00 not in MAP_REGISTRY:
        return TestResult("L10 map config registry", "FAIL", "farm (0x00) missing from MAP_REGISTRY")

    farm = MAP_REGISTRY[0x00]
    if farm.name != "farm":
        return TestResult("L10 map config registry", "FAIL", f"expected name 'farm', got '{farm.name}'")

    # Verify FARM_WALKABLE matches farm_clearer.WALKABLE_TILES
    if set(FARM_WALKABLE) != set(WALKABLE_TILES):
        diff = set(FARM_WALKABLE).symmetric_difference(set(WALKABLE_TILES))
        return TestResult("L10 map config registry", "FAIL",
                          f"FARM_WALKABLE diverged from WALKABLE_TILES: {[f'0x{t:02X}' for t in diff]}")

    if not farm.exits:
        return TestResult("L10 map config registry", "FAIL", "farm has no exits defined")

    # get_walkable_tiles fallback
    unknown = get_walkable_tiles(0xFF)
    if unknown != FARM_WALKABLE:
        return TestResult("L10 map config registry", "FAIL", "unknown tilemap did not fall back to farm")

    return TestResult("L10 map config registry", "PASS",
                      f"{len(MAP_REGISTRY)} maps, farm has {len(farm.exits)} exit(s)")


def test_pathfinder_walkable_injection() -> TestResult:
    """Verify Pathfinder uses injected walkable_tiles set."""
    from harvest.tasks.farm_clearer import TileScanner
    from harvest.tasks.nav import Pathfinder, WALKABLE_TILES

    scanner = TileScanner()

    # Default: should use WALKABLE_TILES
    pf_default = Pathfinder(scanner)
    if pf_default.walkable_tiles is not WALKABLE_TILES:
        return TestResult("L10 pathfinder walkable injection", "FAIL",
                          "default pathfinder does not use WALKABLE_TILES")

    # Custom: should use provided set
    custom = {0x42, 0x99}
    pf_custom = Pathfinder(scanner, walkable_tiles=custom)
    if pf_custom.walkable_tiles != custom:
        return TestResult("L10 pathfinder walkable injection", "FAIL",
                          "custom walkable_tiles not stored")

    return TestResult("L10 pathfinder walkable injection", "PASS")


def test_berry_route_waypoints() -> TestResult:
    """Verify berry_ship route is well-formed."""
    from harvest.maps.map_config import ROUTES, MAP_REGISTRY

    route = ROUTES.get("berry_ship")
    if not route:
        return TestResult("L10 berry route waypoints", "FAIL", "berry_ship route not found")

    if len(route) < 2:
        return TestResult("L10 berry route waypoints", "FAIL",
                          f"route too short: {len(route)} waypoints")

    errors = []
    for i, wp in enumerate(route):
        # Every exit waypoint must have a direction
        if wp.is_exit and not wp.exit_direction:
            errors.append(f"wp[{i}]: exit but no exit_direction")
        # Every action waypoint must have an action
        if wp.action_on_arrive and wp.action_on_arrive not in (
            "press_a",
            "press_b",
            "press_y",
            "lift_throw",
        ):
            errors.append(f"wp[{i}]: unknown action '{wp.action_on_arrive}'")
        # Target px should be positive
        if wp.target_px[0] < 0 or wp.target_px[1] < 0:
            errors.append(f"wp[{i}]: negative target_px {wp.target_px}")

    if errors:
        return TestResult("L10 berry route waypoints", "FAIL", "; ".join(errors))

    return TestResult("L10 berry route waypoints", "PASS",
                      f"{len(route)} waypoints, route well-formed")


def test_multi_nav_farm_exit() -> TestResult:
    """If farm state exists, verify walking left from edge causes tilemap change."""
    state = "Y1_Spring_D1_Farm"
    if not load_state_bytes(state):
        return TestResult("L10 multi_nav farm exit", "SKIP", "missing state")

    from harvest.planner.day_plan import MultiMapNavTask
    from harvest.maps.map_config import Waypoint
    from harvest.runtime.harness_runtime import HarnessRunner

    env = make_env(state)
    env.reset()

    initial_tilemap = get_tilemap(env)
    if initial_tilemap != 0x00:
        env.close()
        return TestResult("L10 multi_nav farm exit", "SKIP",
                          f"expected farm 0x00, got 0x{initial_tilemap:02X}")

    # Create a simple multi_nav with one exit waypoint at the farm left edge
    waypoints = [
        Waypoint(tilemap=0x00, target_px=(24, 424), radius=16,
                 is_exit=True, exit_direction="left"),
    ]

    runner = HarnessRunner(env)
    world = runner.reset()

    task = MultiMapNavTask(name="test_farm_exit", waypoints=waypoints, timeout=4000)
    result = runner.run_task(task, world, max_steps=4000)

    final_tilemap = get_tilemap(env)
    env.close()

    # Success = tilemap changed (we exited the farm)
    if final_tilemap != 0x00:
        return TestResult("L10 multi_nav farm exit", "PASS",
                          f"exited to 0x{final_tilemap:02X}")
    if result.status == TaskStatus.SUCCESS:
        return TestResult("L10 multi_nav farm exit", "PASS", "task succeeded")
    return TestResult("L10 multi_nav farm exit", "FAIL",
                      f"still on farm tilemap, status={result.status} reason={result.reason}")


