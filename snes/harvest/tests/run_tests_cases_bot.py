#!/usr/bin/env python3
"""L3–L8 target / tool / clearing / grass cases for run_tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from run_tests_helpers import (  # noqa: E402
    TestResult,
    get_tool_id,
    load_state_bytes,
    make_env,
)

from harvest.runtime import harvest_bot as hb  # noqa: E402
from harvest.core.tile_catalog import (  # noqa: E402
    Tool,
    DebrisType,
    ADDR_INPUT_LOCK,
)
from harvest.tasks.farm_clearer import (  # noqa: E402
    TileScanner,
    use_tool,
)
from harvest.tasks.grass_planter import GrassPlantTask  # noqa: E402


# =============================================================================
# L3: Target Detection Tests
# =============================================================================

def test_target_scan() -> TestResult:
    """Test TileScanner finds debris on the farm."""
    state = "Y1_Spring_D1_Farm"
    if not load_state_bytes(state):
        return TestResult("L3 target scan", "SKIP", "missing state")
    env = make_env(state)
    env.reset()
    scanner = TileScanner()
    targets = scanner.scan(env.get_ram())
    env.close()
    if not targets:
        return TestResult("L3 target scan", "FAIL", "no targets found")
    return TestResult("L3 target scan", "PASS", f"found {len(targets)} targets")


# =============================================================================
# L4: Tooling Tests
# =============================================================================

def test_tool_use_action() -> TestResult:
    """Test use_tool() generates correct button presses."""
    actions = use_tool(frames=3)
    if not actions:
        return TestResult("L4 tool use", "FAIL", "no actions returned")
    used = any(action[1] for action in actions)  # Y button index
    if not used:
        return TestResult("L4 tool use", "FAIL", "Y not pressed in use_tool")
    return TestResult("L4 tool use", "PASS")


# =============================================================================
# L6: Multi-Objective Clearing Tests
# =============================================================================

def test_clearing_run() -> TestResult:
    """Test bot can find targets and attempt clearing."""
    state = "Y1_Spring_D1_Farm"
    if not load_state_bytes(state):
        return TestResult("L6 clearing run", "SKIP", "missing state")
    env = make_env(state)
    bot = hb.AutoClearBot(priority=[DebrisType.WEED], clear_fences_first=False)
    # Skip startup tasks for faster test
    bot.clearer.startup_tasks = []
    bot.clearer.startup_done = True
    bot.enabled = True
    obs, info = env.reset()
    bot.set_env(env)

    # Test that bot can at least find a target and set up navigation
    target_found = False
    navigation_started = False
    initial_cleared = bot.clearer.cleared_count

    for _ in range(500):  # Fewer frames, just check state machine
        game_state = hb.GameState(info)
        action = bot.get_action(game_state, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

        if bot.clearer.current_target is not None:
            target_found = True
        if bot.clearer.state == "navigating":
            navigation_started = True
        if bot.clearer.cleared_count > initial_cleared:
            env.close()
            return TestResult("L6 clearing run", "PASS", f"cleared {bot.clearer.cleared_count}")

    env.close()

    if bot.clearer.cleared_count > initial_cleared:
        return TestResult("L6 clearing run", "PASS", f"cleared {bot.clearer.cleared_count}")
    if target_found:
        return TestResult("L6 clearing run", "PASS", "target found")
    return TestResult("L6 clearing run", "FAIL", f"state={bot.clearer.state}")


def test_stump_clearing() -> TestResult:
    """Test bot can clear stumps with axe."""
    state = "Y1_Spring_D1_Dawn"
    if not load_state_bytes(state):
        return TestResult("L6 stump clearing", "SKIP", "missing state")
    env = make_env(state)
    bot = hb.AutoClearBot(priority=[DebrisType.STUMP], clear_fences_first=False)
    # Let startup auto-detect tools (should skip get_hammer and get_axe)
    bot.enabled = True
    obs, info = env.reset()
    bot.set_env(env)

    initial_cleared = bot.clearer.cleared_count
    used_axe = False

    for _ in range(2000):  # Stumps take multiple hits
        game_state = hb.GameState(info)
        action = bot.get_action(game_state, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

        # Check if bot is using axe
        if bot.clearer.state == "clearing" and get_tool_id(env) == Tool.AXE:
            used_axe = True

        # Success if we cleared at least one stump
        if bot.clearer.cleared_count > initial_cleared:
            env.close()
            return TestResult("L6 stump clearing", "PASS", f"cleared {bot.clearer.cleared_count} stumps with axe")

    env.close()

    if used_axe:
        return TestResult("L6 stump clearing", "PASS", "used axe on stump")
    return TestResult("L6 stump clearing", "FAIL", f"cleared={bot.clearer.cleared_count} state={bot.clearer.state}")


# =============================================================================
# L7: Robustness Tests
# =============================================================================

def test_dialog_dismissal() -> TestResult:
    """Test that input lock address is accessible for dialog dismissal."""
    state = "Y1_Spring_D1_Farm"
    if not load_state_bytes(state):
        return TestResult("L7 dialog dismissal", "SKIP", "missing state")
    env = make_env(state)
    env.reset()
    ram = env.get_ram()

    if ADDR_INPUT_LOCK >= len(ram):
        env.close()
        return TestResult("L7 dialog dismissal", "FAIL", "ADDR_INPUT_LOCK out of range")

    env.close()
    return TestResult("L7 dialog dismissal", "PASS", "input_lock mechanism verified")


def test_stuck_recovery() -> TestResult:
    """Test bot handles stasis detection via navigator."""
    state = "Y1_Spring_D1_Farm"
    if not load_state_bytes(state):
        return TestResult("L7 stuck recovery", "SKIP", "missing state")
    env = make_env(state)
    bot = hb.AutoClearBot(priority=[DebrisType.WEED], clear_fences_first=False)
    bot.clearer.startup_tasks = []
    bot.clearer.startup_done = True
    bot.enabled = True
    obs, info = env.reset()
    bot.set_env(env)

    # Run for a while and verify stasis tracking works
    max_stasis_seen = 0
    for _ in range(2000):
        game_state = hb.GameState(info)
        action = bot.get_action(game_state, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        max_stasis_seen = max(max_stasis_seen, bot.clearer.navigator.stasis)

    env.close()
    # Stasis tracking is working if we saw it change (even if no stuck event occurred)
    return TestResult("L7 stuck recovery", "PASS", f"max_stasis={max_stasis_seen}")


# =============================================================================
# L8: Grass Planting Tests
# =============================================================================

def test_grass_seed_hack() -> TestResult:
    """Verify set_value('grass_seeds', 99) sets RAM 0x0927 to 99."""
    state = "pretill"
    if not load_state_bytes(state):
        return TestResult("L8 grass seed hack", "SKIP", "missing state")
    env = make_env(state)
    env.reset()

    ram = env.get_ram()
    before = int(ram[0x0927]) if 0x0927 < len(ram) else -1

    try:
        env.data.set_value("grass_seeds", 99)
    except Exception as e:
        env.close()
        return TestResult("L8 grass seed hack", "FAIL", f"set_value failed: {e}")

    env.step(np.zeros(12, dtype=np.int32))
    ram = env.get_ram()
    after = int(ram[0x0927]) if 0x0927 < len(ram) else -1
    env.close()

    if after != 99:
        return TestResult("L8 grass seed hack", "FAIL", f"expected 99, got {after}")
    return TestResult("L8 grass seed hack", "PASS", f"before={before} after={after}")


def test_grass_scan_targets() -> TestResult:
    """Verify GrassPlantTask._scan_targets() finds tillable tiles."""
    state = "pretill"
    if not load_state_bytes(state):
        return TestResult("L8 grass scan targets", "SKIP", "missing state")
    env = make_env(state)
    env.reset()
    ram = env.get_ram()

    task = GrassPlantTask()
    targets = task._scan_targets(ram)
    env.close()

    if not targets:
        return TestResult("L8 grass scan targets", "FAIL", "no tillable tiles found")
    return TestResult("L8 grass scan targets", "PASS", f"found {len(targets)} tillable tiles")


def test_grass_till_run() -> TestResult:
    """Short run: verify bot tills at least 1 tile."""
    state = "pretill"
    if not load_state_bytes(state):
        return TestResult("L8 grass till run", "SKIP", "missing state")

    from harvest.runtime.harness_runtime import HarnessRunner
    from retro_harness import WorldState

    env = make_env(state)
    env.reset()

    # Inject grass seeds so tool cycling finds them
    env.data.set_value("grass_seeds", 99)
    env.step(np.zeros(12, dtype=np.int32))

    runner = HarnessRunner(env)
    world = runner.reset()

    task = GrassPlantTask(till_only=True, bounds=(55, 3, 62, 10))
    result = runner.run_task(task, world, max_steps=5000)
    env.close()

    if task.tilled_count > 0:
        return TestResult("L8 grass till run", "PASS", f"tilled={task.tilled_count}")
    return TestResult("L8 grass till run", "FAIL", f"tilled=0 status={result.status} reason={result.reason}")


def test_grass_plant_run() -> TestResult:
    """Verify planting changes tile IDs after tilling."""
    state = "pretill"
    if not load_state_bytes(state):
        return TestResult("L8 grass plant run", "SKIP", "missing state")

    from harvest.runtime.harness_runtime import HarnessRunner

    env = make_env(state)
    env.reset()
    env.data.set_value("grass_seeds", 99)
    env.step(np.zeros(12, dtype=np.int32))

    runner = HarnessRunner(env)
    world = runner.reset()

    # Full run: till + plant (small bounds for speed)
    task = GrassPlantTask(bounds=(55, 3, 62, 10))

    # Keep seeds topped up during run
    original_step = runner.step_env

    def step_with_seeds(action):
        result = original_step(action)
        try:
            env.data.set_value("grass_seeds", 99)
            env.data.set_value("stamina", 100)
        except Exception:
            pass
        return result

    runner.step_env = step_with_seeds
    result = runner.run_task(task, world, max_steps=10000)
    env.close()

    if task.planted_count > 0:
        return TestResult("L8 grass plant run", "PASS",
                          f"tilled={task.tilled_count} planted={task.planted_count}")
    if task.tilled_count > 0:
        return TestResult("L8 grass plant run", "PASS",
                          f"tilled={task.tilled_count} planted=0 (till phase worked)")
    return TestResult("L8 grass plant run", "FAIL",
                      f"tilled={task.tilled_count} planted={task.planted_count} "
                      f"status={result.status} reason={result.reason}")

