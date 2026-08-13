#!/usr/bin/env python3
"""L1–L2 replay / navigation cases for run_tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from run_tests_helpers import (  # noqa: E402
    TestResult,
    count_tile_id,
    get_money_from_info,
    get_money_values,
    get_pos,
    get_potato_seeds,
    get_potato_seeds_from_info,
    get_tilemap,
    get_tool_id,
    get_water_can_level,
    load_state_bytes,
    make_env,
    require_task,
    run_task,
)

from harvest.core.tile_catalog import Tool  # noqa: E402
from harvest.tasks.nav import Point, TILE_SIZE  # noqa: E402
from retro_harness import TaskStatus  # noqa: E402


# =============================================================================
# L1: Deterministic Task Replay Tests
# =============================================================================

def test_ship_berry_replay() -> TestResult:
    task = require_task("ship_berry")
    if task is None:
        return TestResult("L1 ship_berry replay", "SKIP", "missing task")
    if not task.start_state or not load_state_bytes(task.start_state):
        return TestResult("L1 ship_berry replay", "SKIP", "missing start_state")
    env = make_env(task.start_state)
    env.reset()
    start_pos = get_pos(env)
    run_task(env, task)
    end_pos = get_pos(env)
    env.close()
    if start_pos == end_pos:
        return TestResult("L1 ship_berry replay", "FAIL", "position did not change")
    return TestResult("L1 ship_berry replay", "PASS")


def test_get_hammer_replay() -> TestResult:
    # Try shed_grab_hammer_smash_rock first (more reliable), fall back to get_hammer
    task = require_task("shed_grab_hammer_smash_rock") or require_task("get_hammer")
    if task is None:
        return TestResult("L1 get_hammer replay", "SKIP", "missing task")
    if not task.start_state or not load_state_bytes(task.start_state):
        return TestResult("L1 get_hammer replay", "SKIP", "missing start_state")
    env = make_env(task.start_state)
    env.reset()
    run_task(env, task)
    tool_id = get_tool_id(env)
    env.close()
    if tool_id != int(Tool.HAMMER):
        return TestResult("L1 get_hammer replay", "FAIL", f"tool_id=0x{tool_id:02X}")
    return TestResult("L1 get_hammer replay", "PASS")


# =============================================================================
# L2: Navigation and Tool Acquisition Tests
# =============================================================================

def test_go_to_barn_replay() -> TestResult:
    task = require_task("go_to_barn")
    if task is None:
        return TestResult("L2 go_to_barn replay", "SKIP", "missing task")
    if not task.start_state or not load_state_bytes(task.start_state):
        return TestResult("L2 go_to_barn replay", "SKIP", "missing start_state")
    env = make_env(task.start_state)
    env.reset()
    run_task(env, task)
    tilemap = get_tilemap(env)
    env.close()
    if tilemap != 0x27:
        return TestResult("L2 go_to_barn replay", "FAIL", f"tilemap=0x{tilemap:02X}")
    return TestResult("L2 go_to_barn replay", "PASS")


def test_go_to_coop_replay() -> TestResult:
    task = require_task("go_to_coop")
    if task is None:
        return TestResult("L2 go_to_coop replay", "SKIP", "missing task")
    if not task.start_state or not load_state_bytes(task.start_state):
        return TestResult("L2 go_to_coop replay", "SKIP", "missing start_state")
    env = make_env(task.start_state)
    env.reset()
    run_task(env, task)
    tilemap = get_tilemap(env)
    env.close()
    if tilemap != 0x28:
        return TestResult("L2 go_to_coop replay", "FAIL", f"tilemap=0x{tilemap:02X}")
    return TestResult("L2 go_to_coop replay", "PASS")


def test_toss_fence_pond_replay() -> TestResult:
    task = require_task("toss_fence_pond")
    if task is None:
        return TestResult("L5 toss_fence_pond replay", "SKIP", "missing task")
    if not task.start_state or not load_state_bytes(task.start_state):
        return TestResult("L5 toss_fence_pond replay", "SKIP", "missing start_state")

    successes = 0
    for _ in range(3):
        env = make_env(task.start_state)
        env.reset()
        before = count_tile_id(env, 0x05)
        run_task(env, task)
        after = count_tile_id(env, 0x05)
        env.close()
        if after < before:
            successes += 1

    if successes < 3:
        return TestResult("L5 toss_fence_pond replay", "FAIL", f"removed {successes}/3 fences")
    return TestResult("L5 toss_fence_pond replay", "PASS")


def test_fence_clear_loop() -> TestResult:
    task = require_task("toss_fence_pond")
    if task is None:
        return TestResult("L6 fence clear loop", "SKIP", "missing toss_fence_pond task")
    if not task.start_state or not load_state_bytes(task.start_state):
        return TestResult("L6 fence clear loop", "SKIP", "missing start_state")

    from harvest.runtime.harness_runtime import HarnessRunner
    from harvest.tasks.fence_flow import FenceClearLoopTask

    env = make_env(task.start_state)
    env.reset()
    initial_fences = count_tile_id(env, 0x05)
    if initial_fences < 3:
        env.close()
        return TestResult("L6 fence clear loop", "SKIP", f"only {initial_fences} fences")

    runner = HarnessRunner(env)
    world = runner.reset()
    clear_task = FenceClearLoopTask(max_fences=3)
    result = runner.run_task(clear_task, world, max_steps=12000)
    after_fences = count_tile_id(env, 0x05)
    env.close()

    if result.status != TaskStatus.SUCCESS:
        reason = result.reason or ""
        return TestResult("L6 fence clear loop", "FAIL", f"status={result.status} {reason}".strip())
    removed = initial_fences - after_fences
    if removed < 3 or clear_task.cleared_count < 3:
        return TestResult("L6 fence clear loop", "FAIL", f"removed={removed} cleared={clear_task.cleared_count}")
    return TestResult("L6 fence clear loop", "PASS", f"removed={removed}")


def test_get_hoe_replay() -> TestResult:
    task = require_task("get_hoe")
    if task is None:
        return TestResult("L2 get_hoe replay", "SKIP", "missing task")
    if not task.start_state or not load_state_bytes(task.start_state):
        return TestResult("L2 get_hoe replay", "SKIP", "missing start_state")
    env = make_env(task.start_state)
    env.reset()
    run_task(env, task)
    tool_id = get_tool_id(env)
    env.close()
    if tool_id != int(Tool.HOE):
        return TestResult("L2 get_hoe replay", "FAIL", f"tool_id=0x{tool_id:02X}")
    return TestResult("L2 get_hoe replay", "PASS")


def test_buy_potato_seeds_replay() -> TestResult:
    task = require_task("buy_potato_seeds")
    if task is None:
        return TestResult("L2 buy_potato_seeds replay", "SKIP", "missing task")
    if not task.start_state or not load_state_bytes(task.start_state):
        return TestResult("L2 buy_potato_seeds replay", "SKIP", "missing start_state")
    env = make_env(task.start_state)
    obs, info = env.reset()
    info_has_money = "money_lo" in info
    info_has_seeds = "potato_seeds" in info
    seeds_before = get_potato_seeds(env)
    seeds_info_before = get_potato_seeds_from_info(info)
    money_before_values = get_money_values(env)
    money_info_before = get_money_from_info(info)
    seeds_after = seeds_before
    seeds_info_after = seeds_info_before
    money_after_values = money_before_values
    money_info_after = money_info_before
    saw_money_drop = False
    saw_money_drop_info = False
    saw_money_nonzero = False
    mismatch = False
    for frame in task.frames:
        action = np.array(frame, dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        money_now_values = get_money_values(env)
        money_info_now = get_money_from_info(info)
        seeds_now = get_potato_seeds(env)
        seeds_info_now = get_potato_seeds_from_info(info)
        for key, now_val in money_now_values.items():
            prev_val = money_after_values.get(key)
            if now_val is not None and prev_val is not None and now_val < prev_val:
                saw_money_drop = True
        if info_has_money and money_info_now < money_info_after:
            saw_money_drop_info = True
        money_bcd = money_now_values.get("money_bcd")
        money_bcd_mirror = money_now_values.get("money_bcd_mirror")
        if money_bcd is not None and money_bcd_mirror is not None and money_bcd != money_bcd_mirror:
            mismatch = True
        if (money_bcd or 0) > 0 or (money_bcd_mirror or 0) > 0:
            saw_money_nonzero = True
        if info_has_seeds and seeds_now is not None and seeds_now != seeds_info_now:
            mismatch = True
        seeds_after = seeds_now
        seeds_info_after = seeds_info_now
        money_after_values = money_now_values
        money_info_after = money_info_now
    env.close()
    if seeds_before is None or seeds_after is None:
        return TestResult("L2 buy_potato_seeds replay", "FAIL", "potato seeds addr out of range")
    if seeds_after <= seeds_before and seeds_info_after <= seeds_info_before:
        return TestResult("L2 buy_potato_seeds replay", "FAIL", "potato seeds did not increase")
    if mismatch:
        return TestResult("L2 buy_potato_seeds replay", "FAIL", "money/seeds info+ram mismatch")
    if not (saw_money_drop or saw_money_drop_info):
        if not saw_money_nonzero:
            return TestResult("L2 buy_potato_seeds replay", "SKIP", "money addr stayed zero")
        return TestResult("L2 buy_potato_seeds replay", "FAIL", "money never decreased")
    return TestResult("L2 buy_potato_seeds replay", "PASS")


def test_get_water_can_replay() -> TestResult:
    task = require_task("get_water_can")
    if task is None:
        return TestResult("L2 get_water_can replay", "SKIP", "missing task")
    if not task.start_state or not load_state_bytes(task.start_state):
        return TestResult("L2 get_water_can replay", "SKIP", "missing start_state")
    env = make_env(task.start_state)
    env.reset()
    water_prev = get_water_can_level(env)
    tool_ids = set()
    saw_decrease = False
    saw_increase = False
    for frame in task.frames:
        action = np.array(frame, dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        tool_ids.add(get_tool_id(env))
        water_now = get_water_can_level(env)
        if water_now < water_prev:
            saw_decrease = True
        if water_now > water_prev:
            saw_increase = True
        water_prev = water_now
    env.close()
    if Tool.WATERING_CAN not in tool_ids:
        return TestResult("L2 get_water_can replay", "FAIL", "watering can not selected")
    if not saw_decrease:
        return TestResult("L2 get_water_can replay", "FAIL", "water can never decreased")
    if not saw_increase:
        return TestResult("L2 get_water_can replay", "FAIL", "water can never increased")
    return TestResult("L2 get_water_can replay", "PASS")


def test_dual_item_swap_replay() -> TestResult:
    task = require_task("dual_item_swap")
    if task is None:
        return TestResult("L2 dual_item_swap replay", "SKIP", "missing task")
    if not task.start_state or not load_state_bytes(task.start_state):
        return TestResult("L2 dual_item_swap replay", "SKIP", "missing start_state")
    env = make_env(task.start_state)
    env.reset()
    tool_ids = set()
    water_prev = get_water_can_level(env)
    saw_decrease = False
    for frame in task.frames:
        action = np.array(frame, dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        tool_ids.add(get_tool_id(env))
        water_now = get_water_can_level(env)
        if water_now < water_prev:
            saw_decrease = True
        water_prev = water_now
    env.close()
    if Tool.HOE not in tool_ids:
        return TestResult("L2 dual_item_swap replay", "FAIL", "hoe not selected")
    if Tool.WATERING_CAN not in tool_ids:
        return TestResult("L2 dual_item_swap replay", "FAIL", "watering can not selected")
    if len(tool_ids) < 2:
        return TestResult("L2 dual_item_swap replay", "FAIL", "tool never changed")
    if not saw_decrease:
        return TestResult("L2 dual_item_swap replay", "FAIL", "water can never decreased")
    return TestResult("L2 dual_item_swap replay", "PASS")


def test_nav_to_shed() -> TestResult:
    """Test bot pathfinding to shed target."""
    state = "Y1_Spring_D1_Farm"
    if not load_state_bytes(state):
        return TestResult("L2 nav to shed", "SKIP", "missing state")
    env = make_env(state)
    env.reset()
    ram = env.get_ram()

    # Test that pathfinding can find a path from player to shed
    from harvest.tasks.farm_clearer import TileScanner
    from harvest.tasks.nav import Pathfinder, Navigator, get_pos_from_ram
    scanner = TileScanner()
    pathfinder = Pathfinder(scanner)
    navigator = Navigator(pathfinder)
    navigator.update(ram)

    target = Point(342, 489)
    target_tile = (target.x // TILE_SIZE, target.y // TILE_SIZE)
    approach = pathfinder.find_approach(ram, target_tile, navigator.current_pos)

    env.close()

    if approach is None:
        return TestResult("L2 nav to shed", "FAIL", "no approach tile found")

    path = pathfinder.find_path(ram, navigator.current_tile, approach)
    if path is None:
        return TestResult("L2 nav to shed", "FAIL", "no path found")

    return TestResult("L2 nav to shed", "PASS", f"path_len={len(path)}")


def test_nav_deep_field_to_shed() -> TestResult:
    """Test bot pathfinding from deep field to shed."""
    state = "Y1_Deep_Field"
    if not load_state_bytes(state):
        return TestResult("L2 nav deep field -> shed", "SKIP", "missing state")
    env = make_env(state)
    env.reset()
    ram = env.get_ram()

    # Test that pathfinding can find a path
    from harvest.tasks.farm_clearer import TileScanner
    from harvest.tasks.nav import Pathfinder, Navigator, get_pos_from_ram
    scanner = TileScanner()
    pathfinder = Pathfinder(scanner)
    navigator = Navigator(pathfinder)
    navigator.update(ram)

    target = Point(342, 489)
    target_tile = (target.x // TILE_SIZE, target.y // TILE_SIZE)
    approach = pathfinder.find_approach(ram, target_tile, navigator.current_pos)

    env.close()

    if approach is None:
        return TestResult("L2 nav deep field -> shed", "FAIL", "no approach tile found")

    path = pathfinder.find_path(ram, navigator.current_tile, approach)
    if path is None:
        return TestResult("L2 nav deep field -> shed", "FAIL", "no path found")

    return TestResult("L2 nav deep field -> shed", "PASS", f"path_len={len(path)}")

