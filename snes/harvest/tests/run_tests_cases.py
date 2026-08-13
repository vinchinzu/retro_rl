#!/usr/bin/env python3
"""Aggregate ordered test callables for the Harvest run_tests suite."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from run_tests_helpers import TestResult  # noqa: E402
from run_tests_cases_replay import (  # noqa: E402
    test_ship_berry_replay,
    test_get_hammer_replay,
    test_go_to_barn_replay,
    test_go_to_coop_replay,
    test_toss_fence_pond_replay,
    test_fence_clear_loop,
    test_get_hoe_replay,
    test_buy_potato_seeds_replay,
    test_get_water_can_replay,
    test_dual_item_swap_replay,
    test_nav_to_shed,
    test_nav_deep_field_to_shed,
)
from run_tests_cases_bot import (  # noqa: E402
    test_target_scan,
    test_tool_use_action,
    test_clearing_run,
    test_stump_clearing,
    test_dialog_dismissal,
    test_stuck_recovery,
    test_grass_seed_hack,
    test_grass_scan_targets,
    test_grass_till_run,
    test_grass_plant_run,
)
from run_tests_cases_plan import (  # noqa: E402
    test_day_plan_can_start,
    test_day_plan_exit_house,
    test_day_plan_nav_phase,
    test_spring4_can_start,
    test_spring4_day_plan,
    test_map_config_registry,
    test_pathfinder_walkable_injection,
    test_berry_route_waypoints,
    test_multi_nav_farm_exit,
)

# Ordered suite consumed by run_tests.main (matches historical registry)
ALL_TESTS: list[Callable[[], TestResult]] = [
    # L1: Deterministic task replay
    test_ship_berry_replay,
    test_get_hammer_replay,
    # L2: Navigation and tool acquisition
    test_go_to_barn_replay,
    test_go_to_coop_replay,
    test_toss_fence_pond_replay,
    test_get_hoe_replay,
    test_buy_potato_seeds_replay,
    test_get_water_can_replay,
    test_dual_item_swap_replay,
    test_nav_to_shed,
    test_nav_deep_field_to_shed,
    # L3: Target detection
    test_target_scan,
    # L4: Tooling
    test_tool_use_action,
    # L6: Multi-objective clearing
    test_clearing_run,
    test_stump_clearing,
    test_fence_clear_loop,
    # L7: Robustness
    test_dialog_dismissal,
    test_stuck_recovery,
    # L8: Grass planting
    test_grass_seed_hack,
    test_grass_scan_targets,
    test_grass_till_run,
    test_grass_plant_run,
    # L9: Day plan
    test_day_plan_can_start,
    test_day_plan_exit_house,
    test_day_plan_nav_phase,
    test_spring4_can_start,
    test_spring4_day_plan,
    # L10: Multi-map navigation
    test_map_config_registry,
    test_pathfinder_walkable_injection,
    test_berry_route_waypoints,
    test_multi_nav_farm_exit,
]
