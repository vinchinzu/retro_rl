"""Offline checks for the Magical Sword detour (coast hearts + 0x21 grave).

The legs themselves are ROM-measured (module docstring); these pin the grid
walk, the skip gates and the per-leg success rules.
"""

from __future__ import annotations

import numpy as np
import pytest

from zelda_i.overworld.graph import neighbor_screens
from zelda_i.overworld.magical_sword import (
    COAST_BACK_HOPS,
    COAST_TO_5F_HOPS,
    FROM_GRAVE_HOPS,
    GRAVE_STAND,
    LADDER_HEART_SHORE,
    TO_GRAVE_HOPS,
    CoastHeartPlan,
    GravePlan,
    GraveSwordController,
    LadderHeartController,
    coast_heart_stages,
    magical_sword_stages,
)
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LADDER,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_RAFT,
    ADDR_SCREEN,
    ADDR_SWORD,
    PLAY_MODE,
    read_snapshot,
)


def _ow(*, screen: int, containers: int = 9, sword: int = 2, x: int = 112, y: int = 93):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 0
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_RAFT] = 1
    ram[ADDR_LADDER] = 1
    ram[ADDR_SWORD] = sword
    ram[ADDR_HEALTH] = ((containers - 1) << 4) | (containers - 1)
    return ram


def _walk(start: int, hops) -> list[int]:
    screens = [start]
    for hop in hops:
        assert hop.target in neighbor_screens(screens[-1]).values(), hex(hop.target)
        screens.append(hop.target)
    return screens


def test_coast_walks_are_grid_paths() -> None:
    assert _walk(0x67, COAST_TO_5F_HOPS)[-1] == 0x5F
    # Back from the dock: 0x4F (off_dock), then the coast west to 0x4A.
    assert _walk(0x4F, COAST_BACK_HOPS)[-1] == 0x4A
    assert _walk(0x33, TO_GRAVE_HOPS)[-1] == 0x21
    assert _walk(0x21, FROM_GRAVE_HOPS)[-1] == 0x32
    # 0x6F's north mouth is x 80..128: an off-lattice align flips forever.
    assert all(h.align_x is None or h.align_x % 8 == 0 for h in COAST_TO_5F_HOPS[-1:])


@pytest.mark.parametrize(
    ("screen", "containers", "raft", "active", "reason"),
    [
        (0x67, 9, 1, True, ""),
        (0x67, 11, 1, False, "containers_enough"),
        (0x67, 9, 0, False, "no_ladder_or_raft"),
        (0x45, 9, 1, False, "not_from_0x67"),
    ],
)
def test_coast_plan_gate(screen, containers, raft, active, reason) -> None:
    ram = _ow(screen=screen, containers=containers)
    ram[ADDR_RAFT] = raft
    plan = CoastHeartPlan()
    assert plan.decide(read_snapshot(ram)) is active
    assert plan.reason == reason


@pytest.mark.parametrize(
    ("screen", "containers", "sword", "active"),
    [(0x33, 12, 2, True), (0x33, 11, 2, False), (0x33, 12, 3, False), (0x22, 12, 2, False)],
)
def test_grave_plan_gate(screen, containers, sword, active) -> None:
    plan = GravePlan()
    ram = _ow(screen=screen, containers=containers, sword=sword)
    assert plan.decide(read_snapshot(ram)) is active


def test_skipped_detours_finish_every_leg_on_frame_one() -> None:
    for stages, ram in (
        (coast_heart_stages(), _ow(screen=0x45)),
        (magical_sword_stages(), _ow(screen=0x33, containers=11)),
    ):
        snap = read_snapshot(ram)
        for _, leg, _ in stages:
            leg.step(snap)
            assert leg.success and not leg.failed and leg.frames == 1


def test_ladder_heart_walks_back_to_shore_after_the_pickup() -> None:
    ctl = LadderHeartController()
    shore = read_snapshot(_ow(screen=0x5F, x=LADDER_HEART_SHORE[0], y=LADDER_HEART_SHORE[1]))
    assert ctl.step(shore).reason == "ladder_right"
    stone = _ow(screen=0x5F, containers=10, x=192, y=141)
    assert ctl.step(read_snapshot(stone)).reason == "ladder_left"
    assert not ctl.success
    back = _ow(screen=0x5F, containers=10, x=128, y=141)
    assert ctl.step(read_snapshot(back)).reason == "ladder_ashore"
    assert ctl.success


def test_grave_push_holds_up_from_the_stand_and_stops_on_the_sword() -> None:
    ctl = GraveSwordController()
    at_stand = read_snapshot(_ow(screen=0x21, containers=12, x=GRAVE_STAND[0], y=GRAVE_STAND[1]))
    assert ctl.step(at_stand).reason == "grave_push"
    sword = read_snapshot(_ow(screen=0x21, containers=12, sword=3, x=120, y=149))
    ctl.step(sword)
    assert ctl.success
