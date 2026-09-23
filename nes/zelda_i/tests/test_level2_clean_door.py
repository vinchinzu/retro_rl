"""Unit tests for Clean door-path hop tables / helpers (no emulator)."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.level2.clean_door import EAST_Y, REJOIN_HOPS
from zelda_i.level2.overworld import (
    LEVEL2_CLEAN_FROM_4A_TO_5A,
    LEVEL2_CLEAN_FROM_5A_TO_3C,
    OverworldToLevel2Controller,
    is_5c_maze_hop,
)
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    PLAY_MODE,
    read_snapshot,
)


def test_rejoin_hops_avoid_4b_and_south_4a() -> None:
    assert REJOIN_HOPS[0].target == 0x49
    assert REJOIN_HOPS[0].direction == "LEFT"
    assert REJOIN_HOPS[1].target == 0x59
    assert REJOIN_HOPS[1].direction == "DOWN"
    assert 0x4B not in {h.target for h in REJOIN_HOPS}
    assert 0x5A not in {h.target for h in REJOIN_HOPS}  # no direct 4A→5A


def test_clean_from_5a_has_maze_and_door() -> None:
    assert LEVEL2_CLEAN_FROM_5A_TO_3C[0].align_y == EAST_Y
    assert LEVEL2_CLEAN_FROM_5A_TO_3C[-1].target == 0x3C
    maze = [h for h in LEVEL2_CLEAN_FROM_5A_TO_3C if is_5c_maze_hop(h)]
    assert len(maze) == 1
    assert LEVEL2_CLEAN_FROM_4A_TO_5A[-1].align_y == EAST_Y


def test_rejoin_from_4a_one_heart_walks_left_not_farm() -> None:
    """Natural leftover (64,125) hp 0x31 on 0x4A: LEFT to 0x49, never farm."""
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 0
    ram[ADDR_SCREEN] = 0x4A
    ram[ADDR_LINK_X] = 64
    ram[ADDR_LINK_Y] = 125
    ram[ADDR_HEALTH] = 0x31
    ram[ADDR_SWORD] = 1
    default = OverworldToLevel2Controller(hops=REJOIN_HOPS)
    default.step(read_snapshot(ram))
    assert default._farm is not None
    nav = OverworldToLevel2Controller(hops=REJOIN_HOPS, farm_below_hearts=0)
    act = nav.step(read_snapshot(ram))
    assert nav._farm is None
    assert "farm" not in str(act.reason)
    walked = act.action in (
        nes_action("LEFT"),
        nes_action("LEFT", "A"),
        nes_action("DOWN"),
        nes_action("DOWN", "A"),
    )
    assert walked, act.reason


def _l2_ram(*, room: int, x: int, y: int, keys: int) -> np.ndarray:
    from zelda_i.ram import ADDR_KEYS

    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 2
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_HEALTH] = 0x33
    ram[ADDR_KEYS] = keys
    return ram


