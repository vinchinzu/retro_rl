"""Unit tests for Clean door-path hop tables / helpers (no emulator)."""

from __future__ import annotations

from zelda_i.level2_clean_door import EAST_Y, REJOIN_HOPS
from zelda_i.level2_overworld import (
    LEVEL2_CLEAN_FROM_4A_TO_5A,
    LEVEL2_CLEAN_FROM_5A_TO_3C,
    is_5c_maze_hop,
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
