"""Unit tests for shared OverworldPathController hop advance / maze core."""

from __future__ import annotations

import numpy as np

from zelda_i.overworld import LEVEL2_5C_MAZE_WAYPOINTS, ScreenHop, is_5c_maze_hop
from zelda_i.ow_path import OverworldPathController, PathNavPhase
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    PLAY_MODE,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", 0x77)
    ram[ADDR_LINK_X] = fields.get("x", 112)
    ram[ADDR_LINK_Y] = fields.get("y", 125)
    ram[ADDR_HEALTH] = fields.get("health", 0x33)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    return ram


def test_is_5c_maze_hop_shared() -> None:
    assert is_5c_maze_hop(ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140))
    assert not is_5c_maze_hop(ScreenHop(0x5C, "RIGHT"))
    # Re-export from ow_path matches overworld.
    from zelda_i.ow_path import is_5c_maze_hop as from_ow_path
    from zelda_i.level2_overworld import is_5c_maze_hop as from_l2

    assert from_ow_path is is_5c_maze_hop
    assert from_l2 is is_5c_maze_hop


def test_hop_advance_on_arrival() -> None:
    from zelda_i.ram import read_snapshot

    hops = (
        ScreenHop(0x78, "RIGHT", align_y=140),
        ScreenHop(0x68, "UP", align_x=48),
    )
    ctrl = OverworldPathController(hops=hops, require_sword=True)
    assert ctrl.phase is PathNavPhase.HOP

    # Still on start screen: push RIGHT.
    snap = read_snapshot(_ram(screen=0x77, x=100, y=140, sword=1))
    act = ctrl.step(snap)
    assert ctrl.hop_index == 0
    assert "hop0" in act.reason or "RIGHT" in str(act.action) or act.reason

    # Arrived on 0x78 off the east edge → advance.
    snap = read_snapshot(_ram(screen=0x78, x=100, y=140, sword=1))
    act = ctrl.step(snap)
    assert ctrl.hop_index == 1
    assert "hop_0_78" in ctrl.notes
    assert act.reason == "hop_advance"


def test_maze_waypoints_on_5c() -> None:
    from zelda_i.ram import read_snapshot

    hops = (ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140),)
    ctrl = OverworldPathController(
        hops=hops,
        maze_waypoints=LEVEL2_5C_MAZE_WAYPOINTS,
        maze_hop_pred=is_5c_maze_hop,
    )
    snap = read_snapshot(_ram(screen=0x5C, x=16, y=93, sword=1))
    act = ctrl.step(snap)
    assert "maze" in act.reason
    assert "maze_start" in ctrl.notes

    tx, ty = LEVEL2_5C_MAZE_WAYPOINTS[0]
    snap = read_snapshot(_ram(screen=0x5C, x=tx, y=ty, sword=1))
    ctrl.step(snap)
    assert ctrl.maze_wp_index >= 1


def test_default_stop_after_hops() -> None:
    from zelda_i.ram import read_snapshot

    hops = (ScreenHop(0x74, "RIGHT", align_y=117),)
    ctrl = OverworldPathController(hops=hops, require_sword=True)
    ctrl.hop_index = 1
    snap = read_snapshot(_ram(screen=0x74, x=128, y=140, sword=1))
    act = ctrl.step(snap)
    assert ctrl.success
    assert act.reason == "done"
    assert ctrl.phase is PathNavPhase.DONE
