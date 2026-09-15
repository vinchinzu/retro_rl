"""Frozen-RAM tests for vertical-hop x-align at the far wall.

``align_and_push`` used to drop ``align_x`` outside ``80 < y < 205``.
A DOWN hop at y=205 (south rock, not EDGE_SOUTH_Y=212) then hammered
DOWN 8px west of the gap; the UP hop is the north-wall mirror.

Wall-column alignment is opt-in per hop (``ScreenHop.align_x_at_wall``);
the hops below set it. A hop that does *not* set it keeps the interior
band — see ``test_ow_path.py`` (the post-L6 0x22 DOWN leftover strafed
onto the L6 cave column when the rule was blanket).
"""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.overworld.common import align_and_push
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.path import OverworldPathController
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


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", 0x48)
    ram[ADDR_LINK_X] = fields.get("x", 112)
    ram[ADDR_LINK_Y] = fields.get("y", 205)
    ram[ADDR_HEALTH] = fields.get("health", 0x33)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    return ram


def _cardinal(act) -> str:
    if act.action == nes_idle_action():
        return "IDLE"
    for name in ("LEFT", "RIGHT", "UP", "DOWN"):
        if act.action == nes_action(name) or act.action == nes_action(name, "A"):
            return name
    return "OTHER"


def _ctrl() -> OverworldPathController:
    return OverworldPathController(
        hops=(ScreenHop(0x58, "DOWN", align_x=120, align_x_at_wall=True),),
        farm_below_hearts=0,
    )


def test_down_hop_off_column_at_south_wall_strafes_right() -> None:
    """0x48 (112,205) DOWN align_x=120: first action RIGHT, not DOWN."""
    ctrl = _ctrl()
    act = ctrl.step(read_snapshot(_ram(screen=0x48, x=112, y=205)))
    assert _cardinal(act) == "RIGHT"
    assert _cardinal(act) != "DOWN"
    assert act.reason.endswith("_ax") or "_occ" in act.reason


def test_level2_0x48_hop_uses_gap_column() -> None:
    """Production 0x48→0x58 hop must keep align_x=120 (the south gap)."""
    from zelda_i.overworld.graph import LEVEL2_PATH_HOPS

    hop = LEVEL2_PATH_HOPS[2]
    assert hop.target == 0x58 and hop.direction == "DOWN" and hop.align_x == 120
    ctrl = OverworldPathController(hops=LEVEL2_PATH_HOPS, farm_below_hearts=0)
    ctrl.hop_index = 2
    act = ctrl.step(read_snapshot(_ram(screen=0x48, x=112, y=205)))
    assert _cardinal(act) == "RIGHT"
    assert _cardinal(act) != "DOWN"


def test_down_hop_on_column_at_south_wall_still_pushes_down() -> None:
    """On-column at x=120 y=205 may still push DOWN."""
    ctrl = _ctrl()
    act = ctrl.step(read_snapshot(_ram(screen=0x48, x=120, y=205)))
    assert _cardinal(act) == "DOWN"


def test_up_hop_off_column_at_north_wall_strafes_right() -> None:
    """UP hop at y<=80 still strafes onto align_x (north-wall mirror)."""
    ctrl = OverworldPathController(
        hops=(ScreenHop(0x38, "UP", align_x=120, align_x_at_wall=True),),
        farm_below_hearts=0,
    )
    act = ctrl.step(read_snapshot(_ram(screen=0x48, x=112, y=80)))
    assert _cardinal(act) == "RIGHT"
    assert _cardinal(act) != "UP"


def test_up_hop_on_column_at_north_wall_still_pushes_up() -> None:
    ctrl = OverworldPathController(
        hops=(ScreenHop(0x38, "UP", align_x=120, align_x_at_wall=True),),
        farm_below_hearts=0,
    )
    act = ctrl.step(read_snapshot(_ram(screen=0x48, x=120, y=80)))
    assert _cardinal(act) == "UP"


def test_down_hop_off_column_in_interior_still_strafes() -> None:
    """The 80<y<205 band is unchanged for a DOWN hop in the interior."""
    ctrl = _ctrl()
    act = ctrl.step(read_snapshot(_ram(screen=0x48, x=112, y=140)))
    assert _cardinal(act) == "RIGHT"


def test_right_hop_at_south_wall_does_not_strafe_x() -> None:
    """Horizontal hops still drop align_x at y>=205 so they hit the mouth."""
    snap = read_snapshot(_ram(screen=0x48, x=128, y=205))
    act = align_and_push(
        snap,
        direction="RIGHT",
        reason="hop0",
        align_x=120,
        stuck=0,
    )
    assert _cardinal(act) == "RIGHT"
    assert not act.reason.endswith("_ax")
