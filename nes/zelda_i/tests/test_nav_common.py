from __future__ import annotations

import numpy as np

from zelda_i.combat import in_sword_hitbox, overworld_threat_objects, should_swing_at
from zelda_i.nav_common import (
    swing_action,
    walk_or_swing,
)
from zelda_i.overworld import ScreenHop, path_screens_from_hops
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _snap(
    *,
    x: int = 120,
    y: int = 140,
    screen: int = 0x37,
    mode: int = PLAY_MODE,
    obj: tuple[int, int, int, int] | None = None,
):
    """Optional ``obj`` is (slot, type_id, ox, oy)."""
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    if obj is not None:
        slot, type_id, ox, oy = obj
        ram[ADDR_OBJ_TYPE + slot] = type_id
        ram[ADDR_LINK_X + slot] = ox
        ram[ADDR_LINK_Y + slot] = oy
    return read_snapshot(ram)


def test_path_screens_from_hops() -> None:
    hops = (
        ScreenHop(0x38, "RIGHT", align_y=140),
        ScreenHop(0x48, "DOWN", align_x=120),
    )
    assert path_screens_from_hops(0x37, hops) == (0x37, 0x38, 0x48)


def test_walk_or_swing_no_enemies_no_slash() -> None:
    snap = _snap(x=120, y=140)
    # Pulse frame would slash under bare swing_action; empty screen stays walk.
    act = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    assert act.reason == "walk"
    assert act.action != swing_action(0, "RIGHT", "walk", period=10, hold=3).action


def test_walk_or_swing_enemy_in_front_slashes() -> None:
    # Enemy just to the right of Link → in sword hitbox for RIGHT.
    snap = _snap(x=120, y=140, obj=(1, 0x06, 135, 140))
    threats = overworld_threat_objects(snap)
    assert in_sword_hitbox(120, 140, "RIGHT", 135, 140)
    assert should_swing_at(120, 140, "RIGHT", threats)
    slash = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    walk = walk_or_swing(3, "RIGHT", "walk", snap, period=10, hold=3)
    assert slash.reason == "walk_slash"
    assert walk.reason == "walk"
