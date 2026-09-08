from __future__ import annotations

import numpy as np

from zelda_i.combat import in_sword_hitbox, overworld_threat_objects, should_swing_at
from zelda_i.overworld.common import (
    swing_action,
    walk_or_swing,
)
from zelda_i.overworld.graph import ScreenHop, path_screens_from_hops
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
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
        ram[ADDR_OBJ_HP + slot] = 0x20
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


def test_walk_or_swing_faces_contact_side_enemy() -> None:
    """Contact-range north of Link: face UP this frame. Far side hitbox must not."""
    snap = _snap(x=120, y=140, obj=(1, 0x07, 120, 132))
    assert in_sword_hitbox(120, 140, "UP", 120, 132)
    slash = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    assert slash.reason == "walk_slash"
    assert slash.action == swing_action(0, "UP", "walk", period=10, hold=3).action


def test_walk_or_swing_far_side_hitbox_keeps_travel() -> None:
    """16px north is in the UP blade but not contact — keep the hop RIGHT."""
    snap = _snap(x=120, y=140, obj=(1, 0x07, 120, 124))
    assert in_sword_hitbox(120, 140, "UP", 120, 124)
    act = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    assert act.reason == "walk"
    assert act.action == swing_action(3, "RIGHT", "walk", period=10, hold=3).action


def test_walk_or_swing_front_slash_not_vetoed_by_nearer_side_enemy() -> None:
    """Nearest off-axis enemy must not cancel a travel-direction hitbox slash."""
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_SCREEN] = 0x37
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 140
    ram[ADDR_OBJ_TYPE + 1] = 0x07
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 100
    ram[ADDR_OBJ_HP + 1] = 0x20
    ram[ADDR_OBJ_TYPE + 2] = 0x07
    ram[ADDR_LINK_X + 2] = 135
    ram[ADDR_LINK_Y + 2] = 140
    ram[ADDR_OBJ_HP + 2] = 0x20
    snap = read_snapshot(ram)
    slash = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    assert slash.reason == "walk_slash"
    assert slash.action == swing_action(0, "RIGHT", "walk", period=10, hold=3).action


def test_walk_or_swing_does_not_slash_far_behind_enemy() -> None:
    snap = _snap(x=120, y=140, obj=(1, 0x07, 80, 140))
    assert not in_sword_hitbox(120, 140, "RIGHT", 80, 140)
    assert not in_sword_hitbox(120, 140, "LEFT", 80, 140)
    act = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    assert act.reason == "walk"
    assert act.action == swing_action(3, "RIGHT", "walk", period=10, hold=3).action
