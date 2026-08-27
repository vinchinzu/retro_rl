"""Powered Main Shaft grate takeoff (west_super).

Take02 fires UP+X at ~(1223, 1860). The bot grate-seat is the hatch-lip
pocket ~(1177, 1883): Wave blocks are LEFT, a wall is RIGHT, UP bonks
the pocket ceiling. Face LEFT and fire, then LEFT+A. Morph later at
~(1189, 1785). Never DOWN-morph on the lip. Alcove x≥1224 stays out.
"""

from __future__ import annotations

from super_metroid.ram import FACING_LEFT
from super_metroid.routes.controller_common import is_morph
from super_metroid.routes.skills.basic_moves import shoot_up_action
from super_metroid.routes.skills.charge_shot import CHARGE_FULL

# Take02–05: crouch/aim-up on the right lip / save-ledge. Alcove starts x=1224.
LIP_SHOT_X = (1170, 1223)
LIP_SHOT_Y = (1852, 1896)
# Take02 fires at (1223, 1860). Grate-seat ~(1177, 1883) is west of that column.
LIP_FIRE_X = (1216, 1223)
# Take02/03 morph ~(1189, 1785) p56; take04 ~(1214, 1801). Not the lip.
MORPH_DROP_X = (1176, 1216)
MORPH_DROP_Y = (1765, 1810)
MORPH_DROP_BOMB_FRAMES = 12
_GROUNDED = frozenset({1, 2, 3, 4, 9, 10})
_CROUCH = frozenset({39, 40})
_DROP_MORPH = frozenset({56, 57})


def at_ws_main_lip_shot_seat(
    samus_x: int, samus_y: int, pose: int, velocity_y: int = 0
) -> bool:
    """Grounded on the right lip / save-ledge shoot seat. Not the save alcove."""
    x, y = int(samus_x), int(samus_y)
    return (
        LIP_SHOT_X[0] <= x <= LIP_SHOT_X[1]
        and LIP_SHOT_Y[0] <= y <= LIP_SHOT_Y[1]
        and int(pose) in _GROUNDED | _CROUCH
        and abs(int(velocity_y)) <= 1
        and not is_morph(int(pose))
    )


def at_ws_main_morph_drop(
    samus_x: int, samus_y: int, pose: int = 0, velocity_y: int = 0
) -> bool:
    """Morph-drop hole after the Wave-block spawn. Not the lip, not the alcove."""
    del pose, velocity_y
    x, y = int(samus_x), int(samus_y)
    return MORPH_DROP_X[0] <= x <= MORPH_DROP_X[1] and MORPH_DROP_Y[0] <= y <= MORPH_DROP_Y[1]


def grate_lip_action(
    pose: int,
    lip_hit: bool,
    facing: int = FACING_LEFT,
    samus_x: int = 1223,
    charge: int = 0,
) -> tuple[str, ...]:
    """Shoot the Wave blocks until a 0xD080-family PLM spawns, then jump LEFT.

    Hatch-lip pocket x<1216: blocks are LEFT (UP hits the ceiling; RIGHT
    is a wall). Take02 column ~(1223): UP+X. Release at CHARGE_FULL so
    the beam actually fires. After spawn: LEFT+A, never DOWN.
    """
    if int(pose) in _CROUCH:
        return ("UP",)
    x = int(samus_x)
    if not lip_hit:
        if x < LIP_FIRE_X[0]:
            if int(facing) != FACING_LEFT:
                return ("LEFT",)
            if int(charge) >= CHARGE_FULL:
                return ("LEFT",)
            return ("LEFT", "X")
        if x > LIP_FIRE_X[1]:
            return ("LEFT",)
        if int(charge) >= CHARGE_FULL:
            return ("UP",)
        return shoot_up_action()
    if is_morph(int(pose)):
        return ("LEFT",)
    if int(facing) != FACING_LEFT:
        return ("LEFT",)
    return ("LEFT", "A")


def grate_morph_action(pose: int, lip_hit: bool) -> tuple[str, ...] | None:
    """DOWN-morph after a 0xD080-family spawn. None before spawn or in air."""
    if not lip_hit:
        return None
    p = int(pose)
    if p in _CROUCH:
        return ("DOWN",)
    if is_morph(p) or p in _DROP_MORPH:
        return ("X",) if is_morph(p) else ("DOWN",)
    if p in _GROUNDED:
        return ("DOWN",)
    return None


__all__ = [
    "LIP_FIRE_X",
    "LIP_SHOT_X",
    "LIP_SHOT_Y",
    "MORPH_DROP_BOMB_FRAMES",
    "MORPH_DROP_X",
    "MORPH_DROP_Y",
    "at_ws_main_lip_shot_seat",
    "at_ws_main_morph_drop",
    "grate_lip_action",
    "grate_morph_action",
]
