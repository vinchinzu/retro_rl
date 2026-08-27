"""Powered Main Shaft grate takeoff (west_super).

Take02 fires UP+X at ~(1223, 1860). Take04 fires UP+X at ~(1195, 1883)
p3, stationary. The bot grate-seat is the hatch-lip pocket ~(1177, 1883):
Wave blocks are LEFT, a wall is RIGHT, UP bonks the pocket ceiling.
Face LEFT and fire in place (no LEFT walk — dual leftover 1169 walked
off the seat), then LEFT+A. Morph later at ~(1189, 1785). Never
DOWN-morph on the lip. Alcove x≥1224 stays out.
"""

from __future__ import annotations

from super_metroid.ram import FACING_LEFT
from super_metroid.routes.controller_common import is_morph
from super_metroid.routes.skills.basic_moves import shoot_up_action
from super_metroid.routes.skills.charge_shot import CHARGE_FULL

# Take02–05: crouch/aim-up on the right lip / save-ledge. Alcove starts x=1224.
# 1164 covers leftover (1169, 1883) p38 after LEFT+X walked 1px past 1170.
LIP_SHOT_X = (1164, 1223)
LIP_SHOT_Y = (1852, 1896)
# Take02 fires at (1223, 1860). Grate-seat ~(1177, 1883) is west of that column.
LIP_FIRE_X = (1216, 1223)
# Take02 holds UP+X ~7f then releases. Pocket cannot UP (ceiling) or walk
# RIGHT (wall). Dual leftover charge 55 never hit CHARGE_FULL=60. Shoulder
# R (pose 6) slid off the lip to (1113, 1899) p156; crystals still up.
# Horizontal X tap in place, release at 8. Do not dual R again.
POCKET_RELEASE_CHARGE = 8
# Take02/03 morph ~(1189, 1785) p56; take04 ~(1214, 1801). Not the lip.
MORPH_DROP_X = (1176, 1216)
MORPH_DROP_Y = (1765, 1810)
MORPH_DROP_BOMB_FRAMES = 12
_GROUNDED = frozenset({1, 2, 3, 4, 9, 10})
_AIM = frozenset({5, 6, 7, 8})  # 6 = face LEFT + shoulder R (up-left)
_CROUCH = frozenset({39, 40})
_TURN = frozenset({37, 38})
_DROP_MORPH = frozenset({56, 57})


def at_ws_main_lip_shot_seat(
    samus_x: int, samus_y: int, pose: int, velocity_y: int = 0
) -> bool:
    """Grounded on the right lip / save-ledge shoot seat. Not the save alcove."""
    x, y = int(samus_x), int(samus_y)
    return (
        LIP_SHOT_X[0] <= x <= LIP_SHOT_X[1]
        and LIP_SHOT_Y[0] <= y <= LIP_SHOT_Y[1]
        and int(pose) in _GROUNDED | _AIM | _CROUCH | _TURN
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
    is a wall). Take02/04 fire stationary; LEFT+X walked off to x=1169.
    Face LEFT, then X in place. Release at POCKET_RELEASE_CHARGE like
    take02's ~7f tap — do not wait for CHARGE_FULL, do not hold LEFT
    (walk-off 1169), do not shoulder R (fell to stairs 1113). Take02
    column ~(1223): UP+X. After spawn: LEFT+A, never DOWN.
    """
    if int(pose) in _CROUCH:
        return ("UP",)
    x = int(samus_x)
    if not lip_hit:
        if x < LIP_FIRE_X[0]:
            if int(facing) != FACING_LEFT:
                return ("LEFT",)
            if int(charge) >= POCKET_RELEASE_CHARGE:
                return ()
            return ("X",)
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
    "POCKET_RELEASE_CHARGE",
    "at_ws_main_lip_shot_seat",
    "at_ws_main_morph_drop",
    "grate_lip_action",
    "grate_morph_action",
]
