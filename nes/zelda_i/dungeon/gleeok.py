"""Shared Gleeok sensors + south-stand helpers (L4 0x43 and L6 0x44).

Body type is dungeon-specific (L4 ``0x43``, L6 ``0x44``). Detached head
``0x46`` and fireball residual ``0x56`` are shared. Fight controllers and
TF suffixes stay on the owning level module.
"""

from __future__ import annotations

from retro_harness.nes import nes_action
from zelda_i.dungeon import ids as _ids
from zelda_i.ram import ZeldaSnapshot

GLEEOK_OBJECT_TYPE = _ids.GLEEOK_OBJECT_TYPE
GLEEOK_HEAD_OBJECT_TYPE = _ids.GLEEOK_HEAD_OBJECT_TYPE
GLEEOK_FIREBALL_TYPE = _ids.MANHANDLA_PROJECTILE_TYPE

# Clean-safe south stand (rr-vdnc): dy=22 UP+A dual-green from GleeokEnter.
STAND_DY = 22
FIREBALL_DODGE_DIST = 14


def gleeok_live(snap: ZeldaSnapshot) -> list:
    """Body slots type 0x43 (HP may be 0 mid/late fight — TYPE presence)."""
    return [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id == GLEEOK_OBJECT_TYPE
    ]


def gleeok_heads_live(snap: ZeldaSnapshot) -> list:
    """Detached head type 0x46 (may show HP=0 while still present)."""
    return [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id == GLEEOK_HEAD_OBJECT_TYPE
    ]


def gleeok_fireballs(snap: ZeldaSnapshot) -> list:
    """Fireball type 0x56 (contact hazard; not a clear target)."""
    return [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id == GLEEOK_FIREBALL_TYPE
    ]


def _fireball_dodge_dir(
    snap: ZeldaSnapshot,
    *,
    thr: int = FIREBALL_DODGE_DIST,
    allow_vertical: bool = False,
) -> str | None:
    """Flee nearest fireball if within ``thr`` (manhattan).

    Default is horizontal-only (south-stand mid-fight). Post-boss residual
    approaches from S/N — set ``allow_vertical=True`` so we don't walk into
    the ball while hunting HC (rr-gjey).
    """
    balls = gleeok_fireballs(snap)
    if not balls:
        return None
    nearest = min(
        balls,
        key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
    )
    dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
    if dist > thr:
        return None
    dx = nearest.x - snap.link_x
    dy = nearest.y - snap.link_y
    if allow_vertical and abs(dy) > abs(dx):
        # Ball mainly N/S of Link. Stepping further on that axis often walks
        # *into* a chasing fireball — prefer perpendicular (horizontal) first
        # (rr-gjey post-boss residual).
        if abs(dx) >= 2:
            if dx >= 0:
                return "LEFT" if snap.link_x > 56 else "RIGHT"
            return "RIGHT" if snap.link_x < 200 else "LEFT"
        # Aligned vertically: step toward room edge (away from center).
        if snap.link_x >= 120:
            return "RIGHT" if snap.link_x < 200 else "LEFT"
        return "LEFT" if snap.link_x > 56 else "RIGHT"
    if nearest.x >= snap.link_x:
        return "LEFT" if snap.link_x > 56 else "RIGHT"
    return "RIGHT" if snap.link_x < 200 else "LEFT"


fireball_dodge_dir = _fireball_dodge_dir


def _south_stand_action(snap: ZeldaSnapshot, body, *, stand_dy: int = STAND_DY):
    """Walk to (body.x, body.y+stand_dy) then face UP + A."""
    sx = int(body.x)
    sy = min(173, int(body.y) + stand_dy)
    if abs(snap.link_x - sx) > 3 or abs(snap.link_y - sy) > 3:
        if abs(snap.link_y - sy) >= abs(snap.link_x - sx):
            face = "DOWN" if snap.link_y < sy else "UP"
        else:
            face = "RIGHT" if snap.link_x < sx else "LEFT"
        return nes_action(face)
    return nes_action("UP", "A")


south_stand_action = _south_stand_action


__all__ = [
    "FIREBALL_DODGE_DIST",
    "GLEEOK_FIREBALL_TYPE",
    "GLEEOK_HEAD_OBJECT_TYPE",
    "GLEEOK_OBJECT_TYPE",
    "STAND_DY",
    "_fireball_dodge_dir",
    "_south_stand_action",
    "fireball_dodge_dir",
    "gleeok_fireballs",
    "gleeok_heads_live",
    "gleeok_live",
    "south_stand_action",
]
