"""Shared Gleeok sensors + south-stand helpers (L4 0x43 and L6 0x44).

Body type is dungeon-specific (L4 ``0x43``, L6 ``0x44``). Detached head
``0x46`` and fireball residual ``0x56`` are shared. Fight controllers and
TF suffixes stay on the owning level module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zelda_i.dungeon.hop_controller import room_step

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import FACING_NORTH
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


def _south_stand_action(
    snap: ZeldaSnapshot,
    body,
    *,
    stand_dy: int = STAND_DY,
    stand_dx: int = 0,
):
    """Walk to (body.x + stand_dx, body.y+stand_dy) then face UP + A."""
    sx = int(body.x) + stand_dx
    sy = min(173, int(body.y) + stand_dy)
    # ROM lattice to the stand: the greedy axis step flipped L/R and U/D
    # around off-grid stands (L8 0x3C: 874 reversals).
    step = room_step(snap, (sx, sy), tol=3)
    if step is not None:
        return nes_action(step)
    return nes_action("UP", "A")


south_stand_action = _south_stand_action

STAND_CLIP_Y = 173
# Under the hanging heads, not among them, with no fireball dodge. Over 12
# offsets x 3 Blue Ring power-on pins each: L8 0x3C ~3900f/21h -> 890f/4.2h,
# L6 0x18 1291f/7.0h -> 914f/4.1h. dy 22 (the heads' own rows) and any dodge
# (walk-backs off the node) were worse in both rooms; dy 38 was close.
GLEEOK_STAND_DY = 30


def gleeok_stand(body: Any, stand_dy: int) -> tuple[int, int]:
    """The turn node nearest ``(body.x, body.y + stand_dy)``."""
    x = int(round(int(body.x) / 8)) * 8
    y = min(STAND_CLIP_Y, int(body.y) + stand_dy)
    return x, y - ((y - 5) % 8)


@dataclass
class GleeokStand:
    """Stand on the turn node under the body, face UP once, pulse A.

    ``_south_stand_action`` stood at body.x (124, off the x%8 lattice): the
    UP turn slid Link to x=120 and the walk pulled him back, and a held UP+A
    swung once per arrival -- 23 swing frames in 4411 (L8 0x3C, Blue Ring
    power-on 14 pin). ``slack`` px above the node still count: the turn
    walks Link up a pixel or two.
    """

    stand_dy: int
    period: int = 8
    hold: int = 2
    slack: int = 4
    _clock: int = 0

    def step(self, snap: ZeldaSnapshot, body: Any, env: Any = None) -> tuple[list[int], str]:
        sx, sy = gleeok_stand(body, self.stand_dy)
        x, y = int(snap.link_x), int(snap.link_y)
        if x != sx or not sy - self.slack <= y <= sy:
            self._clock = 0
            step = room_step(snap, (sx, sy), tol=0, env=env)
            return nes_action(step or "UP"), "south_walk"
        if int(snap.facing) != FACING_NORTH:
            return nes_action("UP"), "south_face"
        self._clock += 1
        if self._clock % self.period <= self.hold:
            return nes_action("A"), "south_swing"
        return nes_idle_action(), "south_stand"


__all__ = [
    "FIREBALL_DODGE_DIST",
    "GLEEOK_STAND_DY",
    "GleeokStand",
    "GLEEOK_FIREBALL_TYPE",
    "GLEEOK_HEAD_OBJECT_TYPE",
    "GLEEOK_OBJECT_TYPE",
    "STAND_DY",
    "_fireball_dodge_dir",
    "_south_stand_action",
    "fireball_dodge_dir",
    "gleeok_fireballs",
    "gleeok_heads_live",
    "gleeok_stand",
    "gleeok_live",
    "south_stand_action",
]
