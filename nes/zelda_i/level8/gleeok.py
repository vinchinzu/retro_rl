"""L8 four-head Gleeok 0x3C south-stand fight (live type 0x45).

Idle until the body is present (arrival census is empty). South-stand
(body.x, body.y+22) face UP then UP+A. Fireball dodge manhattan ≤14.
Stop when type 0x45 is absent; then walk the SW diamond for heart 0x1A.
Do not chase heads while the body remains. OccupancyWalker banned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.gleeok import (
    FIREBALL_DODGE_DIST,
    STAND_DY,
    _fireball_dodge_dir,
    _south_stand_action,
    gleeok_fireballs,
    gleeok_heads_live,
)
from zelda_i.dungeon.ids import GLEEOK_HEAD_OBJECT_TYPE
from zelda_i.level8.dungeon import GLEEOK_FOUR_HEAD_OBJECT_TYPE, LEVEL8
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "GLEEOK_4HEAD_MAX_FRAMES",
    "GLEEOK_ROOM",
    "HEART_REACH",
    "HEART_XY",
    "RAM_CLAIM",
    "Level8FourHeadGleeokController",
    "gleeok_4head_live",
    "make_four_head_gleeok_controller",
]

GLEEOK_ROOM = 0x3C
GLEEOK_4HEAD_MAX_FRAMES = 20000
HEART_ITEM = 0x1A
# Room treasure slot 19 ($83/$97) live F5dump: (32,192) after body-gone
# (state 0). F4 stand (32,180) was 12px north of the item (no pickup).
HEART_XY = (32, 192)
HEART_REACH = 1
CLIP_Y = 173
RAM_CLAIM = (
    "From play 0x3C leftover (120,189), idle until body type 0x45 is "
    "present, south-stand (body.x, body.y+22) face UP+A, Magical Sword, "
    "fireball dodge manhattan ≤14. Body type 0x45 goes absent. Do not "
    "chase heads while body remains. Hearts Survival-refilled. Keys/bombs/"
    "MK/TF unchanged except the natural heart-container +1 when 0x1A is "
    "collected. Deaths 0. progression_writes=0 capacity_writes=0. No HP poke."
)


def gleeok_4head_live(snap: ZeldaSnapshot) -> list:
    """Body slots type 0x45 (HP may be 0 mid-fight — TYPE presence)."""
    want = GLEEOK_FOUR_HEAD_OBJECT_TYPE
    if want is None:
        return []
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and int(obj.type_id) == int(want)
    ]


@dataclass
class Level8FourHeadGleeokController:
    """South-stand 0x45 until absent, then SW-diamond heart. Fixture-live."""

    spec_id: str = "level8_four_head_gleeok"
    room: int = GLEEOK_ROOM
    max_frames: int = GLEEOK_4HEAD_MAX_FRAMES
    stand_dy: int = STAND_DY
    fireball_dodge_dist: int = FIREBALL_DODGE_DIST
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    writes: int = 0
    saw_0x45: bool = False
    saw_0x46: bool = False
    body_gone: bool = False
    hc_in: int | None = None
    hc_out: int | None = None
    _armed: bool = False
    route_eligible: bool = False

    @property
    def observed_body_type(self) -> int | None:
        return GLEEOK_FOUR_HEAD_OBJECT_TYPE

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        bodies = gleeok_4head_live(snap)
        if bodies:
            self.saw_0x45 = True
        if gleeok_heads_live(snap):
            self.saw_0x46 = True
        if force or self.frames <= 2 or self.frames % 250 == 0:
            body = bodies[0] if bodies else None
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "reason": action.reason,
                    "bx": None if body is None else int(body.x),
                    "by": None if body is None else int(body.y),
                    "bhp": None if body is None else int(body.hp),
                    "heads": len(gleeok_heads_live(snap)),
                    "n45": len(bodies),
                    "n56": len(gleeok_fireballs(snap)),
                    "hc": int(snap.heart_containers),
                    "item": int(snap.room_item_id),
                }
            )
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            return self._fail(
                snap,
                f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if GLEEOK_FOUR_HEAD_OBJECT_TYPE is None:
            return self._fail(snap, "l8_gleeok_object_type_unobserved")
        if snap.mode == 17:
            return self._fail(snap, "link_death")
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7, 9, 10, 16):
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL8:
            return self._fail(snap, f"left_level_{snap.level}")
        if snap.screen != self.room:
            return self._fail(snap, f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")
        if self.hc_in is None:
            self.hc_in = int(snap.heart_containers)

        bodies = gleeok_4head_live(snap)
        if bodies:
            self.saw_0x45 = True
            self.body_gone = False
        if not bodies:
            if not self.saw_0x45:
                return self._emit(snap, FrameAction(nes_idle_action(), "wait_body"))
            self.body_gone = True
            self.hc_out = int(snap.heart_containers)
            got_heart = self.hc_out > int(self.hc_in)
            item_gone = int(snap.room_item_id) != HEART_ITEM
            if got_heart or item_gone:
                self.success = True
                self.notes.append(
                    f"body_gone_{snap.link_x}_{snap.link_y}"
                    f"_hc={self.hc_in}->{self.hc_out}_0x46={int(self.saw_0x46)}"
                )
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "body_gone_heart"), force=True
                )
            tx, ty = HEART_XY
            if abs(int(snap.link_x) - tx) > HEART_REACH:
                btn = "RIGHT" if snap.link_x < tx else "LEFT"
                return self._emit(snap, FrameAction(nes_action(btn), "heart_x"))
            if abs(int(snap.link_y) - ty) > HEART_REACH:
                btn = "DOWN" if snap.link_y < ty else "UP"
                return self._emit(snap, FrameAction(nes_action(btn), "heart_y"))
            return self._emit(snap, FrameAction(nes_idle_action(), "heart_stand"))

        dodge = _fireball_dodge_dir(snap, thr=self.fireball_dodge_dist)
        if dodge is not None:
            self._armed = False
            return self._emit(snap, FrameAction(nes_action(dodge), "fb_dodge"))

        # South mouth y=189: walk inland. LEFT+UP only if still on the
        # door lip (L6 clip analog); OccupancyWalker banned.
        if snap.link_y > CLIP_Y:
            self._armed = False
            return self._emit(
                snap, FrameAction(nes_action("LEFT", "UP"), "south_inland")
            )

        act = _south_stand_action(snap, bodies[0], stand_dy=self.stand_dy)
        swinging = list(act) == list(nes_action("UP", "A"))
        if swinging and not self._armed:
            self._armed = True
            return self._emit(snap, FrameAction(nes_action("UP"), "south_face"))
        if not swinging:
            self._armed = False
            return self._emit(snap, FrameAction(act, "south_walk"))
        return self._emit(snap, FrameAction(act, "south_stand"))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "spec_id": self.spec_id,
            "observed_body_type": self.observed_body_type,
            "assumed_0x45": False,
            "writes": int(self.writes),
            "route_eligible": False,
            "natural_entry": False,
            "evidence": "fixture-live",
            "saw_0x45": self.saw_0x45,
            "saw_0x46": self.saw_0x46,
            "body_gone": self.body_gone,
            "hc_in": self.hc_in,
            "hc_out": self.hc_out,
            "stand_dy": self.stand_dy,
            "room": self.room,
            "body_type": GLEEOK_FOUR_HEAD_OBJECT_TYPE,
            "head_type": GLEEOK_HEAD_OBJECT_TYPE,
            "policy": RAM_CLAIM,
        }


def make_four_head_gleeok_controller(
    *, topology: Any = None
) -> Level8FourHeadGleeokController:
    del topology
    return Level8FourHeadGleeokController()
