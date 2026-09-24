"""L8 four-head Gleeok 0x3C south-stand fight (live type 0x45).

Idle until the body is present (arrival census is empty). Stand on the turn
node under the body at dy 30 (below the hanging heads), face UP once, then
pulse A. No fireball dodge. Stop when type 0x45 is absent; then walk to the
heart container 0x1A. Do not chase heads while the body remains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.gleeok import (
    GLEEOK_STAND_DY,
    GleeokStand,
    gleeok_fireballs,
    gleeok_heads_live,
)
from zelda_i.dungeon.ids import GLEEOK_HEAD_OBJECT_TYPE
from zelda_i.level8.dungeon import GLEEOK_FOUR_HEAD_OBJECT_TYPE, LEVEL8
from zelda_i.dungeon.hop_controller import room_step
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, room_item_xy

__all__ = [
    "GLEEOK_4HEAD_MAX_FRAMES",
    "GLEEOK_ROOM",
    "HEART_REACH",
    "HEART_SLOT",
    "HEART_XY",
    "RAM_CLAIM",
    "Level8FourHeadGleeokController",
    "gleeok_4head_live",
    "heart_xy",
    "make_four_head_gleeok_controller",
]

GLEEOK_ROOM = 0x3C
GLEEOK_4HEAD_MAX_FRAMES = 20000
HEART_ITEM = 0x1A
# Room treasure slot 19 ($83/$97). Measured leftover after body-gone is
# (32,192); dest is live RAM x/y + heart-container bit, idle if missing.
HEART_SLOT = 19
HEART_XY = (32, 192)
HEART_REACH = 1


RAM_CLAIM = (
    "From play 0x3C leftover (120,189), idle until body type 0x45 is "
    "present, stand on the turn node at (body.x, body.y+30), face UP, pulse "
    "A; no fireball dodge. Body type 0x45 goes absent. Do not chase heads "
    "while body remains. Hearts Survival-refilled. Keys/bombs/MK/TF "
    "unchanged except the natural heart-container +1 when 0x1A is "
    "collected. Deaths 0. progression_writes=0 capacity_writes=0. No HP poke."
)


def heart_xy(ram: Any | None) -> tuple[int, int] | None:
    """Live treasure slot 19 coordinate from RAM, or None if unbound/empty."""
    if ram is not None:
        x, y = room_item_xy(ram)
        if x or y:
            return (x, y)
    return None


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
    stand_dy: int = GLEEOK_STAND_DY
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    writes: int = 0
    saw_0x45: bool = False
    saw_0x46: bool = False
    saw_heart_item: bool = False
    body_gone: bool = False
    hc_in: int | None = None
    hc_out: int | None = None
    route_eligible: bool = False
    _stand: GleeokStand | None = field(default=None, init=False, repr=False)
    _env: Any = field(default=None, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    @property
    def observed_body_type(self) -> int | None:
        return GLEEOK_FOUR_HEAD_OBJECT_TYPE

    def _heart_dest(self) -> tuple[int, int] | None:
        ram = None if self._env is None else self._env.get_ram()
        return heart_xy(ram)

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
        if int(snap.room_item_id) == HEART_ITEM:
            self.saw_heart_item = True

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
            # F6/F7: room_item_id stays 0x1A after the pickup, so the id is
            # only pickup evidence on a falling edge we actually watched.  A
            # bare ``!= 0x1A`` greens a room that never held the container.
            item_gone = self.saw_heart_item and int(snap.room_item_id) != HEART_ITEM
            if got_heart or item_gone:
                self.success = True
                self.notes.append(
                    f"body_gone_{snap.link_x}_{snap.link_y}"
                    f"_hc={self.hc_in}->{self.hc_out}_0x46={int(self.saw_0x46)}"
                )
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "body_gone_heart"), force=True
                )
            hdest = self._heart_dest()
            if hdest is None:
                return self._emit(snap, FrameAction(nes_idle_action(), "heart_wait"))
            # ROM lattice to the container. The x-then-y presses ran LEFT
            # into the block at (56..72,165) from (80,165) for 18000f
            # (Blue Ring power-on 4 resume): the HC was never taken.
            step = room_step(snap, hdest, tol=HEART_REACH, env=self._env)
            if step is not None:
                return self._emit(snap, FrameAction(nes_action(step), "heart_walk"))
            return self._emit(snap, FrameAction(nes_idle_action(), "heart_stand"))

        if self._stand is None:
            self._stand = GleeokStand(stand_dy=self.stand_dy)
        action, reason = self._stand.step(snap, bodies[0], env=self._env)
        return self._emit(snap, FrameAction(action, reason))

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
            "saw_heart_item": self.saw_heart_item,
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
