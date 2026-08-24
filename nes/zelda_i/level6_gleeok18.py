"""Level 6 Gleeok 0x18 south-stand fight (type 0x44).

Live settle (`l6_settle18_continuous_v1`): body is **0x44**, not L4 0x43.
Fireball residual 0x56. Head 0x46 not seen during idle. Reuse L4 south-stand
policy; do not copy the L4 TF suffix. Do not require Map. Diamond south mouth
uses the live 0x28 LEFT+UP clip (cardinal UP at y=181 is solid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon_ids import (
    GLEEOK_3HEAD_OBJECT_TYPE,
    GLEEOK_HEAD_OBJECT_TYPE,
)
from zelda_i.level4_boss_combat import (
    FIREBALL_DODGE_DIST,
    STAND_DY,
    _fireball_dodge_dir,
    _south_stand_action,
    gleeok_heads_live,
)
from zelda_i.level6_overworld import LEVEL6, LEVEL6_GLEEOK_ROOM
from zelda_i.level6_path import CLIP_CLEAR_Y
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "GLEEOK_18_MAX_FRAMES",
    "Level6Gleeok18Controller",
    "gleeok_3head_live",
    "make_gleeok_18_controller",
]

GLEEOK_18_MAX_FRAMES = 20000


def gleeok_3head_live(snap: ZeldaSnapshot) -> list:
    """Body slots type 0x44 (HP may be 0 mid-fight — TYPE presence)."""
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and int(obj.type_id) == GLEEOK_3HEAD_OBJECT_TYPE
    ]


@dataclass
class Level6Gleeok18Controller:
    """Diamond clip inland, then L4 south-stand on 0x44. Stop when body is gone."""

    spec_id: str = "level6_gleeok_0x18"
    room: int = LEVEL6_GLEEOK_ROOM
    max_frames: int = GLEEOK_18_MAX_FRAMES
    stand_dy: int = STAND_DY
    fireball_dodge_dist: int = FIREBALL_DODGE_DIST
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    saw_0x44: bool = False
    saw_0x46: bool = False

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        bodies = gleeok_3head_live(snap)
        if bodies:
            self.saw_0x44 = True
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
                    "n44": len(bodies),
                }
            )
        return action

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7, 9, 10):
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            self.failed = True
            self.notes.append(f"left_level_{snap.level}")
            return FrameAction(nes_idle_action(), "left_level")
        if snap.screen != self.room:
            self.failed = True
            self.notes.append(f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")
            return FrameAction(nes_idle_action(), f"left_0x{self.room:02x}")

        bodies = gleeok_3head_live(snap)
        if bodies:
            self.saw_0x44 = True
        if not bodies:
            if not self.saw_0x44:
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_body")
                )
            self.success = True
            self.notes.append(
                f"body_gone_{snap.link_x}_{snap.link_y}_0x46={int(self.saw_0x46)}"
            )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "body_gone"), force=True
            )

        if snap.link_y > CLIP_CLEAR_Y:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(
                    f"clip_f{self.frames}_{snap.link_x}_{snap.link_y}"
                )
            return self._emit(
                snap, FrameAction(nes_action("LEFT", "UP"), "diamond_clip")
            )

        dodge = _fireball_dodge_dir(snap, thr=self.fireball_dodge_dist)
        if dodge is not None:
            return self._emit(
                snap, FrameAction(nes_action(dodge), "fb_dodge")
            )

        act = _south_stand_action(snap, bodies[0], stand_dy=self.stand_dy)
        reason = (
            "south_stand"
            if list(act) == list(nes_action("UP", "A"))
            else "south_walk"
        )
        return self._emit(snap, FrameAction(act, reason))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": "LEFT+UP y>173 then L4 south-stand on 0x44",
            "saw_0x44": self.saw_0x44,
            "saw_0x46": self.saw_0x46,
            "stand_dy": self.stand_dy,
            "spec_id": self.spec_id,
            "room": self.room,
            "body_type": GLEEOK_3HEAD_OBJECT_TYPE,
            "head_type": GLEEOK_HEAD_OBJECT_TYPE,
        }


def make_gleeok_18_controller() -> Level6Gleeok18Controller:
    """South-stand 0x44 until the body is gone. Map / Rod residual."""
    return Level6Gleeok18Controller()
