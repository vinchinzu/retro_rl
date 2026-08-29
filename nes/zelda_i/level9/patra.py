"""Final Patra combat policy for Level 9 room ``0x52``.

Live fceumm observations from the disclosed full-loadout recon fixture:

- body type ``0x47`` starts in slot 1 with HP ``0xB0``;
- eight orbiting eyes use type ``0x25`` and HP ``0x60``;
- Magical Sword hits move an eye ``0x60 -> 0x20 -> dead``;
- after the eyes are gone, body hits move ``0xB0 -> 0x70 -> 0x30 -> dead``;
- the game raises north-door bit ``0x08`` after the body disappears.

This controller performs no RAM writes.  The start checkpoint remains an
explicit fixture because its full inventory and room-loader setup are composed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level9.ganon import LEVEL9, ROOM_BEFORE_GANON
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

OBJ_PATRA = 0x47
OBJ_PATRA_EYE = 0x25
PATRA_BODY_HP_START = 0xB0
PATRA_EYE_HP_START = 0x60
PATRA_EYE_COUNT = 8
NORTH_DOOR = 0x08

PATRA_STAND_DY = 30
PATRA_ATTACK_COOLDOWN = 12
PATRA_MAX_FRAMES = 6000


def in_final_patra_room(snap: ZeldaSnapshot) -> bool:
    return (
        snap.mode == PLAY_MODE
        and snap.level == LEVEL9
        and snap.screen == ROOM_BEFORE_GANON
    )


def patra_body(snap: ZeldaSnapshot) -> ZeldaObject | None:
    return next(
        (
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 12 and obj.type_id == OBJ_PATRA
        ),
        None,
    )


def patra_eyes(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id == OBJ_PATRA_EYE
    )


def final_patra_live(snap: ZeldaSnapshot) -> bool:
    return in_final_patra_room(snap) and patra_body(snap) is not None


def final_patra_north_door_earned(snap: ZeldaSnapshot) -> bool:
    return (
        in_final_patra_room(snap)
        and patra_body(snap) is None
        and not patra_eyes(snap)
        and bool(snap.cur_opened_doors & NORTH_DOOR)
    )


def patra_action(
    snap: ZeldaSnapshot,
    *,
    cooldown: int,
    stand_dy: int = PATRA_STAND_DY,
) -> tuple[list[int], str, int]:
    """Choose one frame: follow the body, stand south, and pulse UP+A.

    The orbiting eyes repeatedly cross this sword line.  Staying relative to
    the body avoids chasing individual eyes through room geometry and keeps the
    same policy valid once the vulnerable body is exposed.
    """
    body = patra_body(snap)
    if body is None:
        return nes_idle_action(), "wait_north_door", max(0, cooldown - 1)

    target_x = max(48, min(208, int(body.x)))
    target_y = max(93, min(173, int(body.y) + int(stand_dy)))
    dx = target_x - int(snap.link_x)
    dy = target_y - int(snap.link_y)

    if abs(dx) > 4 and abs(dx) >= abs(dy):
        direction = "RIGHT" if dx > 0 else "LEFT"
        return nes_action(direction), "align_south_x", cooldown
    if abs(dy) > 4:
        direction = "DOWN" if dy > 0 else "UP"
        return nes_action(direction), "align_south_y", cooldown
    if cooldown > 0:
        return nes_idle_action(), "attack_cooldown", cooldown - 1
    return nes_action("UP", "A"), "sword_pulse_up", PATRA_ATTACK_COOLDOWN


@dataclass
class FinalPatraFightController:
    """Controller-input-only final Patra clear and north-door earn."""

    max_frames: int = PATRA_MAX_FRAMES
    stand_dy: int = PATRA_STAND_DY
    frames: int = 0
    sword_pulses: int = 0
    max_eyes_seen: int = 0
    eye_count_changes: list[dict[str, int]] = field(default_factory=list)
    body_hp_changes: list[int] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)

    def run(
        self,
        env: Any,
        *,
        assist: Any | None = None,
        total: list[int] | None = None,
    ) -> dict[str, Any]:
        total_frames = total if total is not None else [0]
        start = read_snapshot(env.get_ram())
        if not final_patra_live(start):
            return self.report(ok=False, snap=start, error="final Patra not live")

        cooldown = 0
        last_eye_count: int | None = None
        last_body_hp: int | None = None

        for _ in range(self.max_frames):
            snap = read_snapshot(env.get_ram())
            eyes = patra_eyes(snap)
            body = patra_body(snap)
            eye_count = len(eyes)
            self.max_eyes_seen = max(self.max_eyes_seen, eye_count)
            if eye_count != last_eye_count:
                self.eye_count_changes.append({"frame": self.frames, "eyes": eye_count})
                last_eye_count = eye_count
            if body is not None and body.hp != last_body_hp:
                self.body_hp_changes.append(int(body.hp))
                last_body_hp = body.hp

            if final_patra_north_door_earned(snap):
                return self.report(ok=True, snap=snap)
            if snap.mode == 17:
                return self.report(ok=False, snap=snap, error="link death")

            action, reason, cooldown = patra_action(
                snap,
                cooldown=cooldown,
                stand_dy=self.stand_dy,
            )
            if reason == "sword_pulse_up":
                self.sword_pulses += 1
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
            env.step(action)
            self.frames += 1
            total_frames[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total_frames[0])

        return self.report(
            ok=False,
            snap=read_snapshot(env.get_ram()),
            error="north door timeout",
        )

    def report(
        self,
        *,
        ok: bool,
        snap: ZeldaSnapshot,
        error: str | None = None,
    ) -> dict[str, Any]:
        body = patra_body(snap)
        result: dict[str, Any] = {
            "ok": ok,
            "frames": self.frames,
            "policy": "south_stand",
            "stand_dy": self.stand_dy,
            "sword_pulses": self.sword_pulses,
            "max_eyes_seen": self.max_eyes_seen,
            "eye_count_changes": list(self.eye_count_changes),
            "body_hp_changes": list(self.body_hp_changes),
            "reasons": dict(self.reasons),
            "north_door_earned": bool(snap.cur_opened_doors & NORTH_DOOR),
            "open_doorway_mask": int(snap.open_doorway_mask),
            "room_all_dead": int(snap.room_all_dead),
            "room_obj_count": int(snap.room_obj_count),
            "remaining_eyes": len(patra_eyes(snap)),
            "body": (
                {"slot": body.slot, "hp": body.hp, "state": body.state}
                if body is not None
                else None
            ),
            "controller_memory_writes": 0,
        }
        if error is not None:
            result["error"] = error
        return result


__all__ = [
    "FinalPatraFightController",
    "NORTH_DOOR",
    "OBJ_PATRA",
    "OBJ_PATRA_EYE",
    "PATRA_ATTACK_COOLDOWN",
    "PATRA_BODY_HP_START",
    "PATRA_EYE_COUNT",
    "PATRA_EYE_HP_START",
    "PATRA_MAX_FRAMES",
    "PATRA_STAND_DY",
    "final_patra_live",
    "final_patra_north_door_earned",
    "in_final_patra_room",
    "patra_action",
    "patra_body",
    "patra_eyes",
]
