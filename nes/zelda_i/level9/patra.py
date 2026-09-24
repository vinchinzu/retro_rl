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

from zelda_i.dungeon.hop_controller import room_step

from dataclasses import dataclass, field
from typing import Any

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import direction_to_facing
from zelda_i.level9.ganon import LEVEL9, ROOM_BEFORE_GANON, hazard_dodge_dir
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

OBJ_PATRA = 0x47
OBJ_PATRA_EYE = 0x25
PATRA_BODY_HP_START = 0xB0
PATRA_EYE_HP_START = 0x60
PATRA_EYE_COUNT = 8
NORTH_DOOR = 0x08

# Stand this far south of the body; the stand clamps to the bottom row
# (y=173), outside the eyes' orbit, and the full-health beam still reaches.
# 30 px stood inside the orbit: over 12 RNG offsets from Blue Ring power-on
# 9 pins, 0x52 went 24.0h -> 6.8h (2351f -> 1761f) and 0x61 23h -> 2h.
PATRA_STAND_DY = 84
# 8, not 12: over 12 offsets x 3 pins, 0x52 1690f/13.8h (the clamped south
# stand, cooldown 12) -> 1650f/1.7h; 0x61 2492f/7.0h -> 2135f/3.0h
# (x<=192) or 2483f/1.6h (x<=208). Cooldown 6 was worse.
PATRA_ATTACK_COOLDOWN = 8
# The body roams, and late in a fight it sits in the bottom rows, where the
# clamped south stand put Link inside it: 22 of 22 hits on the power-on 14
# 0x61 pin. Stand on whichever side of the body still leaves this much room
# (the side Link already faces first), and fire once on its lane: the beam
# flies the room, so the exact stand pixel does not matter.
PATRA_MIN_CLEAR = 56
PATRA_LANE_HALF = 8
# x<=192: 0x52's east column holds the stairs down to cellar 0x77 (208,144),
# and a stand there walked Link out of the fight (6 of 36 runs). 0x61 has
# no stairs there and passes the full box.
PATRA_ROOM = (32, 192, 93, 173)
PATRA_ROOM_FULL = (32, 208, 93, 173)
_PATRA_SIDES = (("UP", 0, 1), ("DOWN", 0, -1), ("LEFT", 1, 0), ("RIGHT", -1, 0))
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


def _patra_stands(snap: ZeldaSnapshot, body: ZeldaObject, dist: int, room):
    """``(cost, (x, y), facing)`` per side the room leaves ``PATRA_MIN_CLEAR``."""
    xlo, xhi, ylo, yhi = room
    bx, by = int(body.x), int(body.y)
    lx, ly = int(snap.link_x), int(snap.link_y)
    held = int(snap.facing)
    out = []
    for facing, sx, sy in _PATRA_SIDES:
        gx = max(xlo, min(xhi, bx + sx * dist))
        gy = max(ylo, min(yhi, by + sy * dist))
        if max(abs(gx - bx), abs(gy - by)) < PATRA_MIN_CLEAR:
            continue
        cost = abs(gx - lx) + abs(gy - ly) - (64 if direction_to_facing(facing) == held else 0)
        out.append((cost, (gx, gy), facing))
    return sorted(out)


def _on_lane(snap: ZeldaSnapshot, body: ZeldaObject, facing: str) -> bool:
    """On the body's row/column, on ``facing``'s side and out of the orbit."""
    dx = int(body.x) - int(snap.link_x)
    dy = int(body.y) - int(snap.link_y)
    along, perp = {"UP": (-dy, dx), "DOWN": (dy, dx), "LEFT": (-dx, dy), "RIGHT": (dx, dy)}[facing]
    return along >= PATRA_MIN_CLEAR and abs(perp) <= PATRA_LANE_HALF


def patra_action(
    snap: ZeldaSnapshot,
    *,
    cooldown: int,
    stand_dy: int = PATRA_STAND_DY,
    room: tuple[int, int, int, int] = PATRA_ROOM,
) -> tuple[list[int], str, int]:
    """One frame: on a lane of the body out of the orbit, face it and pulse A.

    The orbiting eyes cross the lane; the full-health beam carries the A
    press the length of the room. Cooldown dodges nearby eyes or stands.
    """
    body = patra_body(snap)
    next_cd = max(0, cooldown - 1)
    if body is None:
        return nes_idle_action(), "wait_north_door", next_cd

    stands = _patra_stands(snap, body, int(stand_dy), room)
    if not stands:
        # Nowhere in the room clears the orbit: step away from the body.
        return nes_action(hazard_dodge_dir(snap, (body,), thr=255) or "DOWN"), "patra_boxed", next_cd
    _, goal, facing = stands[0]
    if not _on_lane(snap, body, facing):
        # ROM lattice to the stand; the greedy axis step pushed into the 0x61
        # blocks at (144,165) for 1041 frames.
        direction = room_step(snap, goal, tol=4)
        if direction is not None:
            return nes_action(direction), f"align_{facing.lower()}", next_cd

    dodge = hazard_dodge_dir(snap, patra_eyes(snap))
    if cooldown > 0:
        if dodge is not None:
            return nes_action(dodge), "attack_dodge", next_cd
        return nes_idle_action(), "cooldown_stand", next_cd
    if int(snap.facing) != direction_to_facing(facing):
        return nes_action(facing), f"face_{facing.lower()}", 0
    return nes_action("A"), f"sword_pulse_{facing.lower()}", PATRA_ATTACK_COOLDOWN


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
            if reason.startswith("sword_pulse"):
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
            "policy": "south_stand_dodge",
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
    "PATRA_ROOM",
    "PATRA_ROOM_FULL",
    "PATRA_STAND_DY",
    "final_patra_live",
    "final_patra_north_door_earned",
    "in_final_patra_room",
    "patra_action",
    "patra_body",
    "patra_eyes",
]
