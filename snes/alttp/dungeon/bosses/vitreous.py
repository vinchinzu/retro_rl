"""Vitreous in room 0x90.

Fighter sword. A slash leaves an eye at 48. A charge of 64 or 72,
released with Link 20-24px south and within 8px in x, drops that eye
48 to 44. Body offsets 0-32 do not drop the body. Eyes are not poked.
The warp landing is closed door 0xF0 and does not walk, so the fight
stands once on the measured floor cell (120, 5060). $F3CC is not written.

Lightning is sprite 0xBF. It is dodged, not a second damage rule.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/stage_replay.py \\
        Scan_mire_boss alttp.dungeon.bosses.vitreous:fight
"""

from __future__ import annotations

from typing import Any

from alttp.dungeon.bosses import FightResult
from alttp.primitives import active_sprites
from alttp.ram import FOLLOWER, LINK_DIRECTION, LINK_MAX_HP, LINK_X, LINK_Y, assign_u8, wram_index
from alttp.startup import action_for, no_action, snapshot_env, step_frames

BODY = 0xBD
EYE = 0xBE
BOLT = 0xBF
HEART = 0xEA
ROOM = 0x90
CHARGE = 64
SPIN_LOCK = 32
FLEE_FRAMES = 20
BOLT_DODGE = 18
CAP = 40000

__all__ = ["fight"]


def _u8(env: Any, offset: int) -> int:
    return int(env.get_ram()[wram_index(offset)])


def _put_u16(env: Any, offset: int, value: int) -> None:
    if any(int(offset) + i == int(FOLLOWER) for i in range(2)):
        raise RuntimeError("refusing to write $F3CC")
    assign_u8(env, offset, int(value) & 0xFF)
    assign_u8(env, offset + 1, (int(value) >> 8) & 0xFF)


def _press(env: Any, buttons: tuple[str, ...]) -> None:
    step_frames(env, action_for(*buttons) if buttons else no_action(), 1)


def _rows(env: Any) -> list[dict[str, int]]:
    found = []
    for spr in active_sprites(env):
        found.append(
            {
                "type": int(spr.sprite_type),
                "hp": int(spr.hp),
                "ai": int(spr.ai_state),
                "x": int(spr.x),
                "y": int(spr.y),
            }
        )
    return found


def _eyes(rows: list[dict[str, int]]) -> list[dict[str, int]]:
    return [row for row in rows if row["type"] == EYE and row["hp"] > 0]


def _body(rows: list[dict[str, int]]) -> dict[str, int] | None:
    return next((row for row in rows if row["type"] == BODY), None)


def _heart(rows: list[dict[str, int]]) -> dict[str, int] | None:
    prize = [row for row in rows if row["type"] in (HEART, 0xEB)]
    if not prize:
        return None
    return prize[0]


class _Fight:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.frames = 0
        self.held = 0
        self.phase = "charge"
        self.left = 0
        self.flee = "RIGHT"
        self.bolt_left = 0
        self.bolt_dir = "RIGHT"
        self.swings = 0
        self.max_hp = 0
        self.blocker = ""
        self.follower = 0

    def run(self) -> FightResult:
        snap = snapshot_env(self.env)
        self.max_hp = _u8(self.env, LINK_MAX_HP)
        self.follower = _u8(self.env, FOLLOWER)
        if int(snap.link_y) >= 5070:
            _put_u16(self.env, LINK_X, 120)
            _put_u16(self.env, LINK_Y, 5060)
            assign_u8(self.env, LINK_DIRECTION, 0)
        while self.frames < CAP:
            if _u8(self.env, FOLLOWER) != self.follower:
                self.blocker = "$F3CC changed"
                break
            snap = snapshot_env(self.env)
            if int(snap.hp) <= 0:
                self.blocker = (
                    f"dead swings={self.swings} xy=({snap.link_x},{snap.link_y})"
                )
                break
            if _u8(self.env, LINK_MAX_HP) > self.max_hp:
                return self._done(True, "")
            rows = _rows(self.env)
            if int(snap.room_base_id) != ROOM and _u8(self.env, LINK_MAX_HP) <= self.max_hp:
                self.blocker = f"left room 0x{int(snap.room_base_id):02X}"
                break
            heart = _heart(rows)
            eyes = _eyes(rows)
            body = _body(rows)
            if self.frames % 400 == 0:
                print(
                    f"vitreous f={self.frames} swings={self.swings} hp={int(snap.hp)} "
                    f"xy=({snap.link_x},{snap.link_y}) phase={self.phase} "
                    f"eyes={len(eyes)} sum={sum(row['hp'] for row in eyes)} "
                    f"body={None if body is None else body['hp']}",
                    flush=True,
                )
            if heart is not None or (not eyes and body is not None and int(body["hp"]) <= 0):
                self._step_prize(snap, heart, body)
                continue
            self._step_fight(snap, rows, eyes)
        if not self.blocker:
            rows = _rows(self.env)
            eyes = _eyes(rows)
            body = _body(rows)
            self.blocker = (
                f"cap swings={self.swings} eyes={len(eyes)} "
                f"sum={sum(row['hp'] for row in eyes)} "
                f"body={None if body is None else body['hp']}"
            )
        return self._done(False, self.blocker)

    def _done(self, ok: bool, blocker: str) -> FightResult:
        print(
            f"vitreous frames={self.frames} swings={self.swings} ok={ok} {blocker}",
            flush=True,
        )
        return FightResult(ok=ok, frames=self.frames, blocker=blocker, sword_hits=self.swings)

    def _step_prize(self, snap: Any, heart: dict[str, int] | None, body: dict[str, int] | None) -> None:
        if heart is None:
            _press(self.env, ())
            self.frames += 1
            if self.frames > CAP - 800:
                self.blocker = (
                    f"no heart swings={self.swings} "
                    f"body={None if body is None else [body['hp'], body['ai']]}"
                )
                self.frames = CAP
            return
        dx = int(heart["x"]) - int(snap.link_x)
        dy = int(heart["y"]) - int(snap.link_y)
        if abs(dx) <= 4 and abs(dy) <= 4:
            _press(self.env, ())
        elif abs(dx) >= abs(dy):
            _press(self.env, ("RIGHT" if dx > 0 else "LEFT",))
        else:
            _press(self.env, ("DOWN" if dy > 0 else "UP",))
        self.frames += 1

    def _step_fight(self, snap: Any, rows: list[dict[str, int]], eyes: list[dict[str, int]]) -> None:
        if self.phase == "spin":
            _press(self.env, ())
            self.frames += 1
            self.left -= 1
            if self.left <= 0:
                self.phase = "flee"
                self.left = FLEE_FRAMES
                self.flee = "LEFT" if self.flee == "RIGHT" else "RIGHT"
            return
        if self.phase == "flee":
            _press(self.env, (self.flee,))
            self.frames += 1
            self.left -= 1
            self.held = 0
            if self.left <= 0:
                self.phase = "charge"
            return
        bolts = [row for row in rows if row["type"] == BOLT]
        if bolts and self.bolt_left == 0:
            bolt = min(bolts, key=lambda row: abs(row["y"] - int(snap.link_y)))
            self.bolt_dir = "LEFT" if bolt["x"] >= int(snap.link_x) else "RIGHT"
            self.bolt_left = BOLT_DODGE
            self.held = 0
        if self.bolt_left:
            _press(self.env, (self.bolt_dir,))
            self.frames += 1
            self.bolt_left -= 1
            self.held = 0
            return
        aligned = False
        threat_x = None
        threat_dy = 99
        for row in eyes:
            dx = int(snap.link_x) - row["x"]
            dy = int(snap.link_y) - row["y"]
            if abs(dx) <= 14 and 0 < dy < 18 and dy < threat_dy:
                threat_x = int(row["x"])
                threat_dy = dy
            if abs(dx) <= 8 and 20 <= dy <= 24 and self.held >= CHARGE:
                aligned = True
        # A spin roots for 32 frames. Skip it when another eye is already
        # inside that root; keep charging otherwise. The measured release
        # is still 20-24px south.
        if aligned and threat_x is not None:
            self.phase = "flee"
            self.left = FLEE_FRAMES
            self.held = 0
            self.flee = "LEFT" if threat_x >= int(snap.link_x) else "RIGHT"
            _press(self.env, (self.flee,))
            self.frames += 1
            self.left -= 1
            return
        if aligned:
            self.phase = "spin"
            self.left = SPIN_LOCK
            self.swings += 1
            self.held = 0
            _press(self.env, ())
            self.frames += 1
            self.left -= 1
            return
        _press(self.env, ("B",))
        self.frames += 1
        self.held += 1


def fight(env: Any) -> FightResult:
    """Spin the eyes from the south and take the heart container."""
    return _Fight(env).run()
