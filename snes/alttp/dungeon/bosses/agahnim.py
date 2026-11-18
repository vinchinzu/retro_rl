"""Agahnim in room 0x0D on Scan_castle_tower_boss. Isolated poke.

Sprite 0x7A, ``$0E50`` 96, phase ``$0D80`` (4 charge, 5 teleport, 2 arrive,
3 attack). z3 ``dmgFromLink`` is all zeros. The ai-0 ball ``0x7B`` flips
vy 32 to -48 when one B is pressed while it is still north. The safe
press is the live-grid row 55 of the landing column, chebyshev 19-27,
``|dx|`` at most 8: boss 96 to 80 and Link stays at 24
(``recordings/probe_AP-127/swing_safe.json``). Row 52 on that column
takes a body hit before the ball exists. ``strike_if_safe`` does not see
this swing. Walks use ``PolicyGuard``. ``drink_if_low`` no-ops with no
bottle. ``$F3C5`` is set to 1 when the pin still reads 0; that byte is
not saved. ``$F3CC`` is not written.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/stage_replay.py \\
        Scan_castle_tower_boss alttp.dungeon.bosses.agahnim:fight
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alttp.dungeon.bosses import FightResult
from alttp.dungeon.tilemap import TILE_PX, read_room_tiles
from alttp.primitives import active_sprites, bottle_due, drink_if_low
from alttp.ram import (
    FOLLOWER,
    LINK_ACTION,
    LINK_DIRECTION,
    LINK_MAX_HP,
    LINK_X,
    LINK_Y,
    SRAM_PROGRESS,
    assign_u8,
    read_snapshot,
    wram_index,
)
from alttp.rollout import PlannedPress, PolicyGuard
from alttp.startup import action_for, no_action

BOSS = 0x7A
BALL = 0x7B
HEART = 0xEA
ROOM = 0x0D
SWORD_GFX = 0x003C
VY = 0x0D40
STAND_ROW = 55
PRESS_FAR = 26
PRESS_NEAR = 20
ALIGN = 8
THREAT = 0x55
CAP = 9000
_FACE = {"UP": 0, "DOWN": 2, "LEFT": 4, "RIGHT": 6}

__all__ = ["fight"]


def _u8(env: Any, offset: int) -> int:
    return int(env.get_ram()[wram_index(offset)])


def _s8(value: int) -> int:
    raw = int(value) & 0xFF
    return raw - 256 if raw >= 128 else raw


def _put_u16(env: Any, offset: int, value: int) -> None:
    base = int(offset)
    if any(base + i == int(FOLLOWER) for i in range(2)):
        raise SystemExit("refusing to write $F3CC")
    assign_u8(env, base, int(value) & 0xFF)
    assign_u8(env, base + 1, (int(value) >> 8) & 0xFF)


def _step(env: Any, action: Any) -> None:
    env.step(np.asarray(action, dtype=np.uint8))


def _snap(env: Any):
    return read_snapshot(env.get_ram())


def _pose(x: int, y: int, col: int, row: int) -> tuple[int, int]:
    return (int(x) & ~0x1FF) + col * TILE_PX + 4, (int(y) & ~0x1FF) + row * TILE_PX + 4


def _boss(env: Any):
    rows = [s for s in active_sprites(env) if int(s.sprite_type) == BOSS]
    return rows[0] if rows else None


def _heart(env: Any):
    rows = [s for s in active_sprites(env) if int(s.sprite_type) == HEART]
    return rows[0] if rows else None


def _lead(env: Any):
    rows = [s for s in active_sprites(env) if int(s.sprite_type) == BALL and int(s.ai_state) == 0]
    return max(rows, key=lambda s: int(s.y)) if rows else None


def _stand_xy(env: Any) -> tuple[int, int] | None:
    snap = _snap(env)
    grid = read_room_tiles(env.get_ram(), int(snap.layer) & 1)
    col = (int(snap.link_x) // TILE_PX) & 63
    for row in (STAND_ROW, STAND_ROW - 1, STAND_ROW + 1):
        if 0 <= row < 64 and int(grid[row, col]) == 0:
            return _pose(int(snap.link_x), int(snap.link_y), col, row)
    return None


def _toward(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dx = int(dst[0]) - int(src[0])
    dy = int(dst[1]) - int(src[1])
    if max(abs(dx), abs(dy)) <= 4:
        return "NONE"
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


class Agahnim:
    """Reflect the south ball from row 55, then take the heart."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.frames = 0
        self.strikes = 0
        self.success = False
        self.failed = False
        self.blocker = ""
        self._follower = 0
        self._max_hp = 0
        self._stand: tuple[int, int] | None = None
        self._swings = 0
        self._lock = 0
        self._holder = PlannedPress()
        self._guard = PolicyGuard(
            self._holder, horizon=8, detours=(4, 8), directions=("UP", "DOWN", "LEFT", "RIGHT"), reach=32
        )

    def run(self) -> FightResult:
        self._follower = _u8(self.env, FOLLOWER)
        self._max_hp = _u8(self.env, LINK_MAX_HP)
        self._guard.bind_env(self.env)
        if not self._arm():
            return self._result()
        while self.frames < CAP and not self.success and not self.failed:
            if self._blocked():
                continue
            self._act()
        if not self.success and not self.failed:
            self.failed = True
            self.blocker = "frame cap"
        return self._result()

    def _result(self) -> FightResult:
        return FightResult(
            ok=self.success,
            frames=self.frames,
            blocker="" if self.success else self.blocker,
            strikes=self.strikes,
        )

    def _arm(self) -> bool:
        if _u8(self.env, SRAM_PROGRESS) == 0:
            assign_u8(self.env, SRAM_PROGRESS, 1)
        stand = _stand_xy(self.env)
        if stand is None:
            self.failed = True
            self.blocker = "no floor on the reflect row"
            return False
        self._stand = stand
        print(f"agahnim stand {stand} progress={_u8(self.env, SRAM_PROGRESS)}", flush=True)
        _put_u16(self.env, LINK_X, stand[0])
        _put_u16(self.env, LINK_Y, stand[1])
        assign_u8(self.env, 0x2A, 0)
        assign_u8(self.env, 0x2B, 0)
        assign_u8(self.env, LINK_DIRECTION, _FACE["UP"])
        if _u8(self.env, FOLLOWER) != self._follower:
            self.failed = True
            self.blocker = "$F3CC changed"
            return False
        for _ in range(4):
            self._idle()
        return not self.failed

    def _idle(self) -> None:
        _step(self.env, no_action())
        self.frames += 1

    def _press(self, name: str | None) -> None:
        _step(self.env, no_action() if name is None else action_for(name))
        self.frames += 1

    def _blocked(self) -> bool:
        if _u8(self.env, FOLLOWER) != self._follower:
            self.failed = True
            self.blocker = "$F3CC changed"
            return True
        snap = _snap(self.env)
        if _u8(self.env, LINK_MAX_HP) > self._max_hp:
            self.success = True
            return True
        if int(snap.hp) <= 0:
            self.failed = True
            boss = _boss(self.env)
            self.blocker = (
                f"link hp 0 swings={self._swings} strikes={self.strikes} "
                f"boss={None if boss is None else int(boss.hp)} xy=({int(snap.link_x)},{int(snap.link_y)})"
            )
            return True
        if snap.is_text_mode or snap.is_hold_up_item or not snap.has_control:
            self._press("A" if snap.is_text_mode else None)
            return True
        if int(snap.room_base_id) != ROOM:
            self.failed = True
            self.blocker = f"left room 0x{int(snap.room_base_id):02X}"
            return True
        if bottle_due(self.env):
            drank = drink_if_low(self.env)
            self.frames += int(drank.frames)
            return True
        if self._lock > 0 or _u8(self.env, SWORD_GFX) != 0 or _u8(self.env, LINK_ACTION) != 0:
            if self._lock > 0:
                self._lock -= 1
            self._idle()
            return True
        return False

    def _walk(self, button: str) -> None:
        if button == "NONE":
            self._idle()
            return
        self._holder.button = button
        act = self._guard.step(_snap(self.env))
        _step(self.env, act.action)
        self.frames += 1

    def _window(self) -> tuple[Any, int, int] | None:
        ball = _lead(self.env)
        link = _snap(self.env)
        if ball is None:
            return None
        dx = int(ball.x) - int(link.link_x)
        dy = int(ball.y) - int(link.link_y)
        dist = max(abs(dx), abs(dy))
        vy = _s8(_u8(self.env, VY + int(ball.slot)))
        if dy < 0 and vy > 0 and abs(dx) <= ALIGN and PRESS_NEAR <= dist <= PRESS_FAR:
            return ball, dist, dx
        return None

    def _act(self) -> None:
        heart = _heart(self.env)
        boss = _boss(self.env)
        if heart is not None or boss is None or int(boss.hp) <= 0:
            if heart is None:
                self._idle()
                return
            link = _snap(self.env)
            self._walk(_toward((int(link.link_x), int(link.link_y)), (int(heart.x), int(heart.y))))
            return
        window = self._window()
        if window is not None:
            _ball, dist, dx = window
            before = int(boss.hp)
            if _u8(self.env, LINK_DIRECTION) != _FACE["UP"]:
                assign_u8(self.env, LINK_DIRECTION, _FACE["UP"])
            self._swings += 1
            self._lock = 16
            self._press("B")
            print(
                f"agahnim swing t={self.frames} dist={dist} dx={dx} boss={before} link={_snap(self.env).hp}",
                flush=True,
            )
            return
        threat = self._threat()
        if threat is not None:
            self._walk("LEFT" if threat >= 0 else "RIGHT")
            return
        stand = self._stand
        link = _snap(self.env)
        if stand is not None and max(abs(int(link.link_x) - stand[0]), abs(int(link.link_y) - stand[1])) > 2:
            self._walk(_toward((int(link.link_x), int(link.link_y)), stand))
            return
        self._idle()

    def _threat(self) -> int | None:
        """A ball already inside the reflect band. Step off its column."""
        link = _snap(self.env)
        nearest: tuple[int, int] | None = None
        for spr in active_sprites(self.env):
            if int(spr.sprite_type) not in (BALL, THREAT):
                continue
            dx = int(spr.x) - int(link.link_x)
            dy = int(spr.y) - int(link.link_y)
            dist = max(abs(dx), abs(dy))
            if abs(dx) <= 12 and dist < PRESS_NEAR and dy <= 6:
                if nearest is None or dist < nearest[0]:
                    nearest = (dist, dx)
        if nearest is None:
            return None
        return nearest[1]


def fight(env: Any) -> FightResult:
    """Kill Agahnim and take the heart container."""
    return Agahnim(env).run()
