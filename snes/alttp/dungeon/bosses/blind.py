"""Blind in room 0xAC on Scan_thieves_boss. Not a continuous claim.

Body and head are both type ``0xCE``. The head is a second slot whose
``$0D90`` is 2, the same body-versus-detached-head split as
``nes/zelda_i/dungeon/gleeok.py``. ``$0E50`` starts at 90. The measured
sword is 32 px south facing UP: ``$0EA0`` is 14 at frame 10, and frame 14
writes ``$0E50`` 90 to 128 and ``$0F90`` 0 to 1. ``strike_verdict`` is
whiff on that pose. The HP byte rises on frame 14, one frame after the
13-frame swing, and ``$0EF0`` is 0x1F so bit 7 is clear. This fight still
presses B on that pose. It does not press B at 24 px south, and it does
not bomb. The hammer connect is 24 px north facing left. Three south
swords, each after ``$0F10`` and ``$0EA0`` return to 0, walk ``$0F90``
through 1, then 2, then 0.

Walks go through ``PolicyGuard``. ``strike_if_safe`` is consulted on the
measured pose and is whiff there, so the measured B press still runs.
Soldiers are not swung: eleven ``strike_if_safe`` kills left the south
shutter at ``0x18`` and Link unmoved. ``drink_if_low`` runs when a bottle
is owned. ``$F3CC`` is not written.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/stage_replay.py \\
        Scan_thieves_boss alttp.dungeon.bosses.blind:fight
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from alttp.dungeon.bosses import FightResult
from alttp.primitives import active_sprites, bottle_due, drink_if_low, hammer
from alttp.ram import LINK_MAX_HP, read_inventory, wram_index
from alttp.rollout import PlannedPress, PolicyGuard, strike_if_safe
from alttp.startup import action_for, no_action, snapshot_env, step_frames

BLIND = 0xCE
HEART = 0xEA
ROOM = 0xAC
MARK = 0x0D90
HEAD = 2
PHASE = 0x0DB0
RECOIL = 0x0EA0
AUX = 0x0F10
HITS = 0x0F90
ALT = 0x0F70
DOOR = 0x1980
CAP = 3600
# Measured pose. Not 24 px south, which leaves $0F90 at 0 and hurts Link.
SWORD_DY = 32
HAMMER_DY = -24
FACE_UP = 0
FACE_LEFT = 4
LOG = Path(__file__).resolve().parents[2] / "recordings" / "probe_AP-160" / "fight.log"

__all__ = ["fight"]


def _u8(env: Any, offset: int) -> int:
    return int(env.get_ram()[wram_index(offset)])


def _step(env: Any, action: Any, frames: list[int], count: int = 1) -> None:
    step_frames(env, np.asarray(action, dtype=np.int8), count)
    frames[0] += count


def _press(env: Any, name: str, frames: list[int], count: int = 1) -> None:
    action = no_action() if name in ("", "NONE") else action_for(name)
    _step(env, action, frames, count)


def _slot_byte(env: Any, base: int, slot: int) -> int:
    return _u8(env, base + int(slot))


def _cheb(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(int(ax) - int(bx)), abs(int(ay) - int(by)))


def _blind_rows(env: Any) -> list[Any]:
    return [row for row in active_sprites(env) if int(row.sprite_type) == BLIND]


def _is_head(env: Any, row: Any) -> bool:
    return _slot_byte(env, MARK, row.slot) == HEAD


def _ready(env: Any, row: Any) -> bool:
    """Recoiled sprites do not take the next measured swing."""
    slot = int(row.slot)
    if _slot_byte(env, AUX, slot) != 0 or _slot_byte(env, RECOIL, slot) != 0:
        return False
    return (_slot_byte(env, 0x0EF0, slot) & 0x80) == 0


def _sword_window(lx: int, ly: int, row: Any) -> bool:
    return abs(lx - int(row.x)) <= 1 and ly - int(row.y) == SWORD_DY


def _hammer_window(lx: int, ly: int, row: Any) -> bool:
    return abs(lx - int(row.x)) <= 1 and ly - int(row.y) == HAMMER_DY


class Blind:
    """Walk to the measured pose, swing, track the detached head, take the heart."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.frames = [0]
        self.success = False
        self.failed = False
        self.blocker = ""
        self.swings = 0
        self.hammers = 0
        self._max_hp = 0
        self._follower = 0
        self._guard: PolicyGuard | None = None
        self._holder: PlannedPress | None = None
        self._stuck = 0
        self._last_xy: tuple[int, int] | None = None
        self._near: tuple[int, int, int, int] | None = None

    def run(self) -> FightResult:
        self._remember()
        while self.frames[0] < CAP and not self.success and not self.failed:
            self._tick()
        if not self.success and not self.failed:
            self.failed = True
            self.blocker = self.blocker or self._stuck_reason()
        self._log()
        return FightResult(ok=self.success, frames=self.frames[0], blocker=self.blocker)

    def _remember(self) -> None:
        inv = read_inventory(self.env.get_ram())
        self._max_hp = int(inv.max_hp)
        self._follower = int(inv.follower) if hasattr(inv, "follower") else _u8(self.env, 0xF3CC)
        holder = PlannedPress()
        guard = PolicyGuard(
            holder,
            horizon=6,
            detours=(4,),
            directions=("UP", "DOWN", "LEFT", "RIGHT"),
            reach=28,
        )
        guard.bind_env(self.env)
        self._holder = holder
        self._guard = guard

    def _snap(self):
        return snapshot_env(self.env)

    def _busy(self) -> bool:
        snap = self._snap()
        if int(snap.hp) <= 0:
            self.failed = True
            self.blocker = f"link hp 0 xy=({snap.link_x},{snap.link_y})"
            return True
        if int(getattr(snap, "room_base_id", snap.room_id)) != ROOM and not self.success:
            self.failed = True
            self.blocker = f"left room 0x{int(snap.room_id):02X}"
            return True
        if snap.is_text_mode or snap.is_hold_up_item or not snap.has_control:
            _press(self.env, "A" if snap.is_text_mode else "NONE", self.frames)
            self._check_heart()
            return True
        if bottle_due(self.env):
            drank = drink_if_low(self.env)
            self.frames[0] += int(drank.frames)
            self._check_heart()
            return True
        if _u8(self.env, LINK_MAX_HP) > self._max_hp:
            self.success = True
            return True
        return False

    def _check_heart(self) -> None:
        if _u8(self.env, LINK_MAX_HP) > self._max_hp:
            self.success = True

    def _tick(self) -> None:
        if self._busy():
            return
        self._check_heart()
        if self.success:
            return
        snap = self._snap()
        xy = (int(snap.link_x), int(snap.link_y))
        self._stuck = self._stuck + 1 if xy == self._last_xy else 0
        self._last_xy = xy
        heart = self._heart()
        if heart is not None:
            self._walk(self._toward(xy[0], xy[1], int(heart.x), int(heart.y)))
            return
        rows = _blind_rows(self.env)
        heads = [row for row in rows if _is_head(self.env, row)]
        bodies = [row for row in rows if not _is_head(self.env, row)]
        target = self._aim(heads, bodies)
        if target is not None and _sword_window(xy[0], xy[1], target) and _ready(self.env, target):
            self._sword(int(snap.link_direction), int(target.slot))
            return
        if target is not None and _hammer_window(xy[0], xy[1], target) and _ready(self.env, target):
            self._hammer(int(snap.link_direction))
            return
        self._note_gap(xy, heads + bodies)
        goal = target if target is not None else (heads[0] if heads else (bodies[0] if bodies else None))
        if goal is None:
            _press(self.env, "NONE", self.frames)
            if self.frames[0] > CAP - 200:
                self.failed = True
                self.blocker = "no blind"
            return
        self._walk(self._toward(xy[0], xy[1], int(goal.x), int(goal.y) + SWORD_DY))

    def _aim(self, heads: list[Any], bodies: list[Any]) -> Any | None:
        """Head once it exists, otherwise the lowest body slot. Not a new pose."""
        if heads:
            return min(heads, key=lambda row: _cheb(row.x, row.y, *self._last_xy or (0, 0)))
        if not bodies:
            return None
        return min(bodies, key=lambda row: int(row.slot))

    def _note_gap(self, xy: tuple[int, int], rows: list[Any]) -> None:
        for row in rows:
            dx = abs(xy[0] - int(row.x))
            dy = xy[1] - int(row.y)
            rank = dx + abs(dy - SWORD_DY)
            if self._near is None or rank < self._near[0]:
                self._near = (rank, dx, dy, int(row.slot))

    def _sword(self, facing: int, slot: int) -> None:
        if facing != FACE_UP:
            _press(self.env, "UP", self.frames)
            return
        script = strike_if_safe(self.env, slot)
        if script:
            for act in script:
                _step(self.env, act.action, self.frames)
            self.swings += 1
            return
        _press(self.env, "B", self.frames)
        self.swings += 1
        for _ in range(16):
            if self._busy():
                return
            _press(self.env, "NONE", self.frames)

    def _hammer(self, facing: int) -> None:
        if facing != FACE_LEFT:
            _press(self.env, "LEFT", self.frames)
            return
        swung = hammer(self.env)
        self.frames[0] += int(swung.frames)
        self.hammers += 1

    def _heart(self) -> Any | None:
        rows = [row for row in active_sprites(self.env) if int(row.sprite_type) == HEART]
        return rows[0] if rows else None

    def _toward(self, x0: int, y0: int, x1: int, y1: int) -> str:
        dx = int(x1) - int(x0)
        dy = int(y1) - int(y0)
        if dx == 0 and dy == 0:
            return "NONE"
        if abs(dx) >= abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        return "DOWN" if dy > 0 else "UP"

    def _walk(self, button: str) -> None:
        assert self._holder is not None and self._guard is not None
        self._holder.button = button
        act = self._guard.step(self._snap())
        _step(self.env, act.action, self.frames)

    def _stuck_reason(self) -> str:
        snap = self._snap()
        heads = [row for row in _blind_rows(self.env) if _is_head(self.env, row)]
        head = "none" if not heads else f"hp={int(heads[0].hp)} z={_slot_byte(self.env, ALT, heads[0].slot)}"
        near = "none" if self._near is None else (
            f"rank={self._near[0]} dx={self._near[1]} dy={self._near[2]} slot={self._near[3]}"
        )
        return (
            f"cap stuck={self._stuck} swings={self.swings} hammers={self.hammers} "
            f"near={near} xy=({snap.link_x},{snap.link_y}) "
            f"door=0x{_u8(self.env, DOOR):02X} head={head}"
        )

    def _log(self) -> None:
        snap = self._snap()
        bodies = [row for row in _blind_rows(self.env) if not _is_head(self.env, row)]
        heads = [row for row in _blind_rows(self.env) if _is_head(self.env, row)]
        body = bodies[0] if bodies else None
        head = heads[0] if heads else None
        line = (
            f"frames={self.frames[0]} ok={int(self.success)} swings={self.swings} "
            f"hammers={self.hammers} near={self._near} stuck={self._stuck} "
            f"xy=({snap.link_x},{snap.link_y}) hp={snap.hp} max=0x{_u8(self.env, LINK_MAX_HP):02X} "
            f"door=0x{_u8(self.env, DOOR):02X} "
            f"body={None if body is None else (int(body.hp), _slot_byte(self.env, HITS, body.slot), _slot_byte(self.env, PHASE, body.slot))} "
            f"head={None if head is None else (int(head.hp), _slot_byte(self.env, ALT, head.slot), _slot_byte(self.env, MARK, head.slot))} "
            f"blocker={self.blocker}\n"
        )
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(line)
        print(line, end="", flush=True)


def fight(env: Any) -> FightResult:
    """Swing the measured poses and take the heart container."""
    return Blind(env).run()
