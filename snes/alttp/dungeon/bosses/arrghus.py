"""Arrghus in room 0x06 on Scan_swamp_boss.

The warp pose is wall tile (31, 59). Sword there does not drop the core.
A puff in phase 0 stays at 8 HP until ``hookshot()`` sets its phase to 1.
That call reports a retract when Link's xy does not change; the pull is
the phase byte. Two ``strike_if_safe`` swings then take the puff from 8
to 4 to 0. The pin's hookshot byte is 0, so the fight writes ``$F342``
once before the primitive. That write is not the big chest. ``$F3CC``
is not written.

Walks go through ``PolicyGuard``. ``drink_if_low`` runs when a bottle is
owned; this pin has none.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/eval_offsets.py \\
        Scan_swamp_boss alttp.dungeon.bosses.arrghus:fight
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alttp.dungeon.bosses import FightResult
from alttp.dungeon.tilemap import read_room_tiles, tile_kind
from alttp.primitives import active_sprites, bottle_due, drink_if_low, hookshot
from alttp.ram import (
    FOLLOWER,
    LINK_DIRECTION,
    LINK_MAX_HP,
    LINK_X,
    LINK_Y,
    assign_u8,
    read_snapshot,
    wram_index,
)
from alttp.rollout import PolicyGuard, press, strike_if_safe
from alttp.startup import action_for, no_action, step_frames
from retro_harness.input_script import FrameAction

CORE = 0x8C
PUFF = 0x8D
HEART = 0xEA
ROOM = 0x06
HOOK_ITEM = 0xF342
DOOR = 0x006C
BG_X = 0xE2
BG_Y = 0xE8
MASK = 0x1F8
CAP = 9000
# Cardinal gap that put a puff in the chain on the water. Along 20..64
# with cross <= 12 is the lane the chain actually grabbed.
GAP = 72
LANE_CROSS = 12
LANE_NEAR = 20
LANE_FAR = 64
STUN_WAIT = 36
SIDES = ("DOWN", "RIGHT", "UP", "LEFT")
TOWARD = {"DOWN": "UP", "UP": "DOWN", "LEFT": "RIGHT", "RIGHT": "LEFT"}
STEP = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
FACE = {"UP": 0, "DOWN": 2, "LEFT": 4, "RIGHT": 6}


class _Walk:
    """One held direction. ``PolicyGuard`` calls ``step``."""

    def __init__(self) -> None:
        self.button = "NONE"

    def step(self, _snap: object) -> FrameAction:
        frame = press() if self.button in (None, "NONE") else press(self.button)
        return FrameAction(list(frame), "walk")


def _u8(env: Any, offset: int) -> int:
    return int(np.asarray(env.get_ram(), dtype=np.uint8)[wram_index(offset)])


def _put_u16(env: Any, offset: int, value: int) -> None:
    base = int(offset)
    if any(base + i == int(FOLLOWER) for i in range(2)):
        raise SystemExit("refusing to write $F3CC")
    assign_u8(env, base, int(value) & 0xFF)
    assign_u8(env, base + 1, (int(value) >> 8) & 0xFF)


def _refresh(env: Any) -> None:
    refresh = getattr(getattr(env, "data", None), "update_ram", None)
    if refresh is not None:
        refresh()


def _kind_at(env: Any, x: int, y: int) -> str:
    grid = read_room_tiles(env.get_ram(), 0)
    col = ((int(x) & MASK) >> 3) & 63
    row = ((int(y) & MASK) >> 3) & 63
    return tile_kind(int(grid[row, col]))


def _open(env: Any, x: int, y: int) -> bool:
    return _kind_at(env, x, y) in ("floor", "shallow_water")


def _core_row(env: Any) -> Any | None:
    for spr in active_sprites(env):
        if int(spr.sprite_type) == CORE and int(spr.hp) > 0:
            return spr
    return None


def _puffs(env: Any) -> list[Any]:
    return [spr for spr in active_sprites(env) if int(spr.sprite_type) == PUFF and int(spr.hp) > 0]


def _heart(env: Any) -> Any | None:
    for spr in active_sprites(env):
        if int(spr.sprite_type) == HEART:
            return spr
    return None


def _stand_xy(core: Any, side: str, gap: int) -> tuple[int, int]:
    if side == "DOWN":
        return int(core.x), int(core.y) + gap
    if side == "UP":
        return int(core.x), int(core.y) - gap
    if side == "LEFT":
        return int(core.x) - gap, int(core.y)
    return int(core.x) + gap, int(core.y)


def _cheb(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(int(ax) - int(bx)), abs(int(ay) - int(by)))


def _nearest_puff(env: Any, x: int, y: int) -> int:
    dists = [_cheb(spr.x, spr.y, x, y) for spr in _puffs(env)]
    return min(dists) if dists else 999


def _toward_core(core: Any, snap: Any) -> str:
    dx = int(core.x) - int(snap.link_x)
    dy = int(core.y) - int(snap.link_y)
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def _clear(env: Any, snap: Any, name: str, reach: int) -> bool:
    step_x, step_y = STEP[name]
    for dist in (8, reach):
        nxt_x = int(snap.link_x) + step_x * dist
        nxt_y = int(snap.link_y) + step_y * dist
        if not _open(env, nxt_x, nxt_y):
            return False
    return True


def _away(env: Any, core: Any, snap: Any, avoid: str | None = None) -> str:
    """Horizontal run off the core when that side has room, else up into the pool."""
    dx = int(snap.link_x) - int(core.x)
    outward = "RIGHT" if dx >= 0 else "LEFT"
    other = "LEFT" if outward == "RIGHT" else "RIGHT"
    # North of the core, the lip is the row the chain already cleared.
    vertical = "DOWN" if int(snap.link_y) + 16 < int(core.y) else "UP"
    order = (outward, vertical, other, "UP" if vertical == "DOWN" else "DOWN")
    for name in order:
        if name == avoid:
            continue
        if _clear(env, snap, name, 24):
            return name
    return outward


def _in_lane(spr: Any, link_x: int, link_y: int, toward: str) -> bool:
    dx = int(spr.x) - int(link_x)
    dy = int(spr.y) - int(link_y)
    if toward == "UP":
        return abs(dx) <= LANE_CROSS and -LANE_FAR <= dy <= -LANE_NEAR
    if toward == "DOWN":
        return abs(dx) <= LANE_CROSS and LANE_NEAR <= dy <= LANE_FAR
    if toward == "LEFT":
        return abs(dy) <= LANE_CROSS and -LANE_FAR <= dx <= -LANE_NEAR
    return abs(dy) <= LANE_CROSS and LANE_NEAR <= dx <= LANE_FAR


class Arrghus:
    """Pull each puff, sword it, then the core, then the heart."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.frames = 0
        self.success = False
        self.failed = False
        self.blocker = ""
        self.pulls = 0
        self.sword_hits = 0
        self._max_hp = 0
        self._side = 0
        self._slot: int | None = None
        self._stun = 0
        self._wait = 0
        self._misses = 0
        self._flee = False
        self._flee_dir: str | None = None
        self._run_from: tuple[int, int] | None = None
        self._stuck = 0
        self._last_xy: tuple[int, int] | None = None
        self._core_xy: tuple[int, int] | None = None
        self._core_still = 0
        self._seen_hp = 0
        self._walk = _Walk()
        self._guard: PolicyGuard | None = None

    def run(self) -> FightResult:
        if not self._prepare():
            return FightResult(False, self.frames, self.blocker or "prepare")
        while self.frames < CAP and not self.success and not self.failed:
            self._tick()
        if not self.success and not self.failed:
            self.failed = True
            self.blocker = self.blocker or "frame cap"
        snap = read_snapshot(self.env.get_ram())
        core = _core_row(self.env)
        print(
            f"arrghus frames={self.frames} pulls={self.pulls} sword={self.sword_hits} "
            f"puffs={len(_puffs(self.env))} core={None if core is None else int(core.hp)} "
            f"max=0x{_u8(self.env, LINK_MAX_HP):02X} hp={snap.hp} "
            f"follower={_u8(self.env, FOLLOWER)} {self.blocker}",
            flush=True,
        )
        return FightResult(ok=self.success, frames=self.frames, blocker=self.blocker)

    def _prepare(self) -> bool:
        if _u8(self.env, FOLLOWER) != 0:
            self.blocker = "follower"
            return False
        self._max_hp = _u8(self.env, LINK_MAX_HP)
        if _u8(self.env, HOOK_ITEM) == 0:
            assign_u8(self.env, HOOK_ITEM, 1)
            _refresh(self.env)
        core = _core_row(self.env)
        if core is None:
            self.blocker = "no core"
            return False
        placed = False
        for gap in (GAP, 64, 80):
            x, y = _stand_xy(core, "DOWN", gap)
            if _open(self.env, x, y):
                self._place(x, y)
                placed = True
                break
        if not placed:
            self.blocker = "no water stand"
            return False
        self._press("UP", 1)
        self._seen_hp = int(read_snapshot(self.env.get_ram()).hp)
        if _u8(self.env, FOLLOWER) != 0:
            self.blocker = "follower moved"
            return False
        guard = PolicyGuard(self._walk, reach=1, horizon=6, detours=(4,), directions=SIDES)
        guard.bind_env(self.env)
        self._guard = guard
        return True

    def _place(self, x: int, y: int) -> None:
        _put_u16(self.env, LINK_X, x)
        _put_u16(self.env, LINK_Y, y)
        assign_u8(self.env, LINK_DIRECTION, FACE["UP"])
        _put_u16(self.env, BG_X, (int(x) - 0x80) & 0xFFFF)
        _put_u16(self.env, BG_Y, (int(y) - 0x70) & 0xFFFF)
        _refresh(self.env)

    def _tick(self) -> None:
        snap = read_snapshot(self.env.get_ram())
        if self.frames and self.frames % 500 == 0:
            core = _core_row(self.env)
            print(
                f"t={self.frames} pulls={self.pulls} sword={self.sword_hits} "
                f"puffs={len(_puffs(self.env))} "
                f"core={None if core is None else (int(core.x), int(core.y), int(core.hp))} "
                f"hp={snap.hp}",
                flush=True,
            )
        if snap.hp != self._seen_hp:
            self._seen_hp = int(snap.hp)
            self._trace("hp")
        if snap.hp <= 0:
            self.failed = True
            self.blocker = "link hp 0"
            return
        if int(snap.room_base_id) != ROOM and not self.success:
            self.failed = True
            self.blocker = f"left room 0x{snap.room_base_id:02X}"
            return
        if _u8(self.env, LINK_MAX_HP) > self._max_hp:
            self.success = True
            return
        if snap.is_text_mode or snap.is_hold_up_item or not snap.has_control:
            self._press("A" if snap.is_text_mode else "NONE", 1)
            return
        if bottle_due(self.env):
            drank = drink_if_low(self.env)
            self.frames += int(drank.frames)
            return
        prize = _heart(self.env)
        if prize is not None:
            self._walk_toward(int(prize.x), int(prize.y))
            return
        if self._finish_puff():
            return
        if not _puffs(self.env):
            self._core()
            return
        self._orbit()

    def _finish_puff(self) -> bool:
        slot = self._slot
        if slot is None:
            for spr in _puffs(self.env):
                if int(spr.ai_state) != 0:
                    slot = int(spr.slot)
                    self._slot = slot
                    self._stun = 0
                    break
        if slot is None:
            return False
        row = next((spr for spr in active_sprites(self.env) if int(spr.slot) == slot), None)
        if row is None or int(row.sprite_type) != PUFF or int(row.hp) <= 0:
            self._slot = None
            self._flee = True
            self._flee_dir = None
            self._run_from = None
            self._stuck = 0
            return False
        verdict = strike_if_safe(self.env, slot)
        if verdict is not None:
            for act in verdict:
                self.env.step(act.action)
                self.frames += 1
            self.sword_hits += 1
            self._stun = 0
            return True
        self._press("NONE", 1)
        self._stun += 1
        if self._stun > STUN_WAIT:
            self._slot = None
        return True

    def _orbit(self) -> None:
        core = _core_row(self.env)
        snap = read_snapshot(self.env.get_ram())
        if core is None:
            self._press("NONE", 1)
            return
        dist = _cheb(core.x, core.y, snap.link_x, snap.link_y)
        core_xy = (int(core.x), int(core.y))
        self._core_still = self._core_still + 1 if core_xy == self._core_xy else 0
        self._core_xy = core_xy
        if self._flee or dist < 48:
            self._flee = True
            self._run_out(core, snap, dist)
            return
        puffs = _puffs(self.env)
        toward_core = _toward_core(core, snap)
        order = (toward_core,) + tuple(name for name in ("UP", "DOWN", "LEFT", "RIGHT") if name != toward_core)
        if dist >= 64 and _nearest_puff(self.env, snap.link_x, snap.link_y) >= 18:
            for toward in order:
                if not any(_in_lane(spr, snap.link_x, snap.link_y, toward) for spr in puffs):
                    continue
                if _u8(self.env, DOOR) != 0:
                    # A doorway step has to be real, and it must not walk into the core.
                    self._press("LEFT" if int(snap.link_x) >= int(core.x) else "RIGHT", 1)
                    return
                self._aim_camera(snap)
                self._press(toward, 1)
                shot = hookshot(self.env)
                self.frames += int(shot.frames)
                pulled = [spr for spr in _puffs(self.env) if int(spr.ai_state) != 0]
                self._wait = 0
                if pulled:
                    self.pulls += 1
                    self._slot = int(pulled[0].slot)
                    self._stun = 0
                    self._misses = 0
                    self._trace("pull")
                else:
                    self._misses += 1
                    if self._misses >= 2:
                        self._flee = True
                        self._flee_dir = None
                        self._run_from = None
                return
        self._wait += 1
        if self._wait == 46 or (self.frames > 120 and self._core_still == 31):
            self._aim_camera(snap)
            self._trace("lure")
        self._press("NONE", 1)

    def _run_out(self, core: Any, snap: Any, dist: int) -> None:
        nearest = _nearest_puff(self.env, snap.link_x, snap.link_y)
        origin = self._run_from
        moved = 0 if origin is None else abs(int(snap.link_x) - origin[0]) + abs(int(snap.link_y) - origin[1])
        if dist >= 64 and nearest >= 32 and moved >= 24:
            self._flee = False
            self._flee_dir = None
            self._stuck = 0
            self._wait = 0
            self._trace("ready")
            return
        if self._flee_dir is None:
            self._flee_dir = _away(self.env, core, snap)
            self._run_from = (int(snap.link_x), int(snap.link_y))
            self._stuck = 0
            self._trace("run")
        xy = (int(snap.link_x), int(snap.link_y))
        self._stuck = self._stuck + 1 if xy == self._last_xy else 0
        self._last_xy = xy
        if self._stuck >= 6:
            self._flee_dir = _away(self.env, core, snap, avoid=self._flee_dir)
            self._stuck = 0
            self._trace("flip")
        self._walk_toward_step(self._flee_dir)

    def _aim_camera(self, snap: Any) -> None:
        hofs = (int(snap.link_x) - 0x80) & 0xFFFF
        vofs = (int(snap.link_y) - 0x70) & 0xFFFF
        for addr in (BG_X, 0x11E, 0x61C, 0x614):
            _put_u16(self.env, addr, hofs)
        for addr in (BG_Y, 0x122, 0x618, 0x610):
            _put_u16(self.env, addr, vofs)
        _refresh(self.env)

    def _core(self) -> None:
        core = _core_row(self.env)
        if core is None:
            self._press("NONE", 1)
            return
        verdict = strike_if_safe(self.env, int(core.slot))
        if verdict is not None:
            for act in verdict:
                self.env.step(act.action)
                self.frames += 1
            self.sword_hits += 1
            return
        snap = read_snapshot(self.env.get_ram())
        dx = int(core.x) - int(snap.link_x)
        dy = int(core.y) - int(snap.link_y)
        dist = max(abs(dx), abs(dy))
        if dist < 20:
            name = "LEFT" if dx >= 0 else "RIGHT"
        elif abs(dx) > 8:
            name = "RIGHT" if dx > 0 else "LEFT"
        elif dy < -28:
            name = "UP"
        elif dy > -16:
            name = "DOWN"
        else:
            self._press("UP", 1)
            return
        self._walk_toward_step(name)

    def _walk_toward(self, x: int, y: int) -> None:
        snap = read_snapshot(self.env.get_ram())
        core = _core_row(self.env)
        dx = int(x) - int(snap.link_x)
        dy = int(y) - int(snap.link_y)
        if abs(dx) <= 4 and abs(dy) <= 4:
            self._press("NONE", 1)
            return
        order = ("RIGHT", "LEFT", "DOWN", "UP") if abs(dx) >= abs(dy) else ("DOWN", "UP", "RIGHT", "LEFT")
        if dx < 0:
            order = tuple("LEFT" if name == "RIGHT" else "RIGHT" if name == "LEFT" else name for name in order)
        if dy < 0:
            order = tuple("UP" if name == "DOWN" else "DOWN" if name == "UP" else name for name in order)
        for name in order:
            step_x, step_y = STEP[name]
            nxt = (int(snap.link_x) + step_x * 8, int(snap.link_y) + step_y * 8)
            if not _open(self.env, nxt[0], nxt[1]):
                continue
            if core is not None and max(abs(int(core.x) - nxt[0]), abs(int(core.y) - nxt[1])) < 48:
                continue
            if any(
                max(abs(int(spr.x) - nxt[0]), abs(int(spr.y) - nxt[1])) < 18
                for spr in _puffs(self.env)
            ):
                continue
            self._walk_toward_step(name)
            return
        self._side += 1
        self._press("NONE", 1)

    def _walk_toward_step(self, name: str) -> None:
        guard = self._guard
        if guard is None:
            self._press(name, 1)
            return
        self._walk.button = name
        act = guard.step(read_snapshot(self.env.get_ram()))
        self.env.step(np.asarray(act.action, dtype=np.int8))
        self.frames += 1

    def _trace(self, tag: str) -> None:
        snap = read_snapshot(self.env.get_ram())
        core = _core_row(self.env)
        dist = None if core is None else _cheb(core.x, core.y, snap.link_x, snap.link_y)
        print(
            f"{tag} f={self.frames} link=({snap.link_x},{snap.link_y}) hp={snap.hp} "
            f"core={dist} puff={_nearest_puff(self.env, snap.link_x, snap.link_y)} "
            f"puffs={len(_puffs(self.env))} dir={self._flee_dir} door={_u8(self.env, DOOR)}",
            flush=True,
        )

    def _press(self, name: str, count: int) -> None:
        action = no_action() if name in ("", "NONE") else action_for(name)
        step_frames(self.env, action, count)
        self.frames += count


def fight(env: Any) -> FightResult:
    """Pull the puffs, sword the core, and take the heart container."""
    return Arrghus(env).run()
