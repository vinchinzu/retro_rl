"""Helmasaur King in room 0x5A on Scan_pod_boss. Not an entry pin.

Sprite 0x92, ``$0E50`` 48, phase ``$0D80``. A bomb planted while Link is
32–48px south and within 24px on x drops that byte by 4. Sword
``strike_if_safe`` is polled inside 16px; the probe's B presses did not
drop ``$0E50``. The bow's arrow from the same south band did not either.
Walks go through ``PolicyGuard``. ``drink_if_low`` runs when a bottle is
owned. The stand is ``maps/room_5a.json`` ``arena_floor``. Bombs are a
disclosed ``$F343`` write (the pin has 0). ``$F3CC`` is not written.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/stage_replay.py \\
        Scan_pod_boss alttp.dungeon.bosses.helmasaur:fight
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alttp.dungeon.bosses import FightResult
from alttp.dungeon.tilemap import read_room_tiles
from alttp.primitives import (
    active_ancillae,
    active_sprites,
    bottle_due,
    drink_if_low,
    place_bomb,
    select_y_item,
)
from alttp.ram import FOLLOWER, LINK_DIRECTION, LINK_MAX_HP, LINK_X, LINK_Y, assign_u8, wram_index
from alttp.rollout import PlannedPress, PolicyGuard, strike_if_safe
from alttp.room_map import load_room_map
from alttp.startup import action_for, no_action, snapshot_env, step_frames

KING = 0x92
HEART = 0xEA
FIREBALL = 0x70
ROOM = 0x5A
DOORWAY = 0x006C
MASK = 0x00EC
BOMBS = 0xF343
BOMB_ANCILLA = 0x07
CAP = 9000
SOUTH_GAP = (32, 48)
X_BAND = 24
SWORD_BAND = 16
BOMB_STOCK = 20
CLEAR = 64
_STEP = {"UP": (0, -12), "DOWN": (0, 12), "LEFT": (-12, 0), "RIGHT": (12, 0)}

__all__ = ["fight"]


def _u8(env: Any, offset: int) -> int:
    return int(env.get_ram()[wram_index(offset)])


def _u16(env: Any, offset: int) -> int:
    return _u8(env, offset) | (_u8(env, offset + 1) << 8)


def _solid(env: Any, x: int, y: int) -> bool:
    mask = _u16(env, MASK)
    col = (int(x) & mask) >> 3
    row = (int(y) & mask) >> 3
    grid = read_room_tiles(env.get_ram(), 0)
    if not (0 <= row < 64 and 0 <= col < 64):
        return True
    return int(grid[row, col]) != 0


def _put_u16(env: Any, offset: int, value: int) -> None:
    base = int(offset)
    if any(base + i == int(FOLLOWER) for i in range(2)):
        raise SystemExit("refusing to write $F3CC")
    assign_u8(env, base, int(value) & 0xFF)
    assign_u8(env, base + 1, (int(value) >> 8) & 0xFF)


def _refresh(env: Any) -> None:
    data = getattr(env, "data", None)
    refresh = getattr(data, "update_ram", None)
    if refresh is not None:
        refresh()


def _step(env: Any, action: Any, frames: list[int], count: int = 1) -> None:
    step_frames(env, np.asarray(action, dtype=np.int8), count)
    frames[0] += count


def _king(env: Any) -> Any | None:
    rows = [row for row in active_sprites(env) if int(row.sprite_type) == KING]
    return rows[0] if rows else None


def _heart(env: Any) -> Any | None:
    rows = [row for row in active_sprites(env) if int(row.sprite_type) == HEART]
    return rows[0] if rows else None


class Helmasaur:
    """Plant bombs south of the king, then take the heart container."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.phase = "stand"
        self.frames = [0]
        self.success = False
        self.failed = False
        self.blocker = ""
        self.bombs_landed = 0
        self.sword_hits = 0
        self.plants = 0
        self._follower = 0
        self._max_hp = 0
        self._floor: tuple[int, int] | None = None
        self._guard: PolicyGuard | None = None
        self._holder: PlannedPress | None = None
        self._misses = 0

    def run(self) -> FightResult:
        self._follower = _u8(self.env, FOLLOWER)
        self._max_hp = _u8(self.env, LINK_MAX_HP)
        while self.frames[0] < CAP and not self.success and not self.failed:
            if self._follower_moved() or self._busy():
                continue
            if self.phase == "stand":
                self._stand()
            elif self.phase == "fight":
                self._fight()
            else:
                self._prize()
        if not self.success and not self.failed:
            self.failed = True
            self.blocker = self.blocker or "frame cap"
        boss = _king(self.env)
        print(
            f"helmasaur phase={self.phase} frames={self.frames[0]} "
            f"bombs={self.bombs_landed} sword={self.sword_hits} plants={self.plants} "
            f"boss={None if boss is None else int(boss.hp)} "
            f"max=0x{_u8(self.env, LINK_MAX_HP):02X}",
            flush=True,
        )
        return FightResult(
            ok=self.success,
            frames=self.frames[0],
            blocker=self.blocker,
            strikes=self.bombs_landed + self.sword_hits,
            sword_hits=self.sword_hits,
        )

    def _follower_moved(self) -> bool:
        if _u8(self.env, FOLLOWER) == self._follower:
            return False
        self.failed = True
        self.blocker = "$F3CC changed"
        return True

    def _snap(self) -> Any:
        return snapshot_env(self.env)

    def _busy(self) -> bool:
        snap = self._snap()
        if int(snap.hp) <= 0:
            self.failed = True
            self.blocker = "link hp 0"
            return True
        if int(snap.room_base_id) != ROOM and self.phase != "prize":
            self.failed = True
            self.blocker = f"left room 0x{int(snap.room_base_id):02X}"
            return True
        if snap.is_text_mode or snap.is_hold_up_item or not snap.has_control:
            _step(self.env, action_for("A") if snap.is_text_mode else no_action(), self.frames)
            self._check_heart()
            return True
        if bottle_due(self.env):
            drank = drink_if_low(self.env)
            self.frames[0] += int(drank.frames)
            return True
        return False

    def _check_heart(self) -> None:
        if _u8(self.env, LINK_MAX_HP) > self._max_hp:
            self.success = True
            self.phase = "prize"

    def _stand(self) -> None:
        point = load_room_map("room_5a").point("arena_floor")
        if point is None:
            self.failed = True
            self.blocker = "maps/room_5a.json has no arena_floor"
            return
        self._floor = (int(point.x), int(point.y))
        _put_u16(self.env, LINK_X, int(point.x))
        _put_u16(self.env, LINK_Y, int(point.y))
        assign_u8(self.env, LINK_DIRECTION, 0)
        assign_u8(self.env, BOMBS, BOMB_STOCK)
        _refresh(self.env)
        _step(self.env, no_action(), self.frames, 2)
        if self._follower_moved():
            return
        picked = select_y_item(self.env, "bombs")
        self.frames[0] += int(picked.frames)
        if not picked.ok:
            self.failed = True
            self.blocker = picked.reason
            return
        holder = PlannedPress()
        guard = PolicyGuard(holder, horizon=4, detours=(4,), directions=("UP", "DOWN", "LEFT", "RIGHT"), reach=24)
        guard.bind_env(self.env)
        self._holder = holder
        self._guard = guard
        self.phase = "fight"

    def _walk(self, button: str) -> None:
        holder = self._holder
        guard = self._guard
        if holder is None or guard is None:
            self.failed = True
            self.blocker = "no walk guard"
            return
        holder.button = button
        act = guard.step(self._snap())
        _step(self.env, act.action, self.frames)

    def _sword(self, boss: Any) -> None:
        snap = self._snap()
        dist = max(abs(int(boss.x) - int(snap.link_x)), abs(int(boss.y) - int(snap.link_y)))
        if dist > SWORD_BAND or int(snap.link_state) != 0:
            return
        before = int(boss.hp)
        script = strike_if_safe(self.env, int(boss.slot))
        if not script:
            return
        for act in script:
            _step(self.env, act.action, self.frames)
        after = _king(self.env)
        if after is None or int(after.hp) < before:
            self.sword_hits += 1
            print(f"sword {before}->{None if after is None else int(after.hp)} t={self.frames[0]}", flush=True)

    def _in_band(self, snap: Any, boss: Any) -> bool:
        gap_y = int(snap.link_y) - int(boss.y)
        gap_x = abs(int(snap.link_x) - int(boss.x))
        return SOUTH_GAP[0] <= gap_y <= SOUTH_GAP[1] and gap_x <= X_BAND

    def _aim(self, snap: Any, boss: Any) -> str:
        gap_y = int(snap.link_y) - int(boss.y)
        gap_x = int(boss.x) - int(snap.link_x)
        if abs(gap_x) > X_BAND:
            return "RIGHT" if gap_x > 0 else "LEFT"
        if gap_y > SOUTH_GAP[1]:
            return "UP"
        if gap_y < SOUTH_GAP[0]:
            return "DOWN"
        return "NONE"

    def _plant(self) -> bool:
        if any(int(row.ancilla_type) == BOMB_ANCILLA for row in active_ancillae(self.env)):
            return False
        before = _king(self.env)
        if before is None:
            return False
        hp0 = int(before.hp)
        placed = place_bomb(self.env)
        self.frames[0] += int(placed.frames)
        if not placed.ok:
            self._misses += 1
            print(f"plant miss {placed.reason} t={self.frames[0]}", flush=True)
            return False
        self.plants += 1
        self._misses = 0
        bomb = next((row for row in active_ancillae(self.env) if int(row.ancilla_type) == BOMB_ANCILLA), None)
        bomb_xy = None if bomb is None else (int(bomb.x), int(bomb.y))
        for _ in range(240):
            boss = _king(self.env)
            if boss is None or int(boss.hp) < hp0:
                self.bombs_landed += 1
                print(
                    f"bomb {hp0}->{None if boss is None else int(boss.hp)} t={self.frames[0]}",
                    flush=True,
                )
                return True
            if int(self._snap().hp) <= 0:
                self.failed = True
                self.blocker = "link hp 0"
                return True
            button = self._flee(bomb_xy)
            if button == "NONE":
                _step(self.env, no_action(), self.frames)
            else:
                self._walk(button)
            if self.failed:
                return True
        print(f"bomb whiff hp {hp0} t={self.frames[0]}", flush=True)
        return False

    def _flee(self, bomb_xy: tuple[int, int] | None) -> str:
        """Step that increases distance from the bomb, then from the king and fireballs."""
        snap = self._snap()
        lx, ly = int(snap.link_x), int(snap.link_y)
        others: list[tuple[int, int]] = []
        boss = _king(self.env)
        if boss is not None:
            others.append((int(boss.x), int(boss.y)))
        for row in active_sprites(self.env):
            if int(row.sprite_type) == FIREBALL:
                others.append((int(row.x), int(row.y)))
        here = self._clear(lx, ly, bomb_xy, others)
        if here >= CLEAR:
            return "NONE"
        best = "NONE"
        score = here
        for name, (dx, dy) in _STEP.items():
            nx, ny = lx + dx, ly + dy
            if _solid(self.env, nx, ny):
                continue
            got = self._clear(nx, ny, bomb_xy, others)
            if got > score:
                score = got
                best = name
        return best

    @staticmethod
    def _clear(x: int, y: int, bomb_xy: tuple[int, int] | None, others: list[tuple[int, int]]) -> int:
        bomb = CLEAR if bomb_xy is None else max(abs(x - bomb_xy[0]), abs(y - bomb_xy[1]))
        rest = CLEAR if not others else min(max(abs(x - px), abs(y - py)) for px, py in others)
        return bomb * 2 + rest

    def _fight(self) -> None:
        self._check_heart()
        boss = _king(self.env)
        if boss is None or int(boss.hp) <= 0:
            self.phase = "prize"
            return
        if self._misses >= 8:
            self.failed = True
            self.blocker = f"bomb would not spawn, boss hp {int(boss.hp)}"
            return
        snap = self._snap()
        self._sword(boss)
        boss = _king(self.env)
        if boss is None or int(boss.hp) <= 0:
            self.phase = "prize"
            return
        snap = self._snap()
        if (
            self._in_band(snap, boss)
            and int(snap.link_state) == 0
            and _u8(self.env, DOORWAY) == 0
            and int(snap.iframes) < 8
        ):
            self._plant()
            return
        self._walk(self._aim(snap, boss))

    def _prize(self) -> None:
        self._check_heart()
        if self.success:
            return
        heart = _heart(self.env)
        if heart is None:
            _step(self.env, no_action(), self.frames)
            if self.frames[0] > CAP - 400:
                self.failed = True
                boss = _king(self.env)
                self.blocker = (
                    f"no heart boss={None if boss is None else int(boss.hp)} "
                    f"max=0x{_u8(self.env, LINK_MAX_HP):02X}"
                )
            return
        snap = self._snap()
        dx = int(heart.x) - int(snap.link_x)
        dy = int(heart.y) - int(snap.link_y)
        if abs(dx) <= 8 and abs(dy) <= 8:
            self._walk("NONE")
            return
        if abs(dx) > abs(dy):
            self._walk("RIGHT" if dx > 0 else "LEFT")
        else:
            self._walk("DOWN" if dy > 0 else "UP")


def fight(env: Any) -> FightResult:
    """Kill Helmasaur and take the heart container."""
    return Helmasaur(env).run()
