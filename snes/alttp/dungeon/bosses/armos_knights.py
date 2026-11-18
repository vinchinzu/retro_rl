"""Armos Knights in room 0xC8.

Six type ``0x53``. ``$0E50`` starts at 48. A moving knight loses 16 to an
arrow and 4 to a ``strike_if_safe`` swing. The still spawn pose does not:
an arrow into that pose halts as ancilla ``0x0A`` and HP stays 48. This
fight waits until a knight is off its spawn xy, then fires that arrow.
Walks go through ``PolicyGuard``. ``drink_if_low`` runs when a bottle is
owned; ``Whatif_Eastern_armos`` has none. ``$F3CC`` is not written.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/stage_replay.py \\
        Whatif_Eastern_armos alttp.dungeon.bosses.armos_knights:fight
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alttp.dungeon.bosses import FightResult
from alttp.primitives import (
    active_ancillae,
    active_sprites,
    bottle_due,
    drink_if_low,
    select_y_item,
)
from alttp.ram import LINK_MAX_HP, read_inventory, wram_index
from alttp.rollout import PlannedPress, PolicyGuard, strike_if_safe
from alttp.startup import action_for, no_action, snapshot_env, step_frames

ARMOS = 0x53
HEART = 0xEA
ROOM = 0xC8
DOORWAY = 0x006C
HUD_CUR_ITEM = 0x0202
HUD_BOW = 0x01
ARROW_COUNT = 0xF377
PENDANTS = 0xF374
COURAGE = 0x04
PRIZE_ANCILLA = 0x29
ARROW_FLY = 0x09
CAP = 8000
# Chebyshev off the spawn xy before an arrow is allowed. The ±2 shuffle
# is still the statue pose.
LEFT_SPAWN = 8
ALIGN = 10
NEAR = 22
FAR = 88
DODGE = 20
# South-east floor, off the statue block and off the south-center line.
HOLD = (4536, 6572)
BOX = (4404, 4544, 6468, 6580)
# Eye of the room. The last knight jumps; backing into a corner is a pin.
CENTER = (4472, 6524)
SOLO_BOX = (4424, 4520, 6488, 6560)

__all__ = ["ArmosKnights", "fight", "collect_prize", "finish_palace"]


def _u8(env: Any, offset: int) -> int:
    return int(env.get_ram()[wram_index(offset)])


def _step(env: Any, action: Any, frames: list[int], count: int = 1) -> None:
    step_frames(env, np.asarray(action, dtype=np.int8), count)
    frames[0] += count


def _press(env: Any, name: str, frames: list[int], count: int = 1) -> None:
    action = no_action() if name in ("", "NONE") else action_for(name)
    _step(env, action, frames, count)


def _knights(env: Any) -> list[Any]:
    return [row for row in active_sprites(env) if int(row.sprite_type) == ARMOS and int(row.hp) > 0]


def _hp_map(rows: list[Any]) -> dict[int, int]:
    return {int(row.slot): int(row.hp) for row in rows}


def _cheb(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(int(ax) - int(bx)), abs(int(ay) - int(by)))


def _toward(x0: int, y0: int, x1: int, y1: int) -> str:
    dx = int(x1) - int(x0)
    dy = int(y1) - int(y0)
    if dx == 0 and dy == 0:
        return "NONE"
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def _away(x0: int, y0: int, x1: int, y1: int) -> str:
    dx = int(x1) - int(x0)
    dy = int(y1) - int(y0)
    if dx == 0 and dy == 0:
        return "UP"
    if abs(dx) >= abs(dy):
        return "LEFT" if dx > 0 else "RIGHT"
    return "UP" if dy > 0 else "DOWN"


def _left_spawn(row: Any, spawn: dict[int, tuple[int, int]]) -> bool:
    origin = spawn.get(int(row.slot))
    if origin is None:
        return True
    return _cheb(row.x, row.y, origin[0], origin[1]) > LEFT_SPAWN


def _still_blocks(lx: int, ly: int, tx: int, ty: int, still: list[Any]) -> bool:
    """A spawn-pose body on the cardinal segment halts the arrow."""
    vertical = abs(tx - lx) <= abs(ty - ly)
    for row in still:
        if vertical:
            lo, hi = sorted((ly, ty))
            if lo + 8 < int(row.y) < hi - 8 and abs(int(row.x) - lx) <= 16:
                return True
        else:
            lo, hi = sorted((lx, tx))
            if lo + 8 < int(row.x) < hi - 8 and abs(int(row.y) - ly) <= 16:
                return True
    return False


def _clamp(x: int, y: int, button: str, box: tuple[int, int, int, int] = BOX) -> str:
    if button == "LEFT" and x <= box[0]:
        return "NONE"
    if button == "RIGHT" and x >= box[1]:
        return "NONE"
    if button == "UP" and y <= box[2]:
        return "NONE"
    if button == "DOWN" and y >= box[3]:
        return "NONE"
    return button


def _dodge(lx: int, ly: int, tx: int, ty: int, box: tuple[int, int, int, int]) -> str:
    """Step that stays in ``box`` and does not walk into the knight.

    Ties go to the eye of the room so the last knight cannot pin a corner.
    """
    gap = _cheb(lx, ly, tx, ty)
    best_name = "NONE"
    best_key: tuple[int, int] | None = None
    for name, sx, sy in (("UP", 0, -12), ("DOWN", 0, 12), ("LEFT", -12, 0), ("RIGHT", 12, 0)):
        if _clamp(lx, ly, name, box) != name:
            continue
        nx, ny = lx + sx, ly + sy
        nd = _cheb(nx, ny, tx, ty)
        if nd + 6 < gap:
            continue
        key = (_cheb(nx, ny, CENTER[0], CENTER[1]), -nd)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
    if best_name != "NONE":
        return best_name
    return _clamp(lx, ly, _toward(lx, ly, CENTER[0], CENTER[1]), box)


class ArmosKnights:
    """arm, then fight, then prize. ``success`` is the heart container."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.phase = "arm"
        self.frames = [0]
        self.success = False
        self.failed = False
        self.blocker = ""
        self.hits = {"arrow": 0, "sword": 0}
        self.last_knight: dict[str, object] | None = None
        self._spawn: dict[int, tuple[int, int]] = {}
        self._seen_types: set[int] = set()
        self._max_hp = 0
        self._pendants = 0
        self._guard: PolicyGuard | None = None
        self._holder: PlannedPress | None = None
        self._prev_alive = -1
        self._prev_hp: dict[int, int] = {}
        self._draw_until = 0

    def run(self) -> FightResult:
        self._remember()
        while self.frames[0] < CAP and not self.success and not self.failed:
            if self.phase == "arm":
                self._arm()
            elif self.phase == "fight":
                self._fight()
            else:
                self._prize()
        if not self.success and not self.failed:
            self.failed = True
            self.blocker = self.blocker or "frame cap"
        snap = self._snap()
        rows = _knights(self.env)
        print(
            f"armos phase={self.phase} frames={self.frames[0]} "
            f"sword={self.hits['sword']} arrow={self.hits['arrow']} "
            f"n={len(rows)} sum={sum(int(row.hp) for row in rows)} "
            f"arrows={_u8(self.env, ARROW_COUNT)} link={snap.hp} "
            f"max=0x{_u8(self.env, LINK_MAX_HP):02X} last={self.last_knight}",
            flush=True,
        )
        return FightResult(
            ok=self.success,
            frames=self.frames[0],
            blocker=self.blocker,
        )

    def _remember(self) -> None:
        inv = read_inventory(self.env.get_ram())
        self._max_hp = int(inv.max_hp)
        self._pendants = int(inv.pendants)
        for row in active_sprites(self.env):
            self._seen_types.add(int(row.sprite_type))
            if int(row.sprite_type) == ARMOS:
                self._spawn[int(row.slot)] = (int(row.x), int(row.y))
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
        self._prev_hp = _hp_map(_knights(self.env))

    def _snap(self):
        return snapshot_env(self.env)

    def _live(self) -> list[Any]:
        rows = _knights(self.env)
        if len(rows) == 1 and self._prev_alive != 1:
            row = rows[0]
            self.last_knight = {
                "slot": int(row.slot),
                "hp": int(row.hp),
                "ai": int(row.ai_state),
                "x": int(row.x),
                "y": int(row.y),
            }
            print(f"last-knight {self.last_knight}", flush=True)
        if len(rows) != self._prev_alive and self._prev_alive >= 0:
            print(
                f"alive {self._prev_alive}->{len(rows)} t={self.frames[0]}",
                flush=True,
            )
        self._prev_alive = len(rows)
        return rows

    def _note_hp(self, rows: list[Any]) -> None:
        now = _hp_map(rows)
        for slot, prev in self._prev_hp.items():
            cur = now.get(slot)
            if cur is None or cur == prev:
                continue
            if cur > prev:
                print(f"hp-reset slot={slot} {prev}->{cur} t={self.frames[0]}", flush=True)
                continue
            if prev - cur >= 8:
                self.hits["arrow"] += 1
            else:
                self.hits["sword"] += 1
            print(
                f"drop slot={slot} {prev}->{cur} t={self.frames[0]}",
                flush=True,
            )
        self._prev_hp = now

    def _busy(self) -> bool:
        snap = self._snap()
        if snap.hp <= 0:
            self.failed = True
            self.blocker = "link hp 0"
            return True
        if snap.room_base_id != ROOM and self.phase != "prize":
            self.failed = True
            self.blocker = f"left room 0x{snap.room_base_id:02X}"
            return True
        if snap.is_text_mode or snap.is_hold_up_item or not snap.has_control:
            _press(self.env, "A" if snap.is_text_mode else "NONE", self.frames)
            self._check_prize()
            return True
        if bottle_due(self.env):
            drank = drink_if_low(self.env)
            self.frames[0] += int(drank.frames)
            return True
        return False

    def _check_prize(self) -> None:
        if _u8(self.env, LINK_MAX_HP) > self._max_hp:
            self.success = True
            self.phase = "prize"

    def _arm(self) -> None:
        for _ in range(24):
            if _u8(self.env, DOORWAY) == 0:
                break
            if self._busy():
                return
            _press(self.env, "UP", self.frames)
        picked = select_y_item(self.env, "bow")
        self.frames[0] += int(picked.frames)
        if not picked.ok:
            self.failed = True
            self.blocker = picked.reason
            return
        self.phase = "fight"

    def _flying(self) -> bool:
        return any(int(row.ancilla_type) == ARROW_FLY for row in active_ancillae(self.env))

    def _loose(self, facing: str) -> bool:
        """Face a knight that has left spawn, then one Y. No spawn-pose shot."""
        if _u8(self.env, DOORWAY) != 0 or _u8(self.env, ARROW_COUNT) == 0:
            return False
        if _u8(self.env, HUD_CUR_ITEM) != HUD_BOW or self._flying():
            return False
        if self.frames[0] < self._draw_until:
            return False
        _press(self.env, facing, self.frames)
        _press(self.env, "Y", self.frames)
        self._draw_until = self.frames[0] + 14
        for _ in range(14):
            if self._flying():
                return True
            rows = _knights(self.env)
            snap = self._snap()
            near = min(
                (_cheb(row.x, row.y, snap.link_x, snap.link_y) for row in rows),
                default=99,
            )
            if near < DODGE or snap.hp <= 0:
                return True
            _press(self.env, "NONE", self.frames)
        return True

    def _sword(self, slot: int) -> bool:
        script = strike_if_safe(self.env, slot)
        if not script:
            return False
        for act in script:
            _step(self.env, act.action, self.frames)
        return True

    def _walk(self, button: str) -> None:
        assert self._holder is not None and self._guard is not None
        snap = self._snap()
        self._holder.button = _clamp(int(snap.link_x), int(snap.link_y), button)
        act = self._guard.step(snap)
        _step(self.env, act.action, self.frames)

    def _shot(
        self,
        snap: Any,
        movers: list[Any],
        still: list[Any],
        *,
        align: int = ALIGN,
        near: int = NEAR,
        far: int = FAR,
    ) -> tuple[str, Any] | None:
        best: tuple[str, Any] | None = None
        best_key: tuple[int, int, int] | None = None
        lx, ly = int(snap.link_x), int(snap.link_y)
        for row in movers:
            if int(row.hit_timer) & 0x80:
                continue
            dx = int(row.x) - lx
            dy = int(row.y) - ly
            vertical = abs(dx) <= align and near <= abs(dy) <= far
            horizontal = abs(dy) <= align and near <= abs(dx) <= far
            if vertical:
                face = "UP" if dy < 0 else "DOWN"
                flight = abs(dy)
                error = abs(dx)
            elif horizontal:
                face = "LEFT" if dx < 0 else "RIGHT"
                flight = abs(dx)
                error = abs(dy)
            else:
                continue
            if _still_blocks(lx, ly, int(row.x), int(row.y), still):
                continue
            key = (flight, int(row.hp), error)
            if best_key is None or key < best_key:
                best_key = key
                best = (face, row)
        return best

    def _align(self, snap: Any, row: Any) -> str:
        dx = int(row.x) - int(snap.link_x)
        dy = int(row.y) - int(snap.link_y)
        adx, ady = abs(dx), abs(dy)
        if adx <= ady:
            if adx > 2:
                return "RIGHT" if dx > 0 else "LEFT"
            if ady < NEAR:
                return "UP" if dy > 0 else "DOWN"
            return "NONE"
        if ady > 2:
            return "DOWN" if dy > 0 else "UP"
        if adx < NEAR:
            return "LEFT" if dx > 0 else "RIGHT"
        return "NONE"

    def _solo(self, snap: Any, row: Any) -> None:
        """Last knight. It jumps and its HP returns to 48. Stay in the eye."""
        lx, ly = int(snap.link_x), int(snap.link_y)
        gap = _cheb(lx, ly, int(row.x), int(row.y))
        off_eye = _cheb(lx, ly, CENTER[0], CENTER[1]) > 40
        if gap < 26:
            self._walk(_dodge(lx, ly, int(row.x), int(row.y), SOLO_BOX))
            return
        if self._flying() or self.frames[0] < self._draw_until:
            self._walk(_dodge(lx, ly, int(row.x), int(row.y), SOLO_BOX) if off_eye else "NONE")
            return
        arrows = _u8(self.env, ARROW_COUNT)
        shot = self._shot(snap, [row], [], align=18, near=26, far=120)
        if shot is not None and arrows > 0:
            face, _target = shot
            if self._loose(face):
                return
        if arrows == 0 and gap <= 36 and int(row.hit_timer) & 0x80 == 0:
            _press(self.env, _toward(lx, ly, int(row.x), int(row.y)), self.frames)
            if self._sword(int(row.slot)):
                return
        if off_eye:
            self._walk(_dodge(lx, ly, int(row.x), int(row.y), SOLO_BOX))
            return
        self._walk(self._align(snap, row))

    def _wake(self, snap: Any) -> str:
        if int(snap.link_x) < HOLD[0] - 4:
            return "RIGHT"
        if int(snap.link_y) < HOLD[1] - 2:
            return "DOWN"
        if int(snap.link_y) > HOLD[1] + 6:
            return "UP"
        return "NONE"

    def _fight(self) -> None:
        """Shoot only a knight that has left its spawn xy. Dodge inside the box."""
        if self._busy():
            return
        self._check_prize()
        if self.success:
            return
        snap = self._snap()
        rows = self._live()
        self._note_hp(rows)
        if not rows:
            self.phase = "prize"
            return
        if len(rows) == 1 and _left_spawn(rows[0], self._spawn):
            self._solo(snap, rows[0])
            return
        movers = [row for row in rows if _left_spawn(row, self._spawn)]
        still = [row for row in rows if row not in movers]
        nearest = min(rows, key=lambda row: _cheb(row.x, row.y, snap.link_x, snap.link_y))
        gap = _cheb(nearest.x, nearest.y, snap.link_x, snap.link_y)
        if gap < DODGE or (not movers and gap < 28):
            self._walk(_away(snap.link_x, snap.link_y, int(nearest.x), int(nearest.y)))
            return
        if self._flying() or self.frames[0] < self._draw_until:
            self._walk("NONE")
            return
        if movers:
            shot = self._shot(snap, movers, still)
            arrows = _u8(self.env, ARROW_COUNT)
            if shot is not None and arrows > 0:
                face, _row = shot
                if self._loose(face):
                    return
            if arrows == 0 and gap <= 36 and _left_spawn(nearest, self._spawn):
                if int(nearest.hit_timer) & 0x80 == 0:
                    _press(
                        self.env,
                        _toward(snap.link_x, snap.link_y, int(nearest.x), int(nearest.y)),
                        self.frames,
                    )
                    if self._sword(int(nearest.slot)):
                        return
            target = min(
                movers,
                key=lambda row: (
                    min(
                        abs(int(row.x) - int(snap.link_x)),
                        abs(int(row.y) - int(snap.link_y)),
                    ),
                    int(row.hp),
                ),
            )
            self._walk(self._align(snap, target))
            return
        self._walk(self._wake(snap))

    def _prize(self) -> None:
        if self._busy():
            return
        self._check_prize()
        if self.success:
            return
        snap = self._snap()
        if snap.room_base_id != ROOM and _u8(self.env, LINK_MAX_HP) <= self._max_hp:
            self.failed = True
            self.blocker = f"prize left room 0x{snap.room_base_id:02X}"
            return
        prize = [
            row
            for row in active_sprites(self.env)
            if int(row.sprite_type) == HEART or int(row.sprite_type) not in self._seen_types
        ]
        if not prize:
            _press(self.env, "NONE", self.frames)
            if self.frames[0] > CAP - 400:
                self.failed = True
                self.blocker = (
                    f"no heart max=0x{_u8(self.env, LINK_MAX_HP):02X} "
                    f"pendant=0x{_u8(self.env, PENDANTS):02X}"
                )
            return
        row = min(prize, key=lambda item: max(abs(item.x - snap.link_x), abs(item.y - snap.link_y)))
        self._walk(_toward(snap.link_x, snap.link_y, int(row.x), int(row.y)))


def fight(env: Any) -> FightResult:
    """Kill the six knights and take the heart container."""
    return ArmosKnights(env).run()


def collect_prize(env: Any, *, max_frames: int = 4500) -> FightResult:
    """Take the falling pendant after the heart, then finish its dialogue.

    The heart-container receipt is earlier than the pendant award. The
    pendant is ancilla ``0x29``, rather than an ordinary sprite. Its actual
    room coordinates determine the approach; no boss-drop position is
    assumed. Acceptance needs Courage's inventory bit and returned control.
    """
    frames = [0]
    while frames[0] < max_frames:
        snap = snapshot_env(env)
        if _u8(env, PENDANTS) & COURAGE and snap.has_control and snap.link_action == 0:
            return FightResult(True, frames[0])
        if snap.hp <= 0:
            return FightResult(False, frames[0], "link hp 0 before pendant")
        if snap.is_text_mode:
            # Dialogue pages consume fresh edges, not a permanently held A.
            _press(env, "A" if frames[0] % 2 == 0 else "NONE", frames)
        elif snap.is_hold_up_item or not snap.has_control:
            _press(env, "NONE", frames)
        else:
            # active_ancillae intentionally exposes Link's projectiles only.
            # Read the prize slots here without widening that shared contract.
            ram = env.get_ram()
            prizes = [
                (
                    int(ram[0x0C04 + slot]) | (int(ram[0x0C18 + slot]) << 8),
                    int(ram[0x0BFA + slot]) | (int(ram[0x0C0E + slot]) << 8),
                )
                for slot in range(10)
                if int(ram[0x0C4A + slot]) == PRIZE_ANCILLA
            ]
            if prizes:
                x, y = min(prizes, key=lambda item: abs(item[0] - snap.link_x) + abs(item[1] - snap.link_y))
                _press(env, _toward(snap.link_x, snap.link_y, x, y), frames)
            else:
                _press(env, "NONE", frames)
    return FightResult(False, frames[0], f"pendant timeout pendants=0x{_u8(env, PENDANTS):02X}")


def finish_palace(env: Any) -> FightResult:
    """Defeat Armos, earn the heart and Courage, and return control."""
    battle = fight(env)
    if not battle.ok:
        return battle
    prize = collect_prize(env)
    return FightResult(prize.ok, battle.frames + prize.frames, prize.blocker)
