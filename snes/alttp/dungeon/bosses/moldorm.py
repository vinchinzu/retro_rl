"""Moldorm on an isolated Whatif_Hera_moldorm pin. Not a continuous claim.

Room 0x07. Sword damage lands on the tail: history index
``($0E80 - 0x30) & 0x7f`` in the ``$1FC00`` ring. A rollout that changes
``$A4``, the room, or enters a fall submodule is a hit and is not committed.
Walk frames go through ``PolicyGuard``. Swings commit only through
``strike_if_safe``, and only from the placed-hit window: 8–16 px east
facing LEFT, or south facing UP. ``rel=(-4, 0)`` facing UP is a measured
whiff and is not a commit. ``drink_if_low`` is the bottle guard and no-ops
when no red or blue potion is owned.

``fight(env)`` steps the env itself. ``stage_replay`` / ``eval_offsets``
load it as ``alttp.dungeon.bosses.moldorm:fight``. A swing is committed only when
the tail is moving into the blade and ``strike_if_safe`` reports a hit.
``rel=(-4, 0)`` facing UP is not a pose. A head parked at screen Y >= 304
gets defl bit 0x80 so the AI runs again. On-screen defl is not written.
Below 3 HP, a tail sitting 0–5 px east is walked east into the fast window
before the turn. The north-rim UP ban does not apply while the heart
container is on the floor. ``rel=(-4, 0)`` facing UP stays out.
While the heart is unspawned, a committed UP at y<=220 or a step that
does not raise an edge already under 24 is replaced with a step toward center.
A head whose ``$0F00`` pause stays set for 24 frames is not chased: Link
stands 12 px east of the tail and swings. That path does not write
``$0F00`` or defl. No ``$F3CC``.

``$001A`` is 0x30 after settle on this pin and gains one per idle frame.
Offsets 3, 5, 6, and 8 refuse a rim strike pose and will not flee downhill.
Offset 0 keeps the path above.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alttp.dungeon.bosses import FightResult
from alttp.dungeon.tilemap import TILE_COLS, TILE_PX, TILE_ROWS, read_room_tiles, tile_kind
from alttp.primitives import (
    SPRITE_HIT_TIMER,
    SPRITE_HP,
    SPRITE_SLOTS,
    SPRITE_STATE,
    SPRITE_TYPE,
    SPRITE_X_HIGH,
    SPRITE_X_LOW,
    SPRITE_Y_HIGH,
    SPRITE_Y_LOW,
    active_sprites,
    drink_if_low,
)
from alttp.ram import (
    LINK_DIRECTION,
    LINK_HP,
    LINK_MAX_HP,
    assign_u8,
    read_snapshot,
    read_u16,
    wram_index,
)
from alttp.rollout import PolicyGuard, press, strike_if_safe
from retro_harness.input_script import FrameAction

MOLDORM = 0x09
HEART_CONTAINER = 0xEA
TAIL_LOOKBACK = 0x30
HIST = 0x1FC00
SUBTYPE2 = 0x0E80
AUX2 = 0x0E10
_FACE = {"UP": 0, "DOWN": 2, "LEFT": 4, "RIGHT": 6}
MAX_FRAMES = 7500
# $001A on Whatif_Hera_moldorm after settle, before any idle frame.
_RNG_AT_IDLE0 = 0x30
# These four knock Link off the arena. The other eight stay on the idle-0 path.
_RIM_OFFSETS = frozenset({3, 5, 6, 8})
FALL_SUB = frozenset({0x07, 0x14})
_STEP = {"UP": (0, -2), "DOWN": (0, 2), "LEFT": (-2, 0), "RIGHT": (2, 0)}
_ORDER = ("UP", "DOWN", "LEFT", "RIGHT")


class _Walk:
    """One held direction. PolicyGuard calls ``step``; it does not touch the ROM."""

    def __init__(self) -> None:
        self.button = "NONE"

    def step(self, _snap: object) -> FrameAction:
        frame = press() if self.button in (None, "NONE") else press(self.button)
        return FrameAction(list(frame), "walk")

    def report(self) -> dict[str, object]:
        return {"button": self.button}


def _ram(env: Any) -> np.ndarray:
    return np.asarray(env.get_ram(), dtype=np.uint8)


def _u8(ram: np.ndarray, offset: int) -> int:
    return int(ram[wram_index(offset)])


def _sprite_xy(ram: np.ndarray, slot: int) -> tuple[int, int, int] | None:
    if int(ram[SPRITE_STATE + slot]) == 0:
        return None
    if int(ram[SPRITE_TYPE + slot]) != MOLDORM:
        return None
    x = int(ram[SPRITE_X_LOW + slot]) | (int(ram[SPRITE_X_HIGH + slot]) << 8)
    y = int(ram[SPRITE_Y_LOW + slot]) | (int(ram[SPRITE_Y_HIGH + slot]) << 8)
    return x, y, int(ram[SPRITE_HP + slot])


def _hist(ram: np.ndarray, slot: int, lookback: int) -> tuple[int, int]:
    index = (_u8(ram, SUBTYPE2 + slot) - lookback) & 0x7F
    x = _u8(ram, HIST + index) | (_u8(ram, HIST + 0x80 + index) << 8)
    y = _u8(ram, HIST + 0x100 + index) | (_u8(ram, HIST + 0x180 + index) << 8)
    return x, y


def _tail_xy(ram: np.ndarray, slot: int) -> tuple[int, int]:
    return _hist(ram, slot, TAIL_LOOKBACK)


def _window(dx: int, dy: int, hp: int = 12) -> str | None:
    """Placed B-only cells. ``dx, dy`` are Link minus the tail.

    East 8–16 facing LEFT, or south facing UP. Below 3 HP the tail is fast
    and the east edge starts at 6. ``(-4, 0)`` is the walked whiff.
    """
    # dx 16 with |dy| >= 2 is the walked whiff that phase-locks. (16, 1) still hits.
    east = 6 if hp < 3 else 8
    if east <= dx <= 16 and abs(dy) <= 4 and not (dx >= 16 and abs(dy) >= 2):
        return "LEFT"
    # dy 4–7 facing UP is the walked phase-lock (measured rel=(2, 5)).
    if -2 <= dx <= 2 and 8 <= dy <= 16:
        return "UP"
    return None


def _cheb(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(ax - bx), abs(ay - by))


def _pits(grid: np.ndarray) -> list[tuple[int, int]]:
    found = []
    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            if int(grid[row, col]) == 0x20 or tile_kind(int(grid[row, col])) == "pit":
                found.append((col, row))
    return found


def _edge_px(lx: int, ly: int, pits: list[tuple[int, int]]) -> int:
    best = 10**9
    for col, row in pits:
        dx = max(col * TILE_PX - lx, lx - (col * TILE_PX + TILE_PX - 1), 0)
        dy = max(row * TILE_PX - ly, ly - (row * TILE_PX + TILE_PX - 1), 0)
        best = min(best, max(dx, dy))
    return best


def _center(ram: np.ndarray, boss: tuple[int, int]) -> tuple[int, int]:
    """World pixel of the floor cell farthest from a pit, near the boss."""
    snap = read_snapshot(ram)
    mask = read_u16(ram, wram_index(0x00EC)) or 0x1FF
    grid = read_room_tiles(ram, snap.layer)
    pits = _pits(grid)
    bx = (boss[0] & mask) // TILE_PX
    by = (boss[1] & mask) // TILE_PX
    seen = np.zeros(grid.shape, dtype=np.uint8)
    best_cells: list[tuple[int, int]] = []
    best_key = 10**9
    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            if seen[row, col] or tile_kind(int(grid[row, col])) != "floor":
                continue
            stack = [(col, row)]
            seen[row, col] = 1
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if not (0 <= nx < TILE_COLS and 0 <= ny < TILE_ROWS):
                        continue
                    if seen[ny, nx] or tile_kind(int(grid[ny, nx])) != "floor":
                        continue
                    seen[ny, nx] = 1
                    stack.append((nx, ny))
            key = min(_cheb(c, r, bx, by) for c, r in cells)
            if key < best_key:
                best_key = key
                best_cells = cells
    if not best_cells or not pits:
        return boss
    stand = max(best_cells, key=lambda cell: _edge_px(cell[0] * TILE_PX + 4, cell[1] * TILE_PX + 4, pits))
    origin_x = int(snap.link_x) & ~mask
    origin_y = int(snap.link_y) & ~mask
    return origin_x + stand[0] * TILE_PX + 4, origin_y + stand[1] * TILE_PX + 4


def _fell(ram: np.ndarray, floor0: int, room0: int) -> bool:
    snap = read_snapshot(ram)
    if (int(snap.floor) & 0xFF) != (floor0 & 0xFF):
        return True
    if int(snap.room_id) != room0:
        return True
    if int(snap.submodule) in FALL_SUB and int(snap.game_mode) == 0x07:
        return True
    return False


def _play_raw(env: Any, frame: tuple[int, ...]) -> None:
    env.em.set_button_mask(np.asarray(frame, dtype=np.uint8), 0)
    env.em.step()


def _roll(env: Any, frames: tuple[tuple[int, ...], ...], floor0: int, room0: int) -> str:
    """Step ``frames`` and restore. ``floor`` is a hit, checked before HP."""
    state = bytes(env.em.get_state())
    hp0 = _u8(_ram(env), LINK_HP)
    try:
        for frame in frames:
            _play_raw(env, frame)
            ram = _ram(env)
            if _fell(ram, floor0, room0):
                return "floor"
            if _u8(ram, LINK_HP) < hp0:
                return "hurt"
        return "ok"
    finally:
        env.em.set_state(state)


def _tail_preview(env: Any, slot: int, floor0: int, room0: int) -> str:
    """B-only swing, restored. ``hit`` is a boss HP drop, not a shove."""
    state = bytes(env.em.get_state())
    before = _sprite_xy(_ram(env), slot)
    hp0 = _u8(_ram(env), LINK_HP)
    if before is None:
        return "gone"
    script = (press("B"),) + (press(),) * 12
    hit = False
    try:
        for frame in script:
            _play_raw(env, frame)
            ram = _ram(env)
            if _fell(ram, floor0, room0):
                return "floor"
            if _u8(ram, LINK_HP) < hp0:
                return "hurt"
            viewed = _sprite_xy(ram, slot)
            if viewed is None or viewed[2] < before[2]:
                hit = True
        return "hit" if hit else "whiff"
    finally:
        env.em.set_state(state)


def _toward(x: int, y: int, goal: tuple[int, int]) -> str | None:
    dx = int(goal[0]) - int(x)
    dy = int(goal[1]) - int(y)
    if max(abs(dx), abs(dy)) <= 1:
        return None
    if abs(dx) >= abs(dy) and dx != 0:
        return "RIGHT" if dx > 0 else "LEFT"
    if dy != 0:
        return "DOWN" if dy > 0 else "UP"
    return None


def _moldorm(env: Any) -> tuple[int, tuple[int, int], tuple[int, int], int] | None:
    ram = _ram(env)
    for slot in range(SPRITE_SLOTS):
        row = _sprite_xy(ram, slot)
        if row is None or row[2] <= 0:
            continue
        return slot, (row[0], row[1]), _tail_xy(ram, slot), row[2]
    return None


def _zero_hp_head(env: Any) -> tuple[int, int] | None:
    """Slot and world Y of a Moldorm whose HP is already 0.

    ``_moldorm`` skips that sprite, so the off-camera wake would stop on the
    frame the last hit lands. The prize routine still needs the AI to run.
    """
    ram = _ram(env)
    for slot in range(SPRITE_SLOTS):
        row = _sprite_xy(ram, slot)
        if row is None or row[2] != 0:
            continue
        return slot, row[1]
    return None


def _heart_xy(env: Any) -> tuple[int, int] | None:
    for sprite in active_sprites(env):
        if sprite.sprite_type == HEART_CONTAINER:
            return sprite.x, sprite.y
    return None


def _heart_taken(ram: np.ndarray, max0: int) -> bool:
    return _u8(ram, LINK_MAX_HP) > max0


def _ready(ram: np.ndarray, slot: int) -> bool:
    stun = int(ram[SPRITE_HIT_TIMER + slot]) & 0x80
    return (not stun) and _u8(ram, AUX2 + slot) == 0


def _wake_head(env: Any, ram: np.ndarray, slot: int, head_y: int) -> None:
    """If the head is parked off-camera, set defl bit 0x80 so it walks back.

    Tail draw stores defl as 4. Screen Y >= 304 then leaves ``$0F00`` set
    and the AI never runs again. The bit is left alone while the head is
    on screen, so placed hits stay on the vanilla defl value. No ``$F3CC``.
    """
    if int(ram[0x0F00 + int(slot)]) == 0:
        return
    cam_y = read_u16(ram, wram_index(0x00E8))
    screen_y = (int(head_y) - cam_y) & 0xFFFF
    if screen_y < 304:
        return
    data = getattr(env, "data", None)
    memory = getattr(data, "memory", None)
    blocks = getattr(memory, "blocks", None)
    if not blocks or 0x7E0000 not in blocks:
        return
    addr = 0x0CAA + int(slot)
    current = int(blocks[0x7E0000][addr])
    if current & 0x80:
        return
    assign_u8(env, addr, current | 0x80)


def _index_step(hp: int) -> int:
    """History index rate. Below 3 HP the head writes two samples a frame."""
    return 2 if hp < 3 else 1


def _poses(tail: tuple[int, int], ahead: tuple[int, int]) -> list[tuple[tuple[int, int], str]]:
    """Strike cells the tail is not running away from."""
    vx = ahead[0] - tail[0]
    vy = ahead[1] - tail[1]
    found: list[tuple[tuple[int, int], str]] = []
    if vx > -8:
        found.append(((tail[0] + 12, tail[1]), "LEFT"))
    if vy > -8:
        found.append(((tail[0], tail[1] + 10), "UP"))
    return found


def _into_blade(ram: np.ndarray, slot: int, hp: int, face: str) -> bool:
    """True when the tail's next 8 samples move into ``face``."""
    step = _index_step(hp)
    look = TAIL_LOOKBACK - 8 * step
    if look < 8:
        return False
    tail = _hist(ram, slot, TAIL_LOOKBACK)
    ahead = _hist(ram, slot, look)
    if face == "LEFT":
        return ahead[0] - tail[0] >= 0
    return ahead[1] - tail[1] >= 0


def _intercept(
    link: tuple[int, int],
    ram: np.ndarray,
    slot: int,
    hp: int,
    head: tuple[int, int],
    center: tuple[int, int],
    mask: int,
    pits: list[tuple[int, int]],
) -> tuple[tuple[int, int], str] | None:
    """Pose on the tail's future path that Link can reach and still stand on."""
    step = 2 if hp < 3 else 1
    best: tuple[tuple[int, int], tuple[int, int], str] | None = None
    for lead in range(0, 28):
        look = TAIL_LOOKBACK - lead * step
        if look < 8:
            break
        tail = _hist(ram, slot, look)
        ahead = _hist(ram, slot, look - 8)
        for pose, face in _poses(tail, ahead):
            if _edge_px(pose[0] & mask, pose[1] & mask, pits) < 24:
                continue
            if _cheb(pose[0], pose[1], head[0], head[1]) < 22:
                continue
            if _cheb(pose[0], pose[1], center[0], center[1]) > 40:
                continue
            dist = _cheb(link[0], link[1], pose[0], pose[1])
            budget = (lead * 3) // 2 + 3
            if dist > budget:
                continue
            key = (lead, dist)
            if best is None or key < best[0]:
                best = (key, pose, face)
    if best is None:
        return None
    return best[1], best[2]


def _face_step(link: tuple[int, int], pose: tuple[int, int], face: str, facing: int) -> str | None:
    """Last steps travel along ``face`` so the swing starts aimed at the tail."""
    dx = pose[0] - link[0]
    dy = pose[1] - link[1]
    if face == "LEFT":
        if abs(dy) > 2:
            return "DOWN" if dy > 0 else "UP"
        if dx > 2:
            return "RIGHT"
        if dx < -1:
            return "LEFT"
        if facing != _FACE["LEFT"]:
            return "LEFT"
        return None
    if abs(dx) > 2:
        return "RIGHT" if dx > 0 else "LEFT"
    if dy > 2:
        return "DOWN"
    if dy < -1:
        return "UP"
    if facing != _FACE["UP"]:
        return "UP"
    return None


def _safe_dirs(
    link: tuple[int, int],
    primary: str | None,
    mask: int,
    pits: list[tuple[int, int]],
    center: tuple[int, int],
    descend: bool,
    pit_floor: int = 16,
    hold_at: int = 28,
) -> list[str]:
    """Directions that stay off pits. ``descend`` allows the heart walk downhill."""
    current = _edge_px(link[0] & mask, link[1] & mask, pits)
    names: list[str] = []
    if primary is not None:
        names.append(primary)
    back = _toward(link[0], link[1], center)
    if back and back not in names:
        names.append(back)
    for name in _ORDER:
        if name not in names:
            names.append(name)
    ok: list[str] = []
    for name in names:
        sx, sy = _STEP[name]
        edge = _edge_px((link[0] + sx) & mask, (link[1] + sy) & mask, pits)
        if edge < (8 if descend else pit_floor):
            continue
        if not descend and current < hold_at and edge < current:
            continue
        ok.append(name)
    return ok


def _step_env(env: Any, frame: tuple[int, ...] | np.ndarray) -> None:
    env.step(np.asarray(frame, dtype=np.uint8))


def _dir_of(frame: tuple[int, ...]) -> str:
    index = (("UP", 4), ("DOWN", 5), ("LEFT", 6), ("RIGHT", 7))
    held = [name for name, at in index if at < len(frame) and int(frame[at])]
    if len(held) != 1:
        return "NONE"
    return held[0]


def _retreat(
    link: tuple[int, int],
    blocked: str,
    mask: int,
    pits: list[tuple[int, int]],
    center: tuple[int, int],
    y_cut: int = 220,
    edge_hold: int = 24,
) -> str:
    """One step toward center that does not repeat a rim or pit step."""
    current = _edge_px(link[0] & mask, link[1] & mask, pits)
    names: list[str] = []
    back = _toward(link[0], link[1], center)
    if back and back != blocked:
        names.append(back)
    for name in _ORDER:
        if name != blocked and name not in names:
            names.append(name)
    for name in names:
        if name == "UP" and link[1] <= y_cut:
            continue
        sx, sy = _STEP[name]
        edge = _edge_px((link[0] + sx) & mask, (link[1] + sy) & mask, pits)
        if edge < 16 or (current < edge_hold and edge <= current):
            continue
        return name
    return "NONE"


def _rng_offset(ram: np.ndarray) -> int:
    """Idle frames already stepped. ``$001A`` is 0x30 at idle 0 on this pin."""
    return (_u8(ram, 0x001A) - _RNG_AT_IDLE0) & 0xFF


def _rim_goal(
    link: tuple[int, int],
    goal: tuple[int, int],
    head: tuple[int, int],
    center: tuple[int, int],
    mask: int,
    pits: list[tuple[int, int]],
) -> tuple[int, int]:
    """Keep a flee that holds the pit edge. A downhill flee slides along the rim."""
    current = _edge_px(link[0] & mask, link[1] & mask, pits)
    if _edge_px(int(goal[0]) & mask, int(goal[1]) & mask, pits) >= current:
        return goal
    sx = 16 if link[0] >= head[0] else -16
    sy = 16 if link[1] >= head[1] else -16
    best = center
    best_key = (-1, -1)
    for cand in (
        (link[0] + sx, link[1]),
        (link[0], link[1] + sy),
        (link[0] + sy, link[1]),
        (link[0] - sy, link[1]),
        (link[0], link[1] + sx),
        (link[0], link[1] - sx),
        center,
    ):
        edge = _edge_px(int(cand[0]) & mask, int(cand[1]) & mask, pits)
        if edge < current:
            continue
        key = (edge, _cheb(cand[0], cand[1], head[0], head[1]))
        if key > best_key:
            best_key = key
            best = cand
    return best


class Moldorm:
    def __init__(self, env: Any) -> None:
        self.env = env

    def run(self) -> FightResult:
        env = self.env
        ram0 = _ram(env)
        snap0 = read_snapshot(ram0)
        floor0 = int(snap0.floor) & 0xFF
        room0 = int(snap0.room_id)
        max0 = _u8(ram0, LINK_MAX_HP)
        # Idle offset from $001A. False on offset 0, so that fight is unchanged.
        rim = _rng_offset(ram0) in _RIM_OFFSETS
        mask = read_u16(ram0, wram_index(0x00EC)) or 0x1FF
        live = _moldorm(env)
        center = _center(ram0, live[1] if live else (int(snap0.link_x), int(snap0.link_y)))
        grid = read_room_tiles(ram0, snap0.layer)
        pits = _pits(grid)
        walk = _Walk()
        guard = PolicyGuard(walk, reach=18, horizon=10)
        guard.bind_env(env)
        strikes = 0
        floor_hits = 0
        frames = 0
        whiff_cd = 0
        leave_cd = 0
        tap_fails = 0
        park = 0
        still = 0
        frozen_pose: tuple[int, int, int, int] | None = None
        script: list[tuple[int, ...]] = []
        while frames < MAX_FRAMES:
            ram = _ram(env)
            awake = _moldorm(env)
            if awake is not None:
                _wake_head(env, ram, awake[0], awake[1][1])
            else:
                dead = _zero_hp_head(env)
                if dead is not None:
                    _wake_head(env, ram, dead[0], dead[1])
            snap = read_snapshot(ram)
            if _heart_taken(ram, max0):
                print(
                    f"moldorm done frames={frames} strikes={strikes} floor_hits={floor_hits}",
                    flush=True,
                )
                return FightResult(
                    ok=True,
                    frames=frames,
                    strikes=strikes,
                    floor_hits=floor_hits,
                )
            if _fell(ram, floor0, room0):
                floor_hits += 1
                print(
                    f"moldorm floor hit frames={frames} strikes={strikes} floor_hits={floor_hits}",
                    flush=True,
                )
                return FightResult(
                    ok=False,
                    frames=frames,
                    blocker="floor",
                    strikes=strikes,
                    floor_hits=floor_hits,
                )
            if int(snap.hp) == 0:
                return FightResult(
                    ok=False,
                    frames=frames,
                    blocker="dead",
                    strikes=strikes,
                    floor_hits=floor_hits,
                )
            if script:
                _step_env(env, script.pop(0))
                frames += 1
                continue
            if not snap.has_control or int(snap.knockback):
                walk.button = "NONE"
                _step_env(env, press())
                frames += 1
                continue
            bottle = drink_if_low(env)
            frames += int(bottle.frames)
            if frames >= MAX_FRAMES:
                break
            ram = _ram(env)
            if _heart_taken(ram, max0) or _fell(ram, floor0, room0):
                continue
            if whiff_cd:
                whiff_cd -= 1
            boss = _moldorm(env)
            heart = _heart_xy(env)
            snap = read_snapshot(ram)
            link = (int(snap.link_x), int(snap.link_y))
            facing = _u8(ram, LINK_DIRECTION)
            goal = center
            face: str | None = None
            window = ""
            rel = (0, 0)
            head_d = -1
            head = link
            tail = link
            frozen = False
            if boss is not None and heart is None:
                slot, head, tail, hp = boss
                rel = (link[0] - tail[0], link[1] - tail[1])
                window = _window(rel[0], rel[1], hp) or ""
                head_d = _cheb(link[0], link[1], head[0], head[1])
                pose = (head[0], head[1], tail[0], tail[1])
                paused = int(ram[0x0F00 + slot]) != 0
                if paused and pose == frozen_pose:
                    still += 1
                else:
                    still = 0
                    frozen_pose = pose
                frozen = still >= 24
                if leave_cd:
                    leave_cd -= 1
                # A leaving tail in the window whiffs and, if we retreat to
                # center, the next pass repeats that cell. Slide instead.
                # A paused head keeps the tail still, so the blade check is the window.
                into = bool(window) and (frozen or _into_blade(ram, slot, hp, window))
                # The last two HP sit inside the flee radius. Do not walk off the blade.
                finish = hp < 3 and bool(window) and facing == _FACE[window]
                if not frozen and head_d < 20 and not finish:
                    away = (link[0] - head[0], link[1] - head[1])
                    if away == (0, 0):
                        away = (center[0] - head[0], center[1] - head[1])
                    goal = (link[0] + (16 if away[0] >= 0 else -16), link[1] + (16 if away[1] >= 0 else -16))
                    if rim:
                        goal = _rim_goal(link, goal, head, center, mask, pits)
                    tap_fails = 0
                elif leave_cd:
                    goal = center
                    tap_fails = 0
                elif (into or hp < 3) and bool(window) and facing != _FACE[window]:
                    # One LEFT step from dx < 10 leaves the blade. Step east first.
                    if window == "LEFT" and rel[0] < 10:
                        tap_fails = 0
                        goal = (link[0] + 8, link[1])
                    elif tap_fails >= 6:
                        leave_cd = 24
                        tap_fails = 0
                        goal = center
                    else:
                        face = window
                        goal = link
                        tap_fails += 1
                elif (into or hp < 3) and bool(window) and _ready(ram, slot) and whiff_cd == 0 and facing == _FACE[window]:
                    tap_fails = 0
                    preview = _tail_preview(env, slot, floor0, room0)
                    if preview == "hit":
                        committed = strike_if_safe(env, slot)
                        if committed:
                            script = [tuple(int(v) for v in act.action) for act in committed]
                            strikes += 1
                            continue
                        # Preview can see the tail hit while the full swing gets Link hurt.
                        # Standing in that cell is how offsets 5 and 8 fall off the north rim.
                        if rim:
                            whiff_cd = 8
                            leave_cd = 12
                            goal = _rim_goal(link, center, head, center, mask, pits)
                    elif preview == "hurt":
                        goal = center
                        whiff_cd = 8
                        leave_cd = 24
                    else:
                        whiff_cd = 8
                        if frozen:
                            goal = (tail[0] + 12, tail[1])
                        else:
                            slide = 8 if ((strikes + frames // 16) % 2 == 0) else -8
                            if window == "LEFT":
                                goal = (link[0], link[1] + slide)
                            else:
                                goal = (link[0] + slide, link[1])
                else:
                    tap_fails = 0
                    # The fast window starts at dx 6. A tail parked at dx 0–5 never
                    # opens it, so step east until the turn branch can take over.
                    # A paused head leaves the tail short of that window too.
                    if frozen:
                        park = 0
                        goal = link if window else (tail[0] + 12, tail[1])
                    elif hp < 3 and abs(rel[1]) <= 4 and 0 <= rel[0] < 6:
                        park = 0
                        goal = (link[0] + 8, link[1])
                    else:
                        # A tail leaving the blade never trips ``into``, and intercept
                        # will stand on that same cell. Slide off after a short park.
                        if window and not into:
                            park += 1
                        else:
                            park = 0
                        if park >= 12:
                            slide = 8 if ((strikes + park // 12) % 2 == 0) else -8
                            if window == "LEFT":
                                goal = (link[0], link[1] + slide)
                            else:
                                goal = (link[0] + slide, link[1])
                        else:
                            picked = _intercept(link, ram, slot, hp, head, center, mask, pits)
                            if picked is not None:
                                goal, face = picked
                                pose_edge = _edge_px(goal[0] & mask, goal[1] & mask, pits)
                                if rim and (pose_edge < 36 or goal[1] <= 240):
                                    goal, face = center, None
            elif heart is not None:
                goal = heart
            primary = _face_step(link, goal, face, facing) if face else _toward(link[0], link[1], goal)
            chosen = "NONE"
            current_edge = _edge_px(link[0] & mask, link[1] & mask, pits)
            # Arrived. Do not step toward center or the pose is abandoned.
            descend = heart is not None
            if rim:
                options = (
                    ()
                    if primary is None
                    else _safe_dirs(
                        link, primary, mask, pits, center, descend, pit_floor=28, hold_at=36
                    )
                )
            else:
                options = () if primary is None else _safe_dirs(link, primary, mask, pits, center, descend)
            y_cut = 240 if rim else 220
            for name in options:
                # North of this line the room loads 0x17. The pit edge does not see it.
                # The heart can sit on that side; the pickup walk is allowed through.
                if name == "UP" and link[1] <= y_cut and not descend:
                    continue
                sx, sy = _STEP[name]
                edge = _edge_px((link[0] + sx) & mask, (link[1] + sy) & mask, pits)
                if edge < 28 or current_edge < 32:
                    rolled = _roll(env, (press(name),) * 4, floor0, room0)
                    if rolled == "floor":
                        floor_hits += 1
                        continue
                chosen = name
                break
            if chosen == "NONE" and current_edge < 36:
                rescue = -1
                rescue_dirs = (
                    _safe_dirs(
                        link,
                        _toward(link[0], link[1], center),
                        mask,
                        pits,
                        center,
                        False,
                        pit_floor=28,
                        hold_at=36,
                    )
                    if rim
                    else _safe_dirs(link, _toward(link[0], link[1], center), mask, pits, center, False)
                )
                for name in rescue_dirs:
                    if name == "UP" and link[1] <= y_cut and heart is None:
                        continue
                    sx, sy = _STEP[name]
                    edge = _edge_px((link[0] + sx) & mask, (link[1] + sy) & mask, pits)
                    rolled = _roll(env, (press(name),) * 4, floor0, room0)
                    if rolled == "floor":
                        floor_hits += 1
                        continue
                    if edge > rescue:
                        rescue = edge
                        chosen = name
            walk.button = chosen
            # Facing the blade is a tap, not a walk. PolicyGuard holds the old facing
            # while the fast head sits inside its horizon.
            turning = (
                bool(window)
                and chosen == window
                and facing != _FACE[window]
            )
            if turning:
                step_frame = press(chosen)
            else:
                act = guard.step(read_snapshot(_ram(env)))
                step_frame = tuple(int(v) for v in act.action)
            # PolicyGuard can commit the rim step the ban already refused.
            # Passing offsets never press it while the boss is alive.
            if heart is None:
                held = _dir_of(step_frame)
                bad = held == "UP" and link[1] <= y_cut
                edge_hold = 36 if rim else 24
                if not bad and held in _STEP and current_edge < edge_hold:
                    sx, sy = _STEP[held]
                    nxt = _edge_px((link[0] + sx) & mask, (link[1] + sy) & mask, pits)
                    bad = nxt <= current_edge
                if bad:
                    guard.drop_detour()
                    if rim:
                        held = _retreat(link, held, mask, pits, center, y_cut=240, edge_hold=36)
                    else:
                        held = _retreat(link, held, mask, pits, center)
                    chosen = held
                    walk.button = held
                    step_frame = press() if held == "NONE" else press(held)
            _step_env(env, step_frame)
            frames += 1
            if frames % 400 == 0:
                boss_hp = boss[3] if boss is not None else -1
                print(
                    f"moldorm f={frames} boss_hp={boss_hp} strikes={strikes} "
                    f"floor_hits={floor_hits} link={link} edge={current_edge} "
                    f"rel={rel} head={head_d} win={window} face={facing} btn={chosen} "
                    f"link_hp={int(snap.hp)}",
                    flush=True,
                )
        print(
            f"moldorm cap frames={frames} strikes={strikes} floor_hits={floor_hits}",
            flush=True,
        )
        return FightResult(
            ok=False,
            frames=frames,
            blocker="frame cap",
            strikes=strikes,
            floor_hits=floor_hits,
        )

def fight(env: Any) -> FightResult:
    """Kill Moldorm and take the heart container. Floor changes count as hits."""
    return Moldorm(env).run()

