"""Lanmolas on an isolated Whatif_Desert_lanmolas pin. Not a continuous claim.

Room 0x33. Three sprite 0x54. Sword damage is checked only while ``$0D80``
is 2 or 3 and ``$0F70`` is low. State 4 is buried. Commits land at origin
distance 21–25 while ``$0F70`` is 1–11, and only on a cardinal: one axis
in that band, the other near zero. Indoor boulder ``0xC2`` moves on the
four diagonals; ``$0D50`` is X and ``$0D40`` is Y, and pixel speed is that
signed byte over 16. A boulder whose dx and dy are both nonzero is on a
diagonal ray: step off that ray before the box arrives. A step that enters
a worm is not a dodge. Swings commit only through ``strike_if_safe``. Walk
frames go through ``PolicyGuard``. ``$F3CC`` is not written.

``fight(env)`` is ``alttp.dungeon.bosses.lanmolas:fight`` for stage_replay.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alttp.dungeon.bosses import FightResult
from alttp.primitives import SPRITE_AI, SPRITE_HP, active_sprites, drink_if_low
from alttp.ram import FOLLOWER, LINK_MAX_HP, read_snapshot, wram_index
from alttp.rollout import PolicyGuard, press, strike_if_safe
from retro_harness.input_script import FrameAction

LANMOLAS = 0x54
ROCK = 0xC2
HEART = 0xEA
ROOM = 0x33
Z = 0x0F70
DELAY = 0x0DF0
VEL_Y = 0x0D40
VEL_X = 0x0D50
MAX_FRAMES = 12000
# Left-pit floor, inset off the center wall and the south door.
_X0, _X1 = 1584, 1728
_Y0, _Y1 = 1856, 1984
# South floor. Idle here survived the opening rocks on the poke.
_CAMP = (1664, 1960)
_FACE = {"UP": 0, "DOWN": 2, "LEFT": 4, "RIGHT": 6}
_NAME = {0: "UP", 2: "DOWN", 4: "LEFT", 6: "RIGHT"}
_AWAY = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
# Face vector points from Link toward the worm. The stand is the opposite tile.
_CARD = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
# Cardinal sword window. Cross-axis wider than this is a diagonal, not a gap.
_AXIS = 6
_NEAR = 21
_FAR = 25
_Z_LO = 1
_Z_HI = 11
_STAND = 23
# Origin chebyshev 17 with a low head was a body hit. Do not step inside 20.
_BODY = 20
_HEAD = 14
_DIRS = ("UP", "DOWN", "LEFT", "RIGHT")
# Frames of boulder travel to leave before contact. Link is the slow side.
_LEAD = 28
_WORM_LOOK = 4


class _Walk:
    """One held direction. PolicyGuard calls ``step``; it does not touch RAM."""

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


def _s8(value: int) -> int:
    return int(value) - 256 if int(value) >= 128 else int(value)


def _cheb(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(ax - bx), abs(ay - by))


def _toward(dx: int, dy: int) -> str:
    if abs(dx) >= abs(dy) and dx != 0:
        return "RIGHT" if dx > 0 else "LEFT"
    if dy > 0:
        return "DOWN"
    if dy < 0:
        return "UP"
    return "DOWN"


def _worms(env: Any) -> list[dict[str, int]]:
    ram = _ram(env)
    rows = []
    for spr in active_sprites(env):
        if int(spr.sprite_type) != LANMOLAS or int(spr.hp) <= 0:
            continue
        slot = int(spr.slot)
        rows.append(
            {
                "slot": slot,
                "x": int(spr.x),
                "y": int(spr.y),
                "hp": int(spr.hp),
                "ai": int(ram[SPRITE_AI + slot]),
                "z": int(ram[Z + slot]),
                "delay": int(ram[DELAY + slot]),
                "vx": _s8(int(ram[VEL_X + slot])),
                "vy": _s8(int(ram[VEL_Y + slot])),
            }
        )
    return rows


def _rocks(env: Any) -> list[tuple[int, int, int, int, int]]:
    ram = _ram(env)
    rows = []
    for spr in active_sprites(env):
        if int(spr.sprite_type) != ROCK:
            continue
        slot = int(spr.slot)
        rows.append(
            (
                slot,
                int(spr.x),
                int(spr.y),
                _s8(int(ram[VEL_X + slot])),
                _s8(int(ram[VEL_Y + slot])),
            )
        )
    return rows


def _nearest(env: Any, kind: int, link: tuple[int, int]) -> tuple[int, int] | None:
    best = None
    dist = 999
    for spr in active_sprites(env):
        if int(spr.sprite_type) != kind:
            continue
        here = _cheb(int(spr.x), int(spr.y), link[0], link[1])
        if here < dist:
            dist = here
            best = (int(spr.x), int(spr.y))
    return best


def _rock_xy(rock: tuple[int, int, int, int, int], t: int) -> tuple[int, int]:
    _slot, x, y, vx, vy = rock
    return x + (vx * t) // 16, y + (vy * t) // 16


def _in_rock(px: int, py: int, lx: int, ly: int, margin: int = 0) -> bool:
    """Sprite_C2_Boulder. ``link_x`` in ``[x-11, x+4]``, ``link_y`` in ``[y-15, y-4]``."""
    return (px - 11 - margin) <= lx <= (px + 4 + margin) and (py - 15 - margin) <= ly <= (py - 4 + margin)


def _rock_eta(
    rocks: list[tuple[int, int, int, int, int]],
    lx: int,
    ly: int,
    horizon: int = 14,
    margin: int = 0,
) -> int:
    """Frames until a boulder box covers ``(lx, ly)``. ``horizon + 1`` if clear."""
    eta = horizon + 1
    for rock in rocks:
        for t in range(horizon):
            rx, ry = _rock_xy(rock, t)
            if _in_rock(rx, ry, lx, ly, margin):
                eta = min(eta, t)
                break
    return eta


def _body_sep(worms: list[dict[str, int]], lx: int, ly: int) -> int:
    """Origin chebyshev of the nearest low head. High ``$0F70`` is not the body."""
    best = 999
    for worm in worms:
        if worm["ai"] not in (2, 3) or worm["z"] > 16:
            continue
        best = min(best, _cheb(worm["x"], worm["y"], lx, ly))
    return best


def _inside(x: int, y: int) -> bool:
    return _X0 <= x <= _X1 and _Y0 <= y <= _Y1


def _toward_floor(x: int, y: int, face: str | None) -> bool:
    """A step that stays on the floor, or walks back onto it."""
    nx, ny = _nudge(x, y, face)
    if face is None or _inside(nx, ny):
        return True
    if x < _X0 and face == "RIGHT":
        return True
    if x > _X1 and face == "LEFT":
        return True
    if y < _Y0 and face == "DOWN":
        return True
    if y > _Y1 and face == "UP":
        return True
    return False


def _recover(link: tuple[int, int]) -> str | None:
    x, y = link
    if x < _X0 + 4:
        return "RIGHT"
    if x > _X1 - 4:
        return "LEFT"
    if y < _Y0 + 4:
        return "DOWN"
    if y > _Y1 - 4:
        return "UP"
    return None


def _nudge(x: int, y: int, face: str | None, pixels: int = 2) -> tuple[int, int]:
    if face == "LEFT":
        return x - pixels, y
    if face == "RIGHT":
        return x + pixels, y
    if face == "UP":
        return x, y - pixels
    if face == "DOWN":
        return x, y + pixels
    return x, y


def _clamp(link: tuple[int, int], face: str | None) -> str | None:
    if face is None:
        return None
    x, y = link
    blocked = {
        "LEFT": x <= _X0 + 6,
        "RIGHT": x >= _X1 - 6,
        "UP": y <= _Y0 + 6,
        "DOWN": y >= _Y1 - 6,
    }
    if not blocked[face]:
        return face
    for name in _DIRS:
        if not blocked[name]:
            return name
    return None


def _window(link: tuple[int, int], worm: dict[str, int]) -> str | None:
    """Cardinal sword gap. Both axes large is a diagonal ray, not a commit."""
    if worm["ai"] not in (2, 3) or not (_Z_LO <= worm["z"] <= _Z_HI):
        return None
    dx = int(worm["x"]) - link[0]
    dy = int(worm["y"]) - link[1]
    if abs(dx) <= _AXIS and _NEAR <= abs(dy) <= _FAR:
        return "DOWN" if dy > 0 else "UP"
    if abs(dy) <= _AXIS and _NEAR <= abs(dx) <= _FAR:
        return "RIGHT" if dx > 0 else "LEFT"
    return None


def _long(link: tuple[int, int], worm: dict[str, int], face: str) -> int:
    if face in ("LEFT", "RIGHT"):
        return abs(int(worm["x"]) - link[0])
    return abs(int(worm["y"]) - link[1])


def _target(worms: list[dict[str, int]]) -> dict[str, int] | None:
    air = [row for row in worms if row["ai"] in (2, 3)]
    air.sort(key=lambda row: (0 if _Z_LO <= row["z"] <= _Z_HI else 1, row["z"]))
    if air:
        return air[0]
    soon = [row for row in worms if row["ai"] == 1]
    soon.sort(key=lambda row: int(row["delay"]))
    return soon[0] if soon else None


def _gap(
    worm: dict[str, int],
    rocks: list[tuple[int, int, int, int, int]],
    link: tuple[int, int],
    prefer: str | None,
) -> tuple[str, int, int] | None:
    """One cardinal at sword range. The north stand sits under a high head."""
    best: tuple[str, int, int] | None = None
    best_key: tuple[int, int, int, int, int] | None = None
    wx, wy = int(worm["x"]), int(worm["y"])
    for face, (dx, dy) in _CARD.items():
        gx, gy = wx - dx * _STAND, wy - dy * _STAND
        if not _inside(gx, gy):
            continue
        eta = _rock_eta(rocks, gx, gy, 36, margin=2)
        behind = 1 if dx * int(worm["vx"]) + dy * int(worm["vy"]) < 0 else 0
        north = 1 if face == "DOWN" and int(worm["z"]) > _Z_HI else 0
        key = (
            eta,
            1 if face == prefer else 0,
            -north,
            behind,
            -_cheb(gx, gy, link[0], link[1]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = (face, gx, gy)
    return best


def _align(link: tuple[int, int], face: str, gx: int, gy: int) -> str | None:
    """Cross axis first. Cutting the corner walks the boulder diagonal."""
    ex, ey = gx - link[0], gy - link[1]
    if face in ("UP", "DOWN"):
        if abs(ex) > 2:
            return "RIGHT" if ex > 0 else "LEFT"
        if abs(ey) > 2:
            return "DOWN" if ey > 0 else "UP"
        return None
    if abs(ey) > 2:
        return "DOWN" if ey > 0 else "UP"
    if abs(ex) > 2:
        return "RIGHT" if ex > 0 else "LEFT"
    return None


def _flee(link: tuple[int, int], worms: list[dict[str, int]]) -> str | None:
    """Step straight out on the long axis. A perp step is the old diagonal."""
    lx, ly = link
    worst: dict[str, int] | None = None
    worst_d = 999
    for worm in worms:
        if worm["ai"] not in (2, 3) or _window(link, worm) is not None:
            continue
        origin = _cheb(worm["x"], worm["y"], lx, ly)
        head = _cheb(worm["x"], worm["y"] - worm["z"], lx, ly)
        close = (worm["z"] <= 16 and origin < _BODY) or head <= _HEAD
        if close and min(origin, head) < worst_d:
            worst_d = min(origin, head)
            worst = worm
    if worst is None:
        return None
    dx, dy = lx - int(worst["x"]), ly - int(worst["y"])
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx >= 0 else "LEFT"
    return "DOWN" if dy >= 0 else "UP"


def _danger(worms: list[dict[str, int]], x: int, y: int) -> int:
    """Smaller is closer. 999 is clear of a low head and its origin."""
    best = 999
    for worm in worms:
        if worm["ai"] not in (2, 3) or worm["z"] > 16:
            continue
        origin = _cheb(worm["x"], worm["y"], x, y)
        head = _cheb(worm["x"], worm["y"] - worm["z"], x, y)
        best = min(best, origin, head)
    return best


def _worm_hit(worms: list[dict[str, int]], x: int, y: int) -> bool:
    for worm in worms:
        if worm["ai"] not in (2, 3) or worm["z"] > 16:
            continue
        origin = _cheb(worm["x"], worm["y"], x, y)
        head = _cheb(worm["x"], worm["y"] - worm["z"], x, y)
        if origin < _BODY or head <= _HEAD:
            return True
    return False


def _deeper_worm(
    worms: list[dict[str, int]], lx: int, ly: int, face: str | None
) -> bool:
    """One step that newly touches a worm, or closes one already touching."""
    if face is None:
        return False
    nx, ny = _nudge(lx, ly, face)
    if not _worm_hit(worms, nx, ny):
        return False
    if not _worm_hit(worms, lx, ly):
        return True
    return _danger(worms, nx, ny) < _danger(worms, lx, ly)


def _rated(
    link: tuple[int, int],
    face: str | None,
    rocks: list[tuple[int, int, int, int, int]],
    worms: list[dict[str, int]],
) -> tuple[int, int] | None:
    if face is not None and _clamp(link, face) != face:
        return None
    px, py = _nudge(link[0], link[1], face)
    if not _toward_floor(link[0], link[1], face):
        return None
    if _deeper_worm(worms, link[0], link[1], face):
        return None
    return _rock_eta(rocks, px, py, 18, margin=2), _body_sep(worms, px, py)


def _step(
    link: tuple[int, int],
    face: str | None,
    rocks: list[tuple[int, int, int, int, int]],
    worms: list[dict[str, int]],
) -> str:
    """Prefer ``face`` when the step is clear. A worm step is not a choice."""
    want = face if face is None or _clamp(link, face) == face else None
    rated = _rated(link, want, rocks, worms)
    if rated is not None:
        eta, sep = rated
        cur = _rated(link, None, rocks, worms)
        cur_sep = cur[1] if cur is not None else 0
        if eta > 12 and (sep >= _BODY or sep >= cur_sep):
            return "NONE" if want is None else want
    best = "NONE"
    best_key: tuple[int, int, int] | None = None
    for name in (None, *_DIRS):
        choice = name if name is None or _clamp(link, name) == name else None
        if name is not None and choice is None:
            continue
        got = _rated(link, choice, rocks, worms)
        if got is None:
            continue
        key = (got[0], got[1], 1 if choice == want else 0)
        if best_key is None or key > best_key:
            best_key = key
            best = "NONE" if choice is None else choice
    return best


def _diag(link: tuple[int, int], worms: list[dict[str, int]]) -> int:
    for worm in worms:
        if worm["ai"] not in (2, 3):
            continue
        dx = abs(int(worm["x"]) - link[0])
        dy = abs(int(worm["y"]) - link[1])
        if dx > _AXIS and dy > _AXIS and max(dx, dy) <= 48:
            return 1
    return 0


def _touch(rocks: list[tuple[int, int, int, int, int]], lx: int, ly: int) -> str:
    parts = []
    for slot, rx, ry, vx, vy in rocks:
        if _in_rock(rx, ry, lx, ly):
            parts.append(f"{slot}:{rx},{ry},{vx},{vy}")
    return ",".join(parts) if parts else "-"


def _ray_text(rocks: list[tuple[int, int, int, int, int]], lx: int, ly: int) -> str:
    """Boulders with both offsets nonzero: slot, dx, dy, vx, vy."""
    parts = []
    for slot, rx, ry, vx, vy in rocks:
        dx, dy = rx - lx, ry - ly
        if dx == 0 or dy == 0 or vx == 0 or vy == 0:
            continue
        parts.append(f"{slot}:{dx},{dy},{vx},{vy}")
    return ";".join(parts) if parts else "-"


def _points(lx: int, ly: int, face: str | None, frames: int, step: int) -> list[tuple[int, int]]:
    x, y = lx, ly
    pts: list[tuple[int, int]] = []
    for _ in range(frames):
        pts.append((x, y))
        if face is None or not _toward_floor(x, y, face):
            continue
        x, y = _nudge(x, y, face, step)
    return pts


def _diag_rocks(
    rocks: list[tuple[int, int, int, int, int]], lx: int, ly: int
) -> list[tuple[int, int, int, int, int]]:
    found = []
    for rock in rocks:
        _slot, rx, ry, vx, vy = rock
        if vx == 0 or vy == 0 or rx == lx or ry == ly:
            continue
        found.append(rock)
    return found


def _covered(rocks: list[tuple[int, int, int, int, int]], x: int, y: int, t: int) -> bool:
    for rock in rocks:
        rx, ry = _rock_xy(rock, t)
        if _in_rock(rx, ry, x, y, margin=1):
            return True
    return False


def _clearance(
    rocks: list[tuple[int, int, int, int, int]], pts: list[tuple[int, int]]
) -> tuple[int, int]:
    first = len(pts) + 1
    safe = 0
    for t, (x, y) in enumerate(pts):
        if _covered(rocks, x, y, t):
            if first > t:
                first = t
        else:
            safe += 1
    return first, safe


def _path_deeper(
    worms: list[dict[str, int]], lx: int, ly: int, face: str | None
) -> bool:
    x, y = lx, ly
    for _ in range(_WORM_LOOK):
        if face is None or not _toward_floor(x, y, face):
            return False
        if _deeper_worm(worms, x, y, face):
            return True
        x, y = _nudge(x, y, face, 2)
    return False


def _leaving(vx: int, vy: int, cross: int) -> set[str]:
    """Cardinals that increase |cross| and leave the diagonal."""
    faces: set[str] = set()
    for face, sign in (("RIGHT", -vy), ("LEFT", vy), ("DOWN", vx), ("UP", -vx)):
        if sign == 0:
            continue
        if (cross >= 0 and sign > 0) or (cross < 0 and sign < 0):
            faces.add(face)
    return faces


def _ray_dodge(
    link: tuple[int, int],
    rocks: list[tuple[int, int, int, int, int]],
    worms: list[dict[str, int]],
    face: str | None,
) -> str | None:
    """Leave a boulder ray when both dx and dy are nonzero. Skip worm steps."""
    lx, ly = link
    diags = _diag_rocks(rocks, lx, ly)
    if not diags:
        return None
    want = face if face in _DIRS else None
    hold_pts = _points(lx, ly, None, _LEAD, 1)
    want_pts = _points(lx, ly, want, _LEAD, 1)
    hold_first, _hold_safe = _clearance(diags, hold_pts)
    want_first, _want_safe = _clearance(diags, want_pts)
    if hold_first > _LEAD and want_first > _LEAD:
        return None
    soon: tuple[int, tuple[int, int, int, int, int]] | None = None
    for pts in (hold_pts, want_pts):
        for t, (x, y) in enumerate(pts):
            for rock in diags:
                rx, ry = _rock_xy(rock, t)
                if not _in_rock(rx, ry, x, y, margin=1):
                    continue
                if soon is None or t < soon[0]:
                    soon = (t, rock)
                break
    leaving: set[str] = set()
    if soon is not None:
        _slot, rx, ry, vx, vy = soon[1]
        cross = vx * (ly - ry) - vy * (lx - rx)
        leaving = _leaving(vx, vy, cross)
    best = "NONE"
    best_key: tuple[int, int, int, int] | None = None
    for name in (None, *_DIRS):
        if name is not None and _clamp(link, name) != name:
            continue
        if not _toward_floor(lx, ly, name):
            continue
        if _path_deeper(worms, lx, ly, name):
            continue
        pts = hold_pts if name is None else _points(lx, ly, name, _LEAD, 1)
        first, safe = _clearance(diags, pts)
        key = (first, safe, 1 if name in leaving else 0, 1 if name == want else 0)
        if best_key is None or key > best_key:
            best_key = key
            best = "NONE" if name is None else name
    return best


def _move(
    link: tuple[int, int],
    face: str | None,
    rocks: list[tuple[int, int, int, int, int]],
    worms: list[dict[str, int]],
) -> str:
    dodge = _ray_dodge(link, rocks, worms, face)
    if dodge is not None:
        return dodge
    return _step(link, face, rocks, worms)


class Lanmolas:
    def __init__(self, env: Any) -> None:
        self.env = env

    def run(self) -> FightResult:
        env = self.env
        ram0 = _ram(env)
        max0 = _u8(ram0, LINK_MAX_HP)
        follower0 = _u8(ram0, FOLLOWER)
        walk = _Walk()
        guard = PolicyGuard(
            walk,
            reach=1,
            horizon=1,
            detours=(1,),
            directions=_DIRS,
        )
        guard.bind_env(env)
        frames = 0
        strikes = 0
        cooldown = 0
        held_face: str | None = None
        held_slot = -1
        hurt = ""

        def note(before: int) -> int:
            nonlocal hurt
            snap_now = read_snapshot(_ram(env))
            hp = int(snap_now.hp)
            if hp < before:
                lx, ly = int(snap_now.link_x), int(snap_now.link_y)
                worms_now = _worms(env)
                rocks_now = _rocks(env)
                name = _NAME.get(int(snap_now.link_direction), str(int(snap_now.link_direction)))
                near = "-"
                if rocks_now:
                    slot, rx, ry, vx, vy = min(rocks_now, key=lambda row: _cheb(row[1], row[2], lx, ly))
                    near = f"{slot}:{rx - lx},{ry - ly},{vx},{vy}"
                hurt = (
                    f"xy=({lx},{ly}) dir={name} f={frames} hp={before}->{hp} "
                    f"rock={_touch(rocks_now, lx, ly)} near={near} "
                    f"rays={_ray_text(rocks_now, lx, ly)} diag={_diag((lx, ly), worms_now)}"
                )
                print(f"HURT {hurt}", flush=True)
            return hp

        while frames < MAX_FRAMES:
            ram = _ram(env)
            if _u8(ram, FOLLOWER) != follower0:
                return FightResult(ok=False, frames=frames, strikes=strikes, blocker="follower")
            snap = read_snapshot(ram)
            if _u8(ram, LINK_MAX_HP) > max0:
                print(f"lanmolas done frames={frames} strikes={strikes}", flush=True)
                return FightResult(ok=True, frames=frames, strikes=strikes)
            if int(snap.hp) <= 0 or int(snap.room_base_id) != ROOM:
                why = "dead " + hurt if int(snap.hp) <= 0 else "left-room"
                return FightResult(ok=False, frames=frames, strikes=strikes, blocker=why)
            if not snap.has_control or int(snap.knockback):
                before = int(snap.hp)
                walk.button = "NONE"
                env.step(np.asarray(press(), dtype=np.uint8))
                frames += 1
                note(before)
                continue
            before = int(snap.hp)
            bottle = drink_if_low(env)
            frames += int(bottle.frames)
            if bottle.frames:
                note(before)
            if frames >= MAX_FRAMES:
                break
            ram = _ram(env)
            snap = read_snapshot(ram)
            if int(snap.hp) <= 0:
                continue
            if _u8(ram, LINK_MAX_HP) > max0:
                continue
            link = (int(snap.link_x), int(snap.link_y))
            worms = _worms(env)
            rocks = _rocks(env)
            heart = _nearest(env, HEART, link)
            if cooldown:
                cooldown -= 1
            face: str | None = None
            head = _target(worms)
            back = _recover(link)
            if back is not None:
                face = back
            elif head is not None:
                if int(head["slot"]) != held_slot:
                    held_face = None
                    held_slot = int(head["slot"])
                gap_face = _window(link, head)
                clear = _rock_eta(rocks, link[0], link[1], 16, margin=1) > 14
                if gap_face is not None and clear and _body_sep(worms, link[0], link[1]) >= _BODY:
                    held_face = gap_face
                    aimed = int(snap.link_direction) == _FACE[gap_face]
                    if aimed and not cooldown and _ray_dodge(link, rocks, worms, None) is None:
                        script = strike_if_safe(env, int(head["slot"]))
                        if script:
                            dx = int(head["x"]) - link[0]
                            dy = int(head["y"]) - link[1]
                            for act in script:
                                step_hp = int(read_snapshot(_ram(env)).hp)
                                env.step(np.asarray(act.action, dtype=np.uint8))
                                frames += 1
                                if note(step_hp) <= 0:
                                    break
                            strikes += 1
                            hp = int(_ram(env)[SPRITE_HP + int(head["slot"])])
                            print(
                                f"strike f={frames} slot={head['slot']} hp={hp} "
                                f"z={head['z']} dx={dx} dy={dy}",
                                flush=True,
                            )
                            cooldown = 8
                            continue
                        cooldown = 2
                        face = None
                    else:
                        face = gap_face if _long(link, head, gap_face) >= 23 else _AWAY[gap_face]
                else:
                    flee = _flee(link, worms)
                    if flee is not None:
                        face = flee
                    else:
                        goal = _gap(head, rocks, link, held_face)
                        if goal is not None:
                            held_face = goal[0]
                            face = _align(link, goal[0], goal[1], goal[2])
            elif heart is not None and not worms:
                face = _toward(heart[0] - link[0], heart[1] - link[1])
            else:
                dist = _cheb(_CAMP[0], _CAMP[1], link[0], link[1])
                face = None if dist <= 4 else _toward(_CAMP[0] - link[0], _CAMP[1] - link[1])
            walk.button = _move(link, face, rocks, worms)
            before = int(read_snapshot(_ram(env)).hp)
            act = guard.step(read_snapshot(_ram(env)))
            env.step(np.asarray(act.action, dtype=np.uint8))
            frames += 1
            note(before)
            if frames % 400 == 0:
                print(
                    f"lanmolas f={frames} strikes={strikes} hp={int(snap.hp)} "
                    f"xy={link} face={face} btn={walk.button} "
                    f"worms={[(w['slot'], w['hp'], w['ai'], w['z']) for w in worms]}",
                    flush=True,
                )
        return FightResult(ok=False, frames=frames, strikes=strikes, blocker="timeout " + hurt)

def fight(env: Any) -> FightResult:
    """Kill the three worms and take the heart. Isolated poke pin only."""
    return Lanmolas(env).run()

