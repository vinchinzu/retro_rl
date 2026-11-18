"""Kholdstare on an isolated Whatif_Ice_kholdstare_magic pin. Not continuous.

The pin is a poke copy of Whatif_Ice_kholdstare with magic ``$F36E=128``.
Eight fire shots (16 magic, 8 shell HP) empty a 64 HP shell. The lip poke
leaves BG2 ``$E8`` south of the eye, so ``$0F00`` stays 1 and the
triplicate at ``$0DF0==1`` never runs. ``fight`` rewrites ``$E2``/``$E8``
so the eye stays on screen. It does not write shell HP, ``$0CAA``, or ``$F3CC``.

A shot lands when one axis is within 4px and the other gap is 16–48.
Master sword from the south (gap 24–34, cross-axis within 8) drops an
eye by 4. The spawn ball sits 15px east, velocity 0, until z hits 0.
The 13-frame swing has to finish while that z is still above 0. An
8px probe stalls a 2px step, so the swing starts late; poll
``strike_if_safe`` on the way in. Step to x <= spawn_x-16 only while
z is above 0. UP, DOWN, LEFT, and RIGHT after the split already died.
After that sword hit, alternate a south-west tile and its north-east
return so the stand does not march into the wall rows. West of x=7456
is wall. Do not step through (7464, 6806). Falling ice
is ``0xA4`` with ``$0DB0`` nonzero. Wall
rows 27–38 are not walked. BG2 writes are ``$E2`` and ``$E8`` only.

``fight(env)`` is ``alttp.dungeon.bosses.kholdstare:fight`` for stage_replay.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alttp.dungeon.bosses import FightResult
from alttp.dungeon.tilemap import TILE_PX, read_room_tiles, tile_kind
from alttp.primitives import active_ancillae, active_sprites, drink_if_low, select_y_item
from alttp.ram import (
    FOLLOWER,
    LINK_MAX_HP,
    LINK_X,
    LINK_Y,
    assign_u8,
    read_inventory,
    read_snapshot,
    wram_index,
)
from alttp.rollout import PolicyGuard, press, strike_if_safe
from retro_harness.input_script import FrameAction

SHELL = 0xA3
EYE = 0xA2
ICE = 0xA4
HEART = 0xEA
FIRE = 0x02
ROOM = 0xDE
HUD_FIRE = 0x06
HUD_CUR_ITEM = 0x0202
SOUTH = 36
FLANK = 40
MAX_FRAMES = 12000
SPRITE_C = 0x0DB0
VEL_X = 0x0D50
VEL_Y = 0x0D40
BG_X = 0xE2
BG_Y = 0xE8
# Sprite_MoveXY steps ``signed($0D50) / 16`` px. Contact was chebyshev 20.
# The sword window is 24–34 south. Separate inside 24 so that pocket can be held.
_LEAD = 4
_BODY = 24
_SOON = 18
_SOUTH = ("LEFT", "RIGHT", "DOWN")
# 16px west of the spawn ball. It sits 15px east of Link until it splits.
_WEST = 16
# Green-mail eye contact is 32. A shard at chebyshev 34 still hit for 16.
_CONTACT = 32
_COLUMN = 36
_STEP = {"LEFT": (-8, 0), "RIGHT": (8, 0), "UP": (0, -8), "DOWN": (0, 8)}
_DIRS = ("UP", "DOWN", "LEFT", "RIGHT")


class _Walk:
    """One held direction. PolicyGuard calls ``step``; it does not touch RAM."""

    def __init__(self) -> None:
        self.button = "NONE"

    def step(self, _snap: object) -> FrameAction:
        frame = press() if self.button in (None, "NONE") else press(self.button)
        return FrameAction(list(frame), "walk")


def _ram(env: Any) -> np.ndarray:
    return np.asarray(env.get_ram(), dtype=np.uint8)


def _u8(ram: np.ndarray, offset: int) -> int:
    return int(ram[wram_index(offset)])


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


def _cheb(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(int(ax) - int(bx)), abs(int(ay) - int(by)))


def _bad(grid: np.ndarray, x: int, y: int) -> bool:
    row = (int(y) // TILE_PX) & 63
    col = (int(x) // TILE_PX) & 63
    if 27 <= row <= 38:
        return True
    return tile_kind(int(grid[row, col])) not in ("floor", "ice")


def _grid(env: Any) -> np.ndarray:
    ram = _ram(env)
    snap = read_snapshot(ram)
    return read_room_tiles(ram, int(snap.layer) & 1)


def _step_blocked(grid: np.ndarray, link: tuple[int, int], name: str, *, eyes: bool, anchor_y: int) -> bool:
    if name not in _STEP:
        return False
    # UP into the shell body. Once the eyes are out, the south window is north of this line.
    if name == "UP" and not eyes and anchor_y and link[1] <= anchor_y + 16:
        return True
    dx, dy = _STEP[name]
    return _bad(grid, link[0] + dx, link[1] + dy)


def _dir_of(frame: tuple[int, ...] | list[int]) -> str:
    index = (("UP", 4), ("DOWN", 5), ("LEFT", 6), ("RIGHT", 7))
    held = [name for name, at in index if at < len(frame) and int(frame[at])]
    if len(held) != 1:
        return "NONE"
    return held[0]


def _rows(env: Any) -> list[dict[str, int]]:
    found = []
    for spr in active_sprites(env):
        kind = int(spr.sprite_type)
        if kind not in (SHELL, EYE, ICE, HEART):
            continue
        found.append(
            {
                "slot": int(spr.slot),
                "type": kind,
                "hp": int(spr.hp),
                "ai": int(spr.ai_state),
                "state": int(spr.state),
                "x": int(spr.x),
                "y": int(spr.y),
                "hit": int(spr.hit_timer),
            }
        )
    return found


def _shell(rows: list[dict[str, int]]) -> dict[str, int] | None:
    for row in rows:
        if row["type"] == SHELL and row["state"] != 0 and row["hp"] > 0:
            return row
    return None


def _ice_at(rows: list[dict[str, int]], shell: dict[str, int] | None, env: Any) -> list[dict[str, int]]:
    """Falling balls and shards. The generator (``$0DB0`` 0) is not a column."""
    out = []
    ram = _ram(env)
    for row in rows:
        if row["type"] != ICE or row["state"] == 0:
            continue
        slot = int(row["slot"])
        if _u8(ram, SPRITE_C + slot) == 0:
            continue
        # Off-screen latch. A paused shard never moves and is not a dodge.
        if int(ram[0x0F00 + slot]) != 0:
            continue
        z = int(ram[0x0F70 + slot])
        near = shell is not None and _cheb(row["x"], row["y"], shell["x"], shell["y"]) <= 20
        if near and z == 0:
            continue
        item = dict(row)
        item["z"] = z
        out.append(item)
    return out


def _on_ice(ice: list[dict[str, int]], x: int, y: int, reach: int = 12) -> bool:
    return any(_cheb(row["x"], row["y"], x, y) <= reach for row in ice)


def _pieces(rows: list[dict[str, int]]) -> list[dict[str, int]]:
    """Eyes that take sword damage. AI 3 is in the shell. AI 2 is the split."""
    return [
        row for row in rows
        if row["type"] == EYE and row["state"] != 0 and row["hp"] > 0 and row["ai"] in (0, 1)
    ]


def _heart(rows: list[dict[str, int]]) -> dict[str, int] | None:
    hearts = [row for row in rows if row["type"] == HEART and row["state"] != 0]
    if not hearts:
        return None
    return hearts[0]


def _face_of(dx: int, dy: int) -> str:
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def _aligned(dx: int, dy: int) -> bool:
    if abs(dx) <= 4 and 16 <= -dy <= 48:
        return True
    return abs(dy) <= 4 and 16 <= abs(dx) <= 48


def _firing_cell(
    grid: np.ndarray,
    shell: dict[str, int],
    ice: list[dict[str, int]],
) -> tuple[int, int] | None:
    """The stand farthest from live ice. Shards stop about 31px out."""
    cells = (
        (int(shell["x"]), int(shell["y"]) + SOUTH),
        (int(shell["x"]) - FLANK, int(shell["y"])),
        (int(shell["x"]) + FLANK, int(shell["y"])),
    )
    best: tuple[int, int] | None = None
    best_d = 35
    for cell in cells:
        if _bad(grid, cell[0], cell[1]):
            continue
        if not ice:
            return cell
        dist = min(_cheb(row["x"], row["y"], cell[0], cell[1]) for row in ice)
        if dist <= best_d:
            continue
        best_d = dist
        best = cell
    return best


def _toward(link: tuple[int, int], cell: tuple[int, int]) -> str:
    """Close x first. Closing y first walks into the shell body."""
    dx = cell[0] - link[0]
    dy = cell[1] - link[1]
    if abs(dx) > 2:
        return "RIGHT" if dx > 0 else "LEFT"
    if abs(dy) > 2:
        return "DOWN" if dy > 0 else "UP"
    return "NONE"


def _away(grid: np.ndarray, link: tuple[int, int], ice: list[dict[str, int]], *, eyes: bool, anchor_y: int) -> str:
    if not ice:
        return "NONE"
    near = min(ice, key=lambda row: _cheb(row["x"], row["y"], link[0], link[1]))
    here = _cheb(near["x"], near["y"], link[0], link[1])
    sideways = "NONE"
    side_d = here
    best = "NONE"
    best_d = here
    for name in _DIRS:
        if _step_blocked(grid, link, name, eyes=eyes, anchor_y=anchor_y):
            continue
        dx, dy = _STEP[name]
        dist = _cheb(near["x"], near["y"], link[0] + dx, link[1] + dy)
        if dist <= here:
            continue
        if name in ("LEFT", "RIGHT") and dist > side_d:
            side_d = dist
            sideways = name
        if dist > best_d:
            best_d = dist
            best = name
    return sideways if sideways != "NONE" else best


def _sword_face(dx: int, dy: int) -> str | None:
    """Measured UP hits. ``dx, dy`` are the eye minus Link.

    Gap 24–34 facing UP, cross-axis within 8. A DOWN stand inside 21 was a
    body hit and is not a swing. Other facings are not a commit.
    """
    if abs(dx) <= 8 and -34 <= dy <= -24:
        return "UP"
    return None


def _eye_note(rows: list[dict[str, int]]) -> str:
    eyes = [f"{row['hp']}/ai{row['ai']}" for row in rows if row["type"] == EYE and row["state"] != 0]
    if not eyes:
        return "eyes none"
    return "eyes " + ",".join(eyes)


def _camera(env: Any, shell: dict[str, int]) -> None:
    """BG2 scroll copies so the eye is inside the on-screen box. Not ``$F3CC``."""
    _put_u16(env, BG_X, int(shell["x"]) - 0x78)
    _put_u16(env, BG_Y, int(shell["y"]) - 0x40)
    _refresh(env)


def _ice_points(ram: np.ndarray, ice: list[dict[str, int]]) -> list[dict[str, int]]:
    """Landing tile plus an 8-frame shard forecast. The ball spawns on Link."""
    out = []
    for row in ice:
        vx, vy = _vel(ram, int(row["slot"]))
        out.append(row)
        out.append(
            {
                "x": int(row["x"]) + (vx * 8) // 16,
                "y": int(row["y"]) + (vy * 8) // 16,
            }
        )
    return out


def _dodge_button(
    grid: np.ndarray,
    ram: np.ndarray,
    link: tuple[int, int],
    ice: list[dict[str, int]],
    shell: dict[str, int] | None, *, eyes: bool, anchor_y: int,
) -> str:
    near = []
    for row in ice:
        dist = _cheb(row["x"], row["y"], link[0], link[1])
        height = int(row.get("z", 0))
        # z 224 is the spawn height. Step off that tile, then stop so a shot
        # can be lined up. A 48px bubble never released the firing cell.
        if height > 16 and dist <= 36:
            near.append(row)
        elif height <= 16 and dist <= 18:
            near.append(row)
    if not near:
        return "NONE"
    return _off_column(grid, link, _ice_points(ram, near), shell, eyes=eyes, anchor_y=anchor_y)


def _into_shell(shell: dict[str, int] | None, x: int, y: int) -> bool:
    """The shell body stops UP. A step that closes inside 28px never leaves."""
    if shell is None:
        return False
    return _cheb(int(shell["x"]), int(shell["y"]), x, y) < 28


def _off_column(
    grid: np.ndarray,
    link: tuple[int, int],
    pts: list[dict[str, int]],
    shell: dict[str, int] | None, *, eyes: bool, anchor_y: int,
) -> str:
    """Leave the landing tile sideways. UP bonks the shell and stays on the ball."""
    here = _gap(pts, link[0], link[1])
    best = "NONE"
    best_d = here
    # Left and right clear the column. UP walks through the dead shell.
    for name in _SOUTH:
        if _step_blocked(grid, link, name, eyes=eyes, anchor_y=anchor_y):
            continue
        dx, dy = _STEP[name]
        nxt = (link[0] + dx, link[1] + dy)
        if _into_shell(shell, nxt[0], nxt[1]):
            continue
        dist = _gap(pts, nxt[0], nxt[1])
        if dist > best_d:
            best_d = dist
            best = name
    if best != "NONE":
        return best
    # An 8px step off a cardinal ball does not raise chebyshev. Slide anyway.
    if here >= 48:
        return "NONE"
    side = "NONE"
    side_d = -1
    for name in ("RIGHT", "LEFT", "DOWN"):
        if _step_blocked(grid, link, name, eyes=eyes, anchor_y=anchor_y):
            continue
        dx, dy = _STEP[name]
        nxt = (link[0] + dx, link[1] + dy)
        if _into_shell(shell, nxt[0], nxt[1]):
            continue
        dist = _gap(pts, nxt[0], nxt[1])
        if dist < here or dist <= side_d:
            continue
        side_d = dist
        side = name
    return side


def _s8(value: int) -> int:
    return int(value) - 256 if int(value) >= 128 else int(value)


def _vel(ram: np.ndarray, slot: int) -> tuple[int, int]:
    return _s8(_u8(ram, VEL_X + int(slot))), _s8(_u8(ram, VEL_Y + int(slot)))


def _gap(eyes: list[dict[str, int]], x: int, y: int) -> int:
    return min(_cheb(row["x"], row["y"], x, y) for row in eyes)


def _forecast(ram: np.ndarray, eyes: list[dict[str, int]]) -> list[dict[str, int]]:
    """Where each eye will be in ``_LEAD`` frames. Velocity is signed/16 px."""
    out = []
    for row in eyes:
        vx, vy = _vel(ram, int(row["slot"]))
        out.append(
            {
                "x": int(row["x"]) + (vx * _LEAD) // 16,
                "y": int(row["y"]) + (vy * _LEAD) // 16,
            }
        )
    return out


def _clearance(eyes: list[dict[str, int]], forecasts: list[dict[str, int]], x: int, y: int) -> int:
    return min(_gap(eyes, x, y), _gap(forecasts, x, y))


def _crowded(eyes: list[dict[str, int]], forecasts: list[dict[str, int]], x: int, y: int) -> bool:
    if _gap(eyes, x, y) < _BODY:
        return True
    return _gap(forecasts, x, y) < _SOON


def _sword_target(
    ram: np.ndarray,
    pieces: list[dict[str, int]],
    link: tuple[int, int],
) -> dict[str, int] | None:
    """South window, and slow enough that the swing finishes outside the body."""
    best: dict[str, int] | None = None
    best_key: tuple[int, int] | None = None
    for row in pieces:
        if int(row["hit"]) & 0x80:
            continue
        dx = int(row["x"]) - link[0]
        dy = int(row["y"]) - link[1]
        if _sword_face(dx, dy) is None:
            continue
        vx, vy = _vel(ram, int(row["slot"]))
        key = (abs(vx) + abs(vy), _cheb(row["x"], row["y"], link[0], link[1]))
        if best_key is None or key < best_key:
            best = row
            best_key = key
    return best


def _hits_body(
    pieces: list[dict[str, int]],
    forecasts: list[dict[str, int]],
    x: int,
    y: int,
) -> bool:
    if not pieces:
        return False
    return _clearance(pieces, forecasts, x, y) < _BODY


def _step_unsafe(
    grid: np.ndarray,
    link: tuple[int, int],
    name: str,
    pieces: list[dict[str, int]],
    forecasts: list[dict[str, int]],
    shell: dict[str, int] | None, *, eyes: bool, anchor_y: int,
) -> bool:
    if name not in _STEP or _step_blocked(grid, link, name, eyes=eyes, anchor_y=anchor_y):
        return True
    dx, dy = _STEP[name]
    nxt = (link[0] + dx, link[1] + dy)
    if _into_shell(shell, nxt[0], nxt[1]):
        return True
    return _hits_body(pieces, forecasts, nxt[0], nxt[1])


def _short_clear(
    grid: np.ndarray,
    link: tuple[int, int],
    name: str,
    pieces: list[dict[str, int]],
    forecasts: list[dict[str, int]],
    shell: dict[str, int] | None,
) -> bool:
    """2px step. An 8px probe refuses a floor step that never enters the body."""
    nudge = {"UP": (0, -2), "DOWN": (0, 2), "LEFT": (-2, 0), "RIGHT": (2, 0)}
    if name not in nudge:
        return False
    dx, dy = nudge[name]
    nxt = (link[0] + dx, link[1] + dy)
    if _bad(grid, nxt[0], nxt[1]) or _into_shell(shell, nxt[0], nxt[1]):
        return False
    if not pieces:
        return True
    return _clearance(pieces, forecasts, nxt[0], nxt[1]) >= _BODY


def _early_target(
    pieces: list[dict[str, int]],
    link: tuple[int, int],
) -> dict[str, int] | None:
    """North eye a 13-frame swing might reach. ``strike_if_safe`` decides."""
    best: dict[str, int] | None = None
    best_key: tuple[int, int] | None = None
    for row in pieces:
        if int(row["hit"]) & 0x80:
            continue
        dx = int(row["x"]) - link[0]
        dy = int(row["y"]) - link[1]
        if abs(dx) > 20 or not -42 <= dy <= -20:
            continue
        key = (abs(dx), abs(dy + 29))
        if best_key is None or key < best_key:
            best = row
            best_key = key
    return best


def _spawn_ball(
    ice: list[dict[str, int]],
    spawn_xy: tuple[int, int] | None,
) -> dict[str, int] | None:
    """Intact ball (ai 0) on its spawn xy. Shards are ai 1. z 1–16 is still in the air."""
    if spawn_xy is None:
        return None
    for row in ice:
        if int(row.get("ai", 0)) != 0 or int(row.get("z", 0)) <= 0:
            continue
        if _cheb(int(row["x"]), int(row["y"]), spawn_xy[0], spawn_xy[1]) <= 4:
            return row
    return None


def _death_tile(x: int, y: int) -> bool:
    """(7464, 6806) is the three-hit death pose. Do not step onto it."""
    return _cheb(x, y, 7464, 6806) <= 8


def _post_step(
    grid: np.ndarray,
    link: tuple[int, int],
    name: str,
    pieces: list[dict[str, int]],
    forecasts: list[dict[str, int]],
    shell: dict[str, int] | None,
    body: bool = True,
) -> bool:
    """2px probe. An 8px probe stalls on the west wall before the floor edge."""
    nudge = {"UP": (0, -2), "DOWN": (0, 2), "LEFT": (-2, 0), "RIGHT": (2, 0)}
    if name not in nudge:
        return False
    dx, dy = nudge[name]
    nxt = (link[0] + dx, link[1] + dy)
    if _bad(grid, nxt[0], nxt[1]) or _death_tile(nxt[0], nxt[1]):
        return False
    if _into_shell(shell, nxt[0], nxt[1]):
        return False
    if not body or not pieces:
        return True
    return _clearance(pieces, forecasts, nxt[0], nxt[1]) >= _BODY


def _post_button(
    grid: np.ndarray,
    link: tuple[int, int],
    ice: list[dict[str, int]],
    pieces: list[dict[str, int]],
    forecasts: list[dict[str, int]],
    shell: dict[str, int] | None,
) -> str | None:
    """After the sword hit. None means the south window may swing in place.

    A falling ball (ai 0) splits about 66 frames after it locks onto Link.
    The hitbox is the band above the shard, so same-y still connects.
    (-16, +40) and its opposite (+16, -40) are each outside that band.
    Track the ball until z hits 0; z under 128 is still in the air.
    """
    ball = None
    for row in ice:
        height = int(row.get("z", 0))
        if int(row.get("ai", 0)) != 0 or height <= 0:
            continue
        if ball is None or height > int(ball["z"]):
            ball = row
    if ball is not None:
        goal = None
        # South-west first. The opposite tile is the return, not a deeper south step.
        for ox, oy in (
            (-16, 40),
            (16, -40),
            (16, 40),
            (-16, -40),
            (-24, 40),
            (24, -40),
            (24, 40),
            (-24, -40),
        ):
            cand = (int(ball["x"]) + ox, int(ball["y"]) + oy)
            if _bad(grid, cand[0], cand[1]) or _death_tile(cand[0], cand[1]):
                continue
            goal = cand
            break
        if goal is None:
            return "NONE"
        dx = goal[0] - link[0]
        dy = goal[1] - link[1]
        if abs(dx) <= 2 and abs(dy) <= 2:
            return "NONE"
        # Leave the vertical ray before traveling y.
        if abs(dx) > 2:
            name = "LEFT" if dx < 0 else "RIGHT"
        else:
            name = "DOWN" if dy > 0 else "UP"
        # The closing eye is inside the body probe and would pin Link on the ball.
        if _post_step(grid, link, name, pieces, forecasts, shell, body=False):
            return name
        other = ("DOWN" if dy > 0 else "UP") if name in ("LEFT", "RIGHT") else (
            "LEFT" if dx < 0 else "RIGHT"
        )
        cross = abs(dy) > 2 if name in ("LEFT", "RIGHT") else abs(dx) > 2
        if cross and _post_step(grid, link, other, pieces, forecasts, shell, body=False):
            return other
        return "NONE"
    shards = [row for row in ice if int(row.get("ai", 0)) == 1 and int(row.get("z", 99)) < 32]
    if shards and _gap(shards, link[0], link[1]) < 16:
        near = min(shards, key=lambda row: _cheb(row["x"], row["y"], link[0], link[1]))
        name = "LEFT" if int(near["x"]) >= link[0] else "RIGHT"
        if _post_step(grid, link, name, pieces, forecasts, shell):
            return name
        return "NONE"
    return None


def _sword_aim(pieces: list[dict[str, int]], link: tuple[int, int]) -> tuple[int, int, str | None] | None:
    best: tuple[int, int, str | None] | None = None
    best_dx = 99
    for row in pieces:
        dx = int(row["x"]) - link[0]
        dy = int(row["y"]) - link[1]
        if dy >= -16:
            continue
        if abs(dx) >= best_dx:
            continue
        best_dx = abs(dx)
        best = (dx, dy, _sword_face(dx, dy))
    return best


def _place(env: Any, shell: dict[str, int]) -> bool:
    grid = _grid(env)
    for gap in (SOUTH, 40, 32, 28):
        x = int(shell["x"])
        y = int(shell["y"]) + gap
        if _bad(grid, x, y):
            continue
        _put_u16(env, LINK_X, x)
        _put_u16(env, LINK_Y, y)
        _refresh(env)
        return True
    return False


def _play(env: Any, frame: tuple[int, ...] | list[int]) -> None:
    env.step(np.asarray(frame, dtype=np.uint8))


def _walk_frame(
    env: Any,
    guard: PolicyGuard,
    walk: _Walk,
    grid: np.ndarray,
    link: tuple[int, int],
    button: str, *, eyes: bool, anchor_y: int,
) -> None:
    if button != "NONE" and _step_blocked(grid, link, button, eyes=eyes, anchor_y=anchor_y):
        button = "NONE"
    walk.button = button
    act = guard.step(read_snapshot(_ram(env)))
    frame = tuple(int(v) for v in act.action)
    held = _dir_of(frame)
    if held != "NONE" and _step_blocked(grid, link, held, eyes=eyes, anchor_y=anchor_y):
        guard.drop_detour()
        side = "NONE"
        # UP is the eye-phase exit. Do not slide it into LEFT or DOWN.
        if walk.button != "UP":
            for name in ("LEFT", "RIGHT"):
                if name != held and not _step_blocked(grid, link, name, eyes=eyes, anchor_y=anchor_y):
                    side = name
                    break
        frame = tuple(press() if side == "NONE" else press(side))
    _play(env, frame)


class Kholdstare:
    def __init__(self, env: Any) -> None:
        self.env = env
        self._anchor_x = 0
        self._anchor_y = 0
        self._eyes = False
        self._west_steps = 0

    def run(self) -> FightResult:
        env = self.env
        ram0 = _ram(env)
        max0 = _u8(ram0, LINK_MAX_HP)
        follower0 = _u8(ram0, FOLLOWER)
        rows = _rows(env)
        shell = _shell(rows)
        if shell is None:
            return FightResult(ok=False, frames=0, blocker="no shell")
        if not _place(env, shell):
            return FightResult(ok=False, frames=0, blocker="no floor stand")
        self._anchor_x = int(shell["x"])
        self._anchor_y = int(shell["y"])
        _camera(env, shell)
        if _u8(_ram(env), FOLLOWER) != follower0:
            return FightResult(ok=False, frames=0, blocker="follower")
        if _u8(_ram(env), HUD_CUR_ITEM) != HUD_FIRE:
            selected = select_y_item(env, "fire_rod")
            if not selected.ok:
                return FightResult(ok=False, frames=int(selected.frames), blocker="fire rod")
        walk = _Walk()
        # Reach 20 stays quiet in the 24–34 sword window. UP walks onto the eye.
        guard = PolicyGuard(walk, reach=20, horizon=2, detours=(2,), directions=_SOUTH)
        guard.bind_env(env)
        frames = 0
        fire_hits = 0
        sword_hits = 0
        shell_hp = int(shell["hp"])
        sword_cool = 0
        spawn_xy: tuple[int, int] | None = None
        while frames < MAX_FRAMES:
            if self._anchor_y:
                _camera(env, {"x": self._anchor_x, "y": self._anchor_y})
            ram = _ram(env)
            if _u8(ram, FOLLOWER) != follower0:
                return _stop(frames, fire_hits, sword_hits, "follower")
            snap = read_snapshot(ram)
            if _u8(ram, LINK_MAX_HP) > max0:
                print(
                    f"kholdstare done frames={frames} fire_hits={fire_hits} sword_hits={sword_hits}",
                    flush=True,
                )
                return FightResult(ok=True, frames=frames, fire_hits=fire_hits, sword_hits=sword_hits)
            if int(snap.hp) <= 0 or int(snap.room_base_id) != ROOM:
                why = "dead" if int(snap.hp) <= 0 else "left-room"
                return _stop(frames, fire_hits, sword_hits, why)
            if int(snap.game_mode) == 0x0E:
                # Walked-into text. A closes a settled page. Same bytes as overworld.
                ready = _u8(ram, 0x1CD4) == 0x03 and _u8(ram, 0x1CE9) == 0
                _play(env, press("A") if ready else press())
                frames += 1
                continue
            if not snap.has_control or int(snap.knockback):
                _play(env, press())
                frames += 1
                continue
            bottle = drink_if_low(env)
            frames += int(bottle.frames)
            if frames >= MAX_FRAMES:
                break
            ram = _ram(env)
            snap = read_snapshot(ram)
            if _u8(ram, LINK_MAX_HP) > max0 or int(snap.hp) <= 0:
                continue
            rows = _rows(env)
            shell = _shell(rows)
            link = (int(snap.link_x), int(snap.link_y))
            grid = _grid(env)
            ice = _ice_at(rows, shell, env)
            landed = [row for row in ice if int(row.get("z", 0)) <= 16]
            magic = int(read_inventory(ram).magic)
            flying = any(int(row.ancilla_type) == FIRE for row in active_ancillae(env))
            if shell is not None and int(shell["hp"]) < shell_hp:
                fire_hits += 1
                shell_hp = int(shell["hp"])
            elif shell is not None:
                shell_hp = int(shell["hp"])
            elif shell_hp > 0 and fire_hits:
                fire_hits += 1
                shell_hp = 0
            pieces = _pieces(rows)
            prize = _heart(rows)
            dodge = _dodge_button(
                grid, ram, link, ice, shell, eyes=self._eyes, anchor_y=self._anchor_y
            )
            guard.reach = 20 if pieces else 28
            if pieces:
                self._eyes = True
                forecasts = _forecast(ram, pieces)
                # The guard's LEFT detour is one of the closed leaves. Press directly.
                guard.drop_detour()
                fresh = [
                    row for row in ice
                    if int(row.get("z", 0)) >= 160
                    and _cheb(row["x"], row["y"], link[0], link[1]) <= 16
                ]
                if spawn_xy is None and fresh:
                    spawn_xy = (int(fresh[0]["x"]), int(fresh[0]["y"]))
                ball = _spawn_ball(ice, spawn_xy)
                clear_x = None if spawn_xy is None else int(spawn_xy[0]) - _WEST
                z = -1 if ball is None else int(ball["z"])
                # Clear x before the swing. The 13-frame swing locks Link until z is already low.
                if (
                    not sword_hits
                    and ball is not None
                    and z > 0
                    and clear_x is not None
                    and link[0] > clear_x
                    and self._west_steps < 3
                    and not _bad(grid, link[0] - 2, link[1])
                ):
                    # Three LEFT taps, then the other leaves own the ball. Not a log cap.
                    self._west_steps += 1
                    _play(env, press("LEFT"))
                    frames += 1
                    continue
                # After the hit, hold the clear x. Do not step once z is 0.
                if sword_hits and ball is not None and z > 0 and clear_x is not None:
                    if link[0] <= clear_x:
                        _play(env, press())
                        frames += 1
                        continue
                    nxt = _clearance(pieces, forecasts, link[0] - 2, link[1]) if pieces else 99
                    here = _clearance(pieces, forecasts, link[0], link[1]) if pieces else 99
                    button = "NONE"
                    if not _bad(grid, link[0] - 2, link[1]) and (nxt >= here or nxt >= _BODY):
                        button = "LEFT"
                    _play(env, press() if button == "NONE" else press(button))
                    frames += 1
                    continue
                if sword_hits and spawn_xy is not None and ball is None:
                    # Opening swing is done. Stand off the next ball's split.
                    button = _post_button(grid, link, ice, pieces, forecasts, shell)
                    if button is None and frames >= sword_cool and int(snap.link_direction) == 0:
                        target = _sword_target(ram, pieces, link)
                        if target is not None:
                            before = int(target["hp"])
                            script = strike_if_safe(env, int(target["slot"]))
                            if script:
                                for act in script:
                                    _play(env, tuple(int(v) for v in act.action))
                                    frames += 1
                                after = next(
                                    (
                                        int(row["hp"])
                                        for row in _rows(env)
                                        if int(row["slot"]) == int(target["slot"])
                                    ),
                                    0,
                                )
                                if after < before:
                                    sword_hits += 1
                                else:
                                    sword_cool = frames + 8
                                continue
                    if button is None:
                        button = "NONE"
                    _play(env, press() if button == "NONE" else press(button))
                    frames += 1
                    continue
                if spawn_xy is not None and ball is None:
                    # z is 0 before the swing. Hold. Cardinals from here already died.
                    _play(env, press())
                    frames += 1
                    continue
                target = None
                early = False
                if frames >= sword_cool:
                    target = _sword_target(ram, pieces, link)
                if target is None and int(snap.link_direction) == 0 and ball is not None:
                    target = _early_target(pieces, link)
                    early = target is not None
                if target is not None and int(snap.link_direction) != 0:
                    if _short_clear(grid, link, "UP", pieces, forecasts, shell):
                        _play(env, press("UP"))
                        frames += 1
                        continue
                if target is not None and int(snap.link_direction) == 0:
                    before = int(target["hp"])
                    script = strike_if_safe(env, int(target["slot"]))
                    if script:
                        for act in script:
                            _play(env, tuple(int(v) for v in act.action))
                            frames += 1
                        after = next(
                            (
                                int(row["hp"])
                                for row in _rows(env)
                                if int(row["slot"]) == int(target["slot"])
                            ),
                            0,
                        )
                        if after < before:
                            sword_hits += 1
                            continue
                        if not early:
                            sword_cool = frames + 8
                        continue
                    if not early:
                        sword_cool = frames + 8
                        continue
                if _crowded(pieces, forecasts, link[0], link[1]):
                    # Hold. A cardinal from this pocket already died.
                    _play(env, press())
                    frames += 1
                    continue
                # 2px UP toward the south window. Not the 8px probe, and not a post-split step.
                button = "NONE"
                aim = _sword_aim(pieces, link)
                # Old log was "(dx,dy)face". ",-" meant dy < 0. "None" meant no sword face.
                if aim is not None and aim[1] < 0 and aim[2] is None:
                    if _short_clear(grid, link, "UP", pieces, forecasts, shell):
                        button = "UP"
                _play(env, press() if button == "NONE" else press(button))
                frames += 1
                continue
            stand = (self._eyes, self._anchor_y)
            if dodge != "NONE":
                _walk_frame(env, guard, walk, grid, link, dodge, eyes=stand[0], anchor_y=stand[1])
                frames += 1
                continue
            if shell is not None and (magic >= 16 or flying or int(shell["hit"]) & 0x80):
                button = _fire_button(
                    grid, link, shell, landed, magic, flying, eyes=stand[0], anchor_y=stand[1]
                )
                if button in ("Y",):
                    face = _face_of(int(shell["x"]) - link[0], int(shell["y"]) - link[1])
                    if not _step_blocked(grid, link, face, eyes=stand[0], anchor_y=stand[1]):
                        _play(env, press(face))
                        frames += 1
                    _play(env, press("Y"))
                    frames += 1
                    continue
                _walk_frame(env, guard, walk, grid, link, button, eyes=stand[0], anchor_y=stand[1])
                frames += 1
                continue
            if prize is not None:
                button = _toward(link, (int(prize["x"]), int(prize["y"])))
                _walk_frame(env, guard, walk, grid, link, button, eyes=stand[0], anchor_y=stand[1])
                frames += 1
                continue
            _walk_frame(env, guard, walk, grid, link, "NONE", eyes=stand[0], anchor_y=stand[1])
            frames += 1
        note = _eye_note(_rows(env))
        return _stop(frames, fire_hits, sword_hits, f"timeout shell hp {shell_hp} {note}")



def fight(env: Any) -> FightResult:
    """Fire the shell, sword the pieces, take the heart. Isolated poke only."""
    return Kholdstare(env).run()

def _fire_button(
    grid: np.ndarray,
    link: tuple[int, int],
    shell: dict[str, int],
    ice: list[dict[str, int]],
    magic: int,
    flying: bool, *, eyes: bool, anchor_y: int,
) -> str:
    dx = int(shell["x"]) - link[0]
    dy = int(shell["y"]) - link[1]
    stunned = int(shell["hit"]) & 0x80
    on = _on_ice(ice, link[0], link[1])
    # A clear cardinal. Stepping "away" on this column walks into the shell
    # and the next ice spawns on that same x.
    if _aligned(dx, dy) and not on and magic >= 16 and not flying and not stunned:
        return "Y"
    if _aligned(dx, dy) and not on:
        return "NONE"
    cell = _firing_cell(grid, shell, ice)
    if cell is not None:
        return _toward(link, cell)
    return _away(grid, link, ice, eyes=eyes, anchor_y=anchor_y)


def _stop(frames: int, fire_hits: int, sword_hits: int, blocker: str) -> FightResult:
    return FightResult(
        ok=False,
        frames=frames,
        blocker=blocker,
        fire_hits=fire_hits,
        sword_hits=sword_hits,
    )
