"""Clean power-on to Sanctuary. No refill, no search restore.

The bush is the aligned pose (2431, 1704). Room 0x80 stays off the closed
lines (303, 4232) and the south-east corner through (405, 4286) and
(410, 4285). The flail is only swung at while retracted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from alttp.opening_route.castle_to_sword import SECRET_HOLE_ENTRY_SCRIPT
from alttp.opening_route.room_engine import in_room
from alttp.overworld import read_overworld_tiles
from alttp.primitives import (
    CASTLE_HOSTILE_TYPES,
    SPRITE_BALL_AND_CHAIN,
    SPRITE_BIG_KEY,
    SPRITE_BLUE_SOLDIER,
    SPRITE_GREEN_SOLDIER,
    SPRITE_HEART,
    SPRITE_KNIGHT,
    CombatResult,
    PrimitiveResult,
    Waypoint,
    collect_nearby,
    fight_nearby,
    interact_until,
    move_to,
    move_until,
    settle_control,
    sprites_of_type,
)
from alttp.ram import (
    SRAM_PROGRESS,
    SRAM_PROGRESS_FLAGS,
    read_inventory,
    snapshot_to_diag,
    wram_index,
)
from alttp.room_map import load_room_map
from alttp.route_report import RoutePhaseResult, SegmentResult, phase_from_primitive
from alttp.scripts.audit_power_on import Survival, story_glance
from alttp.scripts.replay_tape import _CommittedRecorder
from alttp.spine.hops import hops_through
from alttp.spine.tape import LiveRun
from alttp.startup import action_for, no_action, snapshot_env, step_frames
from retro_harness.segment_runner import configure_headless

SOLDIERS = (SPRITE_BLUE_SOLDIER, SPRITE_GREEN_SOLDIER, SPRITE_KNIGHT)
BUSH_X, BUSH_Y = 2431, 1704
# Closed room 0x80 deaths. Do not walk either line.
_DEATH_CELL = (303, 4232)
_SE_X, _SE_Y = 390, 4268
_HIT_TILES = ((405, 4286), (410, 4285))
_TRACE: list[tuple[int, int, int]] = []
SPRITE_A = 0x0D90
SPRITE_B = 0x0DA0
SPRITE_DELAY_AUX2 = 0x0E10
_SIN = (
    0, 3, 6, 9, 12, 15, 18, 21, 25, 28, 31, 34, 37, 40, 40, 46,
    49, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95,
    97, 100, 103, 106, 109, 112, 115, 117, 120, 123, 126, 128, 131, 134, 136, 139,
    142, 144, 147, 149, 152, 155, 157, 159, 162, 164, 167, 169, 171, 174, 176, 178,
    181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211,
    212, 214, 216, 217, 219, 221, 222, 224, 225, 227, 228, 230, 231, 232, 234, 235,
    236, 237, 238, 239, 241, 242, 243, 244, 244, 245, 246, 247, 248, 249, 249, 250,
    251, 251, 252, 252, 253, 253, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255,
    256, 255, 255, 255, 255, 255, 255, 255, 254, 254, 254, 253, 253, 252, 252, 251,
    251, 250, 249, 249, 248, 247, 246, 245, 244, 244, 243, 242, 241, 239, 238, 237,
    236, 235, 234, 232, 231, 230, 228, 227, 225, 224, 222, 221, 219, 217, 216, 214,
    212, 211, 209, 207, 205, 203, 201, 199, 197, 195, 193, 191, 189, 187, 185, 183,
    181, 178, 176, 174, 171, 169, 167, 164, 162, 159, 157, 155, 152, 149, 147, 144,
    142, 139, 136, 134, 131, 128, 126, 123, 120, 117, 115, 112, 109, 106, 103, 100,
    97, 95, 92, 89, 86, 83, 80, 77, 74, 71, 68, 65, 62, 59, 56, 53,
    49, 46, 43, 40, 37, 34, 31, 28, 25, 21, 18, 15, 12, 9, 6, 3,
)
_RADIUS = (
    0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2A, 0x2C, 0x2E,
    0x30, 0x2E, 0x2C, 0x2A, 0x28, 0x26, 0x24, 0x22, 0x20, 0x1E, 0x1C, 0x1A, 0x18, 0x16, 0x14, 0x12,
)
_HAND_X = (4, 4, 12, -5)
_HAND_Y = (-2, -2, -6, -4)


def _u8(value: int) -> int:
    return value & 0xFF


def _s8(value: int) -> int:
    value &= 0xFF
    return value - 256 if value >= 128 else value


def _chain_mult(sine: int, radius: int) -> int:
    if sine >= 256:
        return radius
    product = sine * radius
    return (product >> 8) + ((product >> 7) & 1)


def ball_center(env: object, slot: int) -> tuple[int, int] | None:
    """Next-frame flail center, or None while the hitbox is disabled.

    Sprite_6A draws the 16px ball from sprite_A/B before it advances the
    angle, and it disables that box while ai < 2. The RAM after a step is
    the angle the next hitbox will use.
    """
    guard = next((s for s in sprites_of_type(env, (SPRITE_BALL_AND_CHAIN,)) if s.slot == slot), None)
    if guard is None or guard.hp <= 0:
        return None
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    ai = int(ram[0x0D80 + slot])
    if ai < 2:
        return None
    angle = int(ram[SPRITE_A + slot]) | ((int(ram[SPRITE_B + slot]) & 1) << 8)
    radius = _RADIUS[int(ram[SPRITE_DELAY_AUX2 + slot]) & 31]
    cosine = (angle + 0x80) & 0x1FF
    along = _chain_mult(_SIN[angle & 0xFF], radius)
    across = _chain_mult(_SIN[cosine & 0xFF], radius)
    signed_x = -along if angle & 0x100 else along
    signed_y = -across if cosine & 0x100 else across
    facing = int(ram[0x0DE0 + slot]) & 3
    off_x = _s8(_u8(signed_x) - 4 + _u8(_HAND_X[facing]))
    off_y = _s8(_u8(signed_y) - 4 + _u8(_HAND_Y[facing]))
    return guard.x + off_x + 8, guard.y + off_y + 8


def _forbidden(x: int, y: int) -> bool:
    """Closed doorway cell, the south-east corner line, and both flail tiles."""
    if abs(x - _DEATH_CELL[0]) <= 20 and abs(y - _DEATH_CELL[1]) <= 16:
        return True
    if x >= _SE_X and y >= _SE_Y:
        return True
    for hx, hy in _HIT_TILES:
        if abs(x - hx) <= 6 and abs(y - hy) <= 6:
            return True
    return False


def _predict(x: int, y: int, button: str) -> tuple[int, int]:
    if button == "LEFT":
        return x - 2, y
    if button == "RIGHT":
        return x + 2, y
    if button == "UP":
        return x, y - 2
    if button == "DOWN":
        return x, y + 2
    return x, y


def _escape_button(x: int, y: int) -> str:
    best, score = "", -10**9
    for button in ("LEFT", "UP", "DOWN", "RIGHT"):
        nx, ny = _predict(x, y, button)
        if _forbidden(nx, ny):
            continue
        rank = -abs(nx - 360) // 8 - abs(ny - 4256) // 8
        if x >= _SE_X - 16:
            rank += 4 if button in {"LEFT", "UP"} else 0
        if abs(x - 303) <= 28 and abs(y - 4232) <= 24 and button == "DOWN":
            rank += 4
        if rank > score:
            best, score = button, rank
    return best


def _trace(env: object) -> None:
    snap = snapshot_env(env)
    if snap.room_base_id != 0x80:
        return
    row = (snap.link_x, snap.link_y, snap.hp)
    forbidden = _forbidden(snap.link_x, snap.link_y)
    if (
        _TRACE
        and not forbidden
        and abs(_TRACE[-1][0] - row[0]) < 8
        and abs(_TRACE[-1][1] - row[1]) < 8
        and _TRACE[-1][2] == row[2]
    ):
        return
    _TRACE.append(row)
    if forbidden:
        print(f"FORBIDDEN xy=({snap.link_x},{snap.link_y}) hp={snap.hp}", flush=True)


def _toward(dx: int, dy: int) -> str:
    if dx == 0 and dy == 0:
        return ""
    if abs(dx) >= abs(dy) and dx != 0:
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def _press(env: object, button: str, frames: int = 1) -> None:
    step_frames(env, action_for(button) if button else no_action(), frames)


def _safe_press(env: object, button: str) -> None:
    """One frame. A step that would enter a closed line is turned away."""
    snap = snapshot_env(env)
    if button in {"LEFT", "RIGHT", "UP", "DOWN"}:
        nx, ny = _predict(snap.link_x, snap.link_y, button)
        if _forbidden(nx, ny):
            button = _escape_button(snap.link_x, snap.link_y)
    _press(env, button, 1)
    _trace(env)


def _dead(env: object) -> bool:
    return snapshot_env(env).game_mode == 0x12


def _log_pose(tag: str, env: object) -> None:
    snap = snapshot_env(env)
    print(
        f"{tag} room=0x{snap.room_base_id:02X} xy=({snap.link_x},{snap.link_y}) "
        f"hp={snap.hp} mode=0x{snap.game_mode:02X} dir={snap.link_direction}",
        flush=True,
    )


def _reach_bush_axis(env: object, axis: str) -> int:
    """One axis at a time. X before Y so the last held direction can be UP."""
    frames = 0
    recent: list[tuple[int, int]] = []
    goal = BUSH_X if axis == "x" else BUSH_Y
    for _ in range(120):
        snap = snapshot_env(env)
        if snap.indoors or _dead(env):
            return frames
        cur = snap.link_x if axis == "x" else snap.link_y
        if cur == goal:
            return frames
        recent.append((snap.link_x, snap.link_y))
        if len(recent) >= 6 and len(set(recent[-6:])) <= 2:
            _press(env, "", 1)
            frames += 1
            recent.clear()
            continue
        if axis == "x":
            _press(env, "RIGHT" if goal > cur else "LEFT", 1)
        else:
            _press(env, "DOWN" if goal > cur else "UP", 1)
        frames += 1
    return frames


def _face_up_on_bush(env: object) -> int:
    """$2F=0 on the bush cell. UP must not walk off (2431, 1704)."""
    frames = 0
    for _ in range(8):
        snap = snapshot_env(env)
        if (
            snap.link_direction == 0
            and snap.link_x == BUSH_X
            and snap.link_y == BUSH_Y
        ):
            return frames
        _press(env, "UP", 1)
        frames += 1
        snap = snapshot_env(env)
        if (snap.link_x, snap.link_y) == (BUSH_X, BUSH_Y):
            continue
        frames += _reach_bush_axis(env, "x")
        frames += _reach_bush_axis(env, "y")
    return frames


def aligned_bush_entry(env: object, *, candidates=None) -> object:
    """Face-up A at (2431, 1704). One script, no get_state and no set_state."""
    from alttp.ram import snapshot_to_diag as _diag
    from alttp.route_report import RoutePhaseResult as Phase

    del candidates
    frames = 0
    settle = settle_control(env)
    frames += settle.frames
    snap = settle.snapshot
    if snap.in_secret_passage or (snap.indoors and snap.room_base_id == 0x55):
        return Phase("secret_entrance_entry", True, frames, snap, "already in 0x55", _diag(snap))
    _log_pose("approach", env)
    frames += _reach_bush_axis(env, "x")
    frames += _reach_bush_axis(env, "y")
    frames += _face_up_on_bush(env)
    snap = snapshot_env(env)
    if snap.indoors and snap.room_base_id == 0x55:
        settle = settle_control(env, max_frames=300)
        frames += settle.frames
        snap = settle.snapshot
        return Phase("secret_entrance_entry", True, frames, snap, "entered 0x55 while aligning", _diag(snap))
    if snap.indoors or _dead(env):
        return Phase("secret_entrance_entry", False, frames, snap, "aligned walk left the courtyard", _diag(snap))
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    attr, map16, _lift = read_overworld_tiles(ram).sample(snap.link_x + 7, snap.link_y + 6)
    print(
        f"bush pose=({snap.link_x},{snap.link_y}) probe=({snap.link_x + 7},{snap.link_y + 6}) "
        f"attr=0x{attr:02X} map16=0x{map16:X} dir={snap.link_direction}",
        flush=True,
    )
    if (
        snap.link_x != BUSH_X
        or snap.link_y != BUSH_Y
        or snap.link_direction != 0
        or attr != 0x50
        or map16 != 0x36
    ):
        return Phase(
            "secret_entrance_entry", False, frames, snap,
            "bush alignment missed the facing-up probe", _diag(snap),
        )
    script = __import__("alttp.primitives", fromlist=["run_script"]).run_script(
        env, SECRET_HOLE_ENTRY_SCRIPT, stop_when=lambda s: s.indoors,
    )
    frames += script.frames
    for _ in range(120):
        step_frames(env, no_action(), 1)
        frames += 1
        if snapshot_env(env).indoors:
            break
    settle = settle_control(env, max_frames=300)
    frames += settle.frames
    snap = settle.snapshot
    ok = snap.room_base_id == 0x55 and snap.indoors
    _log_pose("hole", env)
    return Phase(
        "secret_entrance_entry", ok, frames, snap,
        "entered via aligned SECRET_HOLE_ENTRY_SCRIPT" if ok else "aligned hole script missed",
        {**_diag(snap), "bush_attr": attr, "bush_map16": map16},
    )


def _staged(x: int, y: int) -> bool:
    """Open floor west of the south-east line and south of the doorway cell."""
    return 328 <= x <= 384 and 4200 <= y <= 4264 and not _forbidden(x, y)


def _log_guard(env: object) -> None:
    guards = sprites_of_type(env, (SPRITE_BALL_AND_CHAIN,))
    if not guards:
        print("guard none", flush=True)
        return
    for guard in guards:
        print(
            f"guard slot={guard.slot} xy=({guard.x},{guard.y}) hp={guard.hp} "
            f"ai={guard.ai_state} dir={guard.direction} ball={ball_center(env, guard.slot)}",
            flush=True,
        )


def _death_reason(env: object) -> str:
    snap = snapshot_env(env)
    _log_pose("death", env)
    _log_guard(env)
    return (
        f"Link died xy=({snap.link_x},{snap.link_y}) hp={snap.hp} "
        f"mode=0x{snap.game_mode:02X}"
    )


def _ball_near(env: object, snap, guard) -> tuple[bool, tuple[int, int] | None]:
    center = ball_center(env, guard.slot)
    if center is None:
        return False, None
    return max(abs(center[0] - snap.link_x), abs(center[1] - snap.link_y)) < 40, center


def _dodge(snap, center: tuple[int, int] | None, guard) -> str:
    cx, cy = center if center is not None else (guard.x, guard.y)
    best, score = "", -10**9
    for button in ("LEFT", "RIGHT", "UP", "DOWN", ""):
        nx, ny = _predict(snap.link_x, snap.link_y, button)
        if _forbidden(nx, ny):
            continue
        dist = max(abs(nx - cx), abs(ny - cy))
        rank = dist * 4
        if nx < 312 or nx > 388:
            rank -= 24
        if ny > 4264 or ny < 4184:
            rank -= 16
        if rank > score:
            best, score = button, rank
    return best


def _clear_blocker(env: object, snap) -> bool:
    """Sword a soldier or pot that is holding the south-wall walk. No flail step."""
    soldiers = sprites_of_type(env, SOLDIERS, max_distance=40)
    if soldiers and snap.hp > 8:
        target = min(soldiers, key=lambda s: s.distance_to(snap))
        toward = _toward(target.x - snap.link_x, target.y - snap.link_y)
        nx, ny = _predict(snap.link_x, snap.link_y, toward)
        if not _forbidden(nx, ny):
            _safe_press(env, toward)
        _press(env, "B", 4)
        _press(env, "", 4)
        _trace(env)
        return True
    _press(env, "B", 4)
    _press(env, "", 2)
    _trace(env)
    return False


def _stage_for_flail(env: object) -> bool:
    """South of (303, 4232), then north before x reaches the south-east line."""
    _log_guard(env)
    stuck = 0
    for _ in range(1400):
        if _dead(env) or snapshot_env(env).hp <= 0:
            _death_reason(env)
            return False
        snap = snapshot_env(env)
        if not in_room(snap, 0x80):
            _log_pose("left_80", env)
            return False
        if _staged(snap.link_x, snap.link_y):
            _log_pose("staged", env)
            _log_guard(env)
            return True
        guards = [s for s in sprites_of_type(env, (SPRITE_BALL_AND_CHAIN,)) if s.hp > 0]
        guard = min(guards, key=lambda s: s.distance_to(snap)) if guards else None
        near, center = _ball_near(env, snap, guard) if guard is not None else (False, None)
        if near or (center is not None and max(abs(guard.x - snap.link_x), abs(guard.y - snap.link_y)) < 36):
            button = _dodge(snap, center, guard)
        elif snap.link_x < 280 and snap.link_y < 4276:
            button = "DOWN"
        elif snap.link_y >= 4272 and snap.link_x < 348:
            button = "RIGHT"
        elif snap.link_x >= 330 and snap.link_y > 4256:
            button = "UP"
        else:
            button = _toward(360 - snap.link_x, 4256 - snap.link_y)
        nx, ny = _predict(snap.link_x, snap.link_y, button)
        if _forbidden(nx, ny):
            button = _escape_button(snap.link_x, snap.link_y)
        before = (snap.link_x, snap.link_y)
        _safe_press(env, button)
        snap = snapshot_env(env)
        if snap.hp <= 0 or _dead(env):
            _death_reason(env)
            return False
        stuck = stuck + 1 if (snap.link_x, snap.link_y) == before else 0
        if stuck < 8:
            continue
        stuck = 0
        _clear_blocker(env, snap)
        side = "RIGHT" if snap.link_x < 360 else "UP"
        if _forbidden(*_predict(snap.link_x, snap.link_y, side)):
            side = _escape_button(snap.link_x, snap.link_y)
        for _ in range(6):
            _safe_press(env, side)
    _log_pose("stage_timeout", env)
    _log_guard(env)
    return False


def _swing(env: object) -> bool:
    """Tap B and finish the swing. False once HP is already 0."""
    _press(env, "B", 1)
    _trace(env)
    for _ in range(8):
        snap = snapshot_env(env)
        if snap.hp <= 0 or _dead(env):
            _death_reason(env)
            return False
        _press(env, "", 1)
        _trace(env)
    return snapshot_env(env).hp > 0


def _kite_ball(env: object) -> str:
    """Kill sprite 0x6A from the staged floor. Never the two closed lines."""
    last_hp = snapshot_env(env).hp
    idle = 0
    for _ in range(7000):
        if _dead(env) or snapshot_env(env).hp <= 0:
            return _death_reason(env)
        snap = snapshot_env(env)
        if not in_room(snap, 0x80):
            return f"left 0x80 for 0x{snap.room_base_id:02X}"
        if snap.hp != last_hp:
            _log_pose(f"hp_{last_hp}_to_{snap.hp}", env)
            _log_guard(env)
            last_hp = snap.hp
            if snap.hp <= 0:
                return _death_reason(env)
        guards = [s for s in sprites_of_type(env, (SPRITE_BALL_AND_CHAIN,)) if s.hp > 0]
        if not guards:
            _log_pose("guard_down", env)
            return ""
        guard = min(guards, key=lambda s: s.distance_to(snap))
        center = ball_center(env, guard.slot)
        dx, dy = guard.x - snap.link_x, guard.y - snap.link_y
        reach = max(abs(dx), abs(dy))
        near = center is not None and max(
            abs(center[0] - snap.link_x), abs(center[1] - snap.link_y)
        ) < 40
        hearts = sprites_of_type(env, (SPRITE_HEART,), max_distance=160)
        if hearts and snap.hp < 24 and snap.iframes == 0 and not near:
            target = min(hearts, key=lambda s: s.distance_to(snap))
            button = _toward(target.x - snap.link_x, target.y - snap.link_y)
            nx, ny = _predict(snap.link_x, snap.link_y, button)
            if not _forbidden(nx, ny):
                _safe_press(env, button)
                idle = 0
                continue
        if near or (center is not None and reach < 36):
            _safe_press(env, _dodge(snap, center, guard))
            idle = 0
            continue
        if center is None and 18 <= reach <= 32 and snap.hp > 4:
            toward = _toward(dx, dy)
            nx, ny = _predict(snap.link_x, snap.link_y, toward)
            if reach >= 22 and not _forbidden(nx, ny) and not (nx >= _SE_X and ny >= _SE_Y - 8):
                _safe_press(env, toward)
            if not _swing(env):
                return _death_reason(env)
            idle = 0
            continue
        if center is None and reach > 32 and snap.hp > 8:
            toward = _toward(dx, dy)
            nx, ny = _predict(snap.link_x, snap.link_y, toward)
            if _forbidden(nx, ny):
                _safe_press(env, _dodge(snap, None, guard))
            else:
                _safe_press(env, toward)
            idle = 0
            continue
        if center is None and reach < 18:
            _safe_press(env, _dodge(snap, (guard.x, guard.y), guard))
            idle = 0
            continue
        _safe_press(env, "")
        idle += 1
        if idle > 45 and center is not None:
            _safe_press(env, _dodge(snap, center, guard))
            idle = 0
    snap = snapshot_env(env)
    return f"ball fight budget xy=({snap.link_x},{snap.link_y}) hp={snap.hp}"


def _finish_rescue(env: object, record) -> None:
    for _ in range(240):
        if sprites_of_type(env, (SPRITE_BIG_KEY,)) or _dead(env):
            break
        step_frames(env, no_action(), 1)
    record("big_key_pickup", collect_nearby(env, (SPRITE_BIG_KEY,), max_distance=400, max_frames=600))
    record("prize_settle", settle_control(env, max_frames=400))
    point = load_room_map("room_80").point("rescue_cell_align")
    record("rescue_cell_align", move_to(
        env, Waypoint(point.x, point.y, tolerance=1, room=0x80, label="rescue_cell_align"),
        step_size=1, max_frames=900,
    ))
    record("cell_unlock", move_until(
        env, ("UP",), lambda s: s.is_text_mode, max_frames=240, step_size=1,
    ))
    record("zelda_dialogue", interact_until(env, lambda s: s.has_zelda_follower, max_cycles=200))
    record("rescue_settle", settle_control(env, max_frames=480))


def run_from_b1_west(env: object, *, source: str = "natural_boot") -> SegmentResult:
    """Same B1 stairs as the measured route, then an east flail path in 0x80."""
    phases: list[RoutePhaseResult] = []

    def record(label, result):
        phases.append(phase_from_primitive(label, result))
        if not result.ok:
            raise RuntimeError(result.reason)
        return result

    def walk(room, label, tolerance=1):
        point = load_room_map(f"room_{room:02x}").point(label)
        if point is None:
            raise RuntimeError(f"missing geometry room_{room:02x}:{label}")
        waypoint = Waypoint(point.x, point.y, tolerance=tolerance, room=room, label=label)
        moved = move_to(env, waypoint, step_size=1, max_frames=900)
        frames = moved.frames
        for _ in range(2):
            if moved.ok or moved.snapshot.has_control:
                break
            settled = settle_control(env, max_frames=480)
            frames += settled.frames
            if not settled.ok:
                moved = settled
                break
            moved = move_to(env, waypoint, step_size=1, max_frames=900)
            frames += moved.frames
        return record(label, PrimitiveResult(moved.ok, moved.reason, frames, moved.snapshot))

    def push(label, button, predicate, limit=600, settle=True):
        record(label, move_until(env, (button,), predicate, max_frames=limit, step_size=1))
        if settle:
            record(label + "_settle", settle_control(env, max_frames=480))

    def fight(label, distance=180):
        record(label, fight_nearby(
            env, max_distance=distance, attack_distance=50, max_cycles=450,
        ))
        record(label + "_settle", settle_control(env, max_frames=480))

    blocker = ""
    try:
        snap = snapshot_env(env)
        if not in_room(snap, 0x81) or snap.follower != 0 or not snap.has_control:
            raise RuntimeError("rescue requires controllable 0x81 without follower")
        for label in ("rescue_lower_corner", "rescue_lower_north", "rescue_stair_foot"):
            walk(0x81, label)
        push("east_stair", "UP", lambda s: s.layer == 0 and s.has_control)
        walk(0x81, "rescue_stair_head")
        walk(0x81, "rescue_upper_corner", 0)
        walk(0x81, "rescue_upper_south", 0)
        walk(0x81, "rescue_west_stair")
        push("west_stair", "UP", lambda s: s.layer == 1 and s.has_control)
        walk(0x81, "rescue_north_door")
        push("north_to_71", "UP", lambda s: in_room(s, 0x71))
        fight("west_shutter_guard")
        walk(0x71, "rescue_east_door")
        walk(0x71, "rescue_east_guard")
        fight("east_key_guard")
        record("small_key_pickup", collect_nearby(env, (0xE4,), max_distance=300, max_frames=600))
        if snapshot_env(env).num_keys < 1:
            raise RuntimeError("east guard key was not earned")
        walk(0x71, "rescue_alcove_nw")
        record("alcove_wrap", move_until(
            env, ("UP", "LEFT"), lambda s: s.link_x <= 688 and s.has_control,
            max_frames=400, step_size=1,
        ))
        walk(0x71, "rescue_east_door")
        walk(0x71, "rescue_key_door")
        record("key_door_settle", settle_control(env, max_frames=480))
        walk(0x71, "rescue_north_landing")
        push("north_lower_stair", "UP", lambda s: s.layer == 1 and s.has_control)
        walk(0x71, "rescue_spiral_align")
        walk(0x71, "rescue_spiral_mouth")
        push("down_to_70", "UP", lambda s: in_room(s, 0x70))
        walk(0x70, "rescue_stair_hall")
        walk(0x70, "rescue_down_stair")
        push("down_to_80", "UP", lambda s: in_room(s, 0x80))
        _log_pose("entered_80", env)
        if snapshot_env(env).hp <= 0 or not _stage_for_flail(env):
            snap = snapshot_env(env)
            raise RuntimeError(
                f"0x80 stage failed xy=({snap.link_x},{snap.link_y}) "
                f"hp={snap.hp} mode=0x{snap.game_mode:02X}"
            )
        reason = _kite_ball(env)
        snap = snapshot_env(env)
        phases.append(RoutePhaseResult(
            phase="earn_big_key", ok=reason == "" and snap.hp > 0,
            frames=0, snapshot=snap, detail=reason or "flail from the staged floor",
        ))
        if reason:
            raise RuntimeError(reason)
        _finish_rescue(env, record)
        if int(read_inventory(np.asarray(env.get_ram(), dtype=np.uint8)).big_key) == 0:
            raise RuntimeError("big key did not rise")
    except RuntimeError as exc:
        blocker = str(exc)
    snap = snapshot_env(env)
    return SegmentResult(
        ok=not blocker and snap.has_zelda_follower and snap.in_zelda_cell,
        phase="zelda_rescued" if not blocker else (phases[-1].phase if phases else "entry"),
        frames=sum(p.frames for p in phases), snapshot=snap, phases=phases,
        source=source,
        acceptance={"zelda_follower": snap.has_zelda_follower, "in_zelda_cell": snap.in_zelda_cell},
        blocker=blocker,
        notes=["Room 0x80 staged off the closed doorway and south-east flail lines."],
    )


def _nearest_threat(env: object, types, max_distance: int):
    snap = snapshot_env(env)
    found = sprites_of_type(env, types, max_distance=max_distance)
    if not found:
        return None
    return min(found, key=lambda s: s.distance_to(snap))


def _space_step(env: object, target) -> int:
    """Sword from outside contact range. A blocked step cuts the bush."""
    snap = snapshot_env(env)
    hearts = sprites_of_type(env, (SPRITE_HEART,), max_distance=180)
    if hearts and snap.hp < 24 and snap.iframes == 0:
        heart = min(hearts, key=lambda s: s.distance_to(snap))
        _press(env, _toward(heart.x - snap.link_x, heart.y - snap.link_y), 1)
        return 1
    dx, dy = target.x - snap.link_x, target.y - snap.link_y
    reach = max(abs(dx), abs(dy))
    away = _toward(-dx, -dy) or "DOWN"
    if snap.room_base_id == 0x82 and snap.link_y > 4280 and away == "LEFT" and snap.link_x <= 1336:
        away = "UP" if dy > 0 else "DOWN"
    if target.sprite_type == SPRITE_BALL_AND_CHAIN or reach < 18:
        _press(env, away, 2)
        return 2
    if snap.hp <= 8 and reach < 22:
        _press(env, away, 2)
        return 2
    if reach <= 28:
        before = snap.hp
        _press(env, _toward(dx, dy), 1)
        _press(env, "B", 4)
        _press(env, "", 2)
        if snapshot_env(env).hp < before:
            _press(env, away, 4)
            return 11
        return 7
    before_xy = (snap.link_x, snap.link_y)
    _press(env, _toward(dx, dy), 2)
    if (snapshot_env(env).link_x, snapshot_env(env).link_y) == before_xy:
        _press(env, "B", 4)
        _press(env, "", 2)
        return 8
    return 2


def _guard_fight(_orig):
    """Keep a gap while swinging. Contact with green mail is 4, and 4 HP dies."""

    def fight_nearby(
        env,
        *,
        sprite_types=CASTLE_HOSTILE_TYPES,
        room=None,
        max_distance=150,
        attack_distance=60,
        stop_when=None,
        max_cycles=500,
    ):
        del attack_distance
        frames = 0
        limit = max(max_cycles, 240)
        for _ in range(limit):
            snap = snapshot_env(env)
            if snap.game_mode == 0x12:
                _log_pose("death", env)
                return CombatResult(False, "Link died", frames, snap)
            if stop_when is not None and stop_when(env):
                return CombatResult(True, "combat acceptance reached", frames, snap)
            expected = snap.room_base_id if room is None else room
            if snap.room_base_id != expected:
                return CombatResult(
                    False, f"left combat room 0x{expected:02X}", frames, snap,
                )
            target = _nearest_threat(env, sprite_types, max_distance)
            if target is None:
                return CombatResult(True, "no nearby targets", frames, snap)
            frames += _space_step(env, target)
        return CombatResult(False, "combat cycle limit", frames, snapshot_env(env))

    return fight_nearby


def _guard_move(orig):
    """Below 16 HP, clear a nearby soldier before the waypoint step."""

    def move_to(env, waypoint, **kwargs):
        snap = snapshot_env(env)
        if snap.hp > 16:
            return orig(env, waypoint, **kwargs)
        step_size = int(kwargs.get("step_size", 3))
        max_frames = int(kwargs.get("max_frames", 600))
        frames = 0
        unchanged = 0
        previous = None
        while frames < max_frames:
            snap = snapshot_env(env)
            if snap.room_base_id != getattr(move_to, "room", None) or snap.hp != getattr(move_to, "hp", None):
                _log_pose("walk", env)
                move_to.room = snap.room_base_id
                move_to.hp = snap.hp
            if snap.hp > 16:
                rest = orig(env, waypoint, **{**kwargs, "max_frames": max(1, max_frames - frames)})
                return PrimitiveResult(
                    rest.ok, rest.reason, frames + rest.frames, rest.snapshot,
                )
            if waypoint.room is not None and snap.room_base_id != waypoint.room:
                return PrimitiveResult(
                    False, f"left room 0x{waypoint.room:02X} before {waypoint.label or 'waypoint'}",
                    frames, snap,
                )
            if waypoint.reached(snap.link_x, snap.link_y):
                return PrimitiveResult(True, waypoint.label or "waypoint reached", frames, snap)
            if snap.game_mode == 0x12:
                _log_pose("death", env)
                return PrimitiveResult(False, "Link died", frames, snap)
            target = _nearest_threat(env, CASTLE_HOSTILE_TYPES, 96)
            if target is not None:
                reach = max(abs(target.x - snap.link_x), abs(target.y - snap.link_y))
                if reach <= 36:
                    frames += _space_step(env, target)
                    continue
            dx = waypoint.x - snap.link_x
            dy = waypoint.y - snap.link_y
            buttons = []
            if abs(dx) > waypoint.tolerance:
                buttons.append("RIGHT" if dx > 0 else "LEFT")
            if abs(dy) > waypoint.tolerance:
                buttons.append("DOWN" if dy > 0 else "UP")
            step_frames(env, action_for(*buttons) if buttons else no_action(), step_size)
            frames += step_size
            xy = (snap.link_x, snap.link_y)
            unchanged = unchanged + 1 if xy == previous else 0
            previous = xy
            if unchanged >= int(kwargs.get("stuck_cycles", 28)):
                return PrimitiveResult(
                    False, f"stuck before {waypoint.label or 'waypoint'}", frames, snapshot_env(env),
                )
        return PrimitiveResult(
            False, f"timeout before {waypoint.label or 'waypoint'}", frames, snapshot_env(env),
        )

    return move_to


def _guard_cut(orig):
    """Bush cuts walk through spears. Clear a close soldier first."""

    def cut_push(env, face, *, walk=12, swings=2):
        frames = 0
        for _ in range(12):
            snap = snapshot_env(env)
            target = _nearest_threat(env, CASTLE_HOSTILE_TYPES, 64)
            if target is None:
                break
            reach = max(abs(target.x - snap.link_x), abs(target.y - snap.link_y))
            if reach > 30 and snap.hp > 12:
                break
            frames += _space_step(env, target)
        return frames + orig(env, face, walk=walk, swings=swings)

    return cut_push


def _leave_reentered_secret(env: object) -> None:
    """Hole re-entry does not leave on DOWN. Use the room 0x55 stairs."""
    from alttp.opening_route.room_engine import run_room_edge

    _log_pose("reenter", env)
    edge = run_room_edge(
        env, "room_55", "stairs_to_courtyard", clear=True, source="sanctuary_clean",
    )
    _log_pose(f"left55_{edge.ok}", env)


def _install() -> None:
    global fight_nearby, move_to
    import alttp.opening_route.castle_dungeon as dungeon
    import alttp.opening_route.castle_to_sword as hole
    import alttp.opening_route.pocket_to_main_hall as pocket
    import alttp.opening_route.segment as segment
    import alttp.primitives as prim

    wrapped = _guard_fight(prim.fight_nearby)
    prim.fight_nearby = wrapped
    dungeon.fight_nearby = wrapped
    fight_nearby = wrapped
    guarded_move = _guard_move(prim.move_to)
    prim.move_to = guarded_move
    move_to = guarded_move
    pocket.cut_push = _guard_cut(pocket.cut_push)
    hole.attempt_secret_entrance_entry = aligned_bush_entry
    row = segment.segment_registry()["room_81_to_zelda"]
    object.__setattr__(row, "play_fn", run_from_b1_west)
    orig_step = pocket.step_frames
    leaves = {"n": 0}

    def step_frames(env, action, n, *args, **kwargs):
        snap = snapshot_env(env)
        if n == 40 and pocket._reentered_secret(snap) and leaves["n"] < 2:
            leaves["n"] += 1
            _leave_reentered_secret(env)
            if not pocket._reentered_secret(snapshot_env(env)):
                return None
        return orig_step(env, action, n, *args, **kwargs)

    pocket.step_frames = step_frames


class _Recorder(_CommittedRecorder):
    def __init__(self, raw: object) -> None:
        super().__init__(raw)
        self.survival = Survival(False)

    def _step(self, action, *args, **kwargs):
        out = super()._step(action, *args, **kwargs)
        return out


def main() -> None:
    from alttp.opening_route.full_tip import run_to_verified_tip
    from alttp.opening_route.segment import get_segment
    from alttp.startup import build_boot_env

    _install()
    configure_headless()
    raw = build_boot_env(state_name="NONE")
    rec = _Recorder(raw)
    live = LiveRun(rec.attach(), provider="alttp.sanctuary_clean")
    hops = hops_through("sanctuary")
    out = Path("alttp/scratch/sanctuary_clean_082.json")

    def _segment(sid: str):
        snap = snapshot_env(live.env)
        print(
            f"stage={sid} start_frames={len(rec.buttons)} hp={snap.hp} "
            f"room=0x{snap.room_base_id:02X} xy=({snap.link_x},{snap.link_y})",
            flush=True,
        )
        return get_segment(sid)

    try:
        result = run_to_verified_tip(
            live.env, close=False,
            segment_ids=tuple(s for hop in hops for s in hop.stages),
            success=hops[-1].success,
            get_segment_fn=_segment,
        )
        story = story_glance(raw)
        ram = raw.get_ram()
        story["flags_exact"] = int(ram[wram_index(SRAM_PROGRESS_FLAGS)])
        story["progress_exact"] = int(ram[wram_index(SRAM_PROGRESS)])
        audit = live.report()
        snap = result.snapshot
        report = {
            "ok": result.ok,
            "blocker": result.blocker,
            "phase": result.phase,
            "frames": len(rec.buttons),
            "final": snapshot_to_diag(snap),
            "story": story,
            "set_state": audit.set_state,
            "writes": audit.writes,
            "restores": rec.restores,
            "swallowed": rec.swallowed,
            "sites": list(audit.sites),
            "survival": rec.survival.report(),
            "room80_trace": [
                {"x": x, "y": y, "hp": hp} for x, y, hp in _TRACE
            ],
            "forbidden_steps": [
                {"x": x, "y": y, "hp": hp}
                for x, y, hp in _TRACE
                if _forbidden(x, y)
            ],
        }
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"DONE ok={result.ok} room=0x{snap.room_base_id:02X} "
            f"xy=({snap.link_x},{snap.link_y}) hp={snap.hp} mode=0x{snap.game_mode:02X} "
            f"progress={story['progress_exact']} flags=0x{story['flags_exact']:02X} "
            f"follower={snap.follower} set_state={audit.set_state} writes={audit.writes} "
            f"restores={rec.restores} forbidden={len(report['forbidden_steps'])} "
            f"blocker={result.blocker!r}",
            flush=True,
        )
    finally:
        raw.close()


if __name__ == "__main__":
    main()
