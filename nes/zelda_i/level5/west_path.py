"""Level 5 west-door hops: 0x27 → 0x26 → 0x25 → 0x24.

One env-stepping engine over ``WestLeaveSpec`` rows: walk a candidate axis
path, align on the west door, push through it. Room specs and stop
predicates remain in ``level5.dungeon``.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level5.dungeon import (
    LEVEL_5,
    ROOM_L5_WEST_24,
    ROOM_L5_WEST_25,
    ROOM_L5_WEST_26,
)
from zelda_i.level5.path import walk_axis
from zelda_i.ram import PLAY_MODE, read_snapshot as _rs

# West door mouth. y=189 is the south band that clears the x=160 pinch;
# y=109 the north band; y=141 the door channel itself.
WEST_DOOR_X = 32
WEST_DOOR_Y = 141

WEST27_TO_26_PATHS = (
    ("east_wall_south189_west_door", (("x", 208), ("y", 189), ("x", 32), ("y", 141), ("x", 32))),
)

WEST26_TO_25_PATHS = (
    ("y141_west", (("y", 141), ("x", 32))),
    ("south189_west", (("y", 189), ("x", 32), ("y", 141))),
    ("north109_west", (("y", 109), ("x", 32), ("y", 141))),
    ("east208_south189_west", (("x", 208), ("y", 189), ("x", 32), ("y", 141))),
)

WEST25_TO_24_PATHS = (
    ("y141_west", (("y", 141), ("x", 32))),
    ("south189_west", (("y", 189), ("x", 32), ("y", 141))),
    ("north109_west", (("y", 109), ("x", 80), ("y", 141), ("x", 32))),
    ("east208_south189_west", (("x", 208), ("y", 189), ("x", 32), ("y", 141))),
    ("south173_west64", (("y", 173), ("x", 64), ("y", 141), ("x", 32))),
)


@dataclass(frozen=True)
class WestLeaveSpec:
    """One 0x2N west leave: candidate paths, door align, push budget."""

    dest_room: int
    paths: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    push_frames: int
    # None = single proven path, take it without probing the door stand.
    probe_tol: tuple[int, int] | None = None
    # Frames for the second door align before the push (None = skip it).
    align_frames: int | None = None
    fallback_path: str = "y141_west"
    extra: tuple[tuple[str, object], ...] = ()


WEST_27_TO_26 = WestLeaveSpec(
    dest_room=ROOM_L5_WEST_26,
    paths=WEST27_TO_26_PATHS,
    push_frames=220,
    fallback_path="east_wall_south189_west_door",
)

WEST_26_TO_25 = WestLeaveSpec(
    dest_room=ROOM_L5_WEST_25,
    paths=WEST26_TO_25_PATHS,
    push_frames=220,
    probe_tol=(6, 4),
    align_frames=32,
)

WEST_25_TO_24 = WestLeaveSpec(
    dest_room=ROOM_L5_WEST_24,
    paths=WEST25_TO_24_PATHS,
    push_frames=240,
    probe_tol=(8, 8),
    align_frames=28,
    extra=(("fought_digdogger", False),),
)


def _align_door(env, assist, total: list[int], tx: int, ty: int = 141, frames: int = 24) -> list[int]:
    for _ in range(frames):
        snap = _rs(env.get_ram())
        if abs(snap.link_x - tx) <= 2 and abs(snap.link_y - ty) <= 2:
            break
        if abs(snap.link_y - ty) > 2:
            env.step(nes_action("DOWN" if snap.link_y < ty else "UP"))
        else:
            env.step(nes_action("LEFT" if snap.link_x > tx else "RIGHT"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    snap = _rs(env.get_ram())
    return [snap.link_x, snap.link_y]


def _push_left(env, assist, total: list[int], frames: int = 220) -> None:
    room0 = _rs(env.get_ram()).screen
    for _ in range(frames):
        snap = _rs(env.get_ram())
        if snap.screen != room0:
            break
        env.step(nes_action("LEFT"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])


def _idle(env, assist, total: list[int], frames: int = 36) -> None:
    for _ in range(frames):
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])


def walk_west(env, assist, total: list[int], spec: WestLeaveSpec) -> dict:
    """Walk one ``WestLeaveSpec`` west through its door. No combat, no pokes."""
    snap = _rs(env.get_ram())
    keys0 = int(snap.keys)
    log = [{"step": "start", "xy": [snap.link_x, snap.link_y], "room": snap.screen}]
    used = None
    for name, steps in spec.paths:
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=500)
            snap = _rs(env.get_ram())
            log.append(
                {
                    "step": f"{name}:{axis}:{tgt}",
                    "ok": ok,
                    "xy": [snap.link_x, snap.link_y],
                    "room": snap.screen,
                }
            )
        _align_door(env, assist, total, WEST_DOOR_X, WEST_DOOR_Y)
        if spec.probe_tol is None:
            used = name
            break
        snap = _rs(env.get_ram())
        tol_x, tol_y = spec.probe_tol
        if abs(snap.link_x - WEST_DOOR_X) <= tol_x and abs(snap.link_y - WEST_DOOR_Y) <= tol_y:
            used = name
            break
    if spec.align_frames is not None:
        _align_door(env, assist, total, WEST_DOOR_X, WEST_DOOR_Y, frames=spec.align_frames)
    _push_left(env, assist, total, frames=spec.push_frames)
    _idle(env, assist, total, 36)
    snap = _rs(env.get_ram())
    return {
        "path": used or spec.fallback_path,
        "log": log,
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        **dict(spec.extra),
        "success": (
            snap.level == LEVEL_5
            and snap.screen == spec.dest_room
            and snap.mode == PLAY_MODE
        ),
    }


def walk_west_from_27(env, assist, total: list[int]) -> dict:
    """Proven 0x27 leave: east wall, south y=189, west key door → 0x26."""
    return walk_west(env, assist, total, WEST_27_TO_26)


def walk_west_from_26(env, assist, total: list[int]) -> dict:
    """Proven 0x26 leave: y=141 then west open door → 0x25. Moat/C-block fallbacks."""
    return walk_west(env, assist, total, WEST_26_TO_25)


def walk_west_from_25(env, assist, total: list[int]) -> dict:
    """Proven 0x25 leave: y=141 then west key door → 0x24. Door only; no Digdogger."""
    return walk_west(env, assist, total, WEST_25_TO_24)


__all__ = [
    "WEST25_TO_24_PATHS",
    "WEST26_TO_25_PATHS",
    "WEST27_TO_26_PATHS",
    "WEST_25_TO_24",
    "WEST_26_TO_25",
    "WEST_27_TO_26",
    "WEST_DOOR_X",
    "WEST_DOOR_Y",
    "WestLeaveSpec",
    "walk_west",
    "walk_west_from_25",
    "walk_west_from_26",
    "walk_west_from_27",
]
