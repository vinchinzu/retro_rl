"""Level 5 west-door hops: 0x27 → 0x26 → 0x25 → 0x24.

Each hop is the shared ROM-lattice door walk (``dungeon.ops.exit_door``).
The hand axis paths it replaced (y=141 / south 189 / north 109 bands) were
tuned to arrival poses and stalled from any other: R18 pressed LEFT at
(32,173) in 0x25, below the door lane, until the budget ran out.
"""

from __future__ import annotations

from dataclasses import dataclass

from zelda_i.dungeon.ops import exit_door, idle
from zelda_i.level5.dungeon import (
    LEVEL_5,
    ROOM_L5_WEST_24,
    ROOM_L5_WEST_25,
    ROOM_L5_WEST_26,
)
from zelda_i.ram import PLAY_MODE, read_snapshot


@dataclass(frozen=True)
class WestLeaveSpec:
    """One 0x2N west leave: destination room and the door push budget."""

    dest_room: int
    push_frames: int
    extra: tuple[tuple[str, object], ...] = ()


WEST_27_TO_26 = WestLeaveSpec(dest_room=ROOM_L5_WEST_26, push_frames=220)
WEST_26_TO_25 = WestLeaveSpec(dest_room=ROOM_L5_WEST_25, push_frames=220)
WEST_25_TO_24 = WestLeaveSpec(
    dest_room=ROOM_L5_WEST_24,
    push_frames=240,
    extra=(("fought_digdogger", False),),
)


def walk_west(env, assist, total: list[int], spec: WestLeaveSpec) -> dict:
    """Walk one ``WestLeaveSpec`` west through its door. No combat, no pokes."""
    keys0 = int(read_snapshot(env.get_ram()).keys)
    door = exit_door(env, assist, total, "LEFT", push=spec.push_frames)
    idle(env, assist, total, 36)
    snap = read_snapshot(env.get_ram())
    return {
        "path": door.get("via") or door.get("result"),
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
    return walk_west(env, assist, total, WEST_27_TO_26)


def walk_west_from_26(env, assist, total: list[int]) -> dict:
    return walk_west(env, assist, total, WEST_26_TO_25)


def walk_west_from_25(env, assist, total: list[int]) -> dict:
    return walk_west(env, assist, total, WEST_25_TO_24)
