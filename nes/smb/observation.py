"""Shared SMB observation tuple for the approximate stepper and residual lattice.

Lattice (same shape as Super Metroid, SMB addresses):
- Oπ: pixel x/y, pose ($000E), room (world/level/area)
- Oσ: Oπ plus subpixels ($0400, $0416)
- Oσ+: Oσ plus enemy slot 0 flag/type
- O†: lives / death (separate from Oπ)

Speeds ($0057, $009F) live in first-differing-field, not a second σ+.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from smb.ram import (
    ADDR_AREA_POINTER,
    ADDR_ENEMY_FLAG,
    ADDR_ENEMY_TYPE,
    ADDR_FRAME_COUNTER,
    ADDR_JUMP_ORIGIN_Y,
    ADDR_LEVEL,
    ADDR_LIVES,
    ADDR_OPER_MODE,
    ADDR_PLAYER_FACING,
    ADDR_PLAYER_MOTION,
    ADDR_PLAYER_STATE,
    ADDR_PLAYER_X_FRAC,
    ADDR_PLAYER_Y,
    ADDR_PLAYER_Y_FRAC,
    ADDR_RUNNING_SPEED,
    ADDR_VERTICAL_FORCE,
    ADDR_VERTICAL_FORCE_DOWN,
    ADDR_WORLD,
    ADDR_X_FORCE,
    ADDR_Y_MOVE_FORCE,
    PLAYER_STATE_DYING,
    is_dying,
    player_x,
    player_x_speed,
    player_y_speed,
    timer_value,
)

__all__ = [
    "DEFAULT_GROUND_Y",
    "Observation",
    "level1_start_obs",
    "observation_from_ram",
    "pack_room",
    "unpack_room",
]

DEFAULT_GROUND_Y = 176
_PIT_Y = 240


def pack_room(world: int, level: int, area_pointer: int) -> int:
    """Pack area identity into one Oπ room integer."""
    return ((int(world) & 0xFF) << 16) | ((int(level) & 0xFF) << 8) | (int(area_pointer) & 0xFF)


def unpack_room(room: int) -> tuple[int, int, int]:
    """Inverse of :func:`pack_room`."""
    value = int(room) & 0xFFFFFF
    return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF


def _is_dead(player_state: int, oper_mode: int, y: int) -> bool:
    if int(player_state) == PLAYER_STATE_DYING:
        return True
    if int(oper_mode) == 3:
        return True
    return int(y) >= _PIT_Y


@dataclass(frozen=True)
class Observation:
    """One-frame residual observation, readable from RAM or the pure stepper."""

    frame: int
    x: int
    y: int
    pose: int
    room: int
    sub_x: int
    sub_y: int
    velocity_x: int
    velocity_y: int
    energy: int | None
    dead: bool
    frame_counter: int | None
    enemy0_active: int = 0
    enemy0_type: int = 0
    facing: int = 1
    on_ground: bool = True
    x_force: int = 0
    running_speed: int = 0
    a_held: bool = False
    ground_y: int = DEFAULT_GROUND_Y
    vertical_force: int = 0
    vertical_force_down: int = 0x70
    y_move_force: int = 0
    jump_origin_y: int = DEFAULT_GROUND_Y
    oper_mode: int = 1
    timer: int = 400

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(**dict(data))

    @property
    def world(self) -> int:
        return unpack_room(self.room)[0]

    @property
    def level(self) -> int:
        return unpack_room(self.room)[1]

    @property
    def area_pointer(self) -> int:
        return unpack_room(self.room)[2]


def level1_start_obs(*, frame: int = 0) -> Observation:
    """Controllable 1-1 stand (matches live Level1_1.state kinematics)."""
    return Observation(
        frame=frame,
        x=40,
        y=DEFAULT_GROUND_Y,
        pose=0x08,
        room=pack_room(0, 0, 194),
        sub_x=0,
        sub_y=0,
        velocity_x=0,
        velocity_y=0,
        energy=2,
        dead=False,
        frame_counter=178,
        enemy0_active=0,
        enemy0_type=0,
        facing=1,
        on_ground=True,
        x_force=0,
        running_speed=0,
        a_held=False,
        ground_y=DEFAULT_GROUND_Y,
        vertical_force=0,
        vertical_force_down=0x70,
        y_move_force=0,
        jump_origin_y=DEFAULT_GROUND_Y,
        oper_mode=1,
        timer=400,
    )


def observation_from_ram(
    ram: np.ndarray,
    frame: int = 0,
    *,
    a_held: bool = False,
    ground_y: int | None = None,
) -> Observation:
    """Extract a residual observation from a NES RAM snapshot."""
    y = int(ram[ADDR_PLAYER_Y])
    motion = int(ram[ADDR_PLAYER_MOTION])
    on_ground = motion == 0
    if ground_y is None:
        floor = y if on_ground else DEFAULT_GROUND_Y
    else:
        floor = int(ground_y)
    pose = int(ram[ADDR_PLAYER_STATE])
    oper_mode = int(ram[ADDR_OPER_MODE])
    return Observation(
        frame=int(frame),
        x=player_x(ram),
        y=y,
        pose=pose,
        room=pack_room(
            int(ram[ADDR_WORLD]),
            int(ram[ADDR_LEVEL]),
            int(ram[ADDR_AREA_POINTER]),
        ),
        sub_x=int(ram[ADDR_PLAYER_X_FRAC]),
        sub_y=int(ram[ADDR_PLAYER_Y_FRAC]),
        velocity_x=player_x_speed(ram),
        velocity_y=player_y_speed(ram),
        energy=int(ram[ADDR_LIVES]),
        dead=_is_dead(pose, oper_mode, y) or is_dying(ram),
        frame_counter=int(ram[ADDR_FRAME_COUNTER]),
        enemy0_active=int(ram[ADDR_ENEMY_FLAG]),
        enemy0_type=int(ram[ADDR_ENEMY_TYPE]),
        facing=int(ram[ADDR_PLAYER_FACING]),
        on_ground=on_ground,
        x_force=int(ram[ADDR_X_FORCE]),
        running_speed=int(ram[ADDR_RUNNING_SPEED]),
        a_held=bool(a_held),
        ground_y=floor,
        vertical_force=int(ram[ADDR_VERTICAL_FORCE]),
        vertical_force_down=int(ram[ADDR_VERTICAL_FORCE_DOWN]),
        y_move_force=int(ram[ADDR_Y_MOVE_FORCE]),
        jump_origin_y=int(ram[ADDR_JUMP_ORIGIN_Y]) or floor,
        oper_mode=oper_mode,
        timer=timer_value(ram),
    )
