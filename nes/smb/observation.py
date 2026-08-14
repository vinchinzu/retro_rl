"""SMB residual lattice, player physics, and the current flat-ground world.

Observation is RAM-readable R(τ) only (same shape as Super Metroid).
PlayerPhysics is what ``approx.step`` advances. World is the floor
assumption (a constant today; tiles later) — not a player field.

Lattice:
- Oπ: pixel x/y, pose ($000E), room (world/level/area)
- Oσ: Oπ plus subpixels ($0400, $0416)
- Oσ+: Oσ plus enemy slot 0 flag/type
- O†: lives / death (separate from Oπ)

Speeds ($0057, $009F) live in first-differing-field, not a second σ+.
``a_held`` is one-frame controller memory, not RAM.
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
    player_on_ground,
    player_x,
    player_x_speed,
    player_y_speed,
    timer_value,
)

__all__ = [
    "DEFAULT_GROUND_Y",
    "FLAT_1_1",
    "Observation",
    "PlayerPhysics",
    "World",
    "level1_start",
    "level1_start_obs",
    "observation_from_ram",
    "pack_room",
    "player_from_ram",
    "unpack_room",
    "world_from_player",
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
    """RAM-readable residual lattice. Not the stepper state."""

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


@dataclass(frozen=True)
class World:
    """Collision / floor. Flat 1-1 for now; later a tile query."""

    ground_y: int = DEFAULT_GROUND_Y


FLAT_1_1 = World()


@dataclass(frozen=True)
class PlayerPhysics:
    """One-frame player state the approximate stepper advances."""

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
    vertical_force: int = 0
    vertical_force_down: int = 0x70
    y_move_force: int = 0
    jump_origin_y: int = DEFAULT_GROUND_Y
    oper_mode: int = 1
    timer: int = 400

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerPhysics:
        payload = dict(data)
        payload.pop("ground_y", None)
        return cls(**payload)

    def as_observation(self) -> Observation:
        return Observation(
            frame=self.frame,
            x=self.x,
            y=self.y,
            pose=self.pose,
            room=self.room,
            sub_x=self.sub_x,
            sub_y=self.sub_y,
            velocity_x=self.velocity_x,
            velocity_y=self.velocity_y,
            energy=self.energy,
            dead=self.dead,
            frame_counter=self.frame_counter,
            enemy0_active=self.enemy0_active,
            enemy0_type=self.enemy0_type,
        )

    @property
    def world(self) -> int:
        return unpack_room(self.room)[0]

    @property
    def level(self) -> int:
        return unpack_room(self.room)[1]

    @property
    def area_pointer(self) -> int:
        return unpack_room(self.room)[2]


def world_from_player(player: PlayerPhysics) -> World:
    """Freeze the floor from a grounded start; else the 1-1 default."""
    if player.on_ground:
        return World(ground_y=int(player.y))
    return FLAT_1_1


def level1_start(*, frame: int = 0) -> PlayerPhysics:
    """Controllable 1-1 stand (matches live Level1_1.state kinematics)."""
    return PlayerPhysics(
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
        vertical_force=0,
        vertical_force_down=0x70,
        y_move_force=0,
        jump_origin_y=DEFAULT_GROUND_Y,
        oper_mode=1,
        timer=400,
    )


def level1_start_obs(*, frame: int = 0) -> Observation:
    """Lattice view of :func:`level1_start`."""
    return level1_start(frame=frame).as_observation()


def player_from_ram(
    ram: np.ndarray,
    frame: int = 0,
    *,
    a_held: bool = False,
) -> PlayerPhysics:
    """Extract stepper state from a NES RAM snapshot. ``a_held`` is tape memory."""
    y = int(ram[ADDR_PLAYER_Y])
    on_ground = player_on_ground(ram)
    pose = int(ram[ADDR_PLAYER_STATE])
    oper_mode = int(ram[ADDR_OPER_MODE])
    return PlayerPhysics(
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
        vertical_force=int(ram[ADDR_VERTICAL_FORCE]),
        vertical_force_down=int(ram[ADDR_VERTICAL_FORCE_DOWN]),
        y_move_force=int(ram[ADDR_Y_MOVE_FORCE]),
        jump_origin_y=int(ram[ADDR_JUMP_ORIGIN_Y]),
        oper_mode=oper_mode,
        timer=timer_value(ram),
    )


def observation_from_ram(
    ram: np.ndarray,
    frame: int = 0,
    *,
    a_held: bool = False,
) -> Observation:
    """Lattice view of :func:`player_from_ram`."""
    return player_from_ram(ram, frame, a_held=a_held).as_observation()
