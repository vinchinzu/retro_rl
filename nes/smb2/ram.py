"""First-level RAM gates for Super Mario Bros. 2 (NES).

Addresses follow the Data Crystal RAM map. Player Y at ``0x32`` is the slot
immediately before the enemy Y table at ``0x33`` and was confirmed on the
1-1 sky spawn (120, 192) while replaying TASVideos #1724M under fceumm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retro_harness.ram_state import GameMode, GameState

ADDR_PLAYER_X_PAGE = 0x0014
ADDR_PLAYER_Y_PAGE = 0x001E
ADDR_PLAYER_X = 0x0028
ADDR_PLAYER_Y = 0x0032
ADDR_X_SPEED = 0x003C
ADDR_JUMP_PHYSICS = 0x0046
ADDR_CHARACTER = 0x008F
ADDR_HEARTS = 0x04C2
ADDR_TRANSITION = 0x04EC
ADDR_LIVES = 0x04ED
ADDR_AREA = 0x04E7
ADDR_SUBAREA = 0x04E8
ADDR_LEVEL = 0x0531
ADDR_WORLD = 0x0635

LEVEL_1_1 = 0
WORLD_1 = 0
AREA_1_1_START = 0
SUBAREA_1_1_START = 0
TRANSITION_PLAYING = 0
HEARTS_TWO = 0x1F
HEART_VALUES = frozenset({0x0F, 0x1F, 0x2F, 0x3F})
SPAWN_X = 120
SPAWN_Y = 192
CONTROL_X_MIN = 80
CONTROL_X_MAX = 140


@dataclass(frozen=True, slots=True)
class Smb2Snapshot:
    """Compact 1-1 pose used by the TAS checkpoint gates."""

    frame: int
    player_x: int
    player_y: int
    x_page: int
    y_page: int
    x_speed: int
    jump_physics: int
    character: int
    hearts: int
    lives: int
    transition: int
    area: int
    subarea: int
    level: int
    world: int
    obs_mean: float | None = None

    @property
    def abs_x(self) -> int:
        return self.x_page * 256 + self.player_x

    @property
    def abs_y(self) -> int:
        return self.y_page * 256 + self.player_y


def read_u8(ram, addr: int) -> int:
    return int(ram[addr])


def read_snapshot(
    ram,
    frame: int = 0,
    obs_mean: float | None = None,
) -> Smb2Snapshot:
    """Project the first-level fields from a RAM dump."""
    return Smb2Snapshot(
        frame=frame,
        player_x=read_u8(ram, ADDR_PLAYER_X),
        player_y=read_u8(ram, ADDR_PLAYER_Y),
        x_page=read_u8(ram, ADDR_PLAYER_X_PAGE),
        y_page=read_u8(ram, ADDR_PLAYER_Y_PAGE),
        x_speed=read_u8(ram, ADDR_X_SPEED),
        jump_physics=read_u8(ram, ADDR_JUMP_PHYSICS),
        character=read_u8(ram, ADDR_CHARACTER),
        hearts=read_u8(ram, ADDR_HEARTS),
        lives=read_u8(ram, ADDR_LIVES),
        transition=read_u8(ram, ADDR_TRANSITION),
        area=read_u8(ram, ADDR_AREA),
        subarea=read_u8(ram, ADDR_SUBAREA),
        level=read_u8(ram, ADDR_LEVEL),
        world=read_u8(ram, ADDR_WORLD),
        obs_mean=obs_mean,
    )


def is_level_1_1(snap: Smb2Snapshot) -> bool:
    return (
        snap.level == LEVEL_1_1
        and snap.world == WORLD_1
        and snap.transition == TRANSITION_PLAYING
        and 0 < snap.lives < 20
        and snap.hearts in HEART_VALUES
        and snap.character in {0, 1, 2, 3}
    )


def is_level1_start(snap: Smb2Snapshot) -> bool:
    """True on the 1-1 sky spawn pose, including the pre-physics fade."""
    return (
        is_level_1_1(snap)
        and snap.area == AREA_1_1_START
        and snap.subarea == SUBAREA_1_1_START
        and snap.x_page == 0
        and snap.y_page == 0
        and CONTROL_X_MIN <= snap.player_x <= CONTROL_X_MAX
        and snap.player_y > 0
    )


def is_level1_control(snap: Smb2Snapshot) -> bool:
    """True when 1-1 spawn physics are live (fall / run can begin)."""
    return is_level1_start(snap) and snap.jump_physics != 0


def parse_game_state(
    ram: np.ndarray,
    frame: int = 0,
    obs_mean: float | None = None,
) -> GameState:
    """Project confirmed 1-1 fields into ``GameState``."""
    snap = read_snapshot(ram, frame=frame, obs_mean=obs_mean)
    start = is_level1_start(snap)
    control = is_level1_control(snap)
    extras = {
        "ram_map_partial": True,
        "level1_start": start,
        "level1_control": control,
        "lives": snap.lives,
        "hearts": snap.hearts,
        "character": snap.character,
        "level": snap.level,
        "world": snap.world,
        "area": snap.area,
        "subarea": snap.subarea,
        "transition": snap.transition,
        "jump_physics": snap.jump_physics,
        "x_page": snap.x_page,
        "y_page": snap.y_page,
    }
    if control:
        mode = GameMode.PLAYING
    elif start:
        mode = GameMode.CUTSCENE
    elif snap.lives > 0:
        mode = GameMode.MENU
    else:
        mode = GameMode.BOOT
    return GameState(
        frame=frame,
        mode=mode,
        stage=1,
        room=snap.subarea,
        player_x=snap.abs_x,
        player_y=snap.abs_y,
        velocity_x=snap.x_speed,
        velocity_y=snap.jump_physics,
        health=snap.hearts,
        lives=snap.lives,
        extras=extras,
    )
