"""RAM fields for Super Mario Bros. (NES) — M2 instrumentation.

Addresses align with ``platformer_common.levels.smb.SMB_RAM`` and the classic
SMB disassembly (player page/offset, oper mode, world/level, death state).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

# Player / pose
ADDR_PLAYER_STATE = 0x000E  # 0x08 walking, 0x0B dying, etc.
ADDR_PLAYER_FACING = 0x0033
ADDR_X_PAGE = 0x006D  # 256-pixel page
ADDR_PLAYER_X = 0x0086  # offset within page (legacy name)
ADDR_PLAYER_Y = 0x00CE
ADDR_PLAYER_STATUS = 0x0756  # 0=small, 1=big, 2=fire

# Progress / HUD
ADDR_LIVES = 0x075A
ADDR_LEVEL_LO = 0x075C  # area number used by some loaders
ADDR_AREA_POINTER = 0x0750
ADDR_LEVEL = 0x0760  # 0-indexed level within world
ADDR_WORLD = 0x075F  # 0-indexed world
ADDR_OPER_MODE = 0x0770  # 0=demo/title, 1=playing, 2=end, 3=game over
ADDR_TIMER_HUNDREDS = 0x07F8  # 4 at level start (400)
ADDR_TIMER_TENS = 0x07F9
ADDR_TIMER_ONES = 0x07FA

# Death / completion helpers
PLAYER_STATE_DYING = 0x0B
PLAYER_STATE_ALIVE = frozenset({0x00, 0x01, 0x03, 0x08, 0x0A})
OPER_MODE_PLAYING = 1
OPER_MODE_END = 2
LEVEL_ID_1_1 = 0  # world 0 * 4 + level 0


@dataclass(frozen=True)
class SmbSnapshot:
    """One-frame read of verified SMB progress fields."""

    frame: int
    player_state: int
    player_x: int
    player_y: int
    x_page: int
    x_offset: int
    lives: int
    world: int
    level: int
    level_id: int
    oper_mode: int
    player_power: int
    timer_hundreds: int
    area_pointer: int

    @property
    def playing(self) -> bool:
        return self.oper_mode == OPER_MODE_PLAYING

    @property
    def dying(self) -> bool:
        return self.player_state == PLAYER_STATE_DYING

    @property
    def on_world1_1(self) -> bool:
        return self.world == 0 and self.level == 0


def player_x(ram: np.ndarray) -> int:
    """Absolute horizontal position in pixels (page * 256 + offset)."""
    return int(ram[ADDR_X_PAGE]) * 256 + int(ram[ADDR_PLAYER_X])


def level_id(ram: np.ndarray) -> int:
    """``world * 4 + level`` (matches platformer_common SMB computed value)."""
    return int(ram[ADDR_WORLD]) * 4 + int(ram[ADDR_LEVEL])


def is_dying(ram: np.ndarray) -> bool:
    return int(ram[ADDR_PLAYER_STATE]) == PLAYER_STATE_DYING


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True on controllable 1-1 (timer live; not title)."""
    if int(ram[ADDR_OPER_MODE]) != OPER_MODE_PLAYING:
        return False
    if int(ram[ADDR_PLAYER_STATE]) not in (0x08, 0x01, 0x03, 0x0A):
        return False
    # Title does not run the 400 countdown timer.
    if int(ram[ADDR_TIMER_HUNDREDS]) not in (3, 4):
        return False
    if int(ram[ADDR_LIVES]) > 98:
        return False
    if obs_mean is not None and obs_mean <= 40.0:
        return False
    return True


def read_snapshot(ram: np.ndarray, frame: int = 0) -> SmbSnapshot:
    """Read a full progress snapshot from RAM."""
    x_page = int(ram[ADDR_X_PAGE])
    x_off = int(ram[ADDR_PLAYER_X])
    world = int(ram[ADDR_WORLD])
    level = int(ram[ADDR_LEVEL])
    return SmbSnapshot(
        frame=frame,
        player_state=int(ram[ADDR_PLAYER_STATE]),
        player_x=x_page * 256 + x_off,
        player_y=int(ram[ADDR_PLAYER_Y]),
        x_page=x_page,
        x_offset=x_off,
        lives=int(ram[ADDR_LIVES]),
        world=world,
        level=level,
        level_id=world * 4 + level,
        oper_mode=int(ram[ADDR_OPER_MODE]),
        player_power=int(ram[ADDR_PLAYER_STATUS]),
        timer_hundreds=int(ram[ADDR_TIMER_HUNDREDS]),
        area_pointer=int(ram[ADDR_AREA_POINTER]),
    )


def left_level_1_1(ram: np.ndarray, *, start_level_id: int = LEVEL_ID_1_1) -> bool:
    """True when world/level no longer match the 1-1 start id."""
    return level_id(ram) != start_level_id


def segment_1_1_success(
    ram: np.ndarray,
    *,
    start_lives: int,
    max_player_x: int,
    start_level_id: int = LEVEL_ID_1_1,
    min_progress_x: int = 2500,
) -> bool:
    """M3 success: left 1-1 after flagpole progress, without a lives drop.

    Completion is a ``level_id`` change after sufficient horizontal progress
    (flagpole / castle transition). Lives must not have dropped below start.
    """
    if int(ram[ADDR_LIVES]) < start_lives:
        return False
    if max_player_x < min_progress_x:
        return False
    return left_level_1_1(ram, start_level_id=start_level_id)


# World 4 = index 3 (1-2 warp zone exit). level_id 12 = world*4+level.
WORLD_INDEX_4 = 3
LEVEL_ID_4_1 = 12
WORLD_INDEX_8 = 7
LEVEL_INDEX_8_4 = 3


def reached_world_4(ram: np.ndarray) -> bool:
    """True when the warp-zone pipe delivered Mario to World 4."""
    return int(ram[ADDR_WORLD]) == WORLD_INDEX_4


def segment_1_2_warp_success(
    ram: np.ndarray,
    *,
    start_lives: int,
) -> bool:
    """1-2 secret warp success: World 4 without a lives drop."""
    if int(ram[ADDR_LIVES]) < start_lives:
        return False
    return reached_world_4(ram)


def reached_ending(ram: np.ndarray, *, start_lives: int | None = None) -> bool:
    """True on the stable post-8-4 ending mode without a lives drop."""
    if start_lives is not None and int(ram[ADDR_LIVES]) < start_lives:
        return False
    return (
        int(ram[ADDR_WORLD]) == WORLD_INDEX_8
        and int(ram[ADDR_LEVEL]) == LEVEL_INDEX_8_4
        and int(ram[ADDR_OPER_MODE]) == OPER_MODE_END
    )


def parse_game_state(ram: np.ndarray, frame: int = 0, obs_mean: float | None = None) -> GameState:
    """Project confirmed fields into ``GameState``."""
    snap = read_snapshot(ram, frame=frame)
    ready = is_level1_ready(ram, obs_mean=obs_mean)
    extras = {
        "level1_ready": ready,
        "ram_map_partial": False,
        "player_state": snap.player_state,
        "player_x": snap.player_x,
        "player_y": snap.player_y,
        "x_page": snap.x_page,
        "x_offset": snap.x_offset,
        "lives": snap.lives,
        "level_lo": int(ram[ADDR_LEVEL_LO]),
        "level": snap.level,
        "world": snap.world,
        "level_id": snap.level_id,
        "oper_mode": snap.oper_mode,
        "timer_hundreds": snap.timer_hundreds,
        "area_pointer": snap.area_pointer,
        "player_power": snap.player_power,
        "dying": snap.dying,
    }
    mode = GameMode.PLAYING if ready or snap.playing else GameMode.MENU
    if snap.oper_mode == OPER_MODE_END:
        mode = GameMode.PLAYING
    return GameState(
        frame=frame,
        mode=mode,
        stage=snap.world + 1,
        room=snap.level + 1,
        player_x=snap.player_x,
        player_y=snap.player_y,
        health=0,
        lives=snap.lives,
        enemies=(),
        extras=extras,
        player_dead=snap.dying,
    )
