"""Early RAM fields for Zelda II (NES).

M1 readiness still uses magic ``$0773``. Leave-palace stop adds engine mode
``$0736`` (11 side-scroll play, 5 overworld play) plus x/y/HP as needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retro_harness.ram_state import GameMode, GameState

ADDR_PLAYER_Y = 0x0029  # side-scroll Y
ADDR_PLAYER_X = 0x004D  # side-scroll X (low)
ADDR_PAGE = 0x003B  # side-scroll map page
ADDR_OW_Y = 0x0073
ADDR_OW_X = 0x0074
ADDR_LIVES = 0x0700
ADDR_ENGINE_MODE = 0x0736
ADDR_FLAG = 0x0769  # non-zero once file is active
ADDR_LIFE = 0x0773  # magic meter; probe 127 in North Palace
ADDR_HEALTH = 0x0774  # life meter

MODE_OVERWORLD = 5
MODE_SIDESCROLL = 11
TRANSITION_MODES = frozenset({1, 2, 3, 4, 16})
PLAY_MODES = frozenset({MODE_OVERWORLD, MODE_SIDESCROLL}) | TRANSITION_MODES


def _u8(ram, address: int) -> int:
    return int(ram[address]) if address < len(ram) else 0


@dataclass(frozen=True)
class ZeldaIISnapshot:
    """Thin pose used by the North Palace exit stop predicate."""

    engine_mode: int
    player_x: int
    player_y: int
    page: int
    ow_x: int
    ow_y: int
    magic: int
    health: int
    lives: int
    flag: int

    @property
    def side_scroll(self) -> bool:
        return self.engine_mode == MODE_SIDESCROLL

    @property
    def overworld(self) -> bool:
        return self.engine_mode == MODE_OVERWORLD

    @property
    def dead(self) -> bool:
        return self.health == 0 and self.engine_mode in PLAY_MODES

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "engine_mode": self.engine_mode,
            "player_x": self.player_x,
            "player_y": self.player_y,
            "page": self.page,
            "ow_x": self.ow_x,
            "ow_y": self.ow_y,
            "magic": self.magic,
            "health": self.health,
            "lives": self.lives,
            "flag": self.flag,
            "overworld": self.overworld,
            "side_scroll": self.side_scroll,
            "dead": self.dead,
        }


def read_snapshot(ram: np.ndarray | bytes) -> ZeldaIISnapshot:
    """Read the leave-palace / death fields."""
    return ZeldaIISnapshot(
        engine_mode=_u8(ram, ADDR_ENGINE_MODE),
        player_x=_u8(ram, ADDR_PLAYER_X),
        player_y=_u8(ram, ADDR_PLAYER_Y),
        page=_u8(ram, ADDR_PAGE),
        ow_x=_u8(ram, ADDR_OW_X),
        ow_y=_u8(ram, ADDR_OW_Y),
        magic=_u8(ram, ADDR_LIFE),
        health=_u8(ram, ADDR_HEALTH),
        lives=_u8(ram, ADDR_LIVES),
        flag=_u8(ram, ADDR_FLAG),
    )


def is_overworld(ram) -> bool:
    """True on overworld play (engine mode 5)."""
    return read_snapshot(ram).overworld


def is_side_scroll(ram) -> bool:
    """True on side-scroll play (engine mode 11), including North Palace."""
    return read_snapshot(ram).side_scroll


def is_dead(ram) -> bool:
    """True when the life meter collapsed during play or a transition."""
    return read_snapshot(ram).dead


def palace_exit_success(ram) -> bool:
    """Stop predicate: North Palace left, overworld play latched."""
    return is_overworld(ram)


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once North Palace / side-view play has life initialized."""
    if int(ram[ADDR_LIFE]) <= 0:
        return False
    if obs_mean is not None and obs_mean <= 45.0:
        return False
    return True


def parse_game_state(
    ram: np.ndarray, frame: int = 0, obs_mean: float | None = None
) -> GameState:
    """Project confirmed early fields into ``GameState``."""
    ready = is_level1_ready(ram, obs_mean=obs_mean)
    snap = read_snapshot(ram)
    extras = {
        "level1_ready": ready,
        "ram_map_partial": True,
        "palace_exit": snap.overworld,
        **{k: v for k, v in snap.as_dict().items() if k != "dead"},
    }
    for name in (
        "ADDR_HEALTH_1",
        "ADDR_HEALTH",
        "ADDR_LIFE",
        "ADDR_LIVES",
        "ADDR_MODE",
        "ADDR_ENGINE_MODE",
        "ADDR_SCORE",
        "ADDR_GAMEOVER",
        "ADDR_SCREEN",
        "ADDR_FLAG",
    ):
        addr = globals().get(name)
        if addr is not None and addr < len(ram):
            extras[name.removeprefix("ADDR_").lower()] = int(ram[addr])
    if snap.dead:
        mode = GameMode.GAME_OVER
    elif snap.engine_mode in TRANSITION_MODES:
        mode = GameMode.CUTSCENE
    elif ready or snap.side_scroll or snap.overworld:
        mode = GameMode.PLAYING
    else:
        mode = GameMode.MENU
    if snap.overworld:
        player_x, player_y, room = snap.ow_x, snap.ow_y, 0
    else:
        player_x, player_y, room = snap.player_x, snap.player_y, snap.page
    return GameState(
        frame=frame,
        mode=mode,
        stage=1,
        room=room,
        player_x=player_x,
        player_y=player_y,
        health=snap.health,
        lives=snap.lives,
        enemies=(),
        extras=extras,
        player_dead=snap.dead,
    )
