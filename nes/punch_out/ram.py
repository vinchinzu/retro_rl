"""RAM fields for Mike Tyson's Punch-Out!! (NES).

Addresses drawn from Data Crystal / TASVideos RAM maps and verified in-ring
against Glass Joe from ``Level1`` / ``Match1``.
"""

from __future__ import annotations

import numpy as np

from retro_harness.ram_state import GameMode, GameState

# --- Fight / opponent ---
ADDR_FIGHT_STARTED = 0x0000  # 1 = fight active
ADDR_OPP_ID = 0x0001
ADDR_OPP_TYPE = 0x0002  # 0 = Glass Joe
ADDR_FIGHT_FLAG = 0x0004  # 0xFF in-fight, 0x01 between rounds / cutscene
ADDR_KNOCKDOWN = 0x0005  # 0 = standing; non-zero while a fighter is down
ADDR_ROUND = 0x0006

# --- Opponent pattern (very useful for timing) ---
ADDR_OPP_PATTERN_TIMER = 0x0039  # next action when this hits 0
ADDR_OPP_ACTION = 0x003A
ADDR_OPP_PATTERN_SET = 0x003B  # 115 open, 150 backup/taunt, 185 attack set

# --- Clock ---
ADDR_CLOCK_ON = 0x0300
ADDR_CLOCK_MIN = 0x0302
ADDR_CLOCK_TENTHS = 0x0304  # display ones of seconds in practice
ADDR_CLOCK_SEC = 0x0305  # display tens of seconds in practice

# --- Hearts / stars ---
ADDR_HEARTS_TENS = 0x0323
ADDR_HEARTS_ONES = 0x0324
ADDR_STARS_RAW = 0x034A  # stars + 0x40

# --- Health ---
ADDR_MAC_HEALTH_INIT = 0x0390  # 1 when Mac has fight health
ADDR_HEALTH = 0x0391  # Little Mac (max 0x60 = 96)
ADDR_OPP_HEALTH = 0x0398
ADDR_OPP_HEALTH_CUR = 0x0399

# --- Mac action (observed) ---
ADDR_MAC_ACTION = 0x0050
ADDR_MAC_ACTION2 = 0x0051

OPP_TYPE_GLASS_JOE = 0
FIGHT_IN_RING = 0xFF
FIGHT_BETWEEN = 0x01
PATTERN_TAUNT = 150
HEALTH_MAX = 96


def read_u8(ram, addr: int) -> int:
    return int(ram[addr])


def hearts(ram) -> int:
    return read_u8(ram, ADDR_HEARTS_TENS) * 10 + read_u8(ram, ADDR_HEARTS_ONES)


def stars(ram) -> int:
    raw = read_u8(ram, ADDR_STARS_RAW)
    return max(0, raw - 0x40) if raw >= 0x40 else raw


def clock_on(ram) -> bool:
    return read_u8(ram, ADDR_CLOCK_ON) == 1


def in_fight(ram) -> bool:
    return read_u8(ram, ADDR_FIGHT_FLAG) == FIGHT_IN_RING


def between_rounds(ram) -> bool:
    return read_u8(ram, ADDR_FIGHT_FLAG) == FIGHT_BETWEEN


def is_taunt_window(ram) -> bool:
    """Glass Joe backup / Vive La France window (pattern set 150 only).

    Broader checks (e.g. action==2 with pset>=140) false-trigger on attack
    sets and waste hearts before the real taunt.
    """
    return read_u8(ram, ADDR_OPP_PATTERN_SET) == PATTERN_TAUNT


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once Mac and opponent health bars are live in the ring."""
    health = read_u8(ram, ADDR_HEALTH)
    opp = read_u8(ram, ADDR_OPP_HEALTH)
    if not (0 < health < 255 and 0 < opp < 255):
        return False
    if obs_mean is not None and obs_mean <= 40.0:
        return False
    return True


def is_match_live(ram, obs_mean: float | None = None) -> bool:
    """True when the round clock is running (controllable bout)."""
    if not is_level1_ready(ram, obs_mean=obs_mean):
        return False
    return clock_on(ram) and in_fight(ram)


def parse_game_state(
    ram: np.ndarray,
    frame: int = 0,
    obs_mean: float | None = None,
) -> GameState:
    """Project confirmed fields into ``GameState``."""
    ready = is_level1_ready(ram, obs_mean=obs_mean)
    live = is_match_live(ram, obs_mean=obs_mean)
    mac_hp = read_u8(ram, ADDR_HEALTH)
    opp_hp = read_u8(ram, ADDR_OPP_HEALTH)
    kd = read_u8(ram, ADDR_KNOCKDOWN)
    extras = {
        "level1_ready": ready,
        "match_live": live,
        "ram_map_partial": True,
        "health": mac_hp,
        "opp_health": opp_hp,
        "opp_health_cur": read_u8(ram, ADDR_OPP_HEALTH_CUR),
        "hearts": hearts(ram),
        "stars": stars(ram),
        "round": read_u8(ram, ADDR_ROUND),
        "opp_type": read_u8(ram, ADDR_OPP_TYPE),
        "opp_action": read_u8(ram, ADDR_OPP_ACTION),
        "opp_pattern_timer": read_u8(ram, ADDR_OPP_PATTERN_TIMER),
        "opp_pattern_set": read_u8(ram, ADDR_OPP_PATTERN_SET),
        "knockdown": kd,
        "fight_flag": read_u8(ram, ADDR_FIGHT_FLAG),
        "clock_on": clock_on(ram),
        "clock_min": read_u8(ram, ADDR_CLOCK_MIN),
        "clock_sec": read_u8(ram, ADDR_CLOCK_SEC),
        "clock_tenths": read_u8(ram, ADDR_CLOCK_TENTHS),
        "taunt_window": is_taunt_window(ram),
        "mac_down": mac_hp == 0 and kd != 0,
        "opp_down": opp_hp == 0,
    }
    if between_rounds(ram):
        mode = GameMode.CUTSCENE
    elif live or ready:
        mode = GameMode.PLAYING
    else:
        mode = GameMode.MENU
    return GameState(
        frame=frame,
        mode=mode,
        stage=extras["round"],
        room=0,
        player_x=0,
        player_y=0,
        health=mac_hp,
        lives=0,
        enemies=(),
        player_dead=extras["mac_down"],
        extras=extras,
    )
