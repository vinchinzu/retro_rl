"""Confirmed Mute City race-state fields for F-Zero."""

from __future__ import annotations

import numpy as np

from retro_harness.ram_state import GameMode, GameState

# The raw speed word tracks the on-screen km/h value at roughly 10 raw units
# per displayed km/h. Exact scaling remains to be calibrated.
ADDR_SPEED_RAW = 0x0002
ADDR_RACE_STATE = 0x0046
ADDR_TRACK_STATE = 0x0047
ADDR_GAMESTATE = 0x0054
ADDR_GAMESTATE_SUB = 0x0055
ADDR_LIVES = 0x0059
# Mode7 Y center; LEFT/RIGHT on the start straight move it monotonically.
ADDR_LATERAL = 0x007F
ADDR_CAMERA_X = 0x00A2
ADDR_CAMERA_Y = 0x00A6
ADDR_LATERAL_FINE = ADDR_CAMERA_Y
# HUD overlay. Format rplLfyws; bit 4 ("X laps left") rises at the finish line.
ADDR_SCREEN_TEXT = 0x00B8
ADDR_LAST_SCREEN_TEXT = 0x00BA
# Race finish. Format lefcr??h; bit 6 is exploded, bit 7 is race lost.
ADDR_FINISH_STATE = 0x00C3
ADDR_CHECKPOINT_FACING = 0x00C5
ADDR_POWER = 0x00C9
ADDR_DAMAGE_ANIM = 0x00E0
# Player Angle8 heading; LEFT decreases it, RIGHT increases it.
ADDR_HEADING = 0x0BD1

ANGLE8_MOD = 0xC0
GAMESTATE_RACE = 0x02
GAMESTATE_SUB_COUNTDOWN = 0x02
GAMESTATE_SUB_LIVE = 0x03

SCREEN_REVERSE = 0x80
SCREEN_POWER_DOWN = 0x40
SCREEN_LIMIT_X = 0x20
SCREEN_LAPS_LEFT = 0x10
SCREEN_FINAL_LAP = 0x08
SCREEN_LOST = 0x04
SCREEN_WON = 0x02
SCREEN_SPECIAL = 0x01

FINISH_LOST = 0x80
FINISH_EXPLODED = 0x40
FINISH_RACE = 0x20
DAMAGE_HIT = 0x80


def read_u8(ram: np.ndarray, address: int) -> int:
    """Read one unsigned byte from WRAM."""
    return int(ram[address])


def read_u16le(ram: np.ndarray, address: int) -> int:
    """Read one little-endian unsigned word from WRAM."""
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def read_s16le(ram: np.ndarray, address: int) -> int:
    """Read one little-endian signed word from WRAM."""
    value = read_u16le(ram, address)
    return value - 65536 if value > 32767 else value


def heading_error(heading: int, checkpoint_facing: int) -> int:
    """Signed Angle8 error: positive means heading is right of the checkpoint."""
    return (int(heading) - int(checkpoint_facing) + ANGLE8_MOD // 2) % ANGLE8_MOD - (
        ANGLE8_MOD // 2
    )


def laps_left_text(screen_text: int) -> bool:
    """True while the HUD is showing 'X laps left' (finish-line crossing)."""
    return bool(int(screen_text) & SCREEN_LAPS_LEFT)


def crashed_out(finish_state: int, power: int) -> bool:
    """True on explosion / retire, or when machine power has underflowed."""
    if int(finish_state) & (FINISH_EXPLODED | FINISH_LOST):
        return True
    return int(power) < 0


class LapWatch:
    """Latch finish-line crossings from the rising edge of laps-left HUD text."""

    def __init__(self) -> None:
        self.laps = 0
        self._prev = 0

    def update(self, screen_text: int) -> bool:
        """Count one lap when bit 4 rises. Returns True on that frame."""
        current = int(screen_text)
        rose = bool(current & SCREEN_LAPS_LEFT) and not (
            self._prev & SCREEN_LAPS_LEFT
        )
        self._prev = current
        if rose:
            self.laps += 1
        return rose


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project the confirmed Mute City fields into ``GameState``."""
    race_state = read_u8(ram, ADDR_RACE_STATE)
    track_state = read_u8(ram, ADDR_TRACK_STATE)
    gs_main = read_u8(ram, ADDR_GAMESTATE)
    gs_sub = read_u8(ram, ADDR_GAMESTATE_SUB)
    screen_text = read_u8(ram, ADDR_SCREEN_TEXT)
    finish_state = read_u8(ram, ADDR_FINISH_STATE)
    power = read_s16le(ram, ADDR_POWER)
    heading = read_u8(ram, ADDR_HEADING)
    checkpoint = read_u8(ram, ADDR_CHECKPOINT_FACING)
    damage_anim = read_u8(ram, ADDR_DAMAGE_ANIM)
    lateral = read_u16le(ram, ADDR_LATERAL)
    camera_x = read_u16le(ram, ADDR_CAMERA_X)
    camera_y = read_u16le(ram, ADDR_CAMERA_Y)
    live_gs = gs_main == GAMESTATE_RACE and gs_sub >= GAMESTATE_SUB_COUNTDOWN
    playing = live_gs or (race_state == 1 and track_state == 1)
    exploded = crashed_out(finish_state, power)
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if playing and not exploded else GameMode.MENU,
        stage=0,
        room=0,
        player_x=lateral,
        player_y=camera_y,
        health=max(power, 0),
        lives=read_u8(ram, ADDR_LIVES),
        camera_x=camera_x,
        camera_y=camera_y,
        player_dead=exploded,
        level_complete=laps_left_text(screen_text),
        extras={
            "race_state": race_state,
            "track_state": track_state,
            "gamestate": gs_main,
            "gamestate_sub": gs_sub,
            "racing": gs_main == GAMESTATE_RACE and gs_sub >= GAMESTATE_SUB_LIVE,
            "speed_raw": read_u16le(ram, ADDR_SPEED_RAW),
            "lateral": lateral,
            "lateral_fine": read_u16le(ram, ADDR_LATERAL_FINE),
            "screen_text": screen_text,
            "last_screen_text": read_u8(ram, ADDR_LAST_SCREEN_TEXT),
            "finish_state": finish_state,
            "power": power,
            "heading": heading,
            "checkpoint_facing": checkpoint,
            "heading_error": heading_error(heading, checkpoint),
            "damage_anim": damage_anim,
            "damaged": damage_anim >= DAMAGE_HIT,
            "ram_map_partial": True,
        },
    )
