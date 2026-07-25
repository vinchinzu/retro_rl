"""RAM helpers for Hal's Hole in One Golf.

Addresses are WRAM offsets (bank $7E). Pro Action Replay ``7E10A1`` maps to
offset ``0x10A1``. Stable-retro ``data.json`` and ``env.get_ram()`` use that
same offset (not ``0x7E0000 + offset``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

# Confirmed via Pro Action Replay: stroke count forced hole-in-one at 7E10A1.
WRAM_STROKE_COUNT = 0x10A1

# Confirmed by advancing Hole 1 -> Hole 2. The game stores a zero-based index.
WRAM_HOLE_INDEX = 0x10F5

# Confirmed by timing sweeps whose landing screens showed 298, 215, 161, 204,
# 214, 259, 303, and 317 yards respectively.
WRAM_REST_DISTANCE = 0x11B3

# Surface under the ball: 1=tee, 2=fairway, 3=bunker, 6=putting green.
WRAM_LIE_TYPE = 0x11CD
LIE_PUTTING_GREEN = 6

# Backward-compatible name for callers that imported the old candidate. New
# code must use ``read_hole_number`` because the stored value is zero-based.
WRAM_HOLE_NUMBER = WRAM_HOLE_INDEX

# Red aim-offset byte shown while lining up a shot. Earlier probes mislabeled
# this as the cumulative round score; the bot derives that total by summing
# the peak per-hole stroke count instead.
WRAM_AIM_OFFSET = 0x10B1

# In VS HAL / match play the companion stroke byte tracks the opponent (Hal).
# At hole boundaries it often holds Hal's strokes for the hole just finished.
WRAM_OPPONENT_STROKE_COUNT = 0x10A3


class GameScene(IntEnum):
    """Coarse scene classes used by autoplay recovery."""

    UNKNOWN = 0
    TITLE = 1
    MODE_SELECT = 2
    PLAYER_COUNT = 3
    NAME_ENTRY = 4
    LEVEL_SELECT = 5
    CLUB_SELECT = 6
    HOLE_INTRO = 7
    COMMAND = 8
    AIM = 9
    LIE = 10
    CLUB_PICK = 11
    STANCE = 12
    SWING = 13
    PUTT = 14
    BALL_FLIGHT = 15
    SCORECARD = 16
    RESULTS = 17
    TRANSITION = 18


@dataclass(frozen=True)
class GolfSnapshot:
    """Typed view of golf-relevant RAM / info fields."""

    stroke_count: int
    hole_number: int
    rest_distance: int
    lie_type: int
    aim_offset: int
    scene: GameScene
    raw: dict[str, Any]


def read_u8(ram: np.ndarray, offset: int) -> int:
    """Read one unsigned byte from WRAM."""
    if offset < 0 or offset >= len(ram):
        return 0
    return int(ram[offset])


def read_u16_le(ram: np.ndarray, offset: int) -> int:
    """Read little-endian u16 from WRAM."""
    if offset < 0 or offset + 1 >= len(ram):
        return 0
    return int(ram[offset]) | (int(ram[offset + 1]) << 8)


def read_hole_number(ram: np.ndarray, info: dict[str, Any] | None = None) -> int:
    """Return a one-based hole number, or zero outside an active round."""
    info = info or {}
    if "hole_index" in info:
        index = int(info["hole_index"])
    else:
        index = read_u8(ram, WRAM_HOLE_INDEX)
    return index + 1 if 0 <= index < 18 else 0


def read_rest_distance(ram: np.ndarray, info: dict[str, Any] | None = None) -> int:
    """Return the remaining distance in yards."""
    info = info or {}
    if "rest_distance" in info:
        return int(info["rest_distance"])
    return read_u16_le(ram, WRAM_REST_DISTANCE)


def read_aim_offset(ram: np.ndarray, info: dict[str, Any] | None = None) -> int:
    """Return the raw red aim-offset byte."""
    info = info or {}
    if "aim_offset" in info:
        return int(info["aim_offset"])
    return read_u8(ram, WRAM_AIM_OFFSET)


def snapshot_from_ram(
    ram: np.ndarray,
    *,
    info: dict[str, Any] | None = None,
    scene: GameScene = GameScene.UNKNOWN,
) -> GolfSnapshot:
    """Build a golf snapshot from emulator RAM."""
    info = info or {}
    stroke = int(info.get("stroke_count", read_u8(ram, WRAM_STROKE_COUNT)))
    hole = read_hole_number(ram, info)
    rest = read_rest_distance(ram, info)
    lie = int(info.get("lie_type", read_u8(ram, WRAM_LIE_TYPE)))
    aim_offset = read_aim_offset(ram, info)
    return GolfSnapshot(
        stroke_count=stroke,
        hole_number=hole,
        rest_distance=rest,
        lie_type=lie,
        aim_offset=aim_offset,
        scene=scene,
        raw=dict(info),
    )


def wram_to_data_address(offset: int) -> int:
    """Convert a WRAM offset to a stable-retro data.json address."""
    return offset
