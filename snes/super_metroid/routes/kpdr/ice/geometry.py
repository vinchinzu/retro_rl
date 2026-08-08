"""Ice branch geometry: Business Super door + Gate/Acid bands.

Pins from human tape ``tasks/speed_to_wave_ice_moat_human.json`` Phase B
(Business f9988–10817 → Ice Gate entry ~(18,907); Acid f11231–11964).
Controllers import predicates — do not re-encode magic thresholds inline.
"""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.kpdr.k4_common import _LEDGE_POSES, _STANDING_POSES
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUSINESS,
    ROOM_ICE_ACID,
    ROOM_ICE_GATE,
    ROOM_ICE_SNAKE,
)
from super_metroid.routes.rle import RleScript, load_rle_json

# ---------------------------------------------------------------------------
# Business Center (0xA7DE) — mid-shaft Super green LEFT → Ice Gate
# ---------------------------------------------------------------------------

# Elevator platform after Warehouse / continuous Business tip settle.
BUSINESS_ELEVATOR_Y = 680

# Super green door lip (human Super plant ~(61, 922); entry ~(18, 907)).
# y band is slightly lower than Cathedral top-right blue door (840–900).
ICE_SUPER_Y_MIN = 880
ICE_SUPER_Y_MAX = 960
ICE_SUPER_LIP_X_MAX = 90
ICE_SUPER_DOOR_X = 40
# Mid-platform approach before Super lip (human dwell y≈1067 then climb).
ICE_APPROACH_Y = (980, 1100)
ICE_APPROACH_X = (90, 180)

# Grounded poses for door ledge / elevator settle (exclude pure air spin only).
LEDGE_POSES = _LEDGE_POSES
STANDING_POSES = _STANDING_POSES

# Frame budgets (one-knob pure; assist unlimited ammo).
ELEVATOR_SETTLE_FRAMES = 600
DOOR_BAND_FRAMES = 700
SUPER_PRESSURE_FRAMES = 400
ICE_GATE_SETTLE_FRAMES = 320

# ---------------------------------------------------------------------------
# Ice Beam Acid Room (0xA75D) — floor LEFT → Ice Snake
# ---------------------------------------------------------------------------

# Pure Gate→Acid handoff pin ~(470, 139); floor band for recovery.
ACID_HANDOFF_X = (400, 520)
ACID_FLOOR_Y = (120, 160)
ACID_FLOOR_Y_MAX = 160
# Left blue door to Snake (block [0,7]); tape door walk x≤40 y≈139.
ACID_LEFT_DOOR_X = 40
ACID_SNAKE_SETTLE_FRAMES = 320

_ACID_TO_SNAKE_RLE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "ice_acid_to_snake_rle.json"
)
ACID_TO_SNAKE_RLE: RleScript = load_rle_json(_ACID_TO_SNAKE_RLE_PATH)


def in_business(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_BUSINESS


def in_ice_gate(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_ICE_GATE


def in_ice_acid(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_ICE_ACID


def in_ice_snake(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_ICE_SNAKE


def on_acid_floor(state: SuperMetroidState) -> bool:
    """Standing on Acid main floor band (Gate handoff / mid traverse)."""
    if not in_ice_acid(state):
        return False
    y = int(state.samus_y)
    if not (ACID_FLOOR_Y[0] <= y <= ACID_FLOOR_Y[1]):
        return False
    if int(state.velocity_y) != 0:
        return False
    return int(state.pose) in STANDING_POSES | LEDGE_POSES


def on_ice_super_lip(state: SuperMetroidState) -> bool:
    """Standing / grounded near Business Super green door (left mid shaft)."""
    if not in_business(state):
        return False
    y = int(state.samus_y)
    x = int(state.samus_x)
    if not (ICE_SUPER_Y_MIN <= y <= ICE_SUPER_Y_MAX):
        return False
    if x > ICE_SUPER_LIP_X_MAX:
        return False
    if int(state.velocity_y) != 0:
        return False
    return int(state.pose) in LEDGE_POSES | frozenset({1, 2, 9, 10})


def in_ice_super_band(state: SuperMetroidState) -> bool:
    """Door-height band (any x) — used while dropping from elevator."""
    if not in_business(state):
        return False
    y = int(state.samus_y)
    return ICE_SUPER_Y_MIN <= y <= ICE_SUPER_Y_MAX


__all__ = [
    "ACID_FLOOR_Y",
    "ACID_FLOOR_Y_MAX",
    "ACID_HANDOFF_X",
    "ACID_LEFT_DOOR_X",
    "ACID_SNAKE_SETTLE_FRAMES",
    "ACID_TO_SNAKE_RLE",
    "BUSINESS_ELEVATOR_Y",
    "DOOR_BAND_FRAMES",
    "ELEVATOR_SETTLE_FRAMES",
    "ICE_APPROACH_X",
    "ICE_APPROACH_Y",
    "ICE_GATE_SETTLE_FRAMES",
    "ICE_SUPER_DOOR_X",
    "ICE_SUPER_LIP_X_MAX",
    "ICE_SUPER_Y_MAX",
    "ICE_SUPER_Y_MIN",
    "LEDGE_POSES",
    "STANDING_POSES",
    "SUPER_PRESSURE_FRAMES",
    "in_business",
    "in_ice_acid",
    "in_ice_gate",
    "in_ice_snake",
    "in_ice_super_band",
    "on_acid_floor",
    "on_ice_super_lip",
]
