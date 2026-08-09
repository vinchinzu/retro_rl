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

# ---------------------------------------------------------------------------
# Ice Beam Snake Room (0xA8B9) — floor → top (2WJ / platform-hop bands)
# ---------------------------------------------------------------------------
# Pure Acid→Snake handoff ~(216, 651). Prefer platform-hop / 2WJ climb over
# freeze ladder (operator note + tape thrash f12664–15400 is non-product).
# Land bands from live pure probe + clean first-climb tape f12080–12560.
#
#   floor y651 → L1 ~587 → L2 ~523 → L3 ~459 → L4 ~395 (mid door height)
#        → L5 ~331 → L6 ~267 → L7 ~203 → top ~139
#
# Morph tunnel to Ice (node 2) is mid-right ~(x≥200, y~377); entry is from the
# right column after top cross — left-wall morph at x=171 is solid (rr-5if residual).

SNAKE_HANDOFF_X = (80, 250)
SNAKE_HANDOFF_Y = (600, 720)
# Platform land bands (y windows; x is shaft mid ~40–160).
SNAKE_L1_Y = (560, 600)  # first left ledge
SNAKE_L2_Y = (500, 540)
SNAKE_L3_Y = (440, 480)
SNAKE_L4_Y = (380, 420)  # mid-door / Ice door height
SNAKE_L5_Y = (310, 350)
SNAKE_L6_Y = (250, 290)
SNAKE_L7_Y = (190, 220)
SNAKE_TOP_Y = (120, 160)
SNAKE_TOP_X = (80, 180)
# Morph tunnel (right column, after top cross).
SNAKE_TUNNEL_Y = (360, 420)
SNAKE_TUNNEL_X_MIN = 195
SNAKE_WALL_X = 171  # left face of center structure (solid from left shaft)
# Ice Beam room PLM.
ICE_BEAM_MASK = 0x0002
ICE_PLM_X = 187
ICE_ROOM_SETTLE = 280
SNAKE_CLIMB_FRAMES = 2500
SNAKE_TUNNEL_FRAMES = 800
SNAKE_ICE_COLLECT_FRAMES = 500


def in_business(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_BUSINESS


def in_ice_gate(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_ICE_GATE


def in_ice_acid(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_ICE_ACID


def in_ice_snake(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_ICE_SNAKE


def on_snake_floor(state: SuperMetroidState) -> bool:
    """Acid→Snake pure handoff band ~(216, 651)."""
    if not in_ice_snake(state):
        return False
    x, y = int(state.samus_x), int(state.samus_y)
    if not (SNAKE_HANDOFF_X[0] <= x <= SNAKE_HANDOFF_X[1]):
        return False
    if not (SNAKE_HANDOFF_Y[0] <= y <= SNAKE_HANDOFF_Y[1]):
        return False
    return int(state.velocity_y) == 0


def on_snake_top(state: SuperMetroidState) -> bool:
    """Top shelf of Ice Snake (pre right-column drop)."""
    if not in_ice_snake(state):
        return False
    x, y = int(state.samus_x), int(state.samus_y)
    if not (SNAKE_TOP_Y[0] <= y <= SNAKE_TOP_Y[1]):
        return False
    if not (SNAKE_TOP_X[0] <= x <= SNAKE_TOP_X[1] + 80):
        return False
    return int(state.velocity_y) == 0 and int(state.pose) in STANDING_POSES | LEDGE_POSES


def on_snake_tunnel_band(state: SuperMetroidState) -> bool:
    """Right-side morph tunnel approach (past center wall)."""
    if not in_ice_snake(state):
        return False
    x, y = int(state.samus_x), int(state.samus_y)
    return x >= SNAKE_TUNNEL_X_MIN and SNAKE_TUNNEL_Y[0] <= y <= SNAKE_TUNNEL_Y[1]


def has_ice(state: SuperMetroidState) -> bool:
    return bool(int(state.collected_beams) & ICE_BEAM_MASK)


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
    "ICE_BEAM_MASK",
    "ICE_GATE_SETTLE_FRAMES",
    "ICE_PLM_X",
    "ICE_ROOM_SETTLE",
    "ICE_SUPER_DOOR_X",
    "ICE_SUPER_LIP_X_MAX",
    "ICE_SUPER_Y_MAX",
    "ICE_SUPER_Y_MIN",
    "LEDGE_POSES",
    "SNAKE_CLIMB_FRAMES",
    "SNAKE_HANDOFF_X",
    "SNAKE_HANDOFF_Y",
    "SNAKE_ICE_COLLECT_FRAMES",
    "SNAKE_L1_Y",
    "SNAKE_L2_Y",
    "SNAKE_L3_Y",
    "SNAKE_L4_Y",
    "SNAKE_L5_Y",
    "SNAKE_L6_Y",
    "SNAKE_L7_Y",
    "SNAKE_TOP_X",
    "SNAKE_TOP_Y",
    "SNAKE_TUNNEL_FRAMES",
    "SNAKE_TUNNEL_X_MIN",
    "SNAKE_TUNNEL_Y",
    "SNAKE_WALL_X",
    "STANDING_POSES",
    "SUPER_PRESSURE_FRAMES",
    "has_ice",
    "in_business",
    "in_ice_acid",
    "in_ice_gate",
    "in_ice_snake",
    "in_ice_super_band",
    "on_acid_floor",
    "on_ice_super_lip",
    "on_snake_floor",
    "on_snake_top",
    "on_snake_tunnel_band",
]
