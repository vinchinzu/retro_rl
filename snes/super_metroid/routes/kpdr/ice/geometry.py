"""Ice branch geometry: Business Super door + Gate/Acid bands.

Pins from human tape ``tasks/speed_to_wave_ice_moat_human.json`` Phase B
(Business f9988–10817 → Ice Gate entry ~(18,907); Acid f11231–11964).
Controllers import predicates — do not re-encode magic thresholds inline.
"""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.kpdr.norfair.common import _LEDGE_POSES, _STANDING_POSES
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUSINESS,
    ROOM_ICE_ACID,
    ROOM_ICE_GATE,
    ROOM_ICE_SNAKE,
    ROOM_ICE_TUTORIAL,
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
# Human success (tape f15424–15463): morph pose 30 roll RIGHT y=377 x=204→323.
# False ledge just below tunnel is y~409 — do NOT treat as tunnel floor.
SNAKE_TUNNEL_Y = (365, 395)
SNAKE_TUNNEL_FLOOR_Y = 377
SNAKE_FALSE_LEDGE_Y = (400, 430)  # morph trap below tunnel mouth
SNAKE_TUNNEL_X_MIN = 195
SNAKE_TUNNEL_EXIT_X = 320  # past tunnel mouth into open right column
SNAKE_WALL_X = 171  # left face of center structure (solid from left shaft)
# Right-column mid shelf used for jump-up morph into tunnel (tape f15350).
SNAKE_MID_SHELF_Y = (480, 540)
SNAKE_MID_SHELF_X = (180, 230)
# Ice Beam room PLM (chozo pedestal x≈187 after left settle).
ICE_BEAM_MASK = 0x0002
ICE_PLM_X = 187
ICE_ROOM_SETTLE = 280
SNAKE_CLIMB_FRAMES = 2500
SNAKE_TUNNEL_FRAMES = 900
SNAKE_ICE_COLLECT_FRAMES = 500
SNAKE_DOOR_X = 470  # Ice blue door pressure band on right of Snake

# ---------------------------------------------------------------------------
# Ice Beam Room return leave (0xA890 → Snake 0xA8B9) — K5 stack hop 0 / return
# ---------------------------------------------------------------------------
# Pure PLM handoff ~(187, 120) pose 81; human leave shelf y≈139, left blue door.
# Tape Phase B return hop 19 (f16366–16491 clean leave; avoid thrash f16277–16365).
ICE_LEAVE_DOOR_X = 40  # left blue door pressure band
ICE_LEAVE_FRAMES = 480
ICE_SNAKE_RETURN_SETTLE = 280

# ---------------------------------------------------------------------------
# Ice Snake → Tutorial return (0xA8B9 → 0xA865) — K5 stack hop 1
# ---------------------------------------------------------------------------
# Pure ice-to-snake handoff ~(472, 395) pose 10; tape Phase B hop 20.
# Top-right blue door enter ~(236, 146). Prefer 2WJ climb (not freeze thrash).
SNAKE_TUTORIAL_DOOR_X = 210
SNAKE_TUTORIAL_DOOR_Y = (100, 175)
SNAKE_TO_TUTORIAL_DROP_FRAMES = 700
SNAKE_TOP_TO_TUTORIAL_FRAMES = 500
TUTORIAL_RETURN_SETTLE = 280

# ---------------------------------------------------------------------------
# Ice Tutorial → Gate return (0xA865 → 0xA815) — K5 stack hop 2
# ---------------------------------------------------------------------------
# Pure snake-to-tutorial handoff ~(39, 127) pose 81; tape Phase B hop 21.
# Room is 2 screens (widthBlocks 32); right blue door block [31,7] → x≈494.
# Platform bands from human ordinary path (re-solve; no thrash RLE):
#   left shelf y≈139 → lower floor y≈195 → mid Boyon shelf y≈120–145
#   → gap spin → door shelf y≈139 → Gate entry ~(494, 139).
TUTORIAL_SHELF_Y = (120, 155)
TUTORIAL_FLOOR_Y = (120, 160)  # door-height main shelves
TUTORIAL_LOWER_Y = (180, 220)  # pit floor after first gap
TUTORIAL_DOOR_X = 450
TUTORIAL_DOOR_Y = (100, 175)
TUTORIAL_TO_GATE_FRAMES = 2200
GATE_RETURN_SETTLE = 280

_TUTORIAL_TO_GATE_RLE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "ice_tutorial_to_gate_rle.json"
)
TUTORIAL_TO_GATE_RLE: RleScript = load_rle_json(_TUTORIAL_TO_GATE_RLE_PATH)
# First ~500f: left shelf → lower floor → mid structure lip (before morph tunnel).
_mid_acc = 0
_mid_runs: list[tuple[int, tuple[str, ...]]] = []
for _n, _btns in TUTORIAL_TO_GATE_RLE:
    if _mid_acc >= 500:
        break
    take = min(int(_n), 500 - _mid_acc)
    _mid_runs.append((take, tuple(_btns)))
    _mid_acc += take
TUTORIAL_MID_RLE: RleScript = tuple(_mid_runs)

# ---------------------------------------------------------------------------
# Ice Gate → Business return (0xA815 → 0xA7DE) — K5 stack hop 3
# ---------------------------------------------------------------------------
# Pure tutorial-to-gate handoff ~(807, 131) pose 81 mid-top (not door lip).
# Tape Phase B hop 22: morph drop → tunnel y≈569 roll RIGHT → Super door
# floor ~(1772, 651) → Business Super lip ~(39, 907).
GATE_MID_TOP_X = (450, 900)
GATE_MID_TOP_Y = (100, 200)
GATE_TUNNEL_X = (860, 920)  # shaft / tunnel mouth column
GATE_TUNNEL_Y = (555, 585)  # morph tunnel floor y≈569
GATE_SUPER_DOOR_X = 1740  # right Super door pressure band
GATE_TO_BUSINESS_FRAMES = 2200
BUSINESS_RETURN_SETTLE = 280

_GATE_TO_BUSINESS_RLE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "ice_gate_to_business_rle.json"
)
GATE_TO_BUSINESS_RLE: RleScript = load_rle_json(_GATE_TO_BUSINESS_RLE_PATH)


def in_business(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_BUSINESS


def in_ice_gate(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_ICE_GATE


def in_ice_acid(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_ICE_ACID


def in_ice_snake(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_ICE_SNAKE


def in_ice_tutorial(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_ICE_TUTORIAL


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
    """Right-side morph tunnel floor (y~377 only — not false ledge y409)."""
    if not in_ice_snake(state):
        return False
    x, y = int(state.samus_x), int(state.samus_y)
    return x >= SNAKE_TUNNEL_X_MIN and SNAKE_TUNNEL_Y[0] <= y <= SNAKE_TUNNEL_Y[1]


def on_snake_false_ledge(state: SuperMetroidState) -> bool:
    """Morph trap ledge just below tunnel mouth ~(x≥195, y~409)."""
    if not in_ice_snake(state):
        return False
    x, y = int(state.samus_x), int(state.samus_y)
    return x >= SNAKE_TUNNEL_X_MIN and SNAKE_FALSE_LEDGE_Y[0] <= y <= SNAKE_FALSE_LEDGE_Y[1]


def on_snake_mid_shelf(state: SuperMetroidState) -> bool:
    """Right-column mid shelf ~(197, 507) — jump-up morph launch pad."""
    if not in_ice_snake(state):
        return False
    x, y = int(state.samus_x), int(state.samus_y)
    if not (SNAKE_MID_SHELF_X[0] <= x <= SNAKE_MID_SHELF_X[1]):
        return False
    if not (SNAKE_MID_SHELF_Y[0] <= y <= SNAKE_MID_SHELF_Y[1]):
        return False
    return int(state.velocity_y) == 0


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
    "ICE_LEAVE_DOOR_X",
    "ICE_LEAVE_FRAMES",
    "ICE_PLM_X",
    "ICE_ROOM_SETTLE",
    "ICE_SNAKE_RETURN_SETTLE",
    "ICE_SUPER_DOOR_X",
    "ICE_SUPER_LIP_X_MAX",
    "ICE_SUPER_Y_MAX",
    "ICE_SUPER_Y_MIN",
    "LEDGE_POSES",
    "SNAKE_CLIMB_FRAMES",
    "SNAKE_DOOR_X",
    "SNAKE_FALSE_LEDGE_Y",
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
    "SNAKE_MID_SHELF_X",
    "SNAKE_MID_SHELF_Y",
    "SNAKE_TOP_X",
    "SNAKE_TOP_Y",
    "SNAKE_TOP_TO_TUTORIAL_FRAMES",
    "SNAKE_TO_TUTORIAL_DROP_FRAMES",
    "SNAKE_TUTORIAL_DOOR_X",
    "SNAKE_TUTORIAL_DOOR_Y",
    "TUTORIAL_RETURN_SETTLE",
    "TUTORIAL_SHELF_Y",
    "TUTORIAL_FLOOR_Y",
    "TUTORIAL_LOWER_Y",
    "TUTORIAL_DOOR_X",
    "TUTORIAL_DOOR_Y",
    "TUTORIAL_TO_GATE_FRAMES",
    "TUTORIAL_TO_GATE_RLE",
    "TUTORIAL_MID_RLE",
    "GATE_RETURN_SETTLE",
    "GATE_MID_TOP_X",
    "GATE_MID_TOP_Y",
    "GATE_TUNNEL_X",
    "GATE_TUNNEL_Y",
    "GATE_SUPER_DOOR_X",
    "GATE_TO_BUSINESS_FRAMES",
    "GATE_TO_BUSINESS_RLE",
    "BUSINESS_RETURN_SETTLE",
    "SNAKE_TUNNEL_EXIT_X",
    "SNAKE_TUNNEL_FLOOR_Y",
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
    "in_ice_tutorial",
    "in_ice_super_band",
    "on_acid_floor",
    "on_ice_super_lip",
    "on_snake_false_ledge",
    "on_snake_floor",
    "on_snake_mid_shelf",
    "on_snake_top",
    "on_snake_tunnel_band",
]
