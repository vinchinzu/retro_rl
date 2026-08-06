"""Below Spazer / Spazer Room geometry: named bands and pure predicates.

Single source of truth for x/y/pose contracts used by climb, door approach,
collect, and top-drop hops. Controllers import predicates — do not re-encode
magic thresholds inline.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import WallJumpTiming
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER
from super_metroid.routes.skills.geometry import POSE_KNOCKBACK, TRUE_GROUND

# collected_beams bit for Spazer.
SPAZER_BEAM_MASK = 0x0004

# Item-grab pose (Chozo fanfare); not in shared POSE_KNOCKBACK.
POSE_ITEM_GRAB = frozenset({164})
LAG_POSES = POSE_KNOCKBACK | POSE_ITEM_GRAB
# Crouch / ball variants seen on floor land after top drop.
FLOOR_UNMORPH_POSES = frozenset({31, 39, 40, 41, 42, 65})

# ---------------------------------------------------------------------------
# Named geometry bands (Below Spazer 0xA408)
# ---------------------------------------------------------------------------

# Super-door approach needs y on the upper platform.
TOP_Y_MAX = 160
# Land band after WJ (upper shaft; door sill is tighter).
CLIMB_TOP_Y = 190
# Node-4 left top land (natural stand ~y91 after over-lip crest).
# Air peak ~y124 @ x≈59 is NOT land; x≥70 excludes shaft air.
SOLID_TOP_Y = (88, 150)
SOLID_TOP_X_MIN = 70
# Over-lip settle: once past wall, y≲ this before ground settle.
OVER_LIP_Y_MAX = 110
CREST_LAND_X = (75, 120)

MID_Y = 300
MID_BAND_Y = (210, 300)
MID_BAND_X = (40, 80)
# Mid-right platforms under top (place-green → West / TOP-MID success).
MID_PLATFORM_Y = 220
# Floor water band for West runner handoff.
FLOOR_Y_MIN = 360
# Shaft air too high to unmorph+off-door without dumping height.
HIGH_AIR_Y_MAX = 320
# Fell out of crest attempt.
CREST_FAIL_Y = 420

DOOR_SAFE_X = 48
# Super green door sill — human shoots Super from ~(426,139); pre-door ~460.
DOOR_X_MIN = 420
# Top handoff clear of open Super (~480); return pin x≲400.
HANDOFF_X_MAX = 400
DOOR_TRAP_X_MAX = 430
# Top ledge land band — left top pillar ok, not Bat door face.
TOP_LAND_X = (40, 400)
TOP_LEDGE_X_MAX = 480
# Floor-lip reseat for open-loop spin (guide ~48–55).
FLOOR_LIP_X = (46, 58)
CACATAC_OFF_DOOR_X = 48

# Left-shaft consecutive WJ (place-green from y≈230–280, x≈48–55).
WJ_LEFT = WallJumpTiming(
    into="LEFT", flip="RIGHT", into_frames=12, amid_frames=2, flip_frames=14
)
WJ_RIGHT = WallJumpTiming(
    into="RIGHT", flip="LEFT", into_frames=12, amid_frames=2, flip_frames=12
)
WJ_PAIR: tuple[WallJumpTiming, WallJumpTiming] = (WJ_LEFT, WJ_RIGHT)
# Open-loop period after WJ height: into LEFT / flip RIGHT / spin RIGHT.
CREST_PERIOD = 14
CREST_PERIOD_INTO = 4
CREST_PERIOD_FLIP = 3


def in_below_spazer(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_BELOW_SPAZER


def is_true_ground_pose(state: SuperMetroidState) -> bool:
    """Standing / run poses used as land contracts (pose-only; no vy gate)."""
    return int(state.pose) in TRUE_GROUND


def is_lag_pose(state: SuperMetroidState) -> bool:
    return int(state.pose) in LAG_POSES


def on_top_ledge(state: SuperMetroidState, *, y_max: int | None = None) -> bool:
    ym = CLIMB_TOP_Y if y_max is None else y_max
    return (
        in_below_spazer(state)
        and int(state.samus_y) <= ym
        and TOP_LAND_X[0] <= int(state.samus_x) <= TOP_LEDGE_X_MAX
    )


def on_solid_top(state: SuperMetroidState) -> bool:
    """Walkable Super-door sill / node-4 top (not shaft air peak)."""
    return (
        in_below_spazer(state)
        and SOLID_TOP_Y[0] <= int(state.samus_y) <= SOLID_TOP_Y[1]
        and int(state.samus_x) >= SOLID_TOP_X_MIN
        and is_true_ground_pose(state)
    )


def solid_ish_top(state: SuperMetroidState) -> bool:
    """Solid top or y≤160 grounded on walkable top (not random air)."""
    if on_solid_top(state):
        return True
    return (
        in_below_spazer(state)
        and int(state.samus_y) <= TOP_Y_MAX
        and int(state.samus_x) >= SOLID_TOP_X_MIN
        and is_true_ground_pose(state)
    )


def on_super_door_approach(state: SuperMetroidState) -> bool:
    """Green Super door approach band (high x, y≤160, grounded)."""
    return (
        in_below_spazer(state)
        and int(state.samus_x) >= DOOR_X_MIN
        and int(state.samus_y) <= TOP_Y_MAX
        and is_true_ground_pose(state)
    )


def mid_band(state: SuperMetroidState) -> bool:
    """In-shaft mid height for WJ crest (standing or still spinning)."""
    return (
        in_below_spazer(state)
        and MID_BAND_Y[0] <= int(state.samus_y) <= MID_BAND_Y[1]
        and MID_BAND_X[0] <= int(state.samus_x) <= MID_BAND_X[1]
    )


def standing_mid_seat(state: SuperMetroidState) -> bool:
    """Standing mid WJ seat: mid band + grounded pose."""
    return mid_band(state) and is_true_ground_pose(state)


def on_mid_or_floor(state: SuperMetroidState) -> bool:
    """Past top ledge — mid platform band or floor (TOP-MID success bar)."""
    return in_below_spazer(state) and int(state.samus_y) >= MID_PLATFORM_Y


def has_spazer(state: SuperMetroidState) -> bool:
    return bool(int(state.collected_beams) & SPAZER_BEAM_MASK)


__all__ = [
    "CACATAC_OFF_DOOR_X",
    "CLIMB_TOP_Y",
    "CREST_FAIL_Y",
    "CREST_LAND_X",
    "CREST_PERIOD",
    "CREST_PERIOD_FLIP",
    "CREST_PERIOD_INTO",
    "DOOR_SAFE_X",
    "DOOR_TRAP_X_MAX",
    "DOOR_X_MIN",
    "FLOOR_LIP_X",
    "FLOOR_UNMORPH_POSES",
    "FLOOR_Y_MIN",
    "HANDOFF_X_MAX",
    "HIGH_AIR_Y_MAX",
    "LAG_POSES",
    "MID_BAND_X",
    "MID_BAND_Y",
    "MID_PLATFORM_Y",
    "MID_Y",
    "OVER_LIP_Y_MAX",
    "POSE_ITEM_GRAB",
    "SOLID_TOP_X_MIN",
    "SOLID_TOP_Y",
    "SPAZER_BEAM_MASK",
    "TOP_LAND_X",
    "TOP_LEDGE_X_MAX",
    "TOP_Y_MAX",
    "TRUE_GROUND",
    "WJ_LEFT",
    "WJ_PAIR",
    "WJ_RIGHT",
    "has_spazer",
    "in_below_spazer",
    "is_lag_pose",
    "is_true_ground_pose",
    "mid_band",
    "on_mid_or_floor",
    "on_solid_top",
    "on_super_door_approach",
    "on_top_ledge",
    "solid_ish_top",
    "standing_mid_seat",
]
