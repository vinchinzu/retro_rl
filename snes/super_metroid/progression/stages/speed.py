"""KPDR K4 scaffold: post-Varia Business → Bubble → Speed (+ Wave/Ice branches).

Spine continuous Super+ product edges (business return, frog, bat_cave) come from
``continuous_edges_for_tips``. Hand-authored reverse/shortcut/branch edges stay
here until those stages hop-emit them.
"""

from __future__ import annotations

from super_metroid.progression.graph import RoomProgressionGraph
from super_metroid.progression.stages.kraid import (
    EDGES as _VARIA_EDGES,
    MILESTONES as _VARIA_MILESTONES,
    ROOMS as _VARIA_ROOMS,
    VARIA_CAPS as _VARIA_CAPS,
)
from super_metroid.progression.types import DoorEdge, ProgressCondition, ProgressionMilestone, RoomNode
from super_metroid.ram import BOMBS_MASK, HI_JUMP_MASK, MORPH_BALL_MASK, VARIA_MASK
from super_metroid.routes.kpdr.room_ids import (
    ROOM_ALPHA_PB,
    ROOM_BAT_CAVE,
    ROOM_BUBBLE,
    ROOM_BUSINESS,
    ROOM_CATHEDRAL,
    ROOM_CATHEDRAL_ENTRANCE,
    ROOM_CATERPILLAR,
    ROOM_CRATERIA_KIHUNTER,
    ROOM_DOUBLE_CHAMBER,
    ROOM_FROG_SAVE,
    ROOM_FROG_SPEEDWAY,
    ROOM_HELLWAY,
    ROOM_ICE,
    ROOM_ICE_ACID,
    ROOM_ICE_GATE,
    ROOM_ICE_SNAKE,
    ROOM_ICE_TUTORIAL,
    ROOM_MOAT,
    ROOM_RED_BRINSTAR_ELEVATOR,
    ROOM_RISING_TIDE,
    ROOM_SINGLE_CHAMBER,
    ROOM_SPEED,
    ROOM_SPEED_HALL,
    ROOM_UPPER_NORFAIR_FARM,
    ROOM_WAVE,
    ROOM_WEST_OCEAN,
    ROOM_WS_ENTRANCE,
    ROOM_WS_MAIN,
    ROOM_WS_BASEMENT,
    ROOM_PHANTOON,
)
from super_metroid.routes.kpdr.spine import continuous_edges_for_tips

# Edges start ``unverified``; first reverse hop ``varia_to_kraid`` promotes to
# ``controller_dev`` when pure probe is green. Do not claim continuous until
# composed on power-on via tip recipe.
#
# First Bubble visit is **Cathedral climb** (Business → Cathedral Entrance →
# Cathedral → Rising Tide → Bubble). Frog Speedway is a **post-Speed** shortcut
# only — Boost Blocks hard-lock without Speed (SM-K4.2-PURE residual RED).
_K4_CAPS = _VARIA_CAPS
_K4_SPEED_CAPS = _K4_CAPS | frozenset({"speed_booster"})

ROOMS = _VARIA_ROOMS + (
    # Return path rooms already present through Warehouse/Business on K2.
    RoomNode(ROOM_FROG_SAVE, "Frog Savestation", "Norfair"),
    RoomNode(ROOM_FROG_SPEEDWAY, "Frog Speedway", "Norfair"),
    RoomNode(ROOM_UPPER_NORFAIR_FARM, "Upper Norfair Farming Room", "Norfair"),
    # First-visit Bubble approach (no Speed required)
    RoomNode(ROOM_CATHEDRAL_ENTRANCE, "Cathedral Entrance", "Norfair"),
    RoomNode(ROOM_CATHEDRAL, "Cathedral", "Norfair"),
    RoomNode(ROOM_RISING_TIDE, "Rising Tide", "Norfair"),
    RoomNode(ROOM_BUBBLE, "Bubble Mountain", "Norfair"),
    RoomNode(ROOM_BAT_CAVE, "Bat Cave", "Norfair"),
    RoomNode(ROOM_SPEED_HALL, "Speed Booster Hall", "Norfair"),
    RoomNode(ROOM_SPEED, "Speed Booster Room", "Norfair"),
    RoomNode(ROOM_SINGLE_CHAMBER, "Single Chamber", "Norfair"),
    RoomNode(ROOM_DOUBLE_CHAMBER, "Double Chamber", "Norfair"),
    RoomNode(ROOM_WAVE, "Wave Beam Room", "Norfair"),
    RoomNode(ROOM_ICE_GATE, "Ice Beam Gate Room", "Norfair"),
    RoomNode(ROOM_ICE_ACID, "Ice Beam Acid Room", "Norfair"),
    RoomNode(ROOM_ICE_TUTORIAL, "Ice Beam Tutorial Room", "Norfair"),
    RoomNode(ROOM_ICE_SNAKE, "Ice Beam Snake Room", "Norfair"),
    RoomNode(ROOM_ICE, "Ice Beam Room", "Norfair"),
    RoomNode(ROOM_HELLWAY, "Hellway", "Brinstar"),
    RoomNode(ROOM_CATERPILLAR, "Caterpillar Room", "Brinstar"),
    RoomNode(ROOM_ALPHA_PB, "Alpha Power Bomb Room", "Brinstar"),
    RoomNode(
        ROOM_RED_BRINSTAR_ELEVATOR, "Red Brinstar Elevator Room", "Brinstar"
    ),
    RoomNode(ROOM_CRATERIA_KIHUNTER, "Crateria Kihunter Room", "Crateria"),
    RoomNode(ROOM_MOAT, "The Moat", "Crateria"),
    RoomNode(ROOM_WEST_OCEAN, "West Ocean", "Crateria"),
    RoomNode(ROOM_WS_ENTRANCE, "Wrecked Ship Entrance", "Wrecked Ship"),
    RoomNode(ROOM_WS_MAIN, "Wrecked Ship Main Shaft", "Wrecked Ship"),
    RoomNode(ROOM_WS_BASEMENT, "Wrecked Ship Basement", "Wrecked Ship"),
    RoomNode(ROOM_PHANTOON, "Phantoon's Room", "Wrecked Ship"),
)

# Spine continuous Super+ product edges (business return, frog, bat_cave).
# Warehouse→Business return reuses hijump-stage warehouse_to_business edge.
# Hand-authored: Speedway shortcut, Speed/Wave reverse, Ice Tutorial return.
# frog_save_to_business: spine-emitted continuous on tip ``ice`` (rr-kxge /
# Wave→Business pure return compose) — do not hand-author here.
_BRANCH_EDGES = (
    # --- Post-Speed shortcut: Frog Speedway (Boost Blocks need Speed) ---
    DoorEdge(
        "frog_save_to_speedway",
        ROOM_FROG_SAVE,
        ROOM_FROG_SPEEDWAY,
        "right",
        "left",
        _K4_CAPS,
        "kpdr_k4_speed",
        "unverified",  # pure controller green; not on first Bubble path
    ),
    DoorEdge(
        "speedway_to_farm",
        ROOM_FROG_SPEEDWAY,
        ROOM_UPPER_NORFAIR_FARM,
        "right",
        "left",
        _K4_SPEED_CAPS,  # Boost Blocks; SM-K4.2-PURE RED without Speed
        "kpdr_k4_speed",
        "unverified",
    ),
    DoorEdge(
        "farm_to_bubble",
        ROOM_UPPER_NORFAIR_FARM,
        ROOM_BUBBLE,
        "right",
        "left",
        _K4_SPEED_CAPS,  # only after Speedway; post-Speed farm entry
        "kpdr_k4_speed",
        "unverified",
    ),
    # bat_cave_to_speed_hall + speed_hall_to_speed: spine-emitted continuous
    # (tip ``speed``) via continuous_edges_for_tips — do not hand-author here.
    # Wave branch (bubble→single→double→wave): spine-emitted continuous
    # (tip ``wave``) via continuous_edges_for_tips — do not hand-author here.
    # --- Post-Speed reverse (Speed return hop walks Hall→Bat→Bubble) ---
    # Multi-room hop ``speed_return_to_bubble`` does not emit one DoorEdge; the
    # three reverse doors must be known for continuous integrity.
    DoorEdge(
        "speed_to_speed_hall",
        ROOM_SPEED,
        ROOM_SPEED_HALL,
        "left",
        "right",
        _K4_SPEED_CAPS,
        "kpdr_k4_wave",
        "controller_dev",
    ),
    DoorEdge(
        "speed_hall_to_bat_cave",
        ROOM_SPEED_HALL,
        ROOM_BAT_CAVE,
        "left",
        "right",
        _K4_SPEED_CAPS,
        "kpdr_k4_wave",
        "controller_dev",
    ),
    DoorEdge(
        "bat_cave_to_bubble",
        ROOM_BAT_CAVE,
        ROOM_BUBBLE,
        "left",
        "right",
        _K4_SPEED_CAPS,
        "kpdr_k4_wave",
        "controller_dev",
    ),
    # Ice outbound + Wave→Business return prefix: spine-emitted continuous
    # (tip ``ice``) via continuous_edges_for_tips — do not hand-author here.
    # Pure dual GREEN rr-dbu.11 + rr-vqv3; continuous dual rr-kxge (no STATUS
    # without dual continuous green). Tutorial return path only below.
    DoorEdge(
        "ice_gate_to_tutorial",
        ROOM_ICE_GATE,
        ROOM_ICE_TUTORIAL,
        "left",
        "right",
        _K4_CAPS,
        "kpdr_k4_ice",
        "unverified",
    ),
    DoorEdge(
        "ice_tutorial_to_snake",
        ROOM_ICE_TUTORIAL,
        ROOM_ICE_SNAKE,
        "left",
        "right",
        _K4_CAPS,
        "kpdr_k4_ice",
        "unverified",
    ),
    # play_moat_cross from a Moat standing handoff leaves back to Kihunter
    # for the spark runway (play_leave_moat_to_kihunter). Not a spine hop —
    # same integrity pattern as Speed-return reverse doors.
    DoorEdge(
        "moat_to_kihunter",
        ROOM_MOAT,
        ROOM_CRATERIA_KIHUNTER,
        "left",
        "left",
        _K4_SPEED_CAPS,
        "kpdr_k6_moat",
        "controller_dev",
    ),
)

EDGES = (
    _VARIA_EDGES
    + continuous_edges_for_tips(
        "business",
        "frog",
        "bat_cave",
        "speed",
        "wave",
        "ice",
        "alpha_pb",
        "moat",
        "ws",
        "phantoon",
    )
    + _BRANCH_EDGES
)

MILESTONES = _VARIA_MILESTONES + (
    ProgressionMilestone(
        "business_post_varia",
        "Returned to Business Center after Varia (K4 staging room)",
        ProgressCondition(
            room_id=ROOM_BUSINESS,
            collected_items_mask=(
                MORPH_BALL_MASK | BOMBS_MASK | HI_JUMP_MASK | VARIA_MASK
            ),
            minimum_ammo_capacities=(10, 5, 0),
        ),
        requires=_K4_CAPS,
        timeout_frames=40_000,
        policy_id="kpdr_varia_return",
    ),
    ProgressionMilestone(
        "bubble_mountain_entry",
        "Natural Bubble Mountain entry on post-Varia K4 path",
        ProgressCondition(
            room_id=ROOM_BUBBLE,
            collected_items_mask=(
                MORPH_BALL_MASK | BOMBS_MASK | HI_JUMP_MASK | VARIA_MASK
            ),
            minimum_ammo_capacities=(10, 5, 0),
        ),
        requires=_K4_CAPS,
        timeout_frames=30_000,
        policy_id="kpdr_k4_speed",
    ),
    ProgressionMilestone(
        "speed_collected",
        "Natural Speed Booster collect (K4.5) — STATUS-promoted continuous tip (130388f dual)",
        ProgressCondition(
            room_id=ROOM_SPEED,
            collected_items_mask=(
                MORPH_BALL_MASK | BOMBS_MASK | HI_JUMP_MASK | VARIA_MASK
            ),
            minimum_ammo_capacities=(10, 5, 0),
        ),
        requires=_K4_CAPS,
        acquires=frozenset({"speed_booster"}),
        timeout_frames=40_000,
        policy_id="kpdr_k4_speed",
    ),
)

SPEED_GRAPH = RoomProgressionGraph(
    ROOMS,
    EDGES,
    MILESTONES,
    graph_id="speed",
)

__all__ = ["ROOMS", "EDGES", "MILESTONES", "SPEED_GRAPH"]
