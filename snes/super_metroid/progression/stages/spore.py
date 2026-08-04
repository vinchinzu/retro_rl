"""Spore stage: post-Torizo → Green Brinstar → Spore Spawn Super Room."""

from __future__ import annotations

from super_metroid.progression.graph import RoomProgressionGraph
from super_metroid.progression.stages.early import (
    EDGES as _EARLY_EDGES,
    MILESTONES as _EARLY_MILESTONES,
    ROOMS as _EARLY_ROOMS,
)
from super_metroid.progression.types import DoorEdge, ProgressCondition, ProgressionMilestone, RoomNode
from super_metroid.ram import BOMBS_MASK, MORPH_BALL_MASK
from super_metroid.routes.kpdr.room_ids import (
    ROOM_BIG_PINK,
    ROOM_DACHORA,
    ROOM_GREEN_ELEVATOR,
    ROOM_GREEN_MAIN_SHAFT,
    ROOM_GREEN_PIRATES,
    ROOM_LOWER_MUSHROOMS,
    ROOM_PARLOR,
    ROOM_SPORE_KIHUNTER,
    ROOM_SPORE_SPAWN,
    ROOM_SUPER,
    ROOM_TERMINATOR,
)

ROOMS = _EARLY_ROOMS + (
    RoomNode(ROOM_TERMINATOR, "Terminator Room", "Crateria"),
    RoomNode(ROOM_GREEN_PIRATES, "Green Pirates Shaft", "Crateria"),
    RoomNode(ROOM_LOWER_MUSHROOMS, "Lower Mushrooms", "Crateria"),
    RoomNode(ROOM_GREEN_ELEVATOR, "Elevator To Green Brinstar", "Crateria", frozenset({"elevator"})),
    RoomNode(
        ROOM_GREEN_MAIN_SHAFT, "Green Brinstar Main Shaft", "Brinstar", frozenset({"vertical_shaft"})
    ),
    RoomNode(ROOM_DACHORA, "Dachora Room", "Brinstar"),
    RoomNode(ROOM_BIG_PINK, "Big Pink", "Brinstar", frozenset({"vertical_shaft"})),
    RoomNode(ROOM_SPORE_KIHUNTER, "Spore Spawn Kihunter Room", "Brinstar"),
    RoomNode(ROOM_SPORE_SPAWN, "Spore Spawn Room", "Brinstar", frozenset({"boss_room"})),
    RoomNode(ROOM_SUPER, "Spore Spawn Super Room", "Brinstar", frozenset({"item_room"})),
)

EDGES = _EARLY_EDGES + (
    DoorEdge(
        "parlor_to_terminator",
        ROOM_PARLOR,
        ROOM_TERMINATOR,
        "left",
        "right",
        frozenset({"bombs", "bomb_torizo_defeated"}),
        "post_torizo_controller",
        "continuous",
    ),
    DoorEdge(
        "terminator_to_green_pirates",
        ROOM_TERMINATOR,
        ROOM_GREEN_PIRATES,
        "left",
        "right",
        policy_id="post_torizo_controller",
        verification="continuous",
    ),
    DoorEdge(
        "green_pirates_to_lower_mushrooms",
        ROOM_GREEN_PIRATES,
        ROOM_LOWER_MUSHROOMS,
        "left",
        "right",
        policy_id="post_torizo_controller",
        verification="continuous",
    ),
    DoorEdge(
        "lower_mushrooms_to_green_elevator",
        ROOM_LOWER_MUSHROOMS,
        ROOM_GREEN_ELEVATOR,
        "left",
        "right",
        policy_id="post_torizo_controller",
        verification="continuous",
    ),
    DoorEdge(
        "green_elevator_to_main_shaft",
        ROOM_GREEN_ELEVATOR,
        ROOM_GREEN_MAIN_SHAFT,
        "down",
        "elevator",
        policy_id="post_torizo_controller",
        verification="continuous",
    ),
    DoorEdge(
        "main_shaft_to_dachora",
        ROOM_GREEN_MAIN_SHAFT,
        ROOM_DACHORA,
        "right",
        "left",
        frozenset({"missiles"}),
        "post_torizo_controller",
        "continuous",
    ),
    DoorEdge(
        "dachora_to_big_pink",
        ROOM_DACHORA,
        ROOM_BIG_PINK,
        "right",
        "left",
        frozenset({"morph_ball", "bombs"}),
        "post_torizo_controller",
        "continuous",
    ),
    DoorEdge(
        "big_pink_to_spore_kihunters",
        ROOM_BIG_PINK,
        ROOM_SPORE_KIHUNTER,
        "right",
        "left",
        frozenset({"missiles"}),
        "post_torizo_controller",
        "continuous",
    ),
    DoorEdge(
        "spore_kihunters_to_spore_spawn",
        ROOM_SPORE_KIHUNTER,
        ROOM_SPORE_SPAWN,
        "up",
        "bottom",
        policy_id="post_torizo_controller",
        verification="continuous",
    ),
    DoorEdge(
        "spore_spawn_to_super_room",
        ROOM_SPORE_SPAWN,
        ROOM_SUPER,
        "right",
        "left",
        frozenset({"spore_spawn_defeated"}),
        "post_torizo_controller",
        "continuous",
    ),
)

MILESTONES = _EARLY_MILESTONES + (
    ProgressionMilestone(
        "spore_spawn_clear",
        "Spore Spawn defeated and post-boss room reached naturally",
        ProgressCondition(
            room_id=ROOM_SUPER,
            collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK,
            minimum_ammo_capacities=(10, 0, 0),
        ),
        requires=frozenset({"morph_ball", "bombs", "missiles"}),
        acquires=frozenset({"spore_spawn_defeated"}),
        timeout_frames=40_000,
        policy_id="post_torizo_controller",
    ),
)

SPORE_GRAPH = RoomProgressionGraph(
    ROOMS,
    EDGES,
    MILESTONES,
    graph_id="spore",
)

__all__ = ["ROOMS", "EDGES", "MILESTONES", "SPORE_GRAPH"]
