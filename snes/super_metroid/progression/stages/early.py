"""Bombs stage: Morph → first missiles → Bomb Torizo (hand-authored edges)."""

from __future__ import annotations

from super_metroid.progression.graph import RoomProgressionGraph
from super_metroid.progression.stages.morph import (
    EDGES as _MORPH_EDGES,
    MILESTONES as _MORPH_MILESTONES,
    ROOMS as _MORPH_ROOMS,
)
from super_metroid.progression.types import DoorEdge, ProgressCondition, ProgressionMilestone, RoomNode
from super_metroid.ram import BOMBS_MASK, MORPH_BALL_MASK
from super_metroid.routes.kpdr.room_ids import (
    ROOM_BLUE_BRINSTAR_ELEVATOR,
    ROOM_BLUE_BRINSTAR_ETANK,
    ROOM_BOMB_TORIZO,
    ROOM_CLIMB,
    ROOM_CONSTRUCTION,
    ROOM_FIRST_MISSILE,
    ROOM_FLYWAY,
    ROOM_MORPH,
    ROOM_PARLOR,
    ROOM_PIT,
)

ROOMS = _MORPH_ROOMS + (
    RoomNode(
        ROOM_BLUE_BRINSTAR_ETANK, "Blue Brinstar Energy Tank Room", "Brinstar", frozenset({"item_room"})
    ),
    RoomNode(ROOM_FIRST_MISSILE, "First Missile Room", "Brinstar", frozenset({"item_room"})),
    RoomNode(ROOM_FLYWAY, "Flyway", "Crateria"),
    RoomNode(ROOM_BOMB_TORIZO, "Bomb Torizo Room", "Crateria", frozenset({"boss_item_room"})),
)

EDGES = _MORPH_EDGES + (
    DoorEdge(
        "construction_to_first_missile",
        ROOM_CONSTRUCTION,
        ROOM_FIRST_MISSILE,
        "upper_right",
        "left",
        frozenset({"morph_ball"}),
        "two_missile_detour",
        "continuous",
    ),
    DoorEdge(
        "first_missile_to_construction",
        ROOM_FIRST_MISSILE,
        ROOM_CONSTRUCTION,
        "left",
        "upper_right",
        frozenset({"morph_ball"}),
        "two_missile_detour",
        "continuous",
    ),
    DoorEdge(
        "construction_to_blue_missile",
        ROOM_CONSTRUCTION,
        ROOM_BLUE_BRINSTAR_ETANK,
        "lower_left",
        "left",
        frozenset({"morph_ball"}),
        "two_missile_detour",
        "continuous",
    ),
    DoorEdge(
        "blue_missile_to_construction",
        ROOM_BLUE_BRINSTAR_ETANK,
        ROOM_CONSTRUCTION,
        "left",
        "lower_left",
        frozenset({"morph_ball"}),
        "two_missile_detour",
        "continuous",
    ),
    DoorEdge(
        "construction_to_morph",
        ROOM_CONSTRUCTION,
        ROOM_MORPH,
        "left",
        "right",
        frozenset({"morph_ball"}),
        "construction_return",
        "continuous",
    ),
    DoorEdge(
        "morph_to_elevator",
        ROOM_MORPH,
        ROOM_BLUE_BRINSTAR_ELEVATOR,
        "right",
        "elevator",
        frozenset({"morph_ball"}),
        "morph_return",
        "continuous",
    ),
    DoorEdge(
        "elevator_to_pit_return",
        ROOM_BLUE_BRINSTAR_ELEVATOR,
        ROOM_PIT,
        "left",
        "right",
        frozenset({"morph_ball"}),
        "elevator_return",
        "continuous",
    ),
    DoorEdge(
        "pit_to_climb_return",
        ROOM_PIT,
        ROOM_CLIMB,
        "left",
        "bottom",
        frozenset({"morph_ball"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
    DoorEdge(
        "climb_to_parlor_return",
        ROOM_CLIMB,
        ROOM_PARLOR,
        "top",
        "bottom_left",
        frozenset({"morph_ball"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
    DoorEdge(
        "parlor_to_flyway",
        ROOM_PARLOR,
        ROOM_FLYWAY,
        "right",
        "left",
        frozenset({"morph_ball"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
    DoorEdge(
        "flyway_to_torizo",
        ROOM_FLYWAY,
        ROOM_BOMB_TORIZO,
        "right",
        "left",
        frozenset({"missiles"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
    DoorEdge(
        "torizo_to_flyway",
        ROOM_BOMB_TORIZO,
        ROOM_FLYWAY,
        "left",
        "right",
        frozenset({"bombs", "bomb_torizo_defeated"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
    DoorEdge(
        "flyway_to_parlor_return",
        ROOM_FLYWAY,
        ROOM_PARLOR,
        "left",
        "right",
        frozenset({"bombs", "bomb_torizo_defeated"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
)

MILESTONES = _MORPH_MILESTONES + (
    ProgressionMilestone(
        "first_missiles",
        "First Missile expansion collected naturally",
        ProgressCondition(room_id=ROOM_FIRST_MISSILE, minimum_ammo_capacities=(5, 0, 0)),
        requires=frozenset({"morph_ball"}),
        acquires=frozenset({"missiles"}),
        timeout_frames=2_000,
        policy_id="two_missile_detour",
    ),
    ProgressionMilestone(
        "blue_brinstar_missiles",
        "Blue Brinstar Missile expansion collected naturally",
        ProgressCondition(room_id=ROOM_BLUE_BRINSTAR_ETANK, minimum_ammo_capacities=(10, 0, 0)),
        requires=frozenset({"morph_ball", "missiles"}),
        timeout_frames=3_000,
        policy_id="two_missile_detour",
    ),
    ProgressionMilestone(
        "bombs",
        "Morph Ball Bombs collected naturally",
        ProgressCondition(
            room_id=ROOM_BOMB_TORIZO,
            collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK,
            minimum_ammo_capacities=(10, 0, 0),
        ),
        requires=frozenset({"morph_ball", "missiles"}),
        acquires=frozenset({"bombs"}),
        timeout_frames=20_000,
        policy_id="pit_to_torizo_replay",
    ),
    ProgressionMilestone(
        "bomb_torizo_clear",
        "Bomb Torizo defeated and room exited naturally",
        ProgressCondition(
            room_id=ROOM_FLYWAY,
            collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK,
            minimum_ammo_capacities=(10, 0, 0),
        ),
        requires=frozenset({"bombs"}),
        acquires=frozenset({"bomb_torizo_defeated"}),
        timeout_frames=8_000,
        policy_id="pit_to_torizo_replay",
    ),
)

EARLY_GAME_GRAPH = RoomProgressionGraph(
    ROOMS,
    EDGES,
    MILESTONES,
    graph_id="bombs",
)

__all__ = ["ROOMS", "EDGES", "MILESTONES", "EARLY_GAME_GRAPH"]
