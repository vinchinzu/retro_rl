"""Room progression graph and milestone dataclasses.

The graph separates route choice from room movement policy.  Nodes identify
rooms, edges identify observed transitions and capability requirements, and
milestones describe inventory/event predicates.  It can grow into the full
game without forcing a nonlinear world into a stage-number list.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from super_metroid.ram import BOMBS_MASK, MORPH_BALL_MASK, SuperMetroidState


@dataclass(frozen=True)
class RoomNode:
    room_id: int
    name: str
    area: str
    tags: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class DoorEdge:
    edge_id: str
    source_room_id: int
    target_room_id: int
    exit_direction: str
    entry_direction: str
    requires: frozenset[str] = field(default_factory=frozenset)
    policy_id: str = ""
    verification: str = "unverified"


@dataclass(frozen=True)
class ProgressCondition:
    room_id: int | None = None
    game_states: frozenset[int] = field(default_factory=frozenset)
    collected_items_mask: int = 0
    minimum_ammo_capacities: tuple[int, int, int] = (0, 0, 0)

    def matches(self, state: SuperMetroidState) -> bool:
        if self.room_id is not None and state.room_id != self.room_id:
            return False
        if self.game_states and state.game_state not in self.game_states:
            return False
        if self.collected_items_mask:
            if state.collected_items & self.collected_items_mask != self.collected_items_mask:
                return False
        actual = (
            state.max_missiles,
            state.max_super_missiles,
            state.max_power_bombs,
        )
        return all(value >= minimum for value, minimum in zip(actual, self.minimum_ammo_capacities))


@dataclass(frozen=True)
class ProgressionMilestone:
    milestone_id: str
    label: str
    condition: ProgressCondition
    requires: frozenset[str] = field(default_factory=frozenset)
    acquires: frozenset[str] = field(default_factory=frozenset)
    timeout_frames: int = 0
    policy_id: str = ""


@dataclass(frozen=True)
class ObservedTransition:
    frame: int
    source_room_id: int
    target_room_id: int
    edge_id: str | None


class RoomProgressionGraph:
    """Validated directed room graph with capability-aware pathfinding."""

    def __init__(
        self,
        rooms: tuple[RoomNode, ...],
        edges: tuple[DoorEdge, ...],
        milestones: tuple[ProgressionMilestone, ...],
        *,
        graph_id: str,
    ) -> None:
        self.graph_id = graph_id
        self.rooms = {room.room_id: room for room in rooms}
        self.edges = edges
        self.milestones = milestones
        self._outgoing: dict[int, list[DoorEdge]] = defaultdict(list)
        self._edge_pairs: dict[tuple[int, int], DoorEdge] = {}
        for edge in edges:
            if edge.source_room_id not in self.rooms or edge.target_room_id not in self.rooms:
                raise ValueError(f"edge {edge.edge_id} references an unknown room")
            self._outgoing[edge.source_room_id].append(edge)
            self._edge_pairs[(edge.source_room_id, edge.target_room_id)] = edge

    def room_name(self, room_id: int) -> str:
        room = self.rooms.get(room_id)
        return room.name if room else f"Unknown 0x{room_id:04X}"

    def edge_for(self, source_room_id: int, target_room_id: int) -> DoorEdge | None:
        return self._edge_pairs.get((source_room_id, target_room_id))

    def observe_transition(
        self,
        frame: int,
        source_room_id: int,
        target_room_id: int,
    ) -> ObservedTransition:
        edge = self.edge_for(source_room_id, target_room_id)
        return ObservedTransition(
            frame=frame,
            source_room_id=source_room_id,
            target_room_id=target_room_id,
            edge_id=edge.edge_id if edge else None,
        )

    def shortest_path(
        self,
        source_room_id: int,
        target_room_id: int,
        capabilities: frozenset[str] = frozenset(),
    ) -> tuple[DoorEdge, ...] | None:
        if source_room_id == target_room_id:
            return ()
        queue: deque[int] = deque([source_room_id])
        parent: dict[int, tuple[int, DoorEdge]] = {}
        seen = {source_room_id}
        while queue:
            room_id = queue.popleft()
            for edge in self._outgoing.get(room_id, ()):
                if not edge.requires.issubset(capabilities):
                    continue
                if edge.target_room_id in seen:
                    continue
                seen.add(edge.target_room_id)
                parent[edge.target_room_id] = (room_id, edge)
                if edge.target_room_id == target_room_id:
                    path: list[DoorEdge] = []
                    cursor = target_room_id
                    while cursor != source_room_id:
                        previous, used = parent[cursor]
                        path.append(used)
                        cursor = previous
                    return tuple(reversed(path))
                queue.append(edge.target_room_id)
        return None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready navigation-map representation."""
        rooms = [
            {
                "roomId": room.room_id,
                "roomIdHex": f"0x{room.room_id:04X}",
                "name": room.name,
                "area": room.area,
                "tags": sorted(room.tags),
            }
            for room in self.rooms.values()
        ]
        edges = [
            {
                "edgeId": edge.edge_id,
                "sourceRoomId": edge.source_room_id,
                "sourceRoomIdHex": f"0x{edge.source_room_id:04X}",
                "targetRoomId": edge.target_room_id,
                "targetRoomIdHex": f"0x{edge.target_room_id:04X}",
                "exitDirection": edge.exit_direction,
                "entryDirection": edge.entry_direction,
                "requires": sorted(edge.requires),
                "policyId": edge.policy_id,
                "verification": edge.verification,
            }
            for edge in self.edges
        ]
        milestones = [
            {
                "milestoneId": milestone.milestone_id,
                "label": milestone.label,
                "condition": {
                    "roomId": milestone.condition.room_id,
                    "roomIdHex": (
                        f"0x{milestone.condition.room_id:04X}"
                        if milestone.condition.room_id is not None
                        else None
                    ),
                    "gameStates": sorted(milestone.condition.game_states),
                    "collectedItemsMask": milestone.condition.collected_items_mask,
                    "minimumAmmoCapacities": list(
                        milestone.condition.minimum_ammo_capacities
                    ),
                },
                "requires": sorted(milestone.requires),
                "acquires": sorted(milestone.acquires),
                "timeoutFrames": milestone.timeout_frames,
                "policyId": milestone.policy_id,
            }
            for milestone in self.milestones
        ]
        return {
            "schemaVersion": 1,
            "graphId": self.graph_id,
            "rooms": rooms,
            "edges": edges,
            "milestones": milestones,
        }


_ROOMS = (
    RoomNode(0xDF45, "Ceres Elevator Room", "Ceres"),
    RoomNode(0xDF8D, "Ceres Falling Tile Room", "Ceres"),
    RoomNode(0xDFD7, "Ceres Magnet Stairs Room", "Ceres"),
    RoomNode(0xE021, "Ceres Dead Scientist Room", "Ceres"),
    RoomNode(0xE06B, "Ceres Flat Room", "Ceres"),
    RoomNode(0xE0B5, "Ceres Ridley Room", "Ceres", frozenset({"scripted_boss"})),
    RoomNode(0x91F8, "Landing Site", "Crateria", frozenset({"ship"})),
    RoomNode(0x92FD, "Parlor and Alcatraz", "Crateria"),
    RoomNode(0x96BA, "Climb", "Crateria", frozenset({"vertical_shaft"})),
    RoomNode(0x975C, "Pit Room", "Crateria"),
    RoomNode(0x97B5, "Blue Brinstar Elevator Room", "Crateria"),
    RoomNode(0x9E9F, "Morph Ball Room", "Brinstar", frozenset({"item_room"})),
    RoomNode(0x9F11, "Construction Zone", "Brinstar"),
)

_EDGES = (
    DoorEdge("ceres_elevator_to_falling", 0xDF45, 0xDF8D, "right", "left", policy_id="ceres_outbound", verification="continuous"),
    DoorEdge("ceres_falling_to_magnet", 0xDF8D, 0xDFD7, "right", "left", policy_id="ceres_outbound", verification="continuous"),
    DoorEdge("ceres_magnet_to_scientist", 0xDFD7, 0xE021, "bottom_right", "left", policy_id="ceres_outbound", verification="continuous"),
    DoorEdge("ceres_scientist_to_flat", 0xE021, 0xE06B, "right", "left", policy_id="ceres_outbound", verification="continuous"),
    DoorEdge("ceres_flat_to_ridley", 0xE06B, 0xE0B5, "right", "left", policy_id="ceres_outbound", verification="continuous"),
    DoorEdge("ceres_ridley_to_flat", 0xE0B5, 0xE06B, "left", "right", policy_id="ceres_escape", verification="continuous"),
    DoorEdge("ceres_flat_to_scientist", 0xE06B, 0xE021, "left", "right", policy_id="ceres_escape", verification="continuous"),
    DoorEdge("ceres_scientist_to_magnet", 0xE021, 0xDFD7, "left", "bottom_right", policy_id="ceres_escape", verification="continuous"),
    DoorEdge("ceres_magnet_to_falling", 0xDFD7, 0xDF8D, "upper_left", "right", policy_id="ceres_escape", verification="continuous"),
    DoorEdge("ceres_falling_to_elevator", 0xDF8D, 0xDF45, "left", "bottom", policy_id="ceres_escape", verification="continuous"),
    DoorEdge("ceres_to_landing", 0xDF45, 0x91F8, "elevator", "ship", policy_id="ceres_escape", verification="continuous"),
    DoorEdge("landing_to_parlor", 0x91F8, 0x92FD, "left", "right", policy_id="legacy_seed_adapter", verification="continuous"),
    DoorEdge("parlor_to_climb", 0x92FD, 0x96BA, "bottom_left", "top", policy_id="legacy_room_seed", verification="continuous"),
    DoorEdge("climb_to_pit", 0x96BA, 0x975C, "bottom", "left", policy_id="legacy_room_seed", verification="continuous"),
    DoorEdge("pit_to_elevator", 0x975C, 0x97B5, "right", "left", policy_id="legacy_room_seed", verification="continuous"),
    DoorEdge("elevator_to_morph", 0x97B5, 0x9E9F, "elevator", "right", policy_id="legacy_room_seed", verification="continuous"),
    DoorEdge("morph_to_construction", 0x9E9F, 0x9F11, "right", "left", frozenset({"morph_ball"}), "legacy_room_seed", "continuous"),
)

_MILESTONES = (
    ProgressionMilestone(
        "first_ceres_control",
        "First controllable Ceres frame",
        ProgressCondition(room_id=0xDF45, game_states=frozenset({8})),
        timeout_frames=12_000,
        policy_id="power_on_boot",
    ),
    ProgressionMilestone(
        "ridley_countdown",
        "Natural Ceres countdown",
        ProgressCondition(room_id=0xE0B5, game_states=frozenset({8})),
        timeout_frames=7_000,
        policy_id="ceres_ridley_wait",
    ),
    ProgressionMilestone(
        "zebes_landing",
        "Zebes Landing Site control",
        ProgressCondition(room_id=0x91F8, game_states=frozenset({8})),
        timeout_frames=8_000,
        policy_id="ceres_escape",
    ),
    ProgressionMilestone(
        "morph_ball",
        "Morph Ball collected naturally",
        ProgressCondition(room_id=0x9E9F, collected_items_mask=MORPH_BALL_MASK),
        acquires=frozenset({"morph_ball"}),
        timeout_frames=8_000,
        policy_id="legacy_room_seed",
    ),
)

START_TO_MORPH_GRAPH = RoomProgressionGraph(
    _ROOMS,
    _EDGES,
    _MILESTONES,
    graph_id="start_to_morph",
)

_EARLY_ROOMS = _ROOMS + (
    RoomNode(0x9F64, "Blue Brinstar Energy Tank Room", "Brinstar", frozenset({"item_room"})),
    RoomNode(0xA107, "First Missile Room", "Brinstar", frozenset({"item_room"})),
    RoomNode(0x9879, "Flyway", "Crateria"),
    RoomNode(0x9804, "Bomb Torizo Room", "Crateria", frozenset({"boss_item_room"})),
)

_EARLY_EDGES = _EDGES + (
    DoorEdge(
        "construction_to_first_missile",
        0x9F11,
        0xA107,
        "upper_right",
        "left",
        frozenset({"morph_ball"}),
        "two_missile_detour",
        "continuous",
    ),
    DoorEdge(
        "first_missile_to_construction",
        0xA107,
        0x9F11,
        "left",
        "upper_right",
        frozenset({"morph_ball"}),
        "two_missile_detour",
        "continuous",
    ),
    DoorEdge(
        "construction_to_blue_missile",
        0x9F11,
        0x9F64,
        "lower_left",
        "left",
        frozenset({"morph_ball"}),
        "two_missile_detour",
        "continuous",
    ),
    DoorEdge(
        "blue_missile_to_construction",
        0x9F64,
        0x9F11,
        "left",
        "lower_left",
        frozenset({"morph_ball"}),
        "two_missile_detour",
        "continuous",
    ),
    DoorEdge(
        "construction_to_morph",
        0x9F11,
        0x9E9F,
        "left",
        "right",
        frozenset({"morph_ball"}),
        "construction_return",
        "continuous",
    ),
    DoorEdge(
        "morph_to_elevator",
        0x9E9F,
        0x97B5,
        "right",
        "elevator",
        frozenset({"morph_ball"}),
        "morph_return",
        "continuous",
    ),
    DoorEdge(
        "elevator_to_pit_return",
        0x97B5,
        0x975C,
        "left",
        "right",
        frozenset({"morph_ball"}),
        "elevator_return",
        "continuous",
    ),
    DoorEdge(
        "pit_to_climb_return",
        0x975C,
        0x96BA,
        "left",
        "bottom",
        frozenset({"morph_ball"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
    DoorEdge(
        "climb_to_parlor_return",
        0x96BA,
        0x92FD,
        "top",
        "bottom_left",
        frozenset({"morph_ball"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
    DoorEdge(
        "parlor_to_flyway",
        0x92FD,
        0x9879,
        "right",
        "left",
        frozenset({"morph_ball"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
    DoorEdge(
        "flyway_to_torizo",
        0x9879,
        0x9804,
        "right",
        "left",
        frozenset({"missiles"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
    DoorEdge(
        "torizo_to_flyway",
        0x9804,
        0x9879,
        "left",
        "right",
        frozenset({"bombs", "bomb_torizo_defeated"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
    DoorEdge(
        "flyway_to_parlor_return",
        0x9879,
        0x92FD,
        "left",
        "right",
        frozenset({"bombs", "bomb_torizo_defeated"}),
        "pit_to_torizo_replay",
        "continuous",
    ),
)

_EARLY_MILESTONES = _MILESTONES + (
    ProgressionMilestone(
        "first_missiles",
        "First Missile expansion collected naturally",
        ProgressCondition(room_id=0xA107, minimum_ammo_capacities=(5, 0, 0)),
        requires=frozenset({"morph_ball"}),
        acquires=frozenset({"missiles"}),
        timeout_frames=2_000,
        policy_id="two_missile_detour",
    ),
    ProgressionMilestone(
        "blue_brinstar_missiles",
        "Blue Brinstar Missile expansion collected naturally",
        ProgressCondition(room_id=0x9F64, minimum_ammo_capacities=(10, 0, 0)),
        requires=frozenset({"morph_ball", "missiles"}),
        timeout_frames=3_000,
        policy_id="two_missile_detour",
    ),
    ProgressionMilestone(
        "bombs",
        "Morph Ball Bombs collected naturally",
        ProgressCondition(
            room_id=0x9804,
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
            room_id=0x9879,
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
    _EARLY_ROOMS,
    _EARLY_EDGES,
    _EARLY_MILESTONES,
    graph_id="start_to_bomb_torizo",
)

_SPORE_ROOMS = _EARLY_ROOMS + (
    RoomNode(0x990D, "Terminator Room", "Crateria"),
    RoomNode(0x99BD, "Green Pirates Shaft", "Crateria"),
    RoomNode(0x9969, "Lower Mushrooms", "Crateria"),
    RoomNode(0x9938, "Elevator To Green Brinstar", "Crateria", frozenset({"elevator"})),
    RoomNode(0x9AD9, "Green Brinstar Main Shaft", "Brinstar", frozenset({"vertical_shaft"})),
    RoomNode(0x9CB3, "Dachora Room", "Brinstar"),
    RoomNode(0x9D19, "Big Pink", "Brinstar", frozenset({"vertical_shaft"})),
    RoomNode(0x9D9C, "Spore Spawn Kihunter Room", "Brinstar"),
    RoomNode(0x9DC7, "Spore Spawn Room", "Brinstar", frozenset({"boss_room"})),
    RoomNode(0x9B5B, "Spore Spawn Super Room", "Brinstar", frozenset({"item_room"})),
)

_SPORE_EDGES = _EARLY_EDGES + (
    DoorEdge(
        "parlor_to_terminator",
        0x92FD,
        0x990D,
        "left",
        "right",
        frozenset({"bombs", "bomb_torizo_defeated"}),
        "post_torizo_controller",
        "continuous",
    ),
    DoorEdge(
        "terminator_to_green_pirates",
        0x990D,
        0x99BD,
        "left",
        "right",
        policy_id="post_torizo_controller",
        verification="continuous",
    ),
    DoorEdge(
        "green_pirates_to_lower_mushrooms",
        0x99BD,
        0x9969,
        "left",
        "right",
        policy_id="post_torizo_controller",
        verification="continuous",
    ),
    DoorEdge(
        "lower_mushrooms_to_green_elevator",
        0x9969,
        0x9938,
        "left",
        "right",
        policy_id="post_torizo_controller",
        verification="continuous",
    ),
    DoorEdge(
        "green_elevator_to_main_shaft",
        0x9938,
        0x9AD9,
        "down",
        "elevator",
        policy_id="post_torizo_controller",
        verification="continuous",
    ),
    DoorEdge(
        "main_shaft_to_dachora",
        0x9AD9,
        0x9CB3,
        "right",
        "left",
        frozenset({"missiles"}),
        "post_torizo_controller",
        "continuous",
    ),
    DoorEdge(
        "dachora_to_big_pink",
        0x9CB3,
        0x9D19,
        "right",
        "left",
        frozenset({"morph_ball", "bombs"}),
        "post_torizo_controller",
        "continuous",
    ),
    DoorEdge(
        "big_pink_to_spore_kihunters",
        0x9D19,
        0x9D9C,
        "right",
        "left",
        frozenset({"missiles"}),
        "post_torizo_controller",
        "continuous",
    ),
    DoorEdge(
        "spore_kihunters_to_spore_spawn",
        0x9D9C,
        0x9DC7,
        "up",
        "bottom",
        policy_id="post_torizo_controller",
        verification="continuous",
    ),
    DoorEdge(
        "spore_spawn_to_super_room",
        0x9DC7,
        0x9B5B,
        "right",
        "left",
        frozenset({"spore_spawn_defeated"}),
        "post_torizo_controller",
        "continuous",
    ),
)

_SPORE_MILESTONES = _EARLY_MILESTONES + (
    ProgressionMilestone(
        "spore_spawn_clear",
        "Spore Spawn defeated and post-boss room reached naturally",
        ProgressCondition(
            room_id=0x9B5B,
            collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK,
            minimum_ammo_capacities=(10, 0, 0),
        ),
        requires=frozenset({"morph_ball", "bombs", "missiles"}),
        acquires=frozenset({"spore_spawn_defeated"}),
        timeout_frames=40_000,
        policy_id="post_torizo_controller",
    ),
)

START_TO_SPORE_SPAWN_GRAPH = RoomProgressionGraph(
    _SPORE_ROOMS,
    _SPORE_EDGES,
    _SPORE_MILESTONES,
    graph_id="start_to_spore_spawn",
)
