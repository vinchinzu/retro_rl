"""Convert dungeon door-graph exits into adventure GraphEdge rows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Mapping

from retro_harness.adventure.graph import GraphEdge

from zelda_i.door_graph.core import DungeonDoorGraph, GateKind, RoomExit

NodeIdFn = Callable[[int], str]
CapForGate = Callable[[GateKind], Iterable[str]]
AcquiresFn = Callable[[RoomExit, int], Iterable[str]]

# Arrival-room items / TF bits used when notes mention them.
_ITEM_AT_ROOM: Mapping[tuple[int | None, int], str] = {
    (3, 0x0F): "raft",
    (4, 0x60): "stepladder",
    (4, 0x03): "triforce_shard_4",
    (5, 0x04): "whistle",
    (5, 0x14): "triforce_shard_5",
    (3, 0x3D): "triforce_shard_3",
}

_NOTE_ACQUIRES: tuple[tuple[str, str], ...] = (
    ("raft", "raft"),
    ("addr_raft", "raft"),
    ("stepladder", "stepladder"),
    ("addr_ladder", "stepladder"),
    ("whistle", "whistle"),
    ("recorder", "whistle"),
    ("tf 0x08", "triforce_shard_4"),
    ("tf 0x04", "triforce_shard_3"),
    ("tf 0x10", "triforce_shard_5"),
    ("triforce", "triforce"),
)


def default_node_id_fn(room: int) -> str:
    return f"room_{int(room):02x}"


def default_cap_for_gate(gate: GateKind) -> frozenset[str]:
    if gate is GateKind.KEY:
        return frozenset({"keys"})
    if gate is GateKind.BOMB:
        return frozenset({"bombs"})
    if gate is GateKind.KILL_CLEAR:
        return frozenset({"can_clear"})
    return frozenset()


def default_acquires(exit_: RoomExit, source_room: int) -> frozenset[str]:
    del source_room
    found: set[str] = set()
    notes = (exit_.notes or "").lower()
    for token, cap in _NOTE_ACQUIRES:
        if token in notes:
            found.add(cap)
    return frozenset(found)


def door_graph_to_edges(
    graph: DungeonDoorGraph,
    *,
    node_id_fn: NodeIdFn,
    cap_for_gate: CapForGate,
) -> tuple[GraphEdge, ...]:
    """Bind pathfinding exits to GraphEdge rows with requires / acquires."""
    edges: list[GraphEdge] = []
    for source, exits in graph.rooms.items():
        for exit_ in exits:
            if not exit_.is_pathfinding or exit_.target_room is None:
                continue
            requires = frozenset(cap_for_gate(exit_.gate))
            acquires = set(default_acquires(exit_, source))
            item = _ITEM_AT_ROOM.get((graph.level, int(exit_.target_room)))
            if item:
                acquires.add(item)
            src_id = node_id_fn(int(source))
            dst_id = node_id_fn(int(exit_.target_room))
            edges.append(
                GraphEdge(
                    source_id=src_id,
                    target_id=dst_id,
                    edge_id=f"{src_id}->{dst_id}:{exit_.direction.label}",
                    direction=exit_.direction.label,
                    requires=requires,
                    acquires=frozenset(acquires),
                    verification=exit_.verification,
                    provenance=graph.name or "door_graph",
                    meta={
                        "gate": exit_.gate.value,
                        "source_room": int(source),
                        "target_room": int(exit_.target_room),
                        "key_cost": exit_.key_cost,
                        "bomb_stand": exit_.bomb_stand,
                        "notes": exit_.notes,
                        "level": graph.level,
                    },
                )
            )
    return tuple(edges)
