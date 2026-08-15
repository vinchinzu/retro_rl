"""Bind door_graph exits onto adventure GraphEdge requires/acquires."""

from __future__ import annotations

from retro_harness.adventure.graph import GraphEdge

from zelda_i.door_graph import (
    LEVEL_4_DOOR_GRAPH,
    LEVEL_5_DOOR_GRAPH,
    L4_ENTRY,
    L4_TRIFORCE,
    L5_ENTRY,
    L5_TRIFORCE,
    default_cap_for_gate,
    default_node_id_fn,
    door_graph_to_edges,
)
from zelda_i.door_graph.core import GateKind


def _l4_node(room: int) -> str:
    return f"level4_room_{room:02x}"


def test_bind_emits_graph_edges_with_requires() -> None:
    edges = door_graph_to_edges(
        LEVEL_4_DOOR_GRAPH,
        node_id_fn=_l4_node,
        cap_for_gate=default_cap_for_gate,
    )
    assert edges
    assert all(isinstance(e, GraphEdge) for e in edges)
    key_edges = [e for e in edges if "keys" in (e.requires or frozenset())]
    bomb_edges = [e for e in edges if "bombs" in (e.requires or frozenset())]
    assert key_edges
    assert bomb_edges
    src = _l4_node(L4_ENTRY)
    assert any(e.source_id == src for e in edges)


def test_bind_acquires_tf_and_items() -> None:
    edges = door_graph_to_edges(
        LEVEL_4_DOOR_GRAPH,
        node_id_fn=_l4_node,
        cap_for_gate=default_cap_for_gate,
    )
    tf_edges = [e for e in edges if e.target_id == _l4_node(L4_TRIFORCE)]
    assert tf_edges
    assert "triforce_shard_4" in tf_edges[0].acquires
    ladder = [e for e in edges if "stepladder" in e.acquires]
    assert ladder


def test_bind_l5_tf_and_whistle() -> None:
    def node(room: int) -> str:
        return f"level5_room_{room:02x}"

    edges = door_graph_to_edges(
        LEVEL_5_DOOR_GRAPH,
        node_id_fn=node,
        cap_for_gate=default_cap_for_gate,
    )
    tf = [e for e in edges if e.target_id == node(L5_TRIFORCE)]
    assert tf and "triforce_shard_5" in tf[0].acquires
    whistle = [e for e in edges if "whistle" in e.acquires]
    assert whistle
    assert any(e.source_id == node(L5_ENTRY) for e in edges)


def test_bind_kill_clear_can_be_open() -> None:
    def open_clear(gate: GateKind) -> frozenset[str]:
        if gate is GateKind.KILL_CLEAR:
            return frozenset()
        return default_cap_for_gate(gate)

    edges = door_graph_to_edges(
        LEVEL_4_DOOR_GRAPH,
        node_id_fn=default_node_id_fn,
        cap_for_gate=open_clear,
    )
    kill = [e for e in edges if e.meta.get("gate") == GateKind.KILL_CLEAR.value]
    assert kill
    assert all(not e.requires for e in kill)
